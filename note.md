# HECRL 数据集生成

<br />

# ManipObj 数据收集说明

对应文件：

- `datasets/collect_manipobj.py`
- `datasets/collection_policies/pickplace_cube.py`
- `datasets/collection_policies/push_cube.py`

## 收集流程

1. 用 `ob_type='pixels', multiview=True` 创建 `ManipObjEnv`
2. `env.reset()` 后，从 `info['privileged/target_task']` 读当前子任务
3. 根据子任务选择 oracle：
   - `pickplace_cube` -> `PickPlaceCubeMarkovOracle`
   - `push_cube` -> `PushCubeMarkovOracle`
4. 每步执行：
   - `action = agent.select_action(ob, info)`
   - `next_ob, reward, terminated, truncated, info = env.step(action)`
   - 把当前步数据写进 `dataset`
5. 如果 `agent.done`，但 episode 还没结束，就调用 `env.unwrapped.set_new_target()` 切下一个子任务
6. 全部 episode 收完后，按 step 平铺切 train / val，保存成 `xxx.npz` 和 `xxx-val.npz`

## 两个 collection policy 的作用

### `PickPlaceCubeMarkovOracle`

- 负责抓取并放置方块
- 内部是分阶段状态机：接近物块 -> 抓取 -> 抬起 -> 移动到目标 -> 放下 -> 回到结束位
- 当它认为这轮 pick-place 完成后，`agent.done = True`

### `PushCubeMarkovOracle`

- 负责从某一侧推动方块
- reset 时随机选择从 `right/left/front/back` 哪一侧推
- 先走到侧边，再持续向物块推进
- 主要按内部步数上限结束，不保证一次 oracle 生命周期内一定把块推到最终 target

## collector 每步保存的字段

```python
dataset['observations'].append(ob)
dataset['actions'].append(action)
dataset['terminals'].append(done)
dataset['qpos'].append(info['prev_qpos'])
dataset['state_observations'].append(info['prev_state_clean_observation'])
```

这里要注意：

- 保存的是 `ob_t`，不是 `next_ob_t`
- 保存的 `qpos` 和 `state_observations` 对应动作执行前的状态

## dataset 字典每个 key 的 shape

下面先写单步 shape，再写整个 `.npz` 里的 shape。

### 单步

```python
{
    "observations": (2, 200, 200, 3),   # uint8，多视角 RGB
    "actions": (5,),                     # float32
    "terminals": (),                     # bool
    "qpos": (35,),                       # float32，3-cube 配置
    "state_observations": (21,),         # float32，3-cube 配置
}
```

说明：

- `observations` 的 `2` 来自两个相机视角
- `actions` 是 5 维：`dx, dy, dz, dyaw, gripper`
- `state_observations` 的维度公式是 `6 + 5 * num_objects`
- 3 个 cube 时 `state_observations = 21`，`qpos = 35`

### 整个 `.npz`

设总步数为 `T`，则：

```python
{
    "observations": (T, 2, 200, 200, 3),
    "actions": (T, 5),
    "terminals": (T,),
    "qpos": (T, 35),
    "state_observations": (T, 21),
}
```

## 后续生成新数据时要保持的约束

- `terminals` 必须准确标记每个 episode 的最后一步
- 最好保持定长 episode，因为训练侧默认按第一个 terminal 推出 episode length
- 继续保存 `qpos`，因为 supervised reward 计算会用到
- 继续保存 `state_observations`，这样 state-based 和 image-based 训练都能兼容
- 保持当前对齐方式：第 `t` 行保存的是 `s_t, a_t`，下一行的 `qpos` 对应 `s_{t+1}`

# HECRL 训练流程完整解析

本文档详细解析 HECRL 项目的整个训练流程，从训练开始到结束，梳理所有关键函数的调用关系。

***

## 目录

1. [训练启动](#训练启动)
2. [初始化阶段](#初始化阶段)
3. [训练循环](#训练循环)
4. [环境交互详解](#环境交互详解)
5. [Agent决策流程](#agent决策流程)
6. [模型更新流程](#模型更新流程)
7. [评估流程](#评估流程)
8. [完整调用链路](#完整调用链路)

***

## 训练启动

训练从主脚本开始，核心入口：

- `main()` - 程序入口
- `parse_cfg()` - 解析配置文件
- `make_env()` - 创建环境
- `make_agent()` - 创建 Agent
- `make_logger()` - 创建日志记录器
- `Trainer()` - 创建训练器
- `Trainer.train()` - 开始训练

***

## 初始化阶段

### 1. 环境创建

环境采用包装链结构：

```
TorchObsWrapper → TimeLimit → OrderEnforcing → PassiveEnvChecker → ManipObjEnv
```

**关键函数调用：**

- `make_env()` - 创建环境
- `gym.make()` - 创建基础环境
- `TorchObsWrapper()` - 包装为 Tensor 接口
- `ManipObjEnv.__init__()` - 初始化核心环境
  - 构建 MJCF 模型
  - 设置物理参数
  - 初始化逆运动学控制器
  - 定义动作空间

### 2. Agent 创建

- `make_agent()` - 根据配置创建 Agent
- `SGIQL.__init__()` / `HIQL.__init__()` / `IQL.__init__()` - 初始化 Agent
  - 创建 Actor 网络
  - 创建 Q 网络
  - 创建 V 网络
  - 创建 Goal Actor（SGIQL/HIQL）
  - 初始化优化器

### 3. Buffer 加载

- `load_dataset_to_buffer()` - 加载离线数据集
  - `load_dataset_from_file()` - 从文件加载
  - `preprocess_obs()` - 预处理观测
  - `Buffer()` - 创建 Buffer
  - `Buffer.load()` - 加载数据到 Buffer
  - `RMSNormalizer.set_stats()` - 设置归一化统计量

***

## 训练循环

核心训练循环在 `Trainer.train()` 中执行：

```python
for i in range(cfg.steps):
    # 1. 采样
    batch = buffer.sample()
    
    # 2. 更新
    train_metrics = agent.update(batch, i)
    
    # 3. 评估和保存
    if i % eval_freq == 0:
        eval_metrics, eval_videos = _eval()
        save_agent()
```

### 关键函数：

- `Trainer.train()` - 训练主循环
- `Buffer.sample()` - 从 Buffer 采样 batch
- `Agent.update()` - 更新网络参数
- `Trainer._eval()` - 定期评估
- `Logger.log()` - 记录日志
- `Logger.save_agent()` - 保存模型

***

## 环境交互详解

### Reset 流程

当调用 `env.reset()` 时：

**函数调用链：**

1. `Trainer._eval_ogbench()` → `env.reset(options=dict(task_id=task_id))`
2. `env.reset()` - OGBench 基类的 reset 方法
3. `build_mjcf_model()` - 构建 MJCF 模型（首次调用）
4. `compile_model_and_data()` - 编译模型和数据
5. `initialize_episode()` - 初始化 episode
   - `sample_obj_pos()` - 采样物体位置
   - `set_new_target()` - 设置目标
   - `compute_observation()` - 计算 goal 观测（保存为 `_cur_goal_ob`）
   - `compute_ob_info()` - 计算 goal 信息（保存为 `_cur_goal_ob_info`）
   - `pre_step()` - 保存初始状态的 prev\_ob\_info
6. `compute_observation()` - 计算初始观测（返回给 RL）
   - `compute_ob_info()` - 计算完整状态信息
   - `get_state_clean_observation()` - 提取 21 维 clean observation
7. `get_reset_info()` - 获取 reset 信息（返回给 RL）
   - `compute_ob_info()` - 重新计算状态信息
   - `get_state_clean_observation()` - 提取 state\_obs
   - 添加 `goal` 和 `state_goal` 到 info

**返回：** `(obs, info)`

- `obs`: (21,) 初始观测
- `info`: dict，包含 `goal`, `state_goal` 等

### Step 流程

当调用 `env.step(action)` 时：

**函数调用链：**

1. `set_control(action)` - 设置控制信号
2. `pre_step()` - 步前处理
   - `compute_ob_info()` - 计算并保存 prev\_ob\_info
3. `mujoco.mj_step()` - 物理仿真
4. `mujoco.mj_rnePostConstraint()` - 计算接触力
5. `post_step()` - 步后处理
   - `_compute_successes()` - 检查物体是否到达目标
   - 更新 `_success` 和 `_success_frac`
6. `terminate_episode()` - 检查是否终止
7. `truncate_episode()` - 检查是否超时
8. `compute_observation()` - 计算下一个观测
   - `compute_ob_info()` - 计算完整状态信息
   - `get_state_clean_observation()` - 提取 21 维 observation
9. `compute_reward()` - 计算奖励
   - `_compute_successes()` - 检查成功状态
   - 根据 reward\_mode 计算奖励（sparse/step/dense）
10. `get_step_info()` - 获取 step 信息
    - `compute_ob_info()` - 计算完整状态信息
    - `get_state_clean_observation()` - 提取 state\_obs
    - 添加 `success`, `success_frac`, `was_agent_object_interaction`

**返回：** `(next_obs, reward, terminated, truncated, info)`

***

## Agent决策流程

### 动作生成流程

在评估或数据收集时，Agent 需要生成动作：

**函数调用链：**

1. `Agent.act(TensorDict(obs=obs, goal=goal, step=step))`
2. `goal_pi(obs, goal)` - 高层策略生成 subgoal
   - 如果使用 diffusion：`SubgoalDiffuser.sample()`
   - 如果使用 Actor：`GoalActor.forward()`
3. `pi(obs, subgoal)` - 低层策略生成 action
   - `Actor.forward()` - 前向传播
   - `QNetwork.forward()` - 计算 Q 值（用于训练）
4. 返回 `action`

**关键函数：**

- `Agent.act()` - 动作生成入口
- `goal_pi()` - 高层 Goal Actor
- `pi()` - 低层 Actor
- `SubgoalDiffuser.sample()` - Diffusion 采样（可选）

***

## 模型更新流程

### SGIQL 更新流程

`Agent.update(batch, step)` 执行完整的更新：

**更新顺序：**

1. **更新 Value 网络**
   - `_update_v(batch)`
   - `vf(obs, goal)` - 计算当前 value
   - `qnet_target()` - 计算目标 Q 值
   - 计算 expectile loss
   - 反向传播和优化
2. **更新 Q 网络**
   - `_update_q(batch)`
   - `vf_detach(next_obs, goal)` - 计算 next value
   - Bellman 更新计算 target
   - `qnet(obs, action, goal)` - 计算预测 Q 值
   - 计算 MSE loss
   - 反向传播和优化
3. **更新 Actor（低层策略）**
   - `_update_pi(batch)`
   - `pi(obs, goal)` - 生成动作
   - `qnet(obs, pi_out, goal)` - 计算 Q 值
   - 计算 Q loss + BC loss
   - 反向传播和优化
4. **更新 Goal Actor（高层策略）**
   - `_update_goal_pi(batch)` 或 `_update_goal_pi_diffuser(batch)`
   - `goal_pi(obs, goal)` - 生成 subgoal
   - `vf_detach(obs, goal)` 和 `vf_detach(subgoal, goal)` - 计算 advantage
   - AWR loss 或 diffusion loss
   - 反向传播和优化
5. **更新目标网络**
   - `qnet_target_params.lerp_()` - 软更新 Q 网络目标参数

**关键函数：**

- `Agent.update()` - 更新入口
- `_update_v()` - Value 网络更新
- `_update_q()` - Q 网络更新
- `_update_pi()` - Actor 更新
- `_update_goal_pi()` - Goal Actor 更新
- `lerp_()` - 目标网络软更新

***

## 评估流程

### OGBench 评估协议

**函数调用链：**

1. `Trainer._eval()` - 评估入口
2. `Trainer._eval_ogbench()` - OGBench 特定评估
3. 对每个 task\_id (1-5)：
   - `env.reset(options=dict(task_id=task_id))`
     - 返回 `obs` 和 `goal_info`
   - 对每个 episode：
     - `buffer.normalizer.normalize(obs)` - 归一化观测
     - `agent.act(TensorDict(obs=obs, goal=goal, step=step))`
       - 返回 `action` 和 `act_info`
     - `env.step(action)`
       - 返回 `next_obs, reward, terminated, truncated, info`
     - `visualize_frame()` - 生成视频帧（如果需要）
     - 更新 `obs = next_obs`
   - `flatten(info)` - 扁平化 info
   - `add_to(cur_metrics, ...)` - 累加 metrics
4. 计算平均 metrics
5. 返回 `metrics` 和 `frames_list`

**关键函数：**

- `Trainer._eval_ogbench()` - 评估主函数
- `env.reset()` - 重置环境
- `agent.act()` - 生成动作
- `env.step()` - 执行动作
- `visualize_frame()` - 生成视频帧
- `flatten()` - 扁平化字典
- `add_to()` - 累加 metrics

***

## 完整调用链路

### 训练阶段

```
Trainer.train()
  │
  ├─ Buffer.sample()
  │     └─ 从 ReplayBuffer 采样 batch
  │
  └─ Agent.update(batch, step)
        │
        ├─ _update_v(batch)
        │     ├─ vf(obs, goal)
        │     ├─ qnet_target(obs, action, goal)
        │     └─ 反向传播
        │
        ├─ _update_q(batch)
        │     ├─ vf_detach(next_obs, goal)
        │     ├─ qnet(obs, action, goal)
        │     └─ 反向传播
        │
        ├─ _update_pi(batch)
        │     ├─ pi(obs, goal)
        │     ├─ qnet(obs, pi_out, goal)
        │     └─ 反向传播
        │
        ├─ _update_goal_pi(batch)
        │     ├─ goal_pi(obs, goal)
        │     ├─ vf_detach(obs, goal)
        │     ├─ vf_detach(subgoal, goal)
        │     └─ 反向传播
        │
        └─ qnet_target_params.lerp_()
```

### 评估阶段

```
Trainer._eval_ogbench()
  │
  ├─ for task_id in [1,2,3,4,5]:
  │     │
  │     ├─ env.reset(options)
  │     │     ├─ initialize_episode()
  │     │     │     ├─ sample_obj_pos()
  │     │     │     ├─ compute_observation() → 保存 _cur_goal_ob
  │     │     │     └─ pre_step() → 保存 _prev_ob_info
  │     │     ├─ compute_observation() → 返回 obs
  │     │     └─ get_reset_info() → 返回 info (含 goal)
  │     │
  │     └─ while not done:
  │           │
  │           ├─ agent.act(obs, goal)
  │           │     ├─ goal_pi(obs, goal) → subgoal
  │           │     └─ pi(obs, subgoal) → action
  │           │
  │           └─ env.step(action)
  │                 ├─ pre_step() → compute_ob_info()
  │                 ├─ mujoco.mj_step()
  │                 ├─ post_step() → _compute_successes()
  │                 ├─ compute_observation() → compute_ob_info()
  │                 ├─ compute_reward() → _compute_successes()
  │                 └─ get_step_info() → compute_ob_info()
  │
  └─ 返回 metrics 和 frames_list
```

***

## 关键数据结构

### Buffer 采样的 batch

```python
batch = {
    "obs": (batch_size, obs_dim),          # 当前观测
    "action": (batch_size, action_dim),     # 执行的动作
    "next_obs": (batch_size, obs_dim),      # 下一个观测
    "reward": (batch_size, 1),              # 奖励
    "done": (batch_size, 1),                # 终止标志
    "goal": (batch_size, goal_dim),         # 目标
    "value_goal": (batch_size, goal_dim),   # 用于 V/Q 更新的 goal
    "policy_goal": (batch_size, goal_dim),  # 用于 policy 更新的 goal
    "policy_subgoal": (batch_size, subgoal_dim),  # 用于 subgoal policy 更新
}
```

### Reset 返回的 info

```python
info = {
    # 完整状态信息
    "proprio/joint_pos": (6,),
    "proprio/effector_pos": (3,),
    "privileged/block_0_pos": (3,),
    # ... 其他状态
    
    # Clean observation
    "state_obs": (21,),
    
    # Goal 信息
    "goal": (21,),
    "state_goal": (21,),
    
    # 可选
    "goal_rendered": (128, 128, 3) or None,
}
```

### Step 返回的 info

```python
info = {
    # 完整状态信息
    "proprio/joint_pos": (6,),
    "proprio/effector_pos": (3,),
    "privileged/block_0_pos": (3,),
    # ... 其他状态
    
    # Clean observation
    "state_obs": (21,),
    
    # 评估指标
    "success": bool,
    "success_frac": float,
    "was_agent_object_interaction": bool,
}
```

***

## 总结

### 训练流程核心函数

1. **初始化**
   - `make_env()`, `make_agent()`, `load_dataset_to_buffer()`
2. **训练循环**
   - `Trainer.train()` → `Buffer.sample()` → `Agent.update()`
3. **环境交互**
   - `env.reset()` → `initialize_episode()` → `compute_observation()` → `get_reset_info()`
   - `env.step()` → `pre_step()` → `mj_step()` → `post_step()` → `compute_observation()` → `compute_reward()` → `get_step_info()`
4. **Agent 决策**
   - `agent.act()` → `goal_pi()` → `pi()`
5. **模型更新**
   - `_update_v()` → `_update_q()` → `_update_pi()` → `_update_goal_pi()`
6. **评估**
   - `Trainer._eval_ogbench()` → `env.reset()` → `agent.act()` → `env.step()` → `visualize_frame()`

### 关键设计

- **Goal-conditioned RL**: 所有网络都以 goal 为条件
- **Hierarchical Policy**: 高层生成 subgoal，低层生成 action
- **Offline Training**: 从预采集的数据集中学习
- **Regular Evaluation**: 定期在 5 个任务上评估性能

***

**文档生成时间**: 2026-03-14\
**基于代码版本**: HECRL 项目当前版本
