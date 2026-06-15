from copy import deepcopy

import numpy as np
import torch

import gymnasium as gym
from gymnasium.envs.registration import register

from visual_encoders import load_pretrained_rep_model, extract_dlp_features, extract_dlp_image, extract_vqvae_features


register(
    id='manipobj-v0',
    entry_point='envs.manip_obj.manip_obj_env:ManipObjEnv',
)


class TorchObsWrapper(gym.Wrapper):
    
	def __init__(self, env, cfg):
		super().__init__(env)
		
		self.cfg = cfg
		self.obs_mode = cfg.obs
		self.preprocess_info = {
        	"rep_model": load_pretrained_rep_model(cfg),
			"device": cfg.rep_model_device,
		}

		obs, info = self.env.reset()
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		
		obs = self.preprocess_obs(self._obs_to_tensor(obs), info=self.preprocess_info)
		obs = obs.numpy()
		
		if self.obs_mode == 'rgb':
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
		else:
			self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype)

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))
	
	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs
	
	def reset(self, *, seed=None, options=None):
		observation, info = self.env.reset(seed=seed, options=options)#<bound method TimeLimit.reset of <TimeLimit<OrderEnforcing<PassiveEnvChecker<ManipObjEnv<manipobj-v0>>>>>>
		#info['goal'].shape == observation.shape == (21,)
		#{'proprio/joint_pos': (6,), 'proprio/joint_vel': (6,), 'proprio/effector_pos': (3,), 'proprio/effector_yaw': (1,), 'proprio/gripper_opening': (1,), 'proprio/gripper_vel': (1,), 'proprio/gripper_contact': (1,), 'privileged/block_0_pos': (3,), 'privileged/block_0_quat': (4,), 'privileged/block_0_yaw': (1,), 'privileged/block_1_pos': (3,), 'privileged/block_1_quat': (4,), 'privileged/block_1_yaw': (1,), 'privileged/block_2_pos': (3,), 'privileged/block_2_quat': (4,), 'privileged/block_2_yaw': (1,), 'prev_qpos': (35,), 'prev_qvel': (32,), 'qpos': (35,), 'qvel': (32,), 'control': (7,), 'time': (1,), 'state_obs': (21,), 'goal': (21,), 'state_goal': (21,), 'goal_rendered': (128, 128, 3)}
		#import pdb; pdb.set_trace()
		# preprocess goal
		goal = info.get('goal')
	# 	array([-0.01111675,  2.67847993,  2.11276142, -0.33744247, -0.94134615,
    #     1.43117615,  0.80433965,  2.41439493,  0.19883642,  0.73030542,
    #    -0.68312078,  0.80433965,  2.41439493,  0.59874899, -0.25332417,
    #    -0.96738145,  0.80433965,  2.41439493,  0.99870528,  0.29830412,
    #    -0.95447087])
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_goal').copy())
		goal = self.preprocess_obs(self._obs_to_tensor(goal), info=self.preprocess_info)
		info["goal"] = goal
		# tensor([[-0.0111,  2.6785,  2.1128, -0.3374, -0.9413,  1.4312,  1.0000,  0.0000,
        #   0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        # [ 0.8043,  2.4144,  0.1988,  0.7303, -0.6831,  0.0000,  0.0000,  0.0000,
        #   0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000],
        # [ 0.8043,  2.4144,  0.5987, -0.2533, -0.9674,  0.0000,  0.0000,  0.0000,
        #   0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        # [ 0.8043,  2.4144,  0.9987,  0.2983, -0.9545,  0.0000,  0.0000,  0.0000,
        #   0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000]])

		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, info    
	
	def step(self, action):
		observation, reward, terminated, truncated, info = self.env.step(action.numpy())
		
		# preprocess observation
		self.preprocess_info["state_obs"] = self._obs_to_tensor(info.get('state_obs').copy())
		observation = self.preprocess_obs(self._obs_to_tensor(observation), info=self.preprocess_info)

		return observation, torch.tensor(reward, dtype=torch.float32), terminated, truncated, info
			
	def preprocess_obs(self, obs, info={}, batch=False):
		
		if not batch:
			obs = obs.unsqueeze(0)

		if self.obs_mode == 'state':
			pass

		elif self.obs_mode == 'rgb':
			obs = obs.permute(0, 3, 1, 2)

		elif self.obs_mode == 'ec_state':
			num_obs = obs.shape[0]
			agent_state_dim = self.env.unwrapped.agent_state_dim
			object_state_dim = self.env.unwrapped.object_state_dim

			agent_obs = obs[:, :agent_state_dim].unsqueeze(1)
			object_obs = obs[:, agent_state_dim:].reshape(num_obs, -1, object_state_dim)
			padded_object_obs = torch.cat([object_obs, torch.zeros(num_obs, object_obs.shape[1], agent_state_dim - object_state_dim)], dim=-1)
			entity_obs = torch.cat([agent_obs, padded_object_obs], dim=1)

			num_entities = entity_obs.shape[1]  # num_objects + agent
			id = torch.eye(num_entities).unsqueeze(0).expand(num_obs, -1, -1)
			
			obs = torch.cat([entity_obs, id], dim=-1)

		elif self.obs_mode == 'ec_state_gen':#走的这个
			num_obs = obs.shape[0]#[1,21]
			agent_state_dim = self.env.unwrapped.agent_state_dim #6
			object_state_dim = self.env.unwrapped.object_state_dim #5,这两个数字的含义看get_state_clean_observation
			agent_obs = obs[:, :agent_state_dim].unsqueeze(1)#obs的前6维
			object_obs = obs[:, agent_state_dim:].reshape(num_obs, -1, object_state_dim)#obs剩下的1×3×5维
			padded_object_obs = torch.cat([object_obs, torch.zeros(num_obs, object_obs.shape[1], agent_state_dim - object_state_dim)], dim=-1)#变成1×3×6
			entity_obs = torch.cat([agent_obs, padded_object_obs], dim=1)#拼一起，1×4×6

			num_entities = entity_obs.shape[1]  # num_objects + agent
			num_entity_ids = 9  # 8 objects + agent
			
			all_ids = torch.eye(num_entity_ids, device=obs.device, dtype=obs.dtype)
			perm = torch.randperm(num_entity_ids - 1)[:num_entities - 1] + 1
			ids = torch.cat([all_ids[0:1], all_ids[perm]], dim=0)  # (num_entities, num_entity_ids)
			id = ids.unsqueeze(0).expand(num_obs, -1, -1)  # (num_obs, num_entities, num_entity_ids)
			#(num_obs, num_entities, num_entity_ids)表示(batchsize,物体数量和代理状态的总和,代理和物体的id)
			id = info.get("id", id)

			obs = torch.cat([entity_obs, id], dim=-1)#拼一起，1×4×6的obs 和 1×4×9的id

		elif self.obs_mode == 'dlp':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			if obs.ndim == 5:  # multiview
				n_views = obs.shape[1]
				obs = torch.cat([extract_dlp_features(obs[:, i], info["rep_model"]) for i in range(n_views)], dim=1)
			else:
				obs = extract_dlp_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		elif self.obs_mode == 'vqvae':
			obs = obs.to(torch.device(info.get("device", 'cuda:0')))
			if obs.ndim == 5:  # multiview
				n_views = obs.shape[1]
				obs = torch.cat([extract_vqvae_features(obs[:, i], info["rep_model"]) for i in range(n_views)], dim=1)
			else:
				obs = extract_vqvae_features(obs, info["rep_model"])
			obs = obs.cpu()
		
		else:
			raise NotImplementedError

		if not batch:
			obs = obs.squeeze(0)
		return obs
	
	def render(self, info={}):
		frame = self.env.unwrapped.get_pixel_observation()
		
		if self.obs_mode == 'dlp':
			if frame.ndim == 4:  # multiview
				n_views = frame.shape[0]
				frame = np.stack([
					extract_dlp_image(np.transpose(frame[i], (2, 0, 1)), info["obs_mean"], info["obs_std"], self.preprocess_info["rep_model"], device=self.preprocess_info["device"])
					for i in range(n_views)], axis=0)
			else:
				frame = extract_dlp_image(np.transpose(frame, (2, 0, 1)), info["obs_mean"], info["obs_std"], self.preprocess_info["rep_model"], device=self.preprocess_info["device"])

		return frame
	

def make_env(cfg):
	print("先通过manip_obj的init的make_env制造环境")
	assert cfg.task == "manipobj-v0"
	
	ob_type = 'pixels' if cfg.obs in ['rgb', 'vqvae', 'dlp'] else 'states_clean'

	env = gym.make(cfg.task, ob_type=ob_type, multiview=cfg.multiview, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
	#gym.make会自动寻找env.observation_space,他在envs/manip_obj/manip_obj_env.py
	cfg.env_episode_length = env._max_episode_steps
	#env --> <TimeLimit<OrderEnforcing<PassiveEnvChecker<ManipObjEnv<manipobj-v0>>>>>
	env = TorchObsWrapper(env, cfg)

	cfg.num_objects = env.unwrapped.num_objects
	cfg.num_cubes = env.unwrapped.num_cubes

	return env
