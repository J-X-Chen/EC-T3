import os
import re
import warnings

os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"

warnings.filterwarnings('ignore')

import torch

try:
    torch.compiler.config.recompile_limit = 32
except AttributeError:
    torch._dynamo.config.recompile_limit = 32

import hydra
from termcolor import colored

from envs import make_env
from agents import make_agent
from common.trainer import Trainer
from common.utils.logger import Logger
from common.utils.misc import set_seed
from common.utils.parser import parse_cfg

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def _parse_cube_split(value):
    if not isinstance(value, str):
        return None
    match = re.search(r'cube(\d+)\+(\d+)', value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _recommended_episode_steps(total_cubes):
    mapping = {
        1: 500,
        2: 800,
        3: 1000,
        4: 1200,
        5: 1500,
        6: 2000,
    }
    if total_cubes in mapping:
        return mapping[total_cubes]
    return 2000 + max(total_cubes - 6, 0) * 400


def _recommended_num_subgoals(total_cubes):
    mapping = {
        1: 16,
        2: 28,
        3: 32,
        4: 40,
        5: 52,
        6: 72,
    }
    if total_cubes in mapping:
        return mapping[total_cubes]
    return max(32, 12 * total_cubes)


def _recommended_train_steps(total_cubes):
    return 1_501_000 + 500_000 * max(total_cubes, 1)


def _configure_cubexy_cfg(cfg):
    mode = cfg.env_kwargs.get('mode', '')
    mode_split = _parse_cube_split(mode)
    dataset_split = _parse_cube_split(cfg.train_dataset_name)

    split = mode_split or dataset_split
    if split is None:
        raise ValueError(
            'train_cubexy.py expects either env_kwargs.manipobj.mode=task_cubeX+Y '
            'or train_dataset_name containing cubeX+Y.'
        )
    if mode_split is not None and dataset_split is not None and mode_split != dataset_split:
        raise ValueError(
            f'Mismatch between mode "{mode}" and dataset "{cfg.train_dataset_name}".'
        )

    ppp_count, stack_count = split
    total_cubes = ppp_count + stack_count
    if total_cubes <= 0:
        raise ValueError(f'Hybrid split cube{ppp_count}+{stack_count} has no objects.')

    cfg.env_kwargs['mode'] = f'task_cube{ppp_count}+{stack_count}'

    default_dataset_name = 'ppp-cube-noisy-v0-mv'
    expected_dataset_name = f'cube{ppp_count}+{stack_count}-noisy-v0-mv'
    if cfg.train_dataset_name == default_dataset_name:
        cfg.train_dataset_name = expected_dataset_name

    num_cubes = int(cfg.env_kwargs.get('num_cubes', 0))
    if num_cubes in (0, total_cubes):
        cfg.env_kwargs['num_cubes'] = total_cubes
    else:
        raise ValueError(
            f'Mode cube{ppp_count}+{stack_count} expects num_cubes={total_cubes}, got {num_cubes}.'
        )

    if cfg.obs == 'ec_state_gen' and total_cubes > 8:
        raise ValueError('ec_state_gen supports at most 8 objects (plus the agent).')

    if cfg.max_episode_steps == 1000:
        cfg.max_episode_steps = _recommended_episode_steps(total_cubes)

    if cfg.steps == 10_000_000:
        cfg.steps = _recommended_train_steps(total_cubes)

    if cfg.wandb_name_suffix == "":
        cfg.wandb_name_suffix = f'-cube{ppp_count}+{stack_count}'

    if cfg.agent == 'sgiql':
        if cfg.sgiql.get('alpha', 0.1) == 0.1 and total_cubes > 3:
            cfg.sgiql['alpha'] = 0.05
            cfg.alpha = 0.05
        if cfg.sgiql.get('num_subgoals', 32) == 32 and total_cubes != 3:
            cfg.sgiql['num_subgoals'] = _recommended_num_subgoals(total_cubes)
            cfg.num_subgoals = cfg.sgiql['num_subgoals']
    elif cfg.agent == 'hiql':
        if cfg.hiql.get('alpha', 0.1) == 0.1 and total_cubes > 3:
            cfg.hiql['alpha'] = 0.2
            cfg.alpha = 0.2
        if cfg.hiql.get('num_subgoals', 32) == 32 and total_cubes != 3:
            cfg.hiql['num_subgoals'] = _recommended_num_subgoals(total_cubes)
            cfg.num_subgoals = cfg.hiql['num_subgoals']
    elif cfg.agent == 'iql':
        if cfg.iql.get('alpha', 0.1) == 0.1 and total_cubes > 3:
            cfg.iql['alpha'] = 0.05
            cfg.alpha = 0.05

    return ppp_count, stack_count, total_cubes


class CubeXYTrainer(Trainer):
    def train(self):
        print(f'Training agent for {self.cfg.steps} iterations...')
        best_success_rate = float('-inf')

        for step in range(self.cfg.steps):
            batch = self.buffer.sample()

            torch.compiler.cudagraph_mark_step_begin()
            train_metrics = self.agent.update(batch, step)

            if step % self.cfg.eval_freq == 0 or step % 10_000 == 0:
                train_metrics.update(self._time_metrics(step))
                self.logger.log(train_metrics, 'train')

                if step % self.cfg.eval_freq == 0:
                    eval_metrics, eval_videos = self._eval()
                    eval_metrics.update(self._time_metrics(step))
                    self.logger.log(eval_metrics, 'eval', eval_videos)

                    if step > 0:
                        self.logger.save_agent(self.agent, self.buffer, identifier='latest')
                        success = eval_metrics.get('overall_success', float('-inf'))
                        if success > best_success_rate:
                            self.logger.save_agent(self.agent, self.buffer, identifier='best')
                            best_success_rate = success

        self.logger.finish(self.agent, self.buffer)


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
    assert torch.cuda.is_available()
    assert cfg.task == 'manipobj-v0', 'train_cubexy.py only supports manipobj-v0.'
    assert cfg.steps > 0, 'Must train for at least 1 step.'

    cfg = parse_cfg(cfg)
    ppp_count, stack_count, total_cubes = _configure_cubexy_cfg(cfg)
    set_seed(cfg.seed)

    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
    print(
        colored('Hybrid task:', 'yellow', attrs=['bold']),
        f'cube{ppp_count}+{stack_count} ({total_cubes} cubes)',
    )
    print(colored('Dataset:', 'yellow', attrs=['bold']), cfg.train_dataset_name)
    print(colored('Mode:', 'yellow', attrs=['bold']), cfg.env_kwargs['mode'])

    trainer = CubeXYTrainer(
        cfg=cfg,
        env=make_env(cfg),
        agent=make_agent(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
