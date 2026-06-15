#!/bin/bash
# Tableware 0+n modes use explicit generated asset stack order.

#### PPP-Cube ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection env_kwargs.manipobj.num_cubes=3 multiview=true dataset_name=ppp-cube-noisy-v0-mv dataset_type=noisy num_collect_episodes=7500 max_episode_steps=400

#### Stack-Cube ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_stack env_kwargs.manipobj.num_cubes=3 multiview=true dataset_name=stack-cube-noisy-v0-mv dataset_type=noisy p_random_action=0.02 num_collect_episodes=3000 max_episode_steps=1000 

#### Cube2+3 (2 fixed PPP targets + 3 stack) ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_cube2+3 env_kwargs.manipobj.num_cubes=5 multiview=true dataset_name=cube2+3-noisy-v0-mv dataset_type=noisy p_random_action=0.02 num_collect_episodes=5000 max_episode_steps=1000 use_checkpoint=True checkpoint_interval=500

# #### Tableware-Bowl (1 bowl, reserved naming for future tableware family) ####
# python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_tableware1+0 env_kwargs.manipobj.object_type=tableware env_kwargs.manipobj.local_object_name=bowl multiview=true dataset_name=tableware1+0-noisy-v0-mv dataset_type=noisy p_random_action=0.0 action_noise=0.0 num_collect_episodes=3000 max_episode_steps=700 use_checkpoint=False

# #### Tableware 2+0 (2 pick-place subtasks; objects are generated bowl + generated plate) ####
# python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_tableware2+0 env_kwargs.manipobj.object_type=tableware env_kwargs.manipobj.local_object_name=bowl multiview=true dataset_name=tableware2+0-noisy-v0-mv dataset_type=noisy p_random_action=0.0 action_noise=0.0 num_collect_episodes=3000 max_episode_steps=700 use_checkpoint=False

#### Tableware 0+2 (generated plate bottom + generated bowl top) ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_tableware0+2 env_kwargs.manipobj.object_type=tableware env_kwargs.manipobj.local_object_name=bowl multiview=true dataset_name=tableware0+2-noisy-v0-mv dataset_type=noisy p_random_action=0.0 action_noise=0.0 num_collect_episodes=3000 max_episode_steps=700 use_checkpoint=False

#### Tableware 1+2 (generated mug pick-place + generated plate bottom/generated bowl top stack) ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_tableware1+2 env_kwargs.manipobj.object_type=tableware env_kwargs.manipobj.local_object_name=bowl multiview=true dataset_name=tableware1+2-noisy-v0-mv dataset_type=noisy p_random_action=0.0 action_noise=0.0 num_collect_episodes=3000 max_episode_steps=700 use_checkpoint=False

#### Tableware 3+2 (generated mug/fork/knife pick-place + generated plate bottom/generated bowl top stack) ####
python collect.py task=manipobj-v0 env_kwargs.manipobj.mode=data_collection_tableware3+2 env_kwargs.manipobj.object_type=tableware env_kwargs.manipobj.local_object_name=bowl env_kwargs.manipobj.tableware_ppp_min_offset=0.05 env_kwargs.manipobj.tableware_ppp_max_offset=0.28 multiview=true dataset_name=tableware3+2-success-v0-mv dataset_type=noisy p_random_action=0.0 action_noise=0.0 num_collect_episodes=3000 max_episode_steps=700 save_success_only=true use_checkpoint=True checkpoint_interval=100

#### Scene ####
python collect.py task=visual-scene-v0 env_kwargs.scene.mode=data_collection env_kwargs.scene.num_cubes=1 multiview=false dataset_name=visual-scene-noisy-v0 dataset_type=noisy num_collect_episodes=1000 max_episode_steps=1000

#### PushTetris ####
python collect.py task=pushtetris-v0 env_kwargs.pushtetris.mode=data_collection env_kwargs.pushtetris.num_objects=3 dataset_name=push-tetris-constrained_random-v0 dataset_type=noisy num_collect_episodes=2500 max_episode_steps=400
