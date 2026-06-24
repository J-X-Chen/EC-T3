import os
from pathlib import Path
import zipfile
from copy import deepcopy

import numpy as np
from numpy.lib import format as npformat
import torch
from tensordict.tensordict import TensorDict
from tqdm import trange, tqdm

from common.buffer import Buffer


def get_filename(cfg, validation):
    filename = f'{cfg.train_dataset_name}'
    
    if cfg.obs in ['dlp', 'vqvae']:
        filename = filename + '-' + f'{cfg.obs}'
    
    if validation:
        filename = filename + '-' + 'val'
    
    return filename


def _read_npy_header(stream):
    version = npformat.read_magic(stream)
    if version == (1, 0):
        return npformat.read_array_header_1_0(stream)
    if version in [(2, 0), (3, 0)]:
        return npformat.read_array_header_2_0(stream)
    raise ValueError(f'Unsupported npy format version: {version}')


def _open_npz_array_stream(npz_path, key):
    archive = zipfile.ZipFile(npz_path, 'r')
    member_name = f'{key}.npy'
    if member_name not in archive.namelist():
        archive.close()
        raise KeyError(f'Could not find {member_name} in {npz_path}')

    stream = archive.open(member_name, 'r')
    shape, fortran_order, dtype = _read_npy_header(stream)
    if fortran_order:
        stream.close()
        archive.close()
        raise ValueError(f'Fortran-ordered arrays are not supported for {member_name}')
    return archive, stream, shape, np.dtype(dtype)


def _read_exact(stream, nbytes):
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if len(data) != nbytes:
        raise EOFError(f'Expected {nbytes} bytes but received {len(data)} bytes.')
    return data


def _iter_npz_observation_chunks(npz_path, chunk_size):
    archive, stream, shape, dtype = _open_npz_array_stream(npz_path, 'observations')
    row_shape = shape[1:]
    row_bytes = int(np.prod(row_shape)) * dtype.itemsize
    try:
        for start in range(0, shape[0], chunk_size):
            rows = min(chunk_size, shape[0] - start)
            raw = _read_exact(stream, rows * row_bytes)
            batch = np.frombuffer(raw, dtype=dtype).reshape((rows, *row_shape)).copy()
            yield start, batch
    finally:
        stream.close()
        archive.close()


def load_dataset_from_file(dataset_path, cfg):
    '''
    dataset = {
        "observations": (3000000, 21)
        "actions": (3000000, 5)
        "terminals": (3000000,) #终止标志，1 表示该 step 为 episode 结束，0 表示未结束。
        "qpos": (3000000, 35) #完整系统广义坐标（机器人关节 + 物体位姿）
        "state_observations": (3000000, 21) #(dataset['state_observations']==dataset['observations']).all() --> np.True_
    }
    '''
    import os; cjx_debug = os.environ.get('cjx_debug', ''); cjx_debug.lower() == 'true' and print("\033[93m开始下载数据==========\033[0m")
    file = np.load(dataset_path)
    
    dataset = dict()

    for k in file.keys():
        if k == 'observations':
            if cfg.obs in ['state', 'ec_state', 'ec_state_gen']:
                file_key = 'state_observations'
                dtype = np.float32
            else:
                file_key = 'observations'
                if 'dlp' in dataset_path or 'vqvae' in dataset_path:
                    dtype = np.float32
                else:
                    dtype = np.uint8
        else:
            file_key = k
            if 'image' in k:
                dtype = np.uint8
            else:
                dtype = np.float32
        dataset[k] = file[file_key][...].astype(dtype, copy=False)
        dataset[k] = dataset[k][:dataset[k].shape[0] // cfg.train_dataset_div]  # NOTE: optional reduction of dataset size
    #import pdb; pdb.set_trace()
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)#这个操作只是让倒数第二个0变为1(提前结束，避免跨 episode transition)
    #array([0., 0., 0., ..., 0., 0., 1.], shape=(3000000,), dtype=float32) --> array([0., 0., 0., ..., 0., 1., 1.], shape=(3000000,), dtype=float32)
    return dataset


def get_episode_ids_from_diagnostics(np_dataset):
    """Return contiguous episode ids and per-episode lengths when diagnostics are available."""
    if 'diagnostics_episode_index' not in np_dataset:
        return None, None

    raw_episode_ids = np_dataset['diagnostics_episode_index'].astype(np.int64, copy=False)
    if raw_episode_ids.size == 0:
        return None, None

    boundaries = np.flatnonzero(raw_episode_ids[1:] != raw_episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [raw_episode_ids.size]))
    episode_lengths = (ends - starts).astype(np.int64, copy=False)
    episode_ids = np.repeat(np.arange(episode_lengths.size, dtype=np.int64), episode_lengths)
    return episode_ids, episode_lengths


def preprocess_and_save_dataset(cfg, env, save_path, validation=False):

    split_suffix = '-val' if validation else ''
    fp = os.path.join(cfg.data_dir, f'{cfg.train_dataset_name.replace(f"-{cfg.obs}", "")}{split_suffix}.npz')
    print(f"Loading dataset from {fp}...")
    np_dataset = np.load(fp)
    preprocess_info = deepcopy(env.preprocess_info)

    save_path = Path(save_path)
    tmp_obs_path = save_path.with_name(save_path.name + '.observations.tmp.npy')
    tmp_save_path = save_path.with_name(save_path.name + '.tmp')

    archive, stream, obs_shape, _ = _open_npz_array_stream(fp, 'observations')
    stream.close()
    archive.close()

    total_steps = obs_shape[0]
    chunk_size = int(cfg.get('preprocess_chunk_size', 128))
    if chunk_size <= 0:
        raise ValueError('preprocess_chunk_size must be positive.')

    observations = None
    try:
        progress = tqdm(
            _iter_npz_observation_chunks(fp, chunk_size),
            total=(total_steps + chunk_size - 1) // chunk_size,
            desc=f'Encoding {cfg.obs} observations',
        )
        for start, batch_obs_np in progress:
            batch_obs = torch.from_numpy(batch_obs_np)
            batch_obs = env.preprocess_obs(batch_obs, info=preprocess_info, batch=True)
            batch_obs = batch_obs.numpy().astype(np.float32, copy=False)

            if observations is None:
                observations = np.lib.format.open_memmap(
                    tmp_obs_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=(total_steps, *batch_obs.shape[1:]),
                )
            observations[start:start + batch_obs.shape[0]] = batch_obs

        observations.flush()
        del observations
        observations = np.load(tmp_obs_path, mmap_mode='r')

        saved_dict = {'observations': observations}
        for key in np_dataset:
            if key != 'observations':
                saved_dict[key] = np_dataset[key].copy()

        for key in saved_dict:
            print(key, saved_dict[key].shape)

        with open(tmp_save_path, 'wb') as f:
            np.savez(f, **saved_dict)
        os.replace(tmp_save_path, save_path)
    finally:
        if observations is not None:
            del observations
        if tmp_obs_path.exists():
            tmp_obs_path.unlink()
        if tmp_save_path.exists():
            tmp_save_path.unlink()


def load_dataset_to_buffer(cfg, env, validation=False):
    """Load dataset to buffer for offline training."""
    # Load numpy dataset
    fp = os.path.join(cfg.data_dir, f'{get_filename(cfg, validation)}.npz')
    if not os.path.exists(fp):
        print(f"Dataset file {fp} not found, converting dataset images to {cfg.obs} observations...")
        preprocess_and_save_dataset(cfg, env, fp, validation=validation)
        print(f"Processed dataset saved to: {fp}")
    np_dataset = load_dataset_from_file(fp, cfg)
    
    # Get dataset attributes
    capacity = np_dataset["terminals"].size
    episode_ids, episode_lengths = get_episode_ids_from_diagnostics(np_dataset)
    if episode_lengths is not None:
        if episode_ids.size != capacity:
            raise ValueError(
                f'episode id length mismatch: got {episode_ids.size}, expected {capacity}'
            )
        min_episode_length = int(episode_lengths.min())
        if min_episode_length < 2:
            raise ValueError('All episodes must contain at least two observations.')
        cfg.data_episode_length = min_episode_length - 1
        if not np.all(episode_lengths == episode_lengths[0]):
            print(
                f"Variable-length episodes detected: {episode_lengths.size} episodes, "
                f"min={episode_lengths.min()}, max={episode_lengths.max()}, "
                f"mean={episode_lengths.mean():.1f}. "
                f"Sampling fixed windows of length {cfg.data_episode_length}."
            )
    else:
        cfg.data_episode_length = int(np.nonzero(np_dataset["terminals"])[0][0]) + 1
    
    # Convert dataset to torch tensordict
    obs = torch.from_numpy(np_dataset["observations"])#torch.Size([3000000, 21])
    #import pdb; pdb.set_trace()
    if cfg.obs in ['state', 'ec_state', 'ec_state_gen', 'rgb']:
        # Process observations in case they have not been pre-processed into a dataset
        observations = []
        print(f"Preprocessing observations...")
        if episode_lengths is not None:
            episode_ends = np.cumsum(episode_lengths)
            episode_starts = episode_ends - episode_lengths
            chunks = zip(episode_starts, episode_ends)
            iterator = tqdm(chunks, total=episode_lengths.size)
        else:
            chunk_size = cfg.data_episode_length + 1  # NOTE: for ec_state_gen it is important for the chunk_size to cover exactly one episode for the consistency of object IDs
            iterator = ((i, min(i + chunk_size, len(obs))) for i in range(0, len(obs), chunk_size))
            iterator = tqdm(iterator, total=(len(obs) + chunk_size - 1) // chunk_size)
        for start, end in iterator:
            batch_obs = obs[start:end]#torch.Size([1000, 21])
            batch_obs = env.preprocess_obs(batch_obs, batch=True)#<bound method TorchObsWrapper.preprocess_obs of <TorchObsWrapper<TimeLimit<OrderEnforcing<PassiveEnvChecker<ManipObjEnv<manipobj-v0>>>>>>>
            #变为batch_obs.shape --> torch.Size([1000, 1+3, 6+9])
            observations.append(batch_obs.numpy())
        obs = np.concatenate(observations, axis=0)#obs.shape ---> (3000000, 4, 15)

    dataset = TensorDict({
            "obs": obs,#(3000000, 4, 15)
            "action": torch.from_numpy(np_dataset["actions"]),#(3000000, 5)
        }, batch_size=capacity)#3000000
    if episode_ids is not None:
        dataset["episode"] = torch.from_numpy(episode_ids)
    
    # Add information for reward calculation to tensordict
    if cfg.reward != 'unsupervised':
        if 'manipobj' in cfg.task:
            dataset["qpos"] = torch.from_numpy(np_dataset["qpos"])#(3000000, 35)
        if 'scene' in cfg.task:
            dataset["qpos"] = torch.from_numpy(np_dataset["qpos"])
            dataset["button_states"] = torch.from_numpy(np_dataset["button_states"])
        elif 'pushtetris' in cfg.task:
            dataset["image_bitmasks"] = torch.from_numpy(np_dataset["image_bitmasks"])
            
    # Create buffer for sampling
    buffer = Buffer(cfg, env, capacity)
    buffer.load(dataset)#<common.buffer.Buffer object at 0x7072a0902740>
    
    expected_episodes = episode_lengths.size if episode_lengths is not None else capacity // (cfg.data_episode_length + 1)
    if buffer.num_eps != expected_episodes:
        print(f'WARNING: buffer has {buffer.num_eps} episodes, expected {expected_episodes} episodes for {cfg.task} task.')
    return buffer

    # (Pdb) buffer._buffer
    # ReplayBuffer(
    #     storage=LazyTensorStorage(
    #         data=TensorDict(
    #             fields={
    #                 action: Tensor(shape=torch.Size([3000000, 5]), device=cuda:0, dtype=torch.float32, is_shared=True),
    #                 episode: Tensor(shape=torch.Size([3000000]), device=cuda:0, dtype=torch.int64, is_shared=True),
    #                 obs: Tensor(shape=torch.Size([3000000, 4, 15]), device=cuda:0, dtype=torch.float32, is_shared=True),
    #                 qpos: Tensor(shape=torch.Size([3000000, 35]), device=cuda:0, dtype=torch.float32, is_shared=True)},
    #             batch_size=torch.Size([3000000]),
    #             device=cuda:0,
    #             is_shared=True), 
    #         shape=torch.Size([3000000]), 
    #         len=3000000, 
    #         max_size=3000000), 
    #     sampler=SliceSampler(num_slices=512, slice_len=None, end_key=('next', 'done'), traj_key=episode, truncated_key=None, strict_length=True), 
    #     writer=RoundRobinWriter(cursor=0, full_storage=True), 
    #     batch_size=511488, 
    #     collate_fn=<function _collate_id at 0x7075577ec1f0>)
