from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as npformat
from PIL import Image
from tqdm import tqdm


data_dir = Path('./npz2format')
path_to_npz = Path('./datasets/data/stack-cube-noisy-v0-mv.npz')
path_to_npz_val = Path(str(path_to_npz).replace('.npz', '-val.npz'))
data_div = 3  # Set to 1 to keep the full dataset.


def open_npy_stream(npz_path, member_name):
    archive = zipfile.ZipFile(npz_path, 'r')
    stream = archive.open(member_name, 'r')
    version = npformat.read_magic(stream)
    shape, fortran_order, dtype = npformat._read_array_header(stream, version)
    if fortran_order:
        stream.close()
        archive.close()
        raise ValueError(f'Fortran-ordered arrays are not supported for member: {member_name}')
    return archive, stream, shape, np.dtype(dtype)


def read_exact(stream, nbytes):
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


def convert_split(npz_path, save_dir, data_div):
    print(f'Processing data from: {npz_path}')
    with np.load(npz_path) as np_dataset:
        terminals = np_dataset['terminals']
        actions = np_dataset['actions']

    print('Loaded terminals/actions')

    terminal_indices = np.flatnonzero(terminals)
    if len(terminal_indices) == 0:
        raise ValueError(f'No terminal flags found in dataset: {npz_path}')

    data_episode_length = int(terminal_indices[0]) + 1
    _, action_dim = actions.shape

    archive, stream, obs_shape, obs_dtype = open_npy_stream(npz_path, 'observations.npy')
    try:
        if len(obs_shape) == 5:
            total_steps, n_views, h, w, c = obs_shape
        elif len(obs_shape) == 4:
            total_steps, h, w, c = obs_shape
            n_views = 1
        else:
            raise ValueError(f'Unsupported observation shape: {obs_shape}')

        total_episodes = total_steps // data_episode_length
        kept_episodes = total_episodes // data_div
        if kept_episodes <= 0:
            raise ValueError(
                f'data_div={data_div} keeps zero episodes for dataset with {total_episodes} episodes.'
            )

        kept_steps = kept_episodes * data_episode_length
        frames_per_episode = data_episode_length * n_views
        bytes_per_episode = frames_per_episode * h * w * c * obs_dtype.itemsize

        print(f'Data Episode Length: {data_episode_length}')
        print(f'Number of Views: {n_views}')
        print(f'Total episodes: {total_episodes}')
        print(f'Keeping episodes: {kept_episodes}')
        print(f'Saving processed data in: {save_dir}')

        actions = actions[:kept_steps].reshape(kept_episodes, data_episode_length, action_dim)
        save_dir.mkdir(parents=True, exist_ok=True)

        for ep in tqdm(range(kept_episodes)):
            ep_dir = save_dir / str(ep)
            ep_dir.mkdir(parents=True, exist_ok=True)

            np.savez(ep_dir / 'actions.npz', actions=actions[ep, :-1])

            raw = read_exact(stream, bytes_per_episode)
            images = np.frombuffer(raw, dtype=obs_dtype).reshape(frames_per_episode, h, w, c)
            for i in range(frames_per_episode):
                Image.fromarray(images[i]).save(ep_dir / f'{i}.png')
    finally:
        stream.close()
        archive.close()


def main():
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    if not path_to_npz.exists():
        raise FileNotFoundError(f'Train dataset not found: {path_to_npz}')

    convert_split(path_to_npz, train_dir, data_div)

    if path_to_npz_val.exists():
        convert_split(path_to_npz_val, valid_dir, data_div)
    else:
        print(f'Warning: validation dataset not found, skipping valid split: {path_to_npz_val}')


if __name__ == '__main__':
    main()
