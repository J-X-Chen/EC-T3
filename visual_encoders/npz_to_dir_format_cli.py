import argparse
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm
from numpy.lib import format as npformat


def open_npy_stream(npz_path, member_name):
    archive = zipfile.ZipFile(npz_path, 'r')
    stream = archive.open(member_name, 'r')
    version = npformat.read_magic(stream)
    shape, fortran_order, dtype = npformat._read_array_header(stream, version)
    if fortran_order:
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


def convert_split(npz_path, save_dir, data_div, max_steps_per_episode=None):
    print(f'Processing data from: {npz_path}')
    np_dataset = np.load(npz_path)
    terminals = np_dataset['terminals']
    actions = np_dataset['actions']
    print('Loaded terminals/actions')

    data_episode_length = int(np.nonzero(terminals)[0][0]) + 1
    if max_steps_per_episode is not None:
        if max_steps_per_episode <= 0:
            raise ValueError('max_steps_per_episode must be a positive integer.')
        steps_per_episode = min(data_episode_length, max_steps_per_episode)
    else:
        steps_per_episode = data_episode_length
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
        kept_steps = kept_episodes * data_episode_length
        if kept_episodes <= 0:
            raise ValueError(
                f'data_div={data_div} keeps zero episodes for dataset with {total_episodes} episodes.'
            )

        print(f'Data Episode Length: {data_episode_length}')
        print(f'Kept Steps per Episode: {steps_per_episode}')
        print(f'Number of Views: {n_views}')
        print(f'Total episodes: {total_episodes}')
        print(f'Keeping episodes: {kept_episodes}')
        print(f'Saving processed data in: {save_dir}')

        actions = actions[:kept_steps].reshape(kept_episodes, data_episode_length, action_dim)
        frames_per_episode = steps_per_episode * n_views
        full_frames_per_episode = data_episode_length * n_views
        values_per_episode = frames_per_episode * h * w * c
        bytes_per_episode = values_per_episode * obs_dtype.itemsize
        bytes_to_skip = (full_frames_per_episode - frames_per_episode) * h * w * c * obs_dtype.itemsize

        save_dir.mkdir(parents=True, exist_ok=True)
        for ep in tqdm(range(kept_episodes)):
            ep_dir = save_dir / str(ep)
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.savez(ep_dir / 'actions.npz', actions=actions[ep, :max(steps_per_episode - 1, 0)])

            raw = read_exact(stream, bytes_per_episode)
            images = np.frombuffer(raw, dtype=obs_dtype).reshape(frames_per_episode, h, w, c)
            for i in range(frames_per_episode):
                Image.fromarray(images[i]).save(ep_dir / f'{i}.png')
            if bytes_to_skip > 0:
                read_exact(stream, bytes_to_skip)
    finally:
        stream.close()
        archive.close()


def main():
    parser = argparse.ArgumentParser(description='Convert HECRL npz datasets to image-directory format.')
    parser.add_argument('--train-npz', required=True, help='Path to the train .npz dataset.')
    parser.add_argument('--val-npz', default=None, help='Path to the val .npz dataset. Defaults to "<train>-val.npz".')
    parser.add_argument('--data-dir', required=True, help='Output directory that will contain train/ and valid/.')
    parser.add_argument('--data-div', type=int, default=3, help='Optional dataset subsampling factor.')
    parser.add_argument(
        '--max-steps-per-episode',
        type=int,
        default=None,
        help='Optional truncation for a fast smoke test; keeps only the first N steps of each episode.',
    )
    parser.add_argument('--force', action='store_true', help='Rebuild train/valid even if they already exist.')
    args = parser.parse_args()

    train_npz = Path(args.train_npz).expanduser().resolve()
    val_npz = Path(args.val_npz).expanduser().resolve() if args.val_npz else Path(
        str(train_npz).replace('.npz', '-val.npz')
    )
    data_dir = Path(args.data_dir).expanduser().resolve()
    train_dir = data_dir / 'train'
    valid_dir = data_dir / 'valid'

    if not train_npz.exists():
        raise FileNotFoundError(f'Train dataset not found: {train_npz}')
    if not val_npz.exists():
        raise FileNotFoundError(f'Validation dataset not found: {val_npz}')
    if args.data_div <= 0:
        raise ValueError('--data-div must be a positive integer.')

    if args.force:
        for split_dir in [train_dir, valid_dir]:
            if split_dir.exists():
                for path in sorted(split_dir.glob('*'), reverse=True):
                    if path.is_dir():
                        for child in sorted(path.glob('*'), reverse=True):
                            child.unlink()
                        path.rmdir()
                    else:
                        path.unlink()

    convert_split(train_npz, train_dir, args.data_div, args.max_steps_per_episode)
    convert_split(val_npz, valid_dir, args.data_div, args.max_steps_per_episode)


if __name__ == '__main__':
    main()
