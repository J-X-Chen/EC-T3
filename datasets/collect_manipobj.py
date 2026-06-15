from collections import defaultdict
import glob
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import gymnasium as gym
from gymnasium.envs.registration import register
import imageio.v2 as imageio
import numpy as np
from tqdm import trange

from moviepy import ImageSequenceClip

import ogbench.manipspace  # noqa

from datasets.collection_policies.pickplace_cube import PickPlaceCubeMarkovOracle
from datasets.collection_policies.pickplace_tableware import PickPlaceTablewareMarkovOracle
from datasets.collection_policies.push_cube import PushCubeMarkovOracle


register(
    id='manipobj-v0',
    entry_point='envs.manip_obj.manip_obj_env:ManipObjEnv',
)

TABLEWARE_STACK_SUCCESS_DIST = 0.035
TABLEWARE_PICKPLACE_SUCCESS_DIST = 0.03
TABLEWARE_MIN_UP_Z = 0.5


def get_dataset_dtype(key):
    if key == 'observations':
        return np.uint8
    if key == 'terminals':
        return bool
    return np.float32


def save_npz(path, data, compressed=True, label='file'):
    t0 = time.time()
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)
    dt = time.time() - t0
    mode = 'compressed' if compressed else 'uncompressed'
    print(f'Saved {label} ({mode}) to {path} in {dt:.1f}s')


def get_checkpoint_dir(cfg):
    checkpoint_dir = os.path.join(cfg.data_dir, f'{cfg.dataset_name}_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def parse_checkpoint_shard_name(path):
    basename = os.path.basename(path)
    match = re.search(r'^checkpoint_ep(\d+)_(\d+)\.npz$', basename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_checkpoint_shards(checkpoint_dir):
    shard_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_ep*_*.npz'))
    shard_files = [p for p in shard_files if parse_checkpoint_shard_name(p) is not None]
    shard_files.sort(key=lambda p: (os.path.getmtime(p), p))
    return shard_files


def count_completed_episodes(shard_files):
    total_episodes = 0
    for path in shard_files:
        parsed = parse_checkpoint_shard_name(path)
        if parsed is not None:
            total_episodes += parsed[0]
    return total_episodes


def build_ep_size_counters(shard_files):
    counters = {}
    for path in shard_files:
        parsed = parse_checkpoint_shard_name(path)
        if parsed is None:
            continue
        episodes_in_shard, shard_idx = parsed
        counters[episodes_in_shard] = max(counters.get(episodes_in_shard, 0), shard_idx + 1)
    return counters


def format_gib(num_bytes):
    return num_bytes / (1024 ** 3)


def get_available_memory_bytes():
    """Best-effort query for currently available system RAM in bytes."""
    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        available_pages = os.sysconf('SC_AVPHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return int(page_size * available_pages)


def estimate_uncompressed_shard_bytes(shard_files):
    """Estimate total uncompressed bytes stored across .npz shards."""
    total_uncompressed_bytes = 0
    for shard_path in shard_files:
        try:
            with zipfile.ZipFile(shard_path, 'r') as zf:
                total_uncompressed_bytes += sum(info.file_size for info in zf.infolist())
        except zipfile.BadZipFile:
            print(f'Warning: failed to inspect shard (invalid zip): {shard_path}')
    return int(total_uncompressed_bytes)


def should_skip_full_export(cfg, shard_files):
    """
    Decide whether to skip final full export to avoid OOM.

    The final export path loads all shards, concatenates arrays, and then
    materializes train/val splits, which can require several multiples of the
    raw uncompressed shard bytes at peak.
    """
    if not cfg.get('auto_skip_large_export', True):
        return False

    estimated_raw_bytes = estimate_uncompressed_shard_bytes(shard_files)
    if estimated_raw_bytes <= 0:
        return False

    peak_multiplier = float(cfg.get('export_peak_mem_multiplier', 2.5))
    estimated_peak_bytes = int(estimated_raw_bytes * peak_multiplier)
    print(
        f'Export memory check: raw shard bytes ~{format_gib(estimated_raw_bytes):.2f} GiB, '
        f'estimated peak ~{format_gib(estimated_peak_bytes):.2f} GiB '
        f'(multiplier={peak_multiplier:.2f}).'
    )

    max_raw_gib = cfg.get('max_full_export_raw_gib', None)
    if max_raw_gib is not None and float(max_raw_gib) > 0:
        raw_limit_bytes = int(float(max_raw_gib) * (1024 ** 3))
        if estimated_raw_bytes > raw_limit_bytes:
            print(
                f'Skipping full export: raw shard bytes exceed max_full_export_raw_gib '
                f'({format_gib(estimated_raw_bytes):.2f} > {float(max_raw_gib):.2f} GiB).'
            )
            return True

    available_memory_bytes = get_available_memory_bytes()
    if available_memory_bytes is None:
        print('Export memory check: available RAM unknown, continuing with full export.')
        return False

    print(f'Export memory check: available RAM ~{format_gib(available_memory_bytes):.2f} GiB.')
    if estimated_peak_bytes > available_memory_bytes:
        print(
            'Skipping full export to avoid OOM. '
            'Checkpoint shards are kept for later conversion.'
        )
        print('Override with auto_skip_large_export=false if you want to force full export.')
        return True
    return False


def run_lowmem_export(cfg, checkpoint_dir, num_train_episodes):
    script_path = os.path.join(os.path.dirname(__file__), 'merge_shards_lowmem.py')
    train_path = os.path.join(cfg.data_dir, f'{cfg.dataset_name}.npz')
    val_path = train_path.replace('.npz', '-val.npz')

    cmd = [
        sys.executable,
        script_path,
        '--checkpoint-dir', checkpoint_dir,
        '--train-path', train_path,
        '--val-path', val_path,
        '--num-train-episodes', str(int(num_train_episodes)),
        '--chunk-mib', str(int(cfg.get('lowmem_chunk_mib', 32))),
    ]
    if not cfg.get('dataset_compressed', True):
        cmd.append('--uncompressed')

    print('Running low-memory exporter:')
    print(' '.join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f'Low-memory export failed with exit code {completed.returncode}')
    print(f'Low-memory export finished:\n{train_path}\n{val_path}')


def save_checkpoint_shard(cfg, dataset, checkpoint_dir, episodes_in_shard, ep_size_counters):
    if episodes_in_shard <= 0:
        return None

    shard_data = {}
    for key, values in dataset.items():
        if len(values) == 0:
            continue
        shard_data[f'data_{key}'] = np.asarray(values, dtype=get_dataset_dtype(key))

    if not shard_data:
        return None

    shard_idx = ep_size_counters.get(episodes_in_shard, 0)
    ep_size_counters[episodes_in_shard] = shard_idx + 1
    shard_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episodes_in_shard}_{shard_idx}.npz')
    save_npz(
        shard_path,
        shard_data,
        compressed=cfg.get('checkpoint_compressed', True),
        label=f'checkpoint shard ep{episodes_in_shard}_{shard_idx}',
    )
    return shard_path


def load_full_dataset_from_shards(shard_files):
    arrays_by_key = defaultdict(list)
    total_shards = len(shard_files)

    for idx, shard_path in enumerate(shard_files):
        if total_shards <= 10 or idx % 10 == 0 or idx == total_shards - 1:
            print(f'Loading shard {idx + 1}/{total_shards}: {shard_path}')
        with np.load(shard_path) as shard_data:
            for file_key in shard_data.files:
                if file_key.startswith('data_'):
                    key = file_key[5:]
                    arrays_by_key[key].append(shard_data[file_key])

    full_dataset = {}
    for key in sorted(arrays_by_key.keys()):
        dtype = get_dataset_dtype(key)
        arrays = [np.asarray(arr, dtype=dtype) for arr in arrays_by_key[key]]
        full_dataset[key] = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return full_dataset


def extend_dataset(dst, src):
    for key, values in src.items():
        if len(values) > 0:
            dst[key].extend(values)


def compute_train_steps_from_terminals(terminals, num_train_episodes):
    terminals = np.asarray(terminals, dtype=bool)
    if terminals.size == 0 or num_train_episodes <= 0:
        return 0

    done_indices = np.flatnonzero(terminals)
    if done_indices.size < num_train_episodes:
        return int(terminals.size)
    return int(done_indices[num_train_episodes - 1]) + 1


def save_first_episode_video(frames, mode, target_frame=None, fps=15):
    """Save the first collected episode video to ./data/videos."""
    if len(frames) == 0:
        print('No frames collected for first episode, skip video save.')
        return None

    # Use repository-relative path so Hydra run directories do not change output location.
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'videos'))
    os.makedirs(save_dir, exist_ok=True)

    task_name = get_task_name_from_mode(mode)
    save_path = os.path.join(save_dir, f'collect_video_first_episode_{task_name}.mp4')

    if target_frame is None:
        frames_array = np.array([flatten_multiview_frame(frame) for frame in frames])
    else:
        frames_array = np.array([build_four_panel_video_frame(frame, target_frame) for frame in frames])
    clip = ImageSequenceClip([frames_array[i] for i in range(len(frames_array))], fps=fps)
    clip.write_videofile(save_path)
    print(f'First episode video saved to: {save_path}')
    return save_path


def flatten_multiview_frame(frame):
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = np.concatenate([frame[i] for i in range(len(frame))], axis=1)
    return frame


def get_task_name_from_mode(mode):
    if mode == 'data_collection_stack':
        return 'stackcube'
    elif 'tableware' in mode:
        match = re.search(r'tableware\d+\+\d+', mode)
        return match.group(0) if match is not None else 'tableware'
    elif 'cube' in mode and '+' in mode:
        match = re.search(r'cube\d+\+\d+', mode)
        return match.group(0) if match is not None else 'hybridcube'
    else:
        return 'pppcube'


def save_first_episode_target_image(target_frame, mode):
    """Save one debug image: multiview first frame with projected target positions."""
    if target_frame is None:
        print('No target frame captured for first episode, skip target image save.')
        return None

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'videos'))
    os.makedirs(save_dir, exist_ok=True)
    task_name = get_task_name_from_mode(mode)
    save_path = os.path.join(save_dir, f'collect_target_first_episode_{task_name}.png')

    out = flatten_multiview_frame(target_frame)
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    imageio.imwrite(save_path, out)
    print(f'First episode target image saved to: {save_path}')
    return save_path


def split_multiview_frame(frame):
    """Split a frame into per-camera views while preserving original pixels."""
    arr = np.asarray(frame)
    if arr.ndim == 4:
        return [np.array(arr[i], copy=True) for i in range(arr.shape[0])]
    if arr.ndim == 3 and arr.shape[1] >= int(arr.shape[0] * 1.7):
        # Handle already-concatenated two-view image.
        w_mid = arr.shape[1] // 2
        return [np.array(arr[:, :w_mid], copy=True), np.array(arr[:, w_mid:], copy=True)]
    return [np.array(arr, copy=True)]


def _ensure_uint8_image(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _resize_image_nn(img, out_h, out_w):
    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img
    ys = np.linspace(0, max(in_h - 1, 0), out_h).astype(np.int32)
    xs = np.linspace(0, max(in_w - 1, 0), out_w).astype(np.int32)
    return img[np.ix_(ys, xs)]


def _pick_two_views(frame):
    views = split_multiview_frame(frame)
    if len(views) == 0:
        return None, None
    if len(views) == 1:
        left = _ensure_uint8_image(views[0])
        return left, np.array(left, copy=True)
    return _ensure_uint8_image(views[0]), _ensure_uint8_image(views[1])


def build_four_panel_video_frame(live_frame, target_frame):
    """
    Compose one frame:
    - top-left: live front view
    - top-right: live side view
    - bottom-left: static target front view
    - bottom-right: static target side view
    """
    live_left, live_right = _pick_two_views(live_frame)
    if live_left is None or live_right is None:
        return flatten_multiview_frame(live_frame)

    tgt_left, tgt_right = _pick_two_views(target_frame)
    if tgt_left is None or tgt_right is None:
        tgt_left, tgt_right = np.array(live_left, copy=True), np.array(live_right, copy=True)

    panel_h = max(live_left.shape[0], live_right.shape[0])
    panel_w = max(live_left.shape[1], live_right.shape[1])

    live_left = _resize_image_nn(live_left, panel_h, panel_w)
    live_right = _resize_image_nn(live_right, panel_h, panel_w)
    tgt_left = _resize_image_nn(tgt_left, panel_h, panel_w)
    tgt_right = _resize_image_nn(tgt_right, panel_h, panel_w)

    top = np.concatenate([live_left, live_right], axis=1)
    bottom = np.concatenate([tgt_left, tgt_right], axis=1)
    out = np.concatenate([top, bottom], axis=0)
    return out


def _draw_disk(img, cx, cy, radius, color):
    h, w = img.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h - 1, cy + radius)
    r2 = radius * radius
    for y in range(y0, y1 + 1):
        dy = y - cy
        rem = r2 - dy * dy
        if rem < 0:
            continue
        dx = int(rem ** 0.5)
        x0 = max(0, cx - dx)
        x1 = min(w - 1, cx + dx)
        img[y, x0:x1 + 1] = color


def _draw_x(img, cx, cy, size, color):
    h, w = img.shape[:2]
    for d in range(-size, size + 1):
        x1, y1 = cx + d, cy + d
        x2, y2 = cx + d, cy - d
        if 0 <= x1 < w and 0 <= y1 < h:
            img[y1, x1] = color
        if 0 <= x2 < w and 0 <= y2 < h:
            img[y2, x2] = color


def _draw_plus(img, cx, cy, size, color):
    h, w = img.shape[:2]
    for d in range(-size, size + 1):
        x, y = cx + d, cy
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = color
        x, y = cx, cy + d
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = color


def _draw_line(img, x0, y0, x1, y1, color):
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    h, w = img.shape[:2]
    if steps <= 1:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = color
        return
    xs = np.round(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.round(np.linspace(y0, y1, steps)).astype(np.int32)
    for x, y in zip(xs, ys):
        if 0 <= x < w and 0 <= y < h:
            img[y, x] = color


def _get_multiview_camera_names(unwrapped, num_views):
    if num_views <= 1:
        return ['front_pixels']

    uses_stack_side = (
        getattr(unwrapped, '_mode', None) in ['task_stack', 'task_stack_all', 'data_collection_stack']
        or bool(getattr(unwrapped, '_is_hybrid_mode', False))
    )
    cam_names = ['front_pixels', 'side_pixels_stack' if uses_stack_side else 'side_pixels']

    if num_views > 2:
        existing = {name for name in cam_names}
        for cam_id in range(int(unwrapped._model.ncam)):
            cam_name = unwrapped._model.cam(cam_id).name
            if cam_name not in existing:
                cam_names.append(cam_name)
                existing.add(cam_name)
            if len(cam_names) >= num_views:
                break
    return cam_names[:num_views]


def project_world_to_pixel(unwrapped, camera_name, world_xyz, image_height, image_width):
    """Project a 3D world point to 2D pixel coordinates for a named MuJoCo camera."""
    if world_xyz is None:
        return None

    point = np.asarray(world_xyz, dtype=np.float64).reshape(-1)
    if point.size < 3:
        return None
    point = point[:3]

    try:
        cam_id = int(unwrapped._model.camera(camera_name).id)
    except Exception:
        return None

    cam_pos = np.asarray(unwrapped._data.cam_xpos[cam_id], dtype=np.float64)
    cam_rot = np.asarray(unwrapped._data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)

    # MuJoCo camera convention: points in front have negative z in camera coordinates.
    cam_xyz = cam_rot.T @ (point - cam_pos)
    z = float(cam_xyz[2])
    if z >= -1e-6:
        return None

    h = max(int(image_height), 1)
    w = max(int(image_width), 1)
    fovy = np.deg2rad(float(unwrapped._model.cam_fovy[cam_id]))
    if fovy <= 1e-8:
        return None
    fovx = 2.0 * np.arctan(np.tan(fovy / 2.0) * (float(w) / float(h)))

    x_ndc = (cam_xyz[0] / (-z)) / np.tan(fovx / 2.0)
    y_ndc = (cam_xyz[1] / (-z)) / np.tan(fovy / 2.0)
    px = int(round((x_ndc + 1.0) * 0.5 * (w - 1)))
    py = int(round((1.0 - y_ndc) * 0.5 * (h - 1)))

    if px < 0 or px >= w or py < 0 or py >= h:
        return None
    return px, py


def get_overlay_push_plan(env, info):
    """Recompute the active push plan so the debug overlay matches oracle logic."""
    unwrapped = env.unwrapped
    target_block = int(info.get('privileged/target_block', -1))
    if target_block < 0:
        return None

    mode = getattr(unwrapped, '_mode', '').lower()
    object_type = getattr(unwrapped, '_object_type', 'cube').lower()
    is_tableware = ('tableware' in mode) or (object_type != 'cube')
    if not is_tableware:
        return None

    try:
        overlay_oracle = PickPlaceTablewareMarkovOracle(env=env, min_norm=0.4)
        overlay_oracle.reset(None, info)
        if not overlay_oracle._is_bottom_push_target(info):
            return None
        return overlay_oracle._build_push_plan(info)
    except Exception:
        return None


def get_overlay_start_world_pos(info, object_idx, default_pos, push_plan=None):
    """Return the contact point visualized on the target overlay."""
    if default_pos is None:
        return None

    target_block = int(info.get('privileged/target_block', -1))
    if object_idx != target_block or push_plan is None:
        return np.asarray(default_pos, dtype=np.float32)

    overlay_pos = np.asarray(default_pos, dtype=np.float32).copy()
    overlay_pos[:2] = np.asarray(push_plan['contact_xy'], dtype=np.float32)
    overlay_pos[2] = float(default_pos[2])
    return overlay_pos


def get_overlay_center_world_pos(info, object_idx, default_pos, push_plan=None):
    """Return the midpoint used by the oracle for active tableware push targets."""
    if default_pos is None:
        return None

    target_block = int(info.get('privileged/target_block', -1))
    if object_idx != target_block or push_plan is None:
        return None

    center_pos = np.asarray(default_pos, dtype=np.float32).copy()
    center_pos[:2] = np.asarray(push_plan.get('push_center_xy', default_pos[:2]), dtype=np.float32)
    center_pos[2] = float(default_pos[2])
    return center_pos


def get_multiview_object_pixel_centroids(env, cam_names):
    """Estimate each object's visual center from segmentation masks."""
    unwrapped = env.unwrapped
    geom_ids_list = getattr(unwrapped, '_object_geom_ids_list', None)
    if geom_ids_list is None or len(geom_ids_list) == 0:
        return None

    try:
        seg = np.asarray(unwrapped.render(camera=cam_names, segmentation=True))
    except Exception:
        return None

    if seg.ndim == 3:
        seg = seg[None, ...]
    if seg.ndim != 4 or seg.shape[-1] < 1:
        return None

    centroids = {}
    geom_id_sets = [set(int(gid) for gid in geom_ids) for geom_ids in geom_ids_list]
    for obj_idx, geom_id_set in enumerate(geom_id_sets):
        obj_centroids = []
        for view_idx in range(seg.shape[0]):
            geom_id_img = seg[view_idx, ..., 0]
            mask = np.isin(geom_id_img, list(geom_id_set))
            if not np.any(mask):
                obj_centroids.append(None)
                continue
            ys, xs = np.nonzero(mask)
            obj_centroids.append((int(round(xs.mean())), int(round(ys.mean()))))
        centroids[obj_idx] = obj_centroids
    return centroids


def build_multiview_target_overlay(env, info, frame):
    """
    Build one debug image:
    - background: first multiview frame
    - overlay: per-object start point -> target point in each view
    """
    if frame is None:
        return None

    unwrapped = env.unwrapped
    views = split_multiview_frame(frame)
    if len(views) == 0:
        return None

    cam_names = _get_multiview_camera_names(unwrapped, len(views))
    object_pixel_centroids = get_multiview_object_pixel_centroids(env, cam_names)
    overlay_push_plan = get_overlay_push_plan(env, info)

    num_objects = int(getattr(unwrapped, 'num_objects', getattr(unwrapped, 'num_cubes', 0)))
    colors = getattr(unwrapped, '_object_colors', None)
    if colors is None or len(colors) == 0:
        colors = np.array(
            [
                [0.96, 0.26, 0.33, 1.0],
                [0.06, 0.74, 0.21, 1.0],
                [0.35, 0.55, 0.91, 1.0],
                [1.00, 0.69, 0.21, 1.0],
            ],
            dtype=np.float32,
        )

    target_mocap_ids = list(getattr(unwrapped, '_object_target_mocap_ids', []))
    mocap_pos = getattr(getattr(unwrapped, '_data', None), 'mocap_pos', None)
    active_target_idx = int(info.get('privileged/target_block', -1))
    active_target_pos = info.get('privileged/target_block_pos', None)

    for i in range(num_objects):
        obj_pos = info.get(f'privileged/block_{i}_pos', None)
        overlay_obj_pos = get_overlay_start_world_pos(info, i, obj_pos, push_plan=overlay_push_plan)
        overlay_center_pos = get_overlay_center_world_pos(info, i, obj_pos, push_plan=overlay_push_plan)
        tar_pos = None
        if mocap_pos is not None and i < len(target_mocap_ids):
            maybe_tar = np.asarray(mocap_pos[target_mocap_ids[i]], dtype=np.float32)
            if maybe_tar.shape[0] >= 3 and maybe_tar[2] > -0.2:
                tar_pos = maybe_tar
        if tar_pos is None and i == active_target_idx and active_target_pos is not None:
            tar_pos = np.asarray(active_target_pos, dtype=np.float32)
        if tar_pos is None:
            continue

        color = (np.clip(np.asarray(colors[i % len(colors)][:3]), 0.0, 1.0) * 255).astype(np.uint8)
        marker_color = np.minimum(color.astype(np.int16) + 50, 255).astype(np.uint8)
        emphasis_color = np.array([255, 230, 60], dtype=np.uint8) if i == active_target_idx else marker_color

        for view_idx, (view, cam_name) in enumerate(zip(views, cam_names)):
            h, w = view.shape[:2]
            visual_center_px = None
            if overlay_center_pos is not None:
                visual_center_px = project_world_to_pixel(unwrapped, cam_name, overlay_center_pos, h, w)
            if visual_center_px is None and object_pixel_centroids is not None:
                obj_view_centroids = object_pixel_centroids.get(i, None)
                if obj_view_centroids is not None and view_idx < len(obj_view_centroids):
                    visual_center_px = obj_view_centroids[view_idx]
            oracle_contact_px = None
            if overlay_obj_pos is not None:
                oracle_contact_px = project_world_to_pixel(unwrapped, cam_name, overlay_obj_pos, h, w)
            tar_px = project_world_to_pixel(unwrapped, cam_name, tar_pos, h, w)

            line_start_px = oracle_contact_px if oracle_contact_px is not None else visual_center_px
            if line_start_px is not None and tar_px is not None:
                _draw_line(view, line_start_px[0], line_start_px[1], tar_px[0], tar_px[1], color=marker_color)
            if visual_center_px is not None:
                _draw_disk(view, visual_center_px[0], visual_center_px[1], radius=2, color=color)
            if oracle_contact_px is not None and oracle_contact_px != visual_center_px:
                _draw_plus(view, oracle_contact_px[0], oracle_contact_px[1], size=3, color=np.array([255, 255, 255], dtype=np.uint8))
            if tar_px is not None:
                _draw_x(view, tar_px[0], tar_px[1], size=4 if i == active_target_idx else 3, color=emphasis_color)

    return views[0] if len(views) == 1 else np.concatenate(views, axis=1)


def check_task_success(info, env, target_block=None, target_group=None):
    """Check whether the just-finished subtask is successful."""
    tableware_success = check_tableware_task_success(info, env, target_block=target_block, target_group=target_group)
    if tableware_success is not None:
        return tableware_success

    if target_block is not None and hasattr(env.unwrapped, '_compute_successes'):
        try:
            successes = env.unwrapped._compute_successes()
            target_block = int(target_block)
            if 0 <= target_block < len(successes):
                return bool(successes[target_block])
        except Exception:
            pass

    if 'success' in info:
        return bool(info['success'])

    # Fallback to environment success computation if step info has no success key.
    if hasattr(env.unwrapped, '_compute_successes') and 'privileged/target_block' in info:
        successes = env.unwrapped._compute_successes()
        target_block = int(info['privileged/target_block'])
        if 0 <= target_block < len(successes):
            return bool(successes[target_block])

    return False


def _runtime_block_pos(info, env, block_idx):
    if block_idx < 0:
        return np.full(3, np.nan, dtype=np.float32)
    try:
        return np.asarray(env.unwrapped._data.joint(f'object_joint_{block_idx}').qpos[:3], dtype=np.float32).copy()
    except Exception:
        return _to_vec(info.get(f'privileged/block_{block_idx}_pos'), 3)


def _runtime_target_pos(info, env, block_idx):
    if block_idx < 0:
        return np.full(3, np.nan, dtype=np.float32)
    try:
        target_mocap_ids = getattr(env.unwrapped, '_object_target_mocap_ids', [])
        if block_idx < len(target_mocap_ids):
            return np.asarray(env.unwrapped._data.mocap_pos[target_mocap_ids[block_idx]], dtype=np.float32).copy()
    except Exception:
        pass
    if block_idx == _to_int_scalar(info.get('privileged/target_block', -1), default=-1):
        return _to_vec(info.get('privileged/target_block_pos'), 3)
    return np.full(3, np.nan, dtype=np.float32)


def _runtime_block_up_z(env, block_idx):
    try:
        quat = np.asarray(env.unwrapped._data.joint(f'object_joint_{block_idx}').qpos[3:7], dtype=np.float64)
    except Exception:
        return np.nan
    if quat.size < 4:
        return np.nan
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.nan
    quat = quat / norm
    _, x, y, _ = quat
    return float(1.0 - 2.0 * (x * x + y * y))


def _is_tableware_like_env(env):
    unwrapped = env.unwrapped
    mode = getattr(unwrapped, '_mode', '').lower()
    object_type = getattr(unwrapped, '_object_type', 'cube').lower()
    return ('tableware' in mode) or (object_type != 'cube')


def check_tableware_task_success(info, env, target_block=None, target_group=None):
    if not _is_tableware_like_env(env):
        return None

    target_block = _to_int_scalar(
        info.get('privileged/target_block', -1) if target_block is None else target_block,
        default=-1,
    )
    num_objects = int(getattr(env.unwrapped, 'num_objects', getattr(env.unwrapped, 'num_cubes', 0)))
    if target_block < 0 or target_block >= num_objects:
        return False

    block_pos = _runtime_block_pos(info, env, target_block)
    target_pos = _runtime_target_pos(info, env, target_block)
    dist, xy_dist, z_abs_dist = _compute_position_distances(block_pos, target_pos)
    if not np.isfinite(dist):
        return False

    if target_group is None:
        target_group = info.get('privileged/target_task_group', getattr(env.unwrapped, '_target_task_group', ''))
    target_group = str(target_group).lower()
    if target_group == 'stack':
        pos_success = dist <= TABLEWARE_STACK_SUCCESS_DIST
    else:
        pos_success = dist <= TABLEWARE_PICKPLACE_SUCCESS_DIST

    # A tableware center can be close to the target while the object is visibly tipped.
    up_z = _runtime_block_up_z(env, target_block)
    upright_success = (not np.isfinite(up_z)) or up_z >= TABLEWARE_MIN_UP_Z

    return bool(pos_success and upright_success)


TASK_NAME_CODES = {
    'pickplace_cube': 1,
    'stack_cube': 2,
    'push_cube': 3,
    'pickplace_t': 4,
    'push_t': 5,
}

TASK_GROUP_CODES = {
    'pickplace': 1,
    'stack': 2,
}

OBJECT_KIND_CODES = {
    'cube': 0,
    'block': 0,
    'plate': 1,
    'bowl': 2,
    'mug': 3,
    'tableware': 9,
}

ORACLE_MODE_CODES = {
    'idle': 0,
    'push': 1,
    'push_retreat': 2,
    'push_settle': 3,
    'edge_move_above': 10,
    'edge_descend': 11,
    'edge_close': 12,
    'edge_lift': 13,
    'edge_move_target': 14,
    'edge_descend_target': 15,
    'edge_release': 16,
    'edge_retreat': 17,
}


def _to_float_scalar(value, default=np.nan):
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])
    except (TypeError, ValueError):
        return float(default)


def _to_int_scalar(value, default=-1):
    try:
        arr = np.asarray(value).reshape(-1)
        if arr.size == 0:
            return int(default)
        return int(arr[0])
    except (TypeError, ValueError):
        return int(default)


def _to_vec(value, length, default=np.nan):
    vec = np.full(length, default, dtype=np.float32)
    if value is None:
        return vec
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return vec
    copy_len = min(length, arr.size)
    if copy_len > 0:
        vec[:copy_len] = arr[:copy_len]
    return vec


def _encode_string(value, mapping, default=0):
    if value is None:
        return float(default)
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    key = str(value).lower()
    return float(mapping.get(key, default))


def _get_object_kind(env, block_idx):
    unwrapped = env.unwrapped
    kinds = getattr(unwrapped, '_tableware_object_kinds', None)
    if kinds is not None and 0 <= block_idx < len(kinds):
        return str(kinds[block_idx]).lower()
    return str(getattr(unwrapped, '_object_type', 'cube')).lower()


def _get_object_kind_codes(env, num_objects):
    kind_codes = np.full(num_objects, -1.0, dtype=np.float32)
    for idx in range(num_objects):
        kind_codes[idx] = _encode_string(_get_object_kind(env, idx), OBJECT_KIND_CODES, default=-1)
    return kind_codes


def _get_object_successes(env, num_objects):
    successes = np.full(num_objects, np.nan, dtype=np.float32)
    try:
        computed = env.unwrapped._compute_successes()
    except Exception:
        return successes
    for idx, success in enumerate(computed[:num_objects]):
        successes[idx] = float(bool(success))
    return successes


def _get_block_positions(info, num_objects):
    block_positions = np.full((num_objects, 3), np.nan, dtype=np.float32)
    for idx in range(num_objects):
        block_positions[idx] = _to_vec(info.get(f'privileged/block_{idx}_pos'), 3)
    return block_positions


def _get_target_positions(env, num_objects):
    target_positions = np.full((num_objects, 3), np.nan, dtype=np.float32)
    unwrapped = env.unwrapped
    target_mocap_ids = getattr(unwrapped, '_object_target_mocap_ids', [])
    data = getattr(unwrapped, '_data', None)
    mocap_pos = getattr(data, 'mocap_pos', None)
    if mocap_pos is None:
        return target_positions
    for idx in range(min(num_objects, len(target_mocap_ids))):
        target_positions[idx] = _to_vec(mocap_pos[target_mocap_ids[idx]], 3)
    return target_positions


def _get_stack_target_ids(env, num_objects):
    target_ids = np.full(num_objects, -1.0, dtype=np.float32)
    stack_targets = getattr(env.unwrapped, '_stack_targets', [])
    for idx, block_id in enumerate(stack_targets[:num_objects]):
        target_ids[idx] = float(block_id)
    return target_ids


def _get_stack_target_positions(env, num_objects):
    positions = np.full((num_objects, 3), np.nan, dtype=np.float32)
    stack_positions = getattr(env.unwrapped, '_stack_target_positions', [])
    for idx, pos in enumerate(stack_positions[:num_objects]):
        positions[idx] = _to_vec(pos, 3)
    return positions


def _compute_position_distances(block_pos, target_pos):
    diff = np.asarray(target_pos, dtype=np.float32) - np.asarray(block_pos, dtype=np.float32)
    if not np.all(np.isfinite(diff)):
        return np.nan, np.nan, np.nan
    return float(np.linalg.norm(diff)), float(np.linalg.norm(diff[:2])), float(abs(diff[2]))


def build_collect_diagnostics(
    info,
    env,
    agent,
    p_stack,
    ep_idx,
    episode_step,
    global_step,
    task_success_event=False,
    task_success=-1.0,
    action_target_block=None,
    action_target_task=None,
    action_target_group=None,
):
    """Build fixed-shape scalar/vector diagnostics saved next to the dataset."""
    unwrapped = env.unwrapped
    num_objects = int(getattr(unwrapped, 'num_objects', getattr(unwrapped, 'num_cubes', 0)))

    target_block = _to_int_scalar(info.get('privileged/target_block', -1), default=-1)
    block_positions = _get_block_positions(info, num_objects)
    target_positions = _get_target_positions(env, num_objects)
    object_successes = _get_object_successes(env, num_objects)

    active_block_pos = np.full(3, np.nan, dtype=np.float32)
    if 0 <= target_block < num_objects:
        active_block_pos = block_positions[target_block].copy()

    active_target_pos = _to_vec(info.get('privileged/target_block_pos'), 3)
    if not np.all(np.isfinite(active_target_pos)) and 0 <= target_block < num_objects:
        active_target_pos = target_positions[target_block].copy()

    target_dist, target_xy_dist, target_z_abs_dist = _compute_position_distances(active_block_pos, active_target_pos)

    action_target_block = target_block if action_target_block is None else int(action_target_block)
    action_block_pos = np.full(3, np.nan, dtype=np.float32)
    action_target_pos = np.full(3, np.nan, dtype=np.float32)
    action_target_success = np.nan
    if 0 <= action_target_block < num_objects:
        action_block_pos = block_positions[action_target_block].copy()
        action_target_pos = target_positions[action_target_block].copy()
        if not np.all(np.isfinite(action_target_pos)) and action_target_block == target_block:
            action_target_pos = active_target_pos.copy()
        if action_target_block < len(object_successes):
            action_target_success = float(object_successes[action_target_block])
    action_target_dist, action_target_xy_dist, action_target_z_abs_dist = _compute_position_distances(
        action_block_pos,
        action_target_pos,
    )

    target_task = info.get('privileged/target_task', None)
    target_group = info.get('privileged/target_task_group', None)
    action_target_task = target_task if action_target_task is None else action_target_task
    action_target_group = target_group if action_target_group is None else action_target_group
    target_kind = _get_object_kind(env, target_block) if target_block >= 0 else None
    action_target_kind = _get_object_kind(env, action_target_block) if action_target_block >= 0 else None
    oracle_mode = getattr(agent, '_mode', None)
    gripper_contact = _to_float_scalar(info.get('proprio/gripper_contact', [0.0]), default=0.0)
    action_target_up_z = _runtime_block_up_z(env, action_target_block)
    target_up_z = _runtime_block_up_z(env, target_block)
    strict_dist_threshold = (
        TABLEWARE_STACK_SUCCESS_DIST
        if str(action_target_group).lower() == 'stack'
        else TABLEWARE_PICKPLACE_SUCCESS_DIST
    )
    action_target_strict_pos_success = (
        float(action_target_dist <= strict_dist_threshold) if np.isfinite(action_target_dist) else 0.0
    )
    action_target_upright_success = (
        1.0 if ((not np.isfinite(action_target_up_z)) or action_target_up_z >= TABLEWARE_MIN_UP_Z) else 0.0
    )

    return {
        'diagnostics_episode_index': float(ep_idx),
        'diagnostics_episode_step': float(episode_step),
        'diagnostics_global_step': float(global_step),
        'diagnostics_p_stack': float(p_stack),
        'diagnostics_target_block': float(target_block),
        'diagnostics_target_object_kind': _encode_string(target_kind, OBJECT_KIND_CODES, default=-1),
        'diagnostics_target_task_code': _encode_string(target_task, TASK_NAME_CODES, default=0),
        'diagnostics_target_task_group_code': _encode_string(target_group, TASK_GROUP_CODES, default=0),
        'diagnostics_action_target_block': float(action_target_block),
        'diagnostics_action_target_object_kind': _encode_string(action_target_kind, OBJECT_KIND_CODES, default=-1),
        'diagnostics_action_target_task_code': _encode_string(action_target_task, TASK_NAME_CODES, default=0),
        'diagnostics_action_target_task_group_code': _encode_string(action_target_group, TASK_GROUP_CODES, default=0),
        'diagnostics_is_stacking': float(bool(getattr(unwrapped, '_stacking', False))),
        'diagnostics_stack_target_index': float(getattr(unwrapped, '_stack_target_index', -1)),
        'diagnostics_num_stack_targets': float(len(getattr(unwrapped, '_stack_targets', []))),
        'diagnostics_target_dist': target_dist,
        'diagnostics_target_xy_dist': target_xy_dist,
        'diagnostics_target_z_abs_dist': target_z_abs_dist,
        'diagnostics_action_target_dist': action_target_dist,
        'diagnostics_action_target_xy_dist': action_target_xy_dist,
        'diagnostics_action_target_z_abs_dist': action_target_z_abs_dist,
        'diagnostics_action_target_success': action_target_success,
        'diagnostics_action_target_up_z': action_target_up_z,
        'diagnostics_action_target_strict_pos_success': action_target_strict_pos_success,
        'diagnostics_action_target_upright_success': action_target_upright_success,
        'diagnostics_info_success': float(bool(info.get('success', False))),
        'diagnostics_info_success_frac': _to_float_scalar(info.get('success_frac', np.nan), default=np.nan),
        'diagnostics_agent_done': float(bool(getattr(agent, 'done', False))),
        'diagnostics_task_success_event': float(bool(task_success_event)),
        'diagnostics_task_success': _to_float_scalar(task_success, default=-1.0),
        'diagnostics_episode_complete': float(
            is_tableware_episode_complete(env) if is_tableware_collection_env(env) else bool(info.get('success', False))
        ),
        'diagnostics_oracle_mode_code': _encode_string(oracle_mode, ORACLE_MODE_CODES, default=0),
        'diagnostics_oracle_mode_step': _to_float_scalar(getattr(agent, '_mode_step', np.nan), default=np.nan),
        'diagnostics_oracle_step': _to_float_scalar(getattr(agent, '_step', np.nan), default=np.nan),
        'diagnostics_oracle_active_target_block': _to_float_scalar(
            getattr(agent, '_active_target_block', np.nan),
            default=np.nan,
        ),
        'diagnostics_gripper_contact': gripper_contact,
        'diagnostics_push_drive_started': float(bool(getattr(agent, '_push_drive_started', False))),
        'diagnostics_oracle_has_push_plan': float(getattr(agent, '_active_push_plan', None) is not None),
        'diagnostics_oracle_has_grasp_plan': float(getattr(agent, '_active_grasp_plan', None) is not None),
        'diagnostics_object_successes': object_successes,
        'diagnostics_object_kind_codes': _get_object_kind_codes(env, num_objects),
        'diagnostics_block_positions': block_positions.reshape(-1),
        'diagnostics_target_positions': target_positions.reshape(-1),
        'diagnostics_active_block_pos': active_block_pos,
        'diagnostics_active_target_pos': active_target_pos,
        'diagnostics_target_up_z': target_up_z,
        'diagnostics_stack_target_ids': _get_stack_target_ids(env, num_objects),
        'diagnostics_stack_target_positions': _get_stack_target_positions(env, num_objects).reshape(-1),
    }


def _finite_mean(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _code_counts(values, code_to_name):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    counts = {}
    for code in arr[np.isfinite(arr)].astype(np.int32):
        name = code_to_name.get(int(code), str(int(code)))
        counts[name] = counts.get(name, 0) + 1
    return counts


def build_diagnostics_summary(cfg, full_dataset):
    events = np.asarray(full_dataset.get('diagnostics_task_success_event', []), dtype=np.float32).reshape(-1) > 0.5
    task_success = np.asarray(full_dataset.get('diagnostics_task_success', []), dtype=np.float32).reshape(-1)
    valid_events = events
    if task_success.shape == events.shape:
        valid_events = events & (task_success >= 0.0)

    event_success = task_success[valid_events] if task_success.shape == events.shape else np.array([], dtype=np.float32)
    num_task_events = int(event_success.size)
    num_success_events = int(np.sum(event_success > 0.5)) if num_task_events > 0 else 0
    task_success_rate = None if num_task_events == 0 else float(num_success_events / num_task_events)

    terminals = np.asarray(full_dataset.get('terminals', []), dtype=bool).reshape(-1)
    terminal_indices = np.flatnonzero(terminals)
    final_success_frac = np.asarray(full_dataset.get('diagnostics_info_success_frac', []), dtype=np.float32).reshape(-1)
    final_success = np.asarray(full_dataset.get('diagnostics_info_success', []), dtype=np.float32).reshape(-1)
    episode_complete = np.asarray(full_dataset.get('diagnostics_episode_complete', []), dtype=np.float32).reshape(-1)

    if terminal_indices.size > 0 and final_success_frac.size == terminals.size:
        episode_final_success_frac_mean = _finite_mean(final_success_frac[terminal_indices])
    else:
        episode_final_success_frac_mean = None
    if terminal_indices.size > 0 and final_success.size == terminals.size:
        episode_final_info_success_rate = _finite_mean(final_success[terminal_indices] > 0.5)
    else:
        episode_final_info_success_rate = None
    if terminal_indices.size > 0 and episode_complete.size == terminals.size:
        episode_complete_success_rate = _finite_mean(episode_complete[terminal_indices] > 0.5)
    else:
        episode_complete_success_rate = None

    code_to_kind = {0: 'cube', 1: 'plate', 2: 'bowl', 3: 'mug', 9: 'tableware'}
    code_to_task = {code: name for name, code in TASK_NAME_CODES.items()}
    code_to_group = {code: name for name, code in TASK_GROUP_CODES.items()}
    code_to_mode = {code: name for name, code in ORACLE_MODE_CODES.items()}

    action_kind = np.asarray(full_dataset.get('diagnostics_action_target_object_kind', []), dtype=np.float32)
    action_task = np.asarray(full_dataset.get('diagnostics_action_target_task_code', []), dtype=np.float32)
    action_group = np.asarray(full_dataset.get('diagnostics_action_target_task_group_code', []), dtype=np.float32)
    oracle_mode = np.asarray(full_dataset.get('diagnostics_oracle_mode_code', []), dtype=np.float32)

    if action_kind.shape == events.shape:
        event_kind_counts = _code_counts(action_kind[events], code_to_kind)
    else:
        event_kind_counts = {}
    if action_task.shape == events.shape:
        event_task_counts = _code_counts(action_task[events], code_to_task)
    else:
        event_task_counts = {}
    if action_group.shape == events.shape:
        event_group_counts = _code_counts(action_group[events], code_to_group)
    else:
        event_group_counts = {}
    if oracle_mode.shape == events.shape:
        event_oracle_mode_counts = _code_counts(oracle_mode[events], code_to_mode)
    else:
        event_oracle_mode_counts = {}

    summary = {
        'dataset_name': str(cfg.get('dataset_name', 'unknown')),
        'num_steps': int(terminals.size),
        'num_episodes': int(terminal_indices.size),
        'num_task_success_events': num_task_events,
        'num_successful_task_events': num_success_events,
        'task_success_rate': task_success_rate,
        'episode_complete_success_rate': episode_complete_success_rate,
        'episode_final_success_frac_mean': episode_final_success_frac_mean,
        'episode_final_info_success_rate': episode_final_info_success_rate,
        'event_target_kind_counts': event_kind_counts,
        'event_target_task_counts': event_task_counts,
        'event_target_group_counts': event_group_counts,
        'event_oracle_mode_counts': event_oracle_mode_counts,
        'mean_action_target_dist': _finite_mean(full_dataset.get('diagnostics_action_target_dist', [])),
        'mean_action_target_xy_dist': _finite_mean(full_dataset.get('diagnostics_action_target_xy_dist', [])),
        'mean_post_step_target_dist': _finite_mean(full_dataset.get('diagnostics_target_dist', [])),
        'mean_post_step_target_xy_dist': _finite_mean(full_dataset.get('diagnostics_target_xy_dist', [])),
        'code_maps': {
            'object_kind': {name: int(code) for name, code in OBJECT_KIND_CODES.items()},
            'task': {name: int(code) for name, code in TASK_NAME_CODES.items()},
            'task_group': {name: int(code) for name, code in TASK_GROUP_CODES.items()},
            'oracle_mode': {name: int(code) for name, code in ORACLE_MODE_CODES.items()},
        },
    }
    return summary


def save_collect_diagnostics(cfg, full_dataset):
    diagnostics = {
        key: np.asarray(value, dtype=get_dataset_dtype(key))
        for key, value in full_dataset.items()
        if key.startswith('diagnostics_')
    }
    if len(diagnostics) == 0:
        return None, None

    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'diagnostics'))
    os.makedirs(save_dir, exist_ok=True)
    base_name = str(cfg.get('dataset_name', 'unknown'))
    diagnostics_path = os.path.join(save_dir, f'{base_name}_diagnostics.npz')
    summary_path = os.path.join(save_dir, f'{base_name}_summary.json')

    save_npz(
        diagnostics_path,
        diagnostics,
        compressed=cfg.get('diagnostics_compressed', True),
        label='diagnostics',
    )
    summary = build_diagnostics_summary(cfg, full_dataset)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f'Saved diagnostics summary to {summary_path}')
    return diagnostics_path, summary_path


def get_collect_diagnostics_dir():
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'diagnostics'))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def init_live_diagnostics(cfg, append=False):
    save_dir = get_collect_diagnostics_dir()
    base_name = str(cfg.get('dataset_name', 'unknown'))
    events_path = os.path.join(save_dir, f'{base_name}_events.jsonl')
    summary_path = os.path.join(save_dir, f'{base_name}_live_summary.json')
    if not append:
        with open(events_path, 'w', encoding='utf-8'):
            pass
    return events_path, summary_path


def _json_number(value):
    try:
        value = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None
    if not np.isfinite(value):
        return None
    return value


def build_live_event_record(
    step_diagnostics,
    run_finished_tasks,
    run_successful_tasks,
    run_finished_episodes=0,
    run_successful_episodes=0,
):
    run_task_success_rate = 0.0 if run_finished_tasks <= 0 else run_successful_tasks / run_finished_tasks
    run_episode_success_rate = (
        0.0 if run_finished_episodes <= 0 else run_successful_episodes / run_finished_episodes
    )
    keys = [
        'diagnostics_episode_index',
        'diagnostics_episode_step',
        'diagnostics_global_step',
        'diagnostics_task_success',
        'diagnostics_action_target_block',
        'diagnostics_action_target_object_kind',
        'diagnostics_action_target_task_code',
        'diagnostics_action_target_task_group_code',
        'diagnostics_action_target_dist',
        'diagnostics_action_target_xy_dist',
        'diagnostics_action_target_z_abs_dist',
        'diagnostics_action_target_success',
        'diagnostics_action_target_up_z',
        'diagnostics_action_target_strict_pos_success',
        'diagnostics_action_target_upright_success',
        'diagnostics_target_block',
        'diagnostics_target_dist',
        'diagnostics_target_xy_dist',
        'diagnostics_info_success',
        'diagnostics_info_success_frac',
        'diagnostics_episode_complete',
        'diagnostics_gripper_contact',
        'diagnostics_push_drive_started',
        'diagnostics_oracle_mode_code',
        'diagnostics_oracle_step',
        'diagnostics_oracle_mode_step',
    ]
    record = {key.replace('diagnostics_', ''): _json_number(step_diagnostics.get(key)) for key in keys}
    record['run_finished_tasks'] = int(run_finished_tasks)
    record['run_successful_tasks'] = int(run_successful_tasks)
    record['run_task_success_rate'] = float(run_task_success_rate)
    record['run_finished_episodes'] = int(run_finished_episodes)
    record['run_successful_episodes'] = int(run_successful_episodes)
    record['run_episode_success_rate'] = float(run_episode_success_rate)
    return record


def append_live_event(events_path, record):
    with open(events_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, sort_keys=True) + '\n')


def save_live_summary(
    summary_path,
    cfg,
    run_finished_tasks,
    run_successful_tasks,
    run_finished_episodes=0,
    run_successful_episodes=0,
    last_event=None,
):
    run_task_success_rate = 0.0 if run_finished_tasks <= 0 else run_successful_tasks / run_finished_tasks
    run_episode_success_rate = (
        0.0 if run_finished_episodes <= 0 else run_successful_episodes / run_finished_episodes
    )
    summary = {
        'dataset_name': str(cfg.get('dataset_name', 'unknown')),
        'num_task_success_events': int(run_finished_tasks),
        'num_successful_task_events': int(run_successful_tasks),
        'task_success_rate': float(run_task_success_rate),
        'num_finished_episodes': int(run_finished_episodes),
        'num_successful_episodes': int(run_successful_episodes),
        'episode_success_rate': float(run_episode_success_rate),
        'last_event': last_event,
        'paths': {
            'events_jsonl': summary_path.replace('_live_summary.json', '_events.jsonl'),
            'live_summary_json': summary_path,
        },
        'note': 'Updated during collect; final full diagnostics are written at collect end.',
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def is_tableware_collection_env(env):
    unwrapped = env.unwrapped
    mode = getattr(unwrapped, '_mode', '').lower()
    object_type = getattr(unwrapped, '_object_type', 'cube').lower()
    return ('tableware' in mode) or (object_type != 'cube')


def get_oracle_for_task(task_name, env, agents):
    if is_tableware_collection_env(env) and task_name in {'pickplace_cube', 'stack_cube', 'push_cube'}:
        # Tableware objects need geometry-aware edge push/grasp instead of cube-center oracles.
        return agents['pickplace_tableware']
    return agents[task_name]


def is_tableware_episode_complete(env):
    if not is_tableware_collection_env(env):
        return False

    unwrapped = env.unwrapped
    tableware_split = None
    if hasattr(unwrapped, '_parse_tableware_split_from_mode'):
        tableware_split = unwrapped._parse_tableware_split_from_mode(unwrapped._mode)
    if tableware_split is None:
        return False

    pickplace_count, stack_count = tableware_split
    required_blocks = []
    if pickplace_count > 0:
        required_blocks.extend(getattr(unwrapped, '_tableware_ppp_block_indices', []))
    if stack_count > 0:
        required_blocks.extend(getattr(unwrapped, '_tableware_stack_block_indices', []))
    if len(required_blocks) == 0:
        return False

    try:
        successes = unwrapped._compute_successes()
    except Exception:
        return False
    return all(0 <= int(block) < len(successes) and bool(successes[int(block)]) for block in required_blocks)


def collect(cfg):
    # Initialize environment
    env = gym.make(cfg.task, ob_type='pixels', multiview=True, max_episode_steps=cfg.max_episode_steps, **cfg.env_kwargs)
    num_cubes = env.unwrapped.num_cubes
    tableware_mode = is_tableware_collection_env(env)

    # Initialize oracles
    oracle_type = 'plan' if cfg.dataset_type == 'play' else 'markov'

    agents = {
        'pickplace_cube': PickPlaceCubeMarkovOracle(max_step=200, env=env, min_norm=cfg.min_action_norm),
        'stack_cube': PickPlaceCubeMarkovOracle(max_step=200, env=env, min_norm=cfg.min_action_norm),
        'push_cube': PushCubeMarkovOracle(max_step=100, env=env, min_norm=cfg.min_action_norm),
        'pickplace_tableware': PickPlaceTablewareMarkovOracle(max_step=650, env=env, min_norm=cfg.min_action_norm),
    }

    dataset = defaultdict(list)
    total_steps = 0
    shard_finished_tasks = 0
    shard_successful_tasks = 0
    run_finished_tasks = 0
    run_successful_tasks = 0
    run_finished_episodes = 0
    run_successful_episodes = 0
    current_shard_episode_count = 0

    checkpoint_dir = None
    ep_size_counters = {}
    if cfg.get('use_checkpoint', False):
        checkpoint_dir = get_checkpoint_dir(cfg)
        existing_shards = list_checkpoint_shards(checkpoint_dir)
        start_ep_idx = count_completed_episodes(existing_shards)
        ep_size_counters = build_ep_size_counters(existing_shards)
        print(f'Found {len(existing_shards)} checkpoint shards, recovered episodes: {start_ep_idx}')
    else:
        start_ep_idx = 0

    num_train_episodes = cfg.num_collect_episodes
    num_val_episodes = cfg.num_collect_episodes // 10
    total_target_episodes = num_train_episodes + num_val_episodes
    save_success_only = bool(cfg.get('save_success_only', False))
    saved_episode_count = int(start_ep_idx) if save_success_only else int(start_ep_idx)
    discarded_episode_count = 0
    reset_failure_count = 0
    if save_success_only:
        remaining_target_episodes = max(total_target_episodes - saved_episode_count, 0)
        max_collect_attempts = cfg.get('max_collect_attempts', None)
        if max_collect_attempts is None:
            collect_attempts = max(remaining_target_episodes * 3, remaining_target_episodes)
        else:
            collect_attempts = max(0, int(max_collect_attempts))
        attempt_stop_idx = start_ep_idx + collect_attempts
        print(
            'Success-only collection enabled: '
            f'target_saved_episodes={total_target_episodes}, '
            f'already_saved={saved_episode_count}, '
            f'max_attempts_this_run={collect_attempts}.'
        )
    else:
        attempt_stop_idx = total_target_episodes

    live_events_path, live_summary_path = init_live_diagnostics(
        cfg,
        append=bool(cfg.get('use_checkpoint', False) and start_ep_idx > 0),
    )
    save_live_summary(
        live_summary_path,
        cfg,
        run_finished_tasks,
        run_successful_tasks,
        run_finished_episodes=run_finished_episodes,
        run_successful_episodes=run_successful_episodes,
    )
    print(f'Live diagnostics will be written to:\n{live_events_path}\n{live_summary_path}')

    pbar = trange(start_ep_idx, attempt_stop_idx)

    def update_collect_postfix():
        task_success_rate = 0.0 if run_finished_tasks <= 0 else run_successful_tasks / run_finished_tasks
        postfix = {'task_success_rate': f'{task_success_rate:.1%}'}
        if tableware_mode:
            episode_success_rate = 0.0 if run_finished_episodes <= 0 else run_successful_episodes / run_finished_episodes
            postfix = {
                'episode_success_rate': f'{episode_success_rate:.1%}',
                'task_success_rate': f'{task_success_rate:.1%}',
            }
        if save_success_only:
            postfix['kept'] = f'{saved_episode_count}/{total_target_episodes}'
            postfix['discarded'] = discarded_episode_count
            if reset_failure_count > 0:
                postfix['init_fail'] = reset_failure_count
        pbar.set_postfix(**postfix)

    update_collect_postfix()
    last_live_event = None
    first_episode_video_saved = False

    for ep_idx in pbar:
        if save_success_only and saved_episode_count >= total_target_episodes:
            break

        make_video = bool(cfg.get('save_video', True)) and not first_episode_video_saved
        if make_video:
            frames = []
            target_frame = None

        try:
            ob, info = env.reset()
        except AssertionError as exc:
            if 'Sampling' not in str(exc):
                raise
            reset_failure_count += 1
            discarded_episode_count += 1
            run_finished_episodes += 1
            update_collect_postfix()
            save_live_summary(
                live_summary_path,
                cfg,
                run_finished_tasks=run_finished_tasks,
                run_successful_tasks=run_successful_tasks,
                run_finished_episodes=run_finished_episodes,
                run_successful_episodes=run_successful_episodes,
                last_event=last_live_event,
            )
            continue
        if make_video:
            target_frame = build_multiview_target_overlay(env, info, np.array(ob, copy=True))

        # Set the stacking probability for this episode.
        if env.unwrapped._mode == 'data_collection':
            if num_cubes > 1:
                p_stack = 0.5
            else:
                p_stack = 0.0
        elif env.unwrapped._mode == 'data_collection_stack':
            if num_cubes == 1:
                p_stack = 0.0
            elif num_cubes == 2:
                p_stack = np.random.uniform(0.0, 0.25)
            elif num_cubes == 3:
                # p_stack = np.random.uniform(0.05, 0.35)
                p_stack = 0.9
            elif num_cubes == 4:
                p_stack = np.random.uniform(0.1, 0.5)
            else:
                p_stack = 0.5
        elif tableware_mode:
            tableware_split = None
            if hasattr(env.unwrapped, '_parse_tableware_split_from_mode'):
                tableware_split = env.unwrapped._parse_tableware_split_from_mode(env.unwrapped._mode)
            pickplace_count, stack_count = (0, 0) if tableware_split is None else tableware_split
            if stack_count <= 0:
                p_stack = 0.0
            elif pickplace_count <= 0:
                p_stack = 1.0
            else:
                p_stack = 0.5
        else:
            # For hybrid modes (e.g., data_collection_cube2+3), targets are predefined in env.set_new_target().
            p_stack = 0.0

        if oracle_type == 'markov':
            # Set the action noise level for this episode
            noise_high = min(cfg.action_noise, 0.02) if tableware_mode else cfg.action_noise
            xi = np.random.uniform(0, noise_high)

        agent = get_oracle_for_task(info['privileged/target_task'], env, agents)
        agent.reset(ob, info)

        episode_dataset = defaultdict(list)
        done = False
        step = 0
        last_live_event = None

        try:
            while not done:
                action_target_block = _to_int_scalar(info.get('privileged/target_block', -1), default=-1)
                action_target_task = info.get('privileged/target_task', None)
                action_target_group = info.get('privileged/target_task_group', None)

                p_random = min(cfg.p_random_action, 0.01) if tableware_mode else cfg.p_random_action
                if np.random.rand() < p_random:
                    # Sample a random action
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == 'markov':
                        # Add Gaussian noise to the action
                        action = action + np.random.normal(0, [xi, xi, xi, xi * 3, xi * 5], action.shape)
                action = np.clip(action, -1, 1)
                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if os.environ.get('TABLEWARE_PUSH_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}:
                    if isinstance(agent, PickPlaceTablewareMarkovOracle):
                        target_block = int(info.get('privileged/target_block', -1))
                        if target_block >= 0:
                            block_pos = np.asarray(info[f'privileged/block_{target_block}_pos'], dtype=np.float32)
                            target_pos = np.asarray(info['privileged/target_block_pos'], dtype=np.float32)
                            xy_dist = float(np.linalg.norm(target_pos[:2] - block_pos[:2]))
                            print(
                                '[collect-push] '
                                f'mode={getattr(agent, "_mode", "na")} '
                                f'target_block={target_block} '
                                f'xy_dist={xy_dist:.4f} '
                                f'agent_done={agent.done} '
                                f'info_success={info.get("success", None)}'
                            )

                task_success_event = False
                task_success = -1.0
                if agent.done:
                    task_success = check_task_success(
                        info,
                        env,
                        target_block=action_target_block,
                        target_group=action_target_group,
                    )
                    task_success_event = True
                    shard_finished_tasks += 1
                    shard_successful_tasks += int(task_success)
                    run_finished_tasks += 1
                    run_successful_tasks += int(task_success)
                    if not tableware_mode:
                        update_collect_postfix()

                step_diagnostics = build_collect_diagnostics(
                    info,
                    env,
                    agent,
                    p_stack=p_stack,
                    ep_idx=ep_idx,
                    episode_step=step,
                    global_step=total_steps + step,
                    task_success_event=task_success_event,
                    task_success=task_success,
                    action_target_block=action_target_block,
                    action_target_task=action_target_task,
                    action_target_group=action_target_group,
                )

                if task_success_event:
                    live_event = build_live_event_record(
                        step_diagnostics,
                        run_finished_tasks=run_finished_tasks,
                        run_successful_tasks=run_successful_tasks,
                        run_finished_episodes=run_finished_episodes,
                        run_successful_episodes=run_successful_episodes,
                    )
                    append_live_event(live_events_path, live_event)
                    last_live_event = live_event
                    save_live_summary(
                        live_summary_path,
                        cfg,
                        run_finished_tasks=run_finished_tasks,
                        run_successful_tasks=run_successful_tasks,
                        run_finished_episodes=run_finished_episodes,
                        run_successful_episodes=run_successful_episodes,
                        last_event=live_event,
                    )

                next_agent_ob = next_ob
                next_agent_info = info
                tableware_episode_complete = task_success_event and is_tableware_episode_complete(env)
                if tableware_episode_complete:
                    done = True
                elif task_success_event:
                    # Set a new task when the current task is done
                    if os.environ.get('TABLEWARE_PUSH_DEBUG', '').lower() in {'1', 'true', 'yes', 'on'}:
                        print(
                            '[collect-push] '
                            f'switch_target current_block={info.get("privileged/target_block", None)} '
                            f'task_success={task_success}'
                        )
                    agent_ob, agent_info = env.unwrapped.set_new_target(p_stack=p_stack)
                    agent = get_oracle_for_task(agent_info['privileged/target_task'], env, agents)
                    agent.reset(agent_ob, agent_info)
                    next_agent_ob = agent_ob
                    next_agent_info = agent_info

                episode_dataset['observations'].append(ob)
                episode_dataset['actions'].append(action)
                episode_dataset['terminals'].append(done)
                episode_dataset['qpos'].append(info['prev_qpos'])
                episode_dataset['state_observations'].append(info['prev_state_clean_observation'])
                for key, value in step_diagnostics.items():
                    episode_dataset[key].append(value)

                ob = next_agent_ob
                info = next_agent_info
                step += 1

                if make_video:
                    frame = env.unwrapped.get_pixel_observation()
                    frames.append(frame)
        finally:
            if make_video:
                video_path = save_first_episode_video(
                    frames,
                    mode=env.unwrapped._mode,
                    target_frame=target_frame,
                    fps=15,
                )
                first_episode_video_saved = video_path is not None

        total_steps += step
        episode_success = (
            is_tableware_episode_complete(env) if tableware_mode else bool(info.get('success', False))
        )
        keep_episode = (not save_success_only) or bool(episode_success)
        if keep_episode:
            extend_dataset(dataset, episode_dataset)
            current_shard_episode_count += 1
            if save_success_only:
                saved_episode_count += 1
        else:
            discarded_episode_count += 1

        run_finished_episodes += 1
        run_successful_episodes += int(episode_success)
        update_collect_postfix()
        save_live_summary(
            live_summary_path,
            cfg,
            run_finished_tasks=run_finished_tasks,
            run_successful_tasks=run_successful_tasks,
            run_finished_episodes=run_finished_episodes,
            run_successful_episodes=run_successful_episodes,
            last_event=last_live_event,
        )

        # Save a shard periodically.
        if cfg.get('use_checkpoint', False) and cfg.get('checkpoint_interval', 100) > 0:
            if (ep_idx + 1) % cfg.checkpoint_interval == 0:
                shard_path = save_checkpoint_shard(
                    cfg,
                    dataset,
                    checkpoint_dir,
                    current_shard_episode_count,
                    ep_size_counters,
                )
                if shard_path is not None:
                    dataset = defaultdict(list)
                    current_shard_episode_count = 0
                    shard_finished_tasks = 0
                    shard_successful_tasks = 0
                    update_collect_postfix()

    print('Total steps (this run):', total_steps)
    if save_success_only:
        print(
            'Success-only collection result: '
            f'saved={saved_episode_count}/{total_target_episodes}, '
            f'discarded={discarded_episode_count}, '
            f'init_failures={reset_failure_count}, '
            f'attempts={run_finished_episodes}.'
        )
        if saved_episode_count < total_target_episodes:
            print(
                'Warning: reached the attempt limit before collecting the requested number of '
                'successful episodes. Increase max_collect_attempts or reduce num_collect_episodes.'
            )

    if cfg.get('use_checkpoint', False):
        # Flush remaining in-memory samples into one last shard.
        shard_path = save_checkpoint_shard(
            cfg,
            dataset,
            checkpoint_dir,
            current_shard_episode_count,
            ep_size_counters,
        )
        if shard_path is not None:
            dataset = defaultdict(list)
            current_shard_episode_count = 0
            shard_finished_tasks = 0
            shard_successful_tasks = 0

        # Heavy read only once: final export.
        shard_files = list_checkpoint_shards(checkpoint_dir)
        recovered_episodes = count_completed_episodes(shard_files)
        print(f'Checkpoint shards for export: {len(shard_files)}, episodes by filename sum: {recovered_episodes}')
        if cfg.get('force_lowmem_export', False):
            run_lowmem_export(cfg, checkpoint_dir, num_train_episodes)
            return
        if should_skip_full_export(cfg, shard_files):
            print(f'Final dataset export skipped. Checkpoints available at: {checkpoint_dir}')
            train_path = os.path.join(cfg.data_dir, f'{cfg.dataset_name}.npz')
            val_path = train_path.replace('.npz', '-val.npz')
            print('To export with low memory, run:')
            print(
                'python datasets/merge_shards_lowmem.py '
                f'--checkpoint-dir {checkpoint_dir} '
                f'--train-path {train_path} '
                f'--val-path {val_path} '
                f'--num-train-episodes {num_train_episodes}'
            )
            return
        full_dataset = load_full_dataset_from_shards(shard_files)
    else:
        full_dataset = {}
        for key, value in dataset.items():
            full_dataset[key] = np.asarray(value, dtype=get_dataset_dtype(key))

    if 'terminals' in full_dataset:
        total_train_steps = compute_train_steps_from_terminals(full_dataset['terminals'], num_train_episodes)
    else:
        total_train_steps = 0
    print(f'Total train steps used for split: {total_train_steps}')

    train_path = os.path.join(cfg.data_dir, f'{cfg.dataset_name}.npz')
    val_path = train_path.replace('.npz', '-val.npz')

    # Split the dataset into training and validation sets.
    train_dataset = {}
    for key, value in full_dataset.items():
        dtype = get_dataset_dtype(key)
        train_dataset[key] = np.asarray(value[:total_train_steps], dtype=dtype)
    save_npz(
        train_path,
        train_dataset,
        compressed=cfg.get('dataset_compressed', True),
        label='train dataset',
    )
    del train_dataset

    val_dataset = {}
    for key, value in full_dataset.items():
        dtype = get_dataset_dtype(key)
        val_dataset[key] = np.asarray(value[total_train_steps:], dtype=dtype)
    save_npz(
        val_path,
        val_dataset,
        compressed=cfg.get('dataset_compressed', True),
        label='val dataset',
    )

    diagnostics_path, diagnostics_summary_path = save_collect_diagnostics(cfg, full_dataset)
    print(f"Datasets saved to:\n{train_path}\n{val_path}")
    if diagnostics_path is not None:
        print(f"Diagnostics saved to:\n{diagnostics_path}\n{diagnostics_summary_path}")
