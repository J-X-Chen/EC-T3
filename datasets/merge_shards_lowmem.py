#!/usr/bin/env python3
"""
Low-memory offline merger for checkpoint shards.

This script avoids loading the full dataset into RAM by:
1) scanning shard metadata,
2) copying each key shard-by-shard into train/val memmaps on disk,
3) writing final train/val npz files from memmaps.

Merge progress is recorded per key, and final NPZ writes are transactional
per key so interrupted runs can reuse completed work.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat


MANIFEST_VERSION = 1
MANIFEST_NAME = "merge_manifest.json"


def get_dataset_dtype(key: str):
    if key == "observations":
        return np.uint8
    if key == "terminals":
        return bool
    return np.float32


def parse_checkpoint_shard_name(path: str):
    basename = os.path.basename(path)
    match = re.search(r"^checkpoint_ep(\d+)_(\d+)\.npz$", basename)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def list_checkpoint_shards(checkpoint_dir: Path):
    shard_files = glob.glob(str(checkpoint_dir / "checkpoint_ep*_*.npz"))
    shard_files = [Path(p) for p in shard_files if parse_checkpoint_shard_name(os.path.basename(p))]
    shard_files.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return shard_files


def read_npy_header_from_zip(zf: zipfile.ZipFile, member_name: str):
    with zf.open(member_name, "r") as f:
        major, minor = npformat.read_magic(f)
        if major == 1:
            shape, _fortran, dtype = npformat.read_array_header_1_0(f)
        else:
            shape, _fortran, dtype = npformat.read_array_header_2_0(f)
    return tuple(shape), np.dtype(dtype)


def stream_npy_rows_from_zip(
    zf: zipfile.ZipFile,
    member_name: str,
    target_dtype: np.dtype,
    expected_steps: int,
    expected_tail_shape,
    chunk_rows: int,
):
    """Yield row chunks from one npy entry inside a zip file without full materialization."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    with zf.open(member_name, "r") as f:
        major, minor = npformat.read_magic(f)
        if major == 1:
            shape, fortran_order, file_dtype = npformat.read_array_header_1_0(f)
        else:
            shape, fortran_order, file_dtype = npformat.read_array_header_2_0(f)

        shape = tuple(shape)
        file_dtype = np.dtype(file_dtype)
        if fortran_order:
            raise ValueError(f"Fortran-order array is not supported: {member_name}")
        if len(shape) < 1:
            raise ValueError(f"Invalid shape for {member_name}: {shape}")
        if int(shape[0]) != int(expected_steps):
            raise ValueError(
                f"Step mismatch for {member_name}: expected {expected_steps}, got {shape[0]}"
            )
        if tuple(shape[1:]) != tuple(expected_tail_shape):
            raise ValueError(
                f"Tail-shape mismatch for {member_name}: expected {expected_tail_shape}, got {shape[1:]}"
            )

        tail_shape = tuple(shape[1:])
        row_elems = int(np.prod(tail_shape, dtype=np.int64)) if tail_shape else 1
        row_bytes = row_elems * file_dtype.itemsize
        if row_bytes <= 0:
            raise ValueError(f"Invalid row size for {member_name}: row_bytes={row_bytes}")

        rows_total = int(shape[0])
        rows_done = 0
        buf = bytearray()

        while rows_done < rows_total:
            take_rows = min(chunk_rows, rows_total - rows_done)
            need_bytes = take_rows * row_bytes

            while len(buf) < need_bytes:
                data = f.read(need_bytes - len(buf))
                if not data:
                    break
                buf.extend(data)

            if len(buf) < need_bytes:
                raise ValueError(
                    f"Unexpected EOF while reading {member_name}: need {need_bytes} bytes, got {len(buf)}"
                )

            # Copy out the consumed bytes to avoid resizing `buf` while a view is alive.
            raw = bytes(buf[:need_bytes])
            flat = np.frombuffer(raw, dtype=file_dtype, count=take_rows * row_elems)
            if not file_dtype.isnative:
                flat = flat.byteswap().newbyteorder()
            if tail_shape:
                arr = flat.reshape((take_rows, *tail_shape))
            else:
                arr = flat.reshape((take_rows,))

            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype, copy=False)
            else:
                # arr shares memory with `buf`; copy before mutating `buf`.
                arr = arr.copy()

            del buf[:need_bytes]
            rows_done += take_rows
            yield arr


def infer_chunk_rows(tail_shape, dtype: np.dtype, chunk_mib: int):
    row_elems = int(np.prod(tail_shape, dtype=np.int64)) if tail_shape else 1
    row_bytes = max(1, row_elems * np.dtype(dtype).itemsize)
    target_bytes = max(1, int(chunk_mib) * 1024 * 1024)
    return max(1, target_bytes // row_bytes)


def scan_shards(shard_files):
    if not shard_files:
        raise ValueError("No checkpoint shard files found.")

    all_keys = None
    key_shapes = {}
    shard_steps = []
    total_steps = 0
    total_episodes = 0

    for idx, shard_path in enumerate(shard_files, start=1):
        with np.load(shard_path) as shard_data:
            if "data_terminals" not in shard_data.files:
                raise ValueError(f"Missing data_terminals in shard: {shard_path}")
            terminals = np.asarray(shard_data["data_terminals"], dtype=bool)
            steps = int(terminals.shape[0])
            episodes = int(terminals.sum())
            shard_steps.append(steps)
            total_steps += steps
            total_episodes += episodes

        with zipfile.ZipFile(shard_path, "r") as zf:
            names = [n for n in zf.namelist() if n.startswith("data_") and n.endswith(".npy")]
            keys = set(n[5:-4] for n in names)
            if all_keys is None:
                all_keys = keys
            else:
                if keys != all_keys:
                    missing = sorted(all_keys - keys)
                    extra = sorted(keys - all_keys)
                    raise ValueError(
                        f"Inconsistent keys at shard {shard_path}. "
                        f"Missing={missing}, Extra={extra}"
                    )

            for name in names:
                key = name[5:-4]
                shape, _dtype = read_npy_header_from_zip(zf, name)
                if len(shape) < 1:
                    raise ValueError(f"Invalid shape for key {key} in shard {shard_path}: {shape}")
                tail_shape = tuple(shape[1:])
                old = key_shapes.get(key)
                if old is None:
                    key_shapes[key] = tail_shape
                elif old != tail_shape:
                    raise ValueError(
                        f"Shape mismatch for key '{key}': expected tail {old}, got {tail_shape} "
                        f"in shard {shard_path}"
                    )

        if idx % 10 == 0 or idx == len(shard_files):
            print(f"[scan] {idx}/{len(shard_files)} shards")

    keys_sorted = sorted(all_keys) if all_keys is not None else []
    return {
        "keys": keys_sorted,
        "key_shapes": key_shapes,
        "shard_steps": shard_steps,
        "total_steps": total_steps,
        "total_episodes": total_episodes,
    }


def compute_train_steps(shard_files, num_train_episodes: int):
    steps_before = 0
    episodes_seen = 0

    for shard_path in shard_files:
        with np.load(shard_path) as shard_data:
            terminals = np.asarray(shard_data["data_terminals"], dtype=bool)
        done_indices = np.flatnonzero(terminals)
        done_count = int(done_indices.shape[0])
        if episodes_seen + done_count >= num_train_episodes:
            local_idx = int(done_indices[num_train_episodes - episodes_seen - 1])
            return steps_before + local_idx + 1
        episodes_seen += done_count
        steps_before += int(terminals.shape[0])

    return steps_before


def build_memmap_meta(tmp_dir: Path, keys, key_shapes, train_steps, val_steps):
    train_meta = {}
    val_meta = {}
    for key in keys:
        dtype = np.dtype(get_dataset_dtype(key))
        tail = key_shapes[key]
        train_shape = (train_steps, *tail)
        val_shape = (val_steps, *tail)
        train_path = tmp_dir / f"train_{key}.mmap"
        val_path = tmp_dir / f"val_{key}.mmap"
        train_meta[key] = {"path": train_path, "shape": train_shape, "dtype": dtype}
        val_meta[key] = {"path": val_path, "shape": val_shape, "dtype": dtype}
    return train_meta, val_meta


def fill_key_from_shards(
    key: str,
    shard_files,
    shard_steps,
    train_steps: int,
    train_mm: np.memmap,
    val_mm: np.memmap,
    dtype: np.dtype,
    tail_shape,
    chunk_rows: int,
):
    global_offset = 0
    train_offset = 0
    val_offset = 0
    key_name = f"data_{key}"
    member_name = f"{key_name}.npy"
    total_steps = sum(shard_steps)
    last_print_offset = 0

    for shard_idx, shard_path in enumerate(shard_files):
        expected_steps = int(shard_steps[shard_idx])
        with zipfile.ZipFile(shard_path, "r") as zf:
            if member_name not in zf.namelist():
                raise ValueError(f"Missing {member_name} in shard: {shard_path}")

            for arr in stream_npy_rows_from_zip(
                zf=zf,
                member_name=member_name,
                target_dtype=dtype,
                expected_steps=expected_steps,
                expected_tail_shape=tail_shape,
                chunk_rows=chunk_rows,
            ):
                chunk_n = int(arr.shape[0])
                train_take = min(max(train_steps - global_offset, 0), chunk_n)
                if train_take > 0:
                    train_mm[train_offset : train_offset + train_take] = arr[:train_take]
                    train_offset += train_take

                val_take = chunk_n - train_take
                if val_take > 0:
                    val_mm[val_offset : val_offset + val_take] = arr[train_take:]
                    val_offset += val_take

                global_offset += chunk_n

                # Print progress every 100k steps
                if global_offset - last_print_offset >= 100000:
                    pct = 100.0 * global_offset / total_steps
                    print(f"  [progress] {global_offset:,}/{total_steps:,} steps ({pct:.1f}%)")
                    last_print_offset = global_offset

    if train_offset != train_mm.shape[0]:
        raise ValueError(
            f"Train write mismatch for key '{key}': wrote {train_offset}, expected {train_mm.shape[0]}"
        )
    if val_offset != val_mm.shape[0]:
        raise ValueError(
            f"Val write mismatch for key '{key}': wrote {val_offset}, expected {val_mm.shape[0]}"
        )


def shape_nbytes(shape, dtype: np.dtype):
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize


def memmap_file_is_valid(meta):
    path = meta["path"]
    return path.is_file() and path.stat().st_size == shape_nbytes(meta["shape"], meta["dtype"])


def write_bytes_atomic(path: Path, data: bytes):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, data):
    payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    write_bytes_atomic(path, payload)


def build_merge_spec(
    checkpoint_dir: Path,
    shard_files,
    shard_steps,
    keys,
    key_shapes,
    total_steps: int,
    total_episodes: int,
    num_train_episodes: int,
    train_steps: int,
    val_steps: int,
):
    shards = []
    for path, steps in zip(shard_files, shard_steps):
        stat = path.stat()
        shards.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "steps": int(steps),
            }
        )

    return {
        "version": MANIFEST_VERSION,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "shards": shards,
        "keys": {
            key: {
                "tail_shape": list(key_shapes[key]),
                "dtype": np.dtype(get_dataset_dtype(key)).str,
            }
            for key in keys
        },
        "total_steps": int(total_steps),
        "total_episodes": int(total_episodes),
        "num_train_episodes": int(num_train_episodes),
        "train_steps": int(train_steps),
        "val_steps": int(val_steps),
    }


def write_merge_manifest(manifest_path: Path, spec, completed_keys):
    write_json_atomic(
        manifest_path,
        {
            "spec": spec,
            "completed_keys": sorted(completed_keys),
        },
    )


def prepare_tmp_dir(
    tmp_dir: Path,
    spec,
    train_meta,
    val_meta,
    adopt_existing_tmp: bool,
    rebuild_tmp: bool,
):
    if rebuild_tmp and tmp_dir.exists():
        print(f"[tmp] rebuilding: removing {tmp_dir}")
        shutil.rmtree(tmp_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_dir / MANIFEST_NAME

    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("spec") != spec:
            raise ValueError(
                f"Existing tmp manifest does not match this merge: {manifest_path}\n"
                "Use a different --tmp-dir or pass --rebuild-tmp to start over."
            )
        completed_keys = set(manifest.get("completed_keys", []))
        unknown = completed_keys - set(train_meta)
        if unknown:
            raise ValueError(f"Tmp manifest contains unknown completed keys: {sorted(unknown)}")

        invalid = {
            key
            for key in completed_keys
            if not memmap_file_is_valid(train_meta[key]) or not memmap_file_is_valid(val_meta[key])
        }
        if invalid:
            print(f"[tmp] incomplete/corrupt completed keys will be rebuilt: {sorted(invalid)}")
            completed_keys -= invalid
            write_merge_manifest(manifest_path, spec, completed_keys)

        print(
            f"[tmp] resume manifest found: {len(completed_keys)}/{len(train_meta)} "
            "merged key(s) complete"
        )
        return manifest_path, completed_keys

    existing_mmaps = list(tmp_dir.glob("*.mmap"))
    if existing_mmaps:
        if not adopt_existing_tmp:
            raise ValueError(
                f"Existing mmap files have no resume manifest: {tmp_dir}\n"
                "If they came from a run that had already reached NPZ saving, rerun once with "
                "--adopt-existing-tmp. Otherwise use --rebuild-tmp."
            )

        invalid = [
            str(meta["path"])
            for meta_dict in (train_meta, val_meta)
            for meta in meta_dict.values()
            if not memmap_file_is_valid(meta)
        ]
        if invalid:
            raise ValueError(
                "Cannot adopt existing tmp because expected mmap files are missing or have "
                f"the wrong size: {invalid}"
            )
        completed_keys = set(train_meta)
        write_merge_manifest(manifest_path, spec, completed_keys)
        print(f"[tmp] adopted existing mmap files: {len(completed_keys)} merged key(s) complete")
        return manifest_path, completed_keys

    completed_keys = set()
    write_merge_manifest(manifest_path, spec, completed_keys)
    print(f"[tmp] initialized resume manifest: {manifest_path}")
    return manifest_path, completed_keys


def fill_missing_memmaps(
    keys,
    key_shapes,
    shard_files,
    shard_steps,
    train_steps: int,
    train_meta,
    val_meta,
    chunk_mib: int,
    manifest_path: Path,
    spec,
    completed_keys,
):
    for i, key in enumerate(keys, start=1):
        if key in completed_keys:
            print(f"[merge] key {i}/{len(keys)}: {key} already complete, skipping")
            continue

        dtype = train_meta[key]["dtype"]
        tail = key_shapes[key]
        chunk_rows = infer_chunk_rows(tail, dtype, chunk_mib)
        print(
            f"[merge] key {i}/{len(keys)}: {key}, dtype={dtype}, tail={tail}, "
            f"chunk_rows={chunk_rows} (~{chunk_mib} MiB/chunk)"
        )

        train_mm = np.memmap(
            train_meta[key]["path"],
            mode="w+",
            dtype=dtype,
            shape=train_meta[key]["shape"],
        )
        val_mm = np.memmap(
            val_meta[key]["path"],
            mode="w+",
            dtype=dtype,
            shape=val_meta[key]["shape"],
        )
        try:
            fill_key_from_shards(
                key=key,
                shard_files=shard_files,
                shard_steps=shard_steps,
                train_steps=train_steps,
                train_mm=train_mm,
                val_mm=val_mm,
                dtype=dtype,
                tail_shape=tail,
                chunk_rows=chunk_rows,
            )
            train_mm.flush()
            val_mm.flush()
        finally:
            del train_mm
            del val_mm

        completed_keys.add(key)
        write_merge_manifest(manifest_path, spec, completed_keys)
        print(f"[merge] key {key} complete; resume manifest updated")


def make_npy_header(meta):
    header = {
        "descr": npformat.dtype_to_descr(np.dtype(meta["dtype"])),
        "fortran_order": False,
        "shape": tuple(meta["shape"]),
    }
    out = io.BytesIO()
    npformat.write_array_header_2_0(out, header)
    return out.getvalue()


def npz_member_is_complete(zf: zipfile.ZipFile, key: str, meta):
    member_name = f"{key}.npy"
    try:
        info = zf.getinfo(member_name)
    except KeyError:
        return False

    expected_size = len(make_npy_header(meta)) + shape_nbytes(meta["shape"], meta["dtype"])
    if info.file_size != expected_size:
        return False

    try:
        shape, dtype = read_npy_header_from_zip(zf, member_name)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile):
        return False
    return shape == tuple(meta["shape"]) and dtype == np.dtype(meta["dtype"])


def npz_is_complete(path: Path, meta_dict):
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            if len(names) != len(set(names)):
                return False
            return all(npz_member_is_complete(zf, key, meta) for key, meta in meta_dict.items())
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def zip_transaction_paths(partial_path: Path):
    txn_path = partial_path.with_name(f"{partial_path.name}.txn.json")
    tail_path = partial_path.with_name(f"{partial_path.name}.txn.tail")
    return txn_path, tail_path


def recover_zip_transaction(partial_path: Path):
    txn_path, tail_path = zip_transaction_paths(partial_path)
    if not txn_path.exists():
        if tail_path.exists():
            tail_path.unlink()
        return

    with open(txn_path, "r", encoding="utf-8") as f:
        txn = json.load(f)
    tail = tail_path.read_bytes()
    if len(tail) != int(txn["tail_size"]):
        raise ValueError(f"Invalid ZIP transaction backup: {tail_path}")

    start_dir = int(txn["start_dir"])
    with open(partial_path, "r+b") as f:
        f.seek(start_dir)
        f.write(tail)
        f.truncate(start_dir + len(tail))
        f.flush()
        os.fsync(f.fileno())

    txn_path.unlink()
    tail_path.unlink()
    print(f"[save] recovered interrupted ZIP member in {partial_path}")


def begin_zip_transaction(partial_path: Path):
    txn_path, tail_path = zip_transaction_paths(partial_path)
    with zipfile.ZipFile(partial_path, "r") as zf:
        start_dir = int(zf.start_dir)

    with open(partial_path, "rb") as f:
        f.seek(start_dir)
        tail = f.read()

    write_bytes_atomic(tail_path, tail)
    write_json_atomic(
        txn_path,
        {
            "version": 1,
            "start_dir": start_dir,
            "tail_size": len(tail),
        },
    )


def commit_zip_transaction(partial_path: Path):
    txn_path, tail_path = zip_transaction_paths(partial_path)
    txn_path.unlink(missing_ok=True)
    tail_path.unlink(missing_ok=True)


def stream_memmap_to_zip(
    zf: zipfile.ZipFile,
    key: str,
    meta,
    chunk_mib: int,
):
    mm = np.memmap(meta["path"], mode="r", dtype=meta["dtype"], shape=meta["shape"])
    header = make_npy_header(meta)
    total_bytes = shape_nbytes(meta["shape"], meta["dtype"])
    tail_shape = tuple(meta["shape"][1:])
    chunk_rows = infer_chunk_rows(tail_shape, meta["dtype"], chunk_mib)
    rows_total = int(meta["shape"][0])
    raw_written = 0
    last_print_bytes = 0
    last_print_time = time.time()
    started = last_print_time

    try:
        with zf.open(f"{key}.npy", "w", force_zip64=True) as dest:
            dest.write(header)
            for start in range(0, rows_total, chunk_rows):
                end = min(start + chunk_rows, rows_total)
                chunk = mm[start:end]
                view = memoryview(chunk).cast("B")
                dest.write(view)
                raw_written += view.nbytes

                now = time.time()
                if (
                    raw_written - last_print_bytes >= 10 * 1024**3
                    or now - last_print_time >= 60
                ):
                    pct = 100.0 if total_bytes == 0 else 100.0 * raw_written / total_bytes
                    rate = raw_written / max(now - started, 1e-9) / 1024**2
                    print(
                        f"  [save-progress] {key}: {raw_written / 1024**3:.1f}/"
                        f"{total_bytes / 1024**3:.1f} GiB raw ({pct:.1f}%), {rate:.1f} MiB/s"
                    )
                    last_print_bytes = raw_written
                    last_print_time = now
                del view
                del chunk
    finally:
        del mm


def quarantine_invalid_partial(partial_path: Path):
    quarantine_path = partial_path.with_name(
        f"{partial_path.name}.invalid-{time.time_ns()}"
    )
    os.replace(partial_path, quarantine_path)
    print(f"[save] moved invalid partial archive aside: {quarantine_path}")


def save_npz_resumable(
    path: Path,
    meta_dict,
    compressed: bool,
    compression_level: int,
    chunk_mib: int,
    label: str,
):
    if npz_is_complete(path, meta_dict):
        print(f"[save] {label}: complete output already exists, skipping: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f"{path.name}.partial")
    recover_zip_transaction(partial_path)

    if partial_path.exists():
        try:
            with zipfile.ZipFile(partial_path, "r") as zf:
                names = zf.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("duplicate member names")
                incomplete_members = [
                    key
                    for key, meta in meta_dict.items()
                    if f"{key}.npy" in names and not npz_member_is_complete(zf, key, meta)
                ]
                if incomplete_members:
                    raise ValueError(f"incomplete members: {incomplete_members}")
        except (OSError, ValueError, zipfile.BadZipFile):
            quarantine_invalid_partial(partial_path)

    if not partial_path.exists():
        with zipfile.ZipFile(partial_path, "w", allowZip64=True):
            pass

    with zipfile.ZipFile(partial_path, "r") as zf:
        completed_keys = {
            key for key, meta in meta_dict.items() if npz_member_is_complete(zf, key, meta)
        }
    print(
        f"[save] {label}: resume archive has {len(completed_keys)}/{len(meta_dict)} "
        f"complete key(s): {partial_path}"
    )

    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    mode = f"deflate level {compression_level}" if compressed else "uncompressed"
    for i, (key, meta) in enumerate(meta_dict.items(), start=1):
        if key in completed_keys:
            print(f"[save] {label} key {i}/{len(meta_dict)}: {key} already complete, skipping")
            continue

        raw_gib = shape_nbytes(meta["shape"], meta["dtype"]) / 1024**3
        print(
            f"[save] {label} key {i}/{len(meta_dict)}: {key}, {raw_gib:.1f} GiB raw, "
            f"{mode}, ~{chunk_mib} MiB/write"
        )
        t0 = time.time()
        begin_zip_transaction(partial_path)
        try:
            kwargs = {
                "mode": "a",
                "compression": compression,
                "allowZip64": True,
            }
            if compressed:
                kwargs["compresslevel"] = compression_level
            with zipfile.ZipFile(partial_path, **kwargs) as zf:
                stream_memmap_to_zip(zf, key, meta, chunk_mib)
        except BaseException:
            recover_zip_transaction(partial_path)
            raise
        else:
            commit_zip_transaction(partial_path)

        dt = time.time() - t0
        print(f"[save] {label} key {key} complete in {dt:.1f}s")

    if not npz_is_complete(partial_path, meta_dict):
        raise ValueError(f"Final verification failed for partial archive: {partial_path}")

    os.replace(partial_path, path)
    print(f"[save] {label}: finalized {path}")


def infer_num_train_episodes(total_episodes: int):
    # Collect script uses: num_val = num_train // 10, total = num_train + num_val
    # For common case where num_train is divisible by 10, this equals floor(10/11 * total).
    return int((10 * total_episodes) // 11)


def main():
    parser = argparse.ArgumentParser(description="Merge checkpoint shards with low RAM usage.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, required=True)
    parser.add_argument("--num-train-episodes", type=int, default=None)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--uncompressed", action="store_true")
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument(
        "--save-chunk-mib",
        type=int,
        default=256,
        help="Raw mmap bytes passed to the final NPZ writer per write (default: 256 MiB).",
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=1,
        help="ZIP deflate level for final NPZ files, 0-9 (default: 1; faster than NumPy's 6).",
    )
    parser.add_argument(
        "--adopt-existing-tmp",
        action="store_true",
        help="Trust full-size legacy mmap files and create a resume manifest for them.",
    )
    parser.add_argument(
        "--rebuild-tmp",
        action="store_true",
        help="Discard the selected tmp directory and rebuild all mmap files.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Build/adopt resumable mmap files, but do not write final NPZ files.",
    )
    args = parser.parse_args()

    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be > 0")
    if args.save_chunk_mib <= 0:
        raise ValueError("--save-chunk-mib must be > 0")
    if not 0 <= args.compress_level <= 9:
        raise ValueError("--compress-level must be between 0 and 9")
    if args.adopt_existing_tmp and args.rebuild_tmp:
        raise ValueError("--adopt-existing-tmp and --rebuild-tmp cannot be used together")

    shard_files = list_checkpoint_shards(args.checkpoint_dir)
    print(f"[start] checkpoint_dir={args.checkpoint_dir}")
    print(f"[start] found {len(shard_files)} shard(s)")
    if not shard_files:
        raise SystemExit(1)

    meta = scan_shards(shard_files)
    keys = meta["keys"]
    key_shapes = meta["key_shapes"]
    shard_steps = meta["shard_steps"]
    total_steps = int(meta["total_steps"])
    total_episodes = int(meta["total_episodes"])

    if args.num_train_episodes is None:
        num_train_episodes = infer_num_train_episodes(total_episodes)
        print(
            "[info] --num-train-episodes not provided, inferred from total episodes: "
            f"{num_train_episodes} (total={total_episodes})"
        )
    else:
        num_train_episodes = int(args.num_train_episodes)
        print(f"[info] using num_train_episodes={num_train_episodes}")

    if num_train_episodes <= 0:
        raise ValueError("num_train_episodes must be > 0")
    if num_train_episodes > total_episodes:
        raise ValueError(
            f"num_train_episodes ({num_train_episodes}) > total_episodes ({total_episodes})"
        )

    train_steps = compute_train_steps(shard_files, num_train_episodes)
    val_steps = total_steps - train_steps

    print(f"[info] keys={keys}")
    print(f"[info] total_episodes={total_episodes}, total_steps={total_steps}")
    print(f"[info] train_steps={train_steps}, val_steps={val_steps}")

    if args.tmp_dir is not None:
        tmp_dir = args.tmp_dir
    else:
        tmp_dir = args.checkpoint_dir / ".merge_lowmem_work"
        legacy_tmp_dir = args.checkpoint_dir / ".merge_lowmem_tmp"
        if not tmp_dir.exists() and legacy_tmp_dir.exists():
            tmp_dir = legacy_tmp_dir
            print(f"[info] using legacy tmp directory: {legacy_tmp_dir}")
    print(f"[info] tmp_dir={tmp_dir}")

    train_meta, val_meta = build_memmap_meta(tmp_dir, keys, key_shapes, train_steps, val_steps)
    spec = build_merge_spec(
        checkpoint_dir=args.checkpoint_dir,
        shard_files=shard_files,
        shard_steps=shard_steps,
        keys=keys,
        key_shapes=key_shapes,
        total_steps=total_steps,
        total_episodes=total_episodes,
        num_train_episodes=num_train_episodes,
        train_steps=train_steps,
        val_steps=val_steps,
    )

    outputs_complete = False
    try:
        manifest_path, completed_keys = prepare_tmp_dir(
            tmp_dir=tmp_dir,
            spec=spec,
            train_meta=train_meta,
            val_meta=val_meta,
            adopt_existing_tmp=args.adopt_existing_tmp,
            rebuild_tmp=args.rebuild_tmp,
        )
        fill_missing_memmaps(
            keys=keys,
            key_shapes=key_shapes,
            shard_files=shard_files,
            shard_steps=shard_steps,
            train_steps=train_steps,
            train_meta=train_meta,
            val_meta=val_meta,
            chunk_mib=args.chunk_mib,
            manifest_path=manifest_path,
            spec=spec,
            completed_keys=completed_keys,
        )

        if args.merge_only:
            print("[done] merge-only requested; final NPZ writing skipped")
            return

        args.train_path.parent.mkdir(parents=True, exist_ok=True)
        args.val_path.parent.mkdir(parents=True, exist_ok=True)

        save_npz_resumable(
            args.train_path,
            train_meta,
            compressed=not args.uncompressed,
            compression_level=args.compress_level,
            chunk_mib=args.save_chunk_mib,
            label="train",
        )
        save_npz_resumable(
            args.val_path,
            val_meta,
            compressed=not args.uncompressed,
            compression_level=args.compress_level,
            chunk_mib=args.save_chunk_mib,
            label="val",
        )
        outputs_complete = True
        print(f"[done] wrote:\n  {args.train_path}\n  {args.val_path}")
    finally:
        if outputs_complete and not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[cleanup] removed tmp_dir={tmp_dir}")
        else:
            reason = "--keep-tmp" if outputs_complete else "run incomplete or merge-only"
            print(f"[cleanup] kept tmp_dir={tmp_dir} ({reason})")


if __name__ == "__main__":
    main()
