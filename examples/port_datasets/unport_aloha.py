#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Export a LeRobotDataset to ALOHA-style HDF5 files (one file per episode).

This is the inverse of the legacy ``aloha_hdf5_format`` ingestion. Each episode
is written as a ``episode_{idx:06d}.hdf5`` following the layout used by
https://github.com/tonyzhaozh/act:

    /action                           float (T, A)
    /observations/qpos                float (T, S)
    /observations/qvel                float (T, S)          [if present]
    /observations/effort              float (T, S)          [if present]
    /observations/images/<camera>     uint8 (T, H, W, 3)    uncompressed
                                      uint8 (T, max_bytes)  JPEG-compressed

Example:

    python examples/port_datasets/unport_aloha.py \\
        --repo-id lerobot/aloha_sim_insertion_human \\
        --output-dir /tmp/aloha_sim_insertion_human
"""

import argparse
import io
import logging
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"
VELOCITY_KEY = "observation.velocity"
EFFORT_KEY = "observation.effort"
ACTION_KEY = "action"


def _stack_column(series: pd.Series, dtype=np.float32) -> np.ndarray:
    """Stack a parquet column of vector values into a (T, D) numpy array."""
    return np.stack([np.asarray(x, dtype=dtype) for x in series.to_list()], axis=0)


def _load_episode_parquet(dataset: LeRobotDataset, ep_idx: int) -> pd.DataFrame:
    parquet_path = dataset.root / dataset.meta.get_data_file_path(ep_idx)
    df = pd.read_parquet(parquet_path)
    return (
        df[df["episode_index"] == ep_idx]
        .sort_values("frame_index")
        .reset_index(drop=True)
    )


def _decode_episode_video(dataset: LeRobotDataset, ep_idx: int, image_key: str) -> np.ndarray:
    """Decode every frame of an episode from the backing mp4 as (T, H, W, 3) uint8 RGB."""
    ep = dataset.meta.episodes[ep_idx]
    num_frames = ep["dataset_to_index"] - ep["dataset_from_index"]
    fps = dataset.meta.fps
    from_timestamp = ep[f"videos/{image_key}/from_timestamp"]
    timestamps = [from_timestamp + i / fps for i in range(num_frames)]

    video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, image_key)
    frames = decode_video_frames(
        video_path, timestamps, dataset.tolerance_s, dataset.video_backend
    )
    # (T, C, H, W) float in [0, 1] -> (T, H, W, C) uint8 RGB
    frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _decode_parquet_images(series: pd.Series) -> np.ndarray:
    """Decode a parquet ``datasets.Image()`` column into (T, H, W, 3) uint8 RGB."""
    frames = []
    for entry in series.to_list():
        if isinstance(entry, dict):
            buf = entry.get("bytes")
        else:
            buf = entry
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        frames.append(np.asarray(img))
    return np.stack(frames, axis=0)


def _compress_jpeg(frames: np.ndarray, quality: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode each RGB frame as JPEG and zero-pad to a fixed row width."""
    encoded = []
    for img in frames:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        encoded.append(np.asarray(buf).flatten())
    lengths = np.array([e.size for e in encoded], dtype=np.int32)
    max_len = int(lengths.max())
    padded = np.zeros((len(encoded), max_len), dtype=np.uint8)
    for i, buf in enumerate(encoded):
        padded[i, : buf.size] = buf
    return padded, lengths


def _write_hdf5(
    hdf5_path: Path,
    qpos: np.ndarray,
    action: np.ndarray,
    images: dict[str, np.ndarray],
    qvel: np.ndarray | None,
    effort: np.ndarray | None,
    compress: bool,
    jpeg_quality: int,
    is_sim: bool,
) -> None:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["sim"] = is_sim

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        if qvel is not None:
            obs.create_dataset("qvel", data=qvel)
        if effort is not None:
            obs.create_dataset("effort", data=effort)

        f.create_dataset("action", data=action)

        images_group = obs.create_group("images")
        for cam, frames in images.items():
            if compress:
                padded, lengths = _compress_jpeg(frames, quality=jpeg_quality)
                ds = images_group.create_dataset(cam, data=padded, dtype="uint8")
                ds.attrs["compress_len"] = lengths
            else:
                images_group.create_dataset(cam, data=frames, dtype="uint8")


def _camera_name(image_key: str) -> str:
    return image_key[len(IMAGE_PREFIX):] if image_key.startswith(IMAGE_PREFIX) else image_key


def export_episode(
    dataset: LeRobotDataset,
    ep_idx: int,
    output_dir: Path,
    compress: bool,
    jpeg_quality: int,
    is_sim: bool,
) -> None:
    ep_df = _load_episode_parquet(dataset, ep_idx)
    features = dataset.features

    qpos = _stack_column(ep_df[STATE_KEY])
    action = _stack_column(ep_df[ACTION_KEY])
    qvel = _stack_column(ep_df[VELOCITY_KEY]) if VELOCITY_KEY in features else None
    effort = _stack_column(ep_df[EFFORT_KEY]) if EFFORT_KEY in features else None

    images: dict[str, np.ndarray] = {}
    for key in dataset.meta.video_keys:
        images[_camera_name(key)] = _decode_episode_video(dataset, ep_idx, key)
    for key in dataset.meta.image_keys:
        images[_camera_name(key)] = _decode_parquet_images(ep_df[key])

    hdf5_path = output_dir / f"episode_{ep_idx:06d}.hdf5"
    _write_hdf5(
        hdf5_path,
        qpos=qpos,
        action=action,
        images=images,
        qvel=qvel,
        effort=effort,
        compress=compress,
        jpeg_quality=jpeg_quality,
        is_sim=is_sim,
    )


def unport_aloha(
    repo_id: str,
    output_dir: Path,
    root: Path | None = None,
    episodes: list[int] | None = None,
    compress: bool = True,
    jpeg_quality: int = 50,
    is_sim: bool = False,
) -> None:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_indices = episodes if episodes is not None else list(range(dataset.meta.total_episodes))
    logging.info(f"Exporting {len(ep_indices)} episode(s) from {repo_id} to {output_dir}")

    for ep_idx in tqdm(ep_indices, desc="Exporting episodes"):
        export_episode(dataset, ep_idx, output_dir, compress, jpeg_quality, is_sim)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face Hub repo id (e.g. 'lerobot/aloha_sim_insertion_human').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the 'episode_*.hdf5' files will be written.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local dataset root. If set, data is read from disk instead of the hub.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Subset of 0-indexed episode indices to export. Defaults to all episodes.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Write raw (T, H, W, 3) uint8 image arrays instead of JPEG-compressed bytes.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=50,
        help="JPEG quality in [1, 100] when compressing images.",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Set the HDF5 root attribute 'sim' to True (matches ALOHA simulation datasets).",
    )

    args = parser.parse_args()

    unport_aloha(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        root=args.root,
        episodes=args.episodes,
        compress=not args.no_compress,
        jpeg_quality=args.jpeg_quality,
        is_sim=args.sim,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
