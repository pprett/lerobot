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
is written as a ``episode_{idx:06d}.hdf`` following the layout used by
https://github.com/tonyzhaozh/act:

    /action                           float (T, A)
    /progress                         float (T,)             [if SARM progress is present]
    /observations/qpos                float (T, S)
    /observations/qvel                float (T, S)          [if present]
    /observations/effort              float (T, S)          [if present]
    /observations/images/<camera>     uint8 (T, H, W, 3)    uncompressed
                                      uint8 (T, max_bytes)  JPEG-compressed

Sensor and camera output prefixes are configurable. For example, use
``--sensor-prefix lowdim/robot/follower/state --camera-prefix image/robot/cam``
to write qpos under ``/lowdim/robot/follower/state/qpos`` and cameras under
``/image/robot/cam/<camera>``.

If the LeRobot dataset has a ``sarm_progress.parquet`` sidecar, the selected
progress score column is written into each episode HDF5 file as ``/progress``.

Example:

    python examples/port_datasets/unport_aloha.py \\
        --repo-id lerobot/aloha_sim_insertion_human \\
        --output-dir /tmp/aloha_sim_insertion_human
"""

import argparse
import contextlib
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
DEFAULT_SENSOR_PREFIX = "observations"
DEFAULT_CAMERA_PREFIX = "observations/images"
SARM_PROGRESS_FILENAME = "sarm_progress.parquet"
SARM_PROGRESS_INDEX_COLUMNS = ("episode_index", "frame_index")
DEFAULT_SARM_PROGRESS_COLUMN = "progress_dense"


def _stack_column(series: pd.Series, dtype=np.float32) -> np.ndarray:
    """Stack a parquet column of vector values into a (T, D) numpy array."""
    return np.stack([np.asarray(x, dtype=dtype) for x in series.to_list()], axis=0)


def _join_hdf5_path(prefix: str, name: str) -> str:
    prefix = prefix.strip("/")
    name = name.strip("/")
    return f"{prefix}/{name}" if prefix else name


def _load_episode_parquet(dataset: LeRobotDataset, ep_idx: int) -> pd.DataFrame:
    parquet_path = dataset.root / dataset.meta.get_data_file_path(ep_idx)
    df = pd.read_parquet(parquet_path)
    return df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)


def _episode_num_frames(dataset: LeRobotDataset, ep_idx: int) -> int:
    ep = dataset.meta.episodes[ep_idx]
    return ep["dataset_to_index"] - ep["dataset_from_index"]


def _resolve_sarm_progress_path(dataset: LeRobotDataset, progress_path: Path | None) -> Path | None:
    if progress_path is not None:
        return progress_path

    progress_path = dataset.root / SARM_PROGRESS_FILENAME
    if progress_path.exists():
        return progress_path

    # LeRobotDataset downloads only standard data/video files for selected episodes.
    # Try to fetch the optional sidecar from the Hub before deciding it is absent.
    with contextlib.suppress(Exception):
        dataset.pull_from_repo(allow_patterns=SARM_PROGRESS_FILENAME)
    return progress_path if progress_path.exists() else None


def load_sarm_progress_scores(
    dataset: LeRobotDataset,
    ep_indices: list[int],
    progress_path: Path | None = None,
    progress_column: str = DEFAULT_SARM_PROGRESS_COLUMN,
) -> dict[int, np.ndarray]:
    """Load per-episode SARM progress scores to write into HDF5 files."""
    progress_path = _resolve_sarm_progress_path(dataset, progress_path)
    if progress_path is None:
        logging.info(f"No {SARM_PROGRESS_FILENAME} found; skipping SARM progress score export")
        return {}
    if not progress_path.exists():
        raise FileNotFoundError(f"SARM progress score file not found: {progress_path}")

    df = pd.read_parquet(progress_path)
    missing_columns = [col for col in SARM_PROGRESS_INDEX_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{progress_path} is missing required columns: {missing_columns}")
    if progress_column not in df.columns:
        progress_columns = [col for col in df.columns if "progress" in col.lower()]
        raise ValueError(
            f"{progress_path} does not contain progress column '{progress_column}'. "
            f"Available progress columns: {progress_columns}"
        )
    if df.duplicated(["episode_index", "frame_index"]).any():
        raise ValueError(f"{progress_path} contains duplicate episode/frame progress rows")

    progress_by_episode: dict[int, np.ndarray] = {}
    for ep_idx in ep_indices:
        num_frames = _episode_num_frames(dataset, ep_idx)
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        expected_frame_indices = np.arange(num_frames)
        actual_frame_indices = ep_df["frame_index"].to_numpy(dtype=np.int64)
        if len(ep_df) != num_frames or not np.array_equal(actual_frame_indices, expected_frame_indices):
            raise ValueError(
                f"Progress scores for episode {ep_idx} do not match exported frames: "
                f"expected frame_index 0..{num_frames - 1}, found {len(ep_df)} rows"
            )
        progress_by_episode[ep_idx] = ep_df[progress_column].to_numpy(dtype=np.float32)

    logging.info(
        f"Loaded SARM progress column '{progress_column}' from {progress_path} "
        f"for {len(progress_by_episode)} episode(s)"
    )
    return progress_by_episode


def _decode_episode_video(dataset: LeRobotDataset, ep_idx: int, image_key: str) -> np.ndarray:
    """Decode every frame of an episode from the backing mp4 as (T, H, W, 3) uint8 RGB."""
    ep = dataset.meta.episodes[ep_idx]
    num_frames = _episode_num_frames(dataset, ep_idx)
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
        buf = entry.get("bytes") if isinstance(entry, dict) else entry
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
    progress: np.ndarray | None,
    compress: bool,
    jpeg_quality: int,
    is_sim: bool,
    sensor_prefix: str,
    camera_prefix: str,
) -> None:
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["sim"] = is_sim
        f.attrs["compress"] = compress

        f.create_dataset(_join_hdf5_path(sensor_prefix, "qpos"), data=qpos)
        if qvel is not None:
            f.create_dataset(_join_hdf5_path(sensor_prefix, "qvel"), data=qvel)
        if effort is not None:
            f.create_dataset(_join_hdf5_path(sensor_prefix, "effort"), data=effort)

        f.create_dataset("action", data=action)
        if progress is not None:
            f.create_dataset("progress", data=progress)

        for cam, frames in images.items():
            image_path = _join_hdf5_path(camera_prefix, cam)
            if compress:
                padded, lengths = _compress_jpeg(frames, quality=jpeg_quality)
                ds = f.create_dataset(image_path, data=padded, dtype="uint8")
                ds.attrs["compress_len"] = lengths
            else:
                f.create_dataset(image_path, data=frames, dtype="uint8")


def _camera_name(image_key: str) -> str:
    return image_key[len(IMAGE_PREFIX):] if image_key.startswith(IMAGE_PREFIX) else image_key


def export_episode(
    dataset: LeRobotDataset,
    ep_idx: int,
    output_dir: Path,
    compress: bool,
    jpeg_quality: int,
    is_sim: bool,
    progress: np.ndarray | None = None,
    sensor_prefix: str = DEFAULT_SENSOR_PREFIX,
    camera_prefix: str = DEFAULT_CAMERA_PREFIX,
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

    hdf5_path = output_dir / f"episode_{ep_idx:06d}.hdf"
    _write_hdf5(
        hdf5_path,
        qpos=qpos,
        action=action,
        images=images,
        qvel=qvel,
        effort=effort,
        progress=progress,
        compress=compress,
        jpeg_quality=jpeg_quality,
        is_sim=is_sim,
        sensor_prefix=sensor_prefix,
        camera_prefix=camera_prefix,
    )


def unport_aloha(
    repo_id: str,
    output_dir: Path,
    root: Path | None = None,
    episodes: list[int] | None = None,
    compress: bool = True,
    jpeg_quality: int = 50,
    is_sim: bool = False,
    progress_path: Path | None = None,
    progress_column: str = DEFAULT_SARM_PROGRESS_COLUMN,
    export_progress: bool = True,
    sensor_prefix: str = DEFAULT_SENSOR_PREFIX,
    camera_prefix: str = DEFAULT_CAMERA_PREFIX,
) -> None:
    dataset = LeRobotDataset(repo_id=repo_id, root=root, episodes=episodes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ep_indices = episodes if episodes is not None else list(range(dataset.meta.total_episodes))
    logging.info(f"Exporting {len(ep_indices)} episode(s) from {repo_id} to {output_dir}")

    progress_by_episode = (
        load_sarm_progress_scores(dataset, ep_indices, progress_path, progress_column) if export_progress else {}
    )

    for ep_idx in tqdm(ep_indices, desc="Exporting episodes"):
        export_episode(
            dataset,
            ep_idx,
            output_dir,
            compress,
            jpeg_quality,
            is_sim,
            progress=progress_by_episode.get(ep_idx),
            sensor_prefix=sensor_prefix,
            camera_prefix=camera_prefix,
        )


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
    parser.add_argument(
        "--sensor-prefix",
        type=str,
        default=DEFAULT_SENSOR_PREFIX,
        help=(
            "HDF5 group prefix for sensor datasets qpos/qvel/effort. "
            f"Defaults to '{DEFAULT_SENSOR_PREFIX}'."
        ),
    )
    parser.add_argument(
        "--camera-prefix",
        type=str,
        default=DEFAULT_CAMERA_PREFIX,
        help=(
            "HDF5 group prefix for camera datasets. "
            f"Defaults to '{DEFAULT_CAMERA_PREFIX}'."
        ),
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help=(
            "Optional path to an input SARM progress sidecar parquet. Defaults to "
            f"'<dataset_root>/{SARM_PROGRESS_FILENAME}' and is skipped if absent."
        ),
    )
    parser.add_argument(
        "--progress-column",
        type=str,
        default=DEFAULT_SARM_PROGRESS_COLUMN,
        help="Progress score column from the sidecar parquet to write as '/progress'.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not write optional SARM progress scores into the HDF5 files.",
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
        progress_path=args.progress_path,
        progress_column=args.progress_column,
        export_progress=not args.no_progress,
        sensor_prefix=args.sensor_prefix,
        camera_prefix=args.camera_prefix,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
