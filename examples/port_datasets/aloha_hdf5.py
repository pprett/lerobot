"""
Convert HDF5 datasets to LeRobot format.

Expected HDF5 schema:
├action    [float64: N × action_dim]
├image
│ └robot
│   └cam
│     ├front    [uint8: N × compressed_size] (JPEG compressed)
│     └wrist    [uint8: N × compressed_size] (JPEG compressed)
└lowdim
  └robot
    └follower
      └state
        └qpos    [float64: N × state_dim]
  └state_machine
    └phase   [float64: N × n_phases]  (one-hot, optional)

If `lowdim/state_machine/phase` is present, contiguous runs of the same phase id
are written as SARM **dense** subtask annotations, paired with an auto-generated
single-stage **sparse** `task` annotation per episode (matches the
``--dense-only`` mode of
`lerobot.data_processing.sarm_annotations.subtask_annotation`). Phase ids are
mapped to human-readable names through the optional `--phase-labels` JSON file
(maps str(index) -> name); without it, names default to `phase_<idx>`.

Usage:
    python aloha_hdf5.py \
        --hdf5-data /path/to/hdf5/files \
        --repo-id user/dataset_name \
        --robot so100 \
        --hz 30 \
        --local-dir /tmp/lerobot_datasets \
        --single-task "pick up the cube" \
        --phase-labels examples/port_datasets/phase_labels.json
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from lerobot.data_processing.sarm_annotations.subtask_annotation import (
    Subtask,
    SubtaskAnnotation,
    Timestamp,
    compute_temporal_proportions,
    save_annotations_to_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HDF5 paths for this schema
ACTION_PATH = "action"
STATE_PATH = "lowdim/robot/follower/state/qpos"
IMAGE_BASE_PATH = "image/robot/cam"
PHASE_PATH = "lowdim/state_machine/phase"
IMG_KEY_BLACKLIST = {"front_overlay"}


def load_phase_labels(path: Path | None) -> dict[int, str] | None:
    """Load phase index -> name mapping from a JSON file."""
    if path is None:
        return None
    raw = json.loads(path.read_text())
    return {int(k): str(v) for k, v in raw.items()}


def _format_mmss(seconds: float) -> str:
    """Format seconds as MM:SS to match the SARM annotation schema."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def extract_phase_annotation(
    ep: h5py.File,
    fps: int,
    labels: dict[int, str] | None,
) -> SubtaskAnnotation | None:
    """Convert HDF5 per-frame phase one-hot into a sparse SubtaskAnnotation.

    Contiguous runs of the same phase id become individual subtask segments,
    in temporal order. Returns None if the phase dataset is absent.
    """
    if PHASE_PATH not in ep:
        return None

    phase = ep[PHASE_PATH][:]
    if phase.ndim != 2 or phase.shape[0] == 0:
        return None

    indices = phase.argmax(axis=1)
    n_frames = indices.shape[0]
    change_points = np.where(np.diff(indices) != 0)[0] + 1
    boundaries = np.concatenate([[0], change_points, [n_frames]])

    subtasks: list[Subtask] = []
    for i in range(len(boundaries) - 1):
        start_frame = int(boundaries[i])
        end_frame = int(boundaries[i + 1])
        idx = int(indices[start_frame])
        name = labels[idx] if labels is not None else f"phase_{idx}"
        subtasks.append(
            Subtask(
                name=name,
                timestamps=Timestamp(
                    start=_format_mmss(start_frame / fps),
                    end=_format_mmss(end_frame / fps),
                ),
            )
        )
    return SubtaskAnnotation(subtasks=subtasks)


def get_image_keys(episode_file: Path) -> list[str]:
    """Discover available camera keys from the HDF5 file."""
    with h5py.File(episode_file, "r") as ep:
        image_group = ep[IMAGE_BASE_PATH]
        return [k for k in image_group if k not in IMG_KEY_BLACKLIST]


def decode_image(data: np.ndarray) -> np.ndarray:
    """Decode JPEG-compressed image data."""
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def build_features(episode_file: Path) -> dict:
    """Build feature specification from a sample episode."""
    with h5py.File(episode_file, "r") as ep:
        action = ep[ACTION_PATH]
        state = ep[STATE_PATH]
        image_keys = get_image_keys(episode_file)

        features = {
            "action": {
                "dtype": "float32",
                "shape": (action.shape[1],),
                "names": None,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state.shape[1],),
                "names": None,
            },
        }

        # Get dimensions for each camera separately (they may differ)
        for image_key in image_keys:
            sample_img = decode_image(ep[f"{IMAGE_BASE_PATH}/{image_key}"][0])
            h, w, c = sample_img.shape
            features[f"observation.images.{image_key}"] = {
                "dtype": "video",
                "shape": (c, h, w),
                "names": ["channels", "height", "width"],
            }

    return features


def convert_dataset(
    hdf5_data: Path,
    repo_id: str,
    local_dir: Path,
    hz: int,
    robot: str | None = None,
    single_task: str | None = None,
    phase_labels_path: Path | None = None,
):
    """Convert HDF5 dataset to LeRobot format."""
    # LeRobotDataset.create uses root directly (doesn't append repo_id)
    output_path = local_dir / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not hdf5_data.exists():
        raise FileNotFoundError(f"{hdf5_data} does not exist.")

    # Find and sort episode files (support multiple naming patterns)
    episode_files = list(hdf5_data.glob("episode_*.hdf5"))
    if not episode_files:
        episode_files = list(hdf5_data.glob("rec_*.hdf"))
    if not episode_files:
        episode_files = list(hdf5_data.glob("*.hdf5")) + list(hdf5_data.glob("*.hdf"))

    # Sort by filename (works for both episode_N and timestamp-based names)
    episode_files = sorted(episode_files, key=lambda f: f.name)

    if not episode_files:
        raise FileNotFoundError(f"No HDF5 files found in {hdf5_data}")

    logger.info(f"Found {len(episode_files)} episodes")

    # Build features from first episode
    features = build_features(episode_files[0])
    image_keys = get_image_keys(episode_files[0])
    logger.info(f"Features: {features}")
    logger.info(f"Image keys: {image_keys}")

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=hz,
        robot_type=robot,
        features=features,
        image_writer_threads=4,
    )

    if single_task is None:
        raise ValueError("--single-task is required (per-episode task files not implemented)")

    phase_labels = load_phase_labels(phase_labels_path)
    if phase_labels is not None:
        logger.info(f"Loaded {len(phase_labels)} phase labels from {phase_labels_path}")
    dense_annotations: dict[int, SubtaskAnnotation] = {}
    episode_num_frames: dict[int, int] = {}

    # Convert each episode
    for episode_idx, episode_file in enumerate(tqdm(episode_files, desc="Converting episodes")):
        with h5py.File(episode_file, "r") as ep:
            num_frames = ep[ACTION_PATH].shape[0]

            for frame_idx in range(num_frames):
                frame = {
                    "task": single_task,
                    "action": torch.from_numpy(
                        ep[ACTION_PATH][frame_idx].astype(np.float32)
                    ),
                    "observation.state": torch.from_numpy(
                        ep[STATE_PATH][frame_idx].astype(np.float32)
                    ),
                }

                # Decode and add images
                for image_key in image_keys:
                    img = decode_image(ep[f"{IMAGE_BASE_PATH}/{image_key}"][frame_idx])
                    # Convert HWC -> CHW
                    img = np.transpose(img, (2, 0, 1))
                    frame[f"observation.images.{image_key}"] = torch.from_numpy(img)

                dataset.add_frame(frame)

            annotation = extract_phase_annotation(ep, hz, phase_labels)

        dataset.save_episode()
        episode_num_frames[episode_idx] = num_frames
        if annotation is not None:
            dense_annotations[episode_idx] = annotation

    dataset.finalize()
    logger.info(f"Dataset saved to {output_path}")

    if dense_annotations:
        # Auto-generate single-stage sparse "task" annotations covering each episode.
        sparse_annotations: dict[int, SubtaskAnnotation] = {
            ep_idx: SubtaskAnnotation(
                subtasks=[
                    Subtask(
                        name="task",
                        timestamps=Timestamp(
                            start="00:00",
                            end=_format_mmss(episode_num_frames[ep_idx] / hz),
                        ),
                    )
                ]
            )
            for ep_idx in dense_annotations
        }

        save_annotations_to_dataset(output_path, sparse_annotations, hz, prefix="sparse")
        save_annotations_to_dataset(output_path, dense_annotations, hz, prefix="dense")

        meta_dir = output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        with open(meta_dir / "temporal_proportions_sparse.json", "w") as f:
            json.dump({"task": 1.0}, f, indent=2)

        subtask_order = list(phase_labels.values()) if phase_labels else None
        dense_proportions = compute_temporal_proportions(dense_annotations, hz, subtask_order)
        with open(meta_dir / "temporal_proportions_dense.json", "w") as f:
            json.dump(dense_proportions, f, indent=2)

        logger.info(
            f"Wrote sparse + dense phase annotations for {len(dense_annotations)} episodes "
            f"and temporal proportions to {meta_dir}"
        )
    else:
        logger.info(f"No '{PHASE_PATH}' found in HDF5 files; skipped phase annotation.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 datasets to LeRobot format"
    )
    parser.add_argument(
        "--hdf5-data",
        type=Path,
        required=True,
        help="Directory containing episode_*.hdf5 files",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID (e.g. 'user/dataset_name')",
    )
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="Robot type (e.g. 'koch', 'aloha', 'so100')",
    )
    parser.add_argument(
        "--hz",
        type=int,
        required=True,
        help="Control frequency in Hz",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local directory to store the converted dataset",
    )
    parser.add_argument(
        "--single-task",
        type=str,
        required=True,
        help="Task description for all episodes",
    )
    parser.add_argument(
        "--phase-labels",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping str(phase_index) -> human-readable name. "
            "If provided and the HDF5 files contain '" + PHASE_PATH + "', "
            "contiguous phase runs are written as SARM dense subtask annotations "
            "with a single-stage sparse 'task' annotation per episode."
        ),
    )

    args = parser.parse_args()
    convert_dataset(
        hdf5_data=args.hdf5_data,
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        hz=args.hz,
        robot=args.robot,
        single_task=args.single_task,
        phase_labels_path=args.phase_labels,
    )


if __name__ == "__main__":
    main()
