"""Data registry for the dual-arm LT suction dataset."""

import fcntl
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotSingleDataset,
    ModalityConfig,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.video import get_all_frames


class DiskMMapLeRobotSingleDataset(LeRobotSingleDataset):
    """Persist resized RGB frames on disk and access them through read-only mmap."""

    def __init__(self, cache_dir, img_resize=(224, 224), *args, **kwargs):
        if isinstance(img_resize, int):
            img_resize = (img_resize, img_resize)
        self.img_resize = tuple(int(value) for value in img_resize)
        if len(self.img_resize) != 2:
            raise ValueError(f"img_resize must contain two values, got {self.img_resize!r}")

        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths
        self.cached_frames = {}

        total_frames = int(np.sum(self.trajectory_lengths))
        width, height = self.img_resize
        expected_shape = (total_frames, height, width, 3)
        for video_key in self.modality_keys["video"]:
            clean_key = video_key.removeprefix("video.")
            cache_path = self.cache_dir / f"{clean_key}_{width}x{height}_uint8.npy"
            lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                if not cache_path.exists():
                    self._build_camera_cache(
                        clean_key=clean_key,
                        cache_path=cache_path,
                        expected_shape=expected_shape,
                    )

                frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                if frames.dtype != np.uint8 or tuple(frames.shape) != expected_shape:
                    raise RuntimeError(
                        f"Invalid video cache {cache_path}: "
                        f"dtype={frames.dtype}, shape={tuple(frames.shape)}, "
                        f"expected uint8 {expected_shape}. Remove this cache file and restart."
                    )
                self.cached_frames[clean_key] = frames
                print(f"Using mmap cache {clean_key}: {frames.shape} at {cache_path}")

    def _build_camera_cache(self, clean_key, cache_path, expected_shape):
        incomplete_path = Path(str(cache_path) + ".incomplete")
        mmap_array = np.lib.format.open_memmap(
            incomplete_path,
            mode="w+",
            dtype=np.uint8,
            shape=expected_shape,
        )
        offset = 0
        try:
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Building mmap cache for {clean_key}",
            ):
                video_path = self.get_video_path(trajectory_id, clean_key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=self.img_resize,
                )
                expected_frames = int(trajectory_length)
                if len(frames) != expected_frames:
                    raise RuntimeError(
                        f"Frame count mismatch for {video_path}: "
                        f"decoded={len(frames)}, expected={expected_frames}"
                    )
                mmap_array[offset : offset + expected_frames] = frames
                offset += expected_frames

            if offset != expected_shape[0]:
                raise RuntimeError(
                    f"Cache frame count mismatch for {clean_key}: "
                    f"wrote={offset}, expected={expected_shape[0]}"
                )
            mmap_array.flush()
        finally:
            del mmap_array

        os.replace(incomplete_path, cache_path)
        print(f"Completed mmap cache: {cache_path}")

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(
            step_indices,
            self.trajectory_lengths[trajectory_index] - 1,
        )
        if not key.startswith("video."):
            raise ValueError(f"Video key must start with 'video.', got {key}")
        clean_key = key.removeprefix("video.")
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[clean_key][absolute_indices]


class DualArmLTV1H100Config:
    """v1: full sensor state and active Cartesian/digital actions."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.left_side",
        "video.left_wrist",
        "video.right_wrist",
        "video.right_side",
    ]
    state_keys = [
        "state.arm_0_pose",
        "state.arm_1_pose",
        "state.digital_input",
        "state.force_torque_compensated",
    ]
    action_keys = ["action.arm_0_pose", "action.arm_1_pose", "action.digital_output"]
    state_key_dims = {
        "state.arm_0_pose": 7,
        "state.arm_1_pose": 7,
        "state.digital_input": 17,
        "state.force_torque_compensated": 12,
    }
    action_key_dims = {"action.arm_0_pose": 7, "action.arm_1_pose": 7, "action.digital_output": 4}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(100))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.arm_0_pose": "q99",
                    "state.arm_1_pose": "q99",
                    "state.digital_input": "binary",
                    "state.force_torque_compensated": "q99",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.arm_0_pose": "q99",
                    "action.arm_1_pose": "q99",
                    "action.digital_output": "binary",
                },
            ),
        ])

    def make_dataset(self, **kwargs):
        """Use a persistent mmap video cache when requested by the recipe."""
        kwargs.pop("dataset_name", None)
        data_cfg = kwargs.get("data_cfg")
        cache_videos = bool(
            data_cfg.get("load_all_data_for_training", False)
            if data_cfg is not None
            else False
        )
        if not cache_videos:
            return LeRobotSingleDataset(**kwargs)

        image_size = data_cfg.get("obs_image_size", (224, 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        else:
            image_size = tuple(int(value) for value in image_size)
        if len(image_size) != 2:
            raise ValueError(
                f"obs_image_size must be an int or a 2-item sequence, got {image_size!r}"
            )

        cache_dir = data_cfg.get(
            "video_cache_dir",
            "/root/data/cache/starvla/dualarm_lt_suck_224_mmap",
        )
        return DiskMMapLeRobotSingleDataset(
            cache_dir=cache_dir,
            img_resize=image_size,
            **kwargs,
        )


class DualArmLTV1H50Config(DualArmLTV1H100Config):
    """H=50 variant sharing the same modalities, normalization, and video cache."""

    action_indices = list(range(50))


class DualArmLTV2H100Config(DualArmLTV1H100Config):
    """Suction-only v2: moving arm pose, digital I/O, and no force input."""

    state_keys = ["state.arm_0_pose", "state.digital_input"]
    action_keys = ["action.arm_0_pose", "action.digital_output"]
    state_key_dims = {"state.arm_0_pose": 7, "state.digital_input": 17}
    action_key_dims = {"action.arm_0_pose": 7, "action.digital_output": 4}

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.arm_0_pose": "q99",
                    "state.digital_input": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.arm_0_pose": "q99",
                    "action.digital_output": "binary",
                },
            ),
        ])


class DualArmLTV2H50Config(DualArmLTV2H100Config):
    """H=50 variant of the suction-only v2 mapping."""

    action_indices = list(range(50))


class DualArmLTV2H50ThreeCameraConfig(DualArmLTV2H50Config):
    """QwenPI v3 variant using three cameras and excluding left_wrist."""

    video_keys = [
        "video.left_side",
        "video.right_wrist",
        "video.right_side",
    ]


class DualArmLTV3H50Config(DualArmLTV2H50Config):
    """V3：状态只保留运动臂位姿，动作仍保留位姿和 DO。"""

    state_keys = ["state.arm_0_pose"]
    state_key_dims = {"state.arm_0_pose": 7}

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={"state.arm_0_pose": "q99"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.arm_0_pose": "q99",
                    "action.digital_output": "binary",
                },
            ),
        ])


class DualArmLTV3H50ThreeCameraConfig(DualArmLTV3H50Config):
    """V3 QwenPI：保留三相机顺序，排除 left_wrist。"""

    video_keys = list(DualArmLTV2H50ThreeCameraConfig.video_keys)


ROBOT_TYPE_CONFIG_MAP = {
    "dualarm_lt_v1_h100": DualArmLTV1H100Config(),
    "dualarm_lt_v1_h50": DualArmLTV1H50Config(),
    "dualarm_lt_v2_h100": DualArmLTV2H100Config(),
    "dualarm_lt_v2_h50": DualArmLTV2H50Config(),
    "dualarm_lt_v2_h50_3cam": DualArmLTV2H50ThreeCameraConfig(),
    "dualarm_lt_v3_h50": DualArmLTV3H50Config(),
    "dualarm_lt_v3_h50_3cam": DualArmLTV3H50ThreeCameraConfig(),
}

DATASET_NAMED_MIXTURES = {
    "dualarm_lt_suck_v1_h100": [("suck_lt", 1.0, "dualarm_lt_v1_h100")],
    "dualarm_lt_suck_v1_h50": [("suck_lt", 1.0, "dualarm_lt_v1_h50")],
    "dualarm_lt_suck_v2_h100": [("suck_lt", 1.0, "dualarm_lt_v2_h100")],
    "dualarm_lt_suck_v2_h50": [("suck_lt", 1.0, "dualarm_lt_v2_h50")],
    "dualarm_lt_suck_v2_h50_3cam": [("suck_lt", 1.0, "dualarm_lt_v2_h50_3cam")],
    "dualarm_lt_suck_v3_h50": [("suck_lt", 1.0, "dualarm_lt_v3_h50")],
    "dualarm_lt_suck_v3_h50_3cam": [("suck_lt", 1.0, "dualarm_lt_v3_h50_3cam")],
}
