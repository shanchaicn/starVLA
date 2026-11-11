"""LeRobot v3 pp_white_red: two RGB views and absolute 8D TB6-R5 pose."""

from pathlib import Path

import pandas as pd

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class PPWhiteRedDataset(LeRobotSingleDataset):
    """Keep the LeRobot v3 task column and index instead of duplicating it."""

    def _get_tasks(self) -> pd.DataFrame:
        tasks = pd.read_parquet(Path(self.dataset_path) / "meta/tasks.parquet").reset_index()
        if not {"task_index", "task"}.issubset(tasks.columns):
            raise ValueError("pp_white_red tasks.parquet lacks task_index or task")
        if tasks["task_index"].duplicated().any() or tasks["task"].isna().any():
            raise ValueError("pp_white_red task metadata is ambiguous")
        return tasks[["task_index", "task"]].set_index("task_index")


class PPWhiteRedAbs8H50Config:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.front", "video.wrist"]
    state_keys = ["state.xyz", "state.quat_xyzw", "state.gripper_m"]
    action_keys = ["action.xyz", "action.quat_xyzw", "action.gripper_m"]
    state_key_dims = {"state.xyz": 3, "state.quat_xyzw": 4, "state.gripper_m": 1}
    action_key_dims = {"action.xyz": 3, "action.quat_xyzw": 4, "action.gripper_m": 1}
    language_keys = ["annotation.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(50))

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
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
        ])

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return PPWhiteRedDataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {"pp_white_red_abs8_h50": PPWhiteRedAbs8H50Config()}
DATASET_NAMED_MIXTURES = {
    "pp_white_red_abs8_h50": [("pp_white_red", 1.0, "pp_white_red_abs8_h50")],
}
