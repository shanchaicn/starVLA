"""StarVLA registrations for combined and task-filtered Push/Pull datasets."""
from pathlib import Path

import pandas as pd

from examples.realRobots.dualarmLT.train_files.data_registry.data_config import DualArmLTV2H50Config
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class DualArmLTPushPullDataset(LeRobotSingleDataset):
    """Index v3 task metadata by its explicit task_index column.

    The base v3 loader happens to work for the original two-row table because
    its RangeIndex equals task_index 0/1. A one-task filtered table may retain
    task_index 1, so relying on the row number would drop every Push episode.
    """

    def _get_tasks(self) -> pd.DataFrame:
        tasks = pd.read_parquet(Path(self.dataset_path) / "meta/tasks.parquet").reset_index()
        if not {"task_index", "task"}.issubset(tasks.columns):
            raise ValueError("Push/Pull tasks.parquet lacks task_index or task")
        if tasks["task_index"].duplicated().any() or tasks["task"].isna().any():
            raise ValueError("Push/Pull task metadata is ambiguous")
        return tasks[["task_index", "task"]].set_index("task_index")


class DualArmLTPushPullV2H50Config(DualArmLTV2H50Config):
    video_keys = ["video.left_wrist", "video.left_side"]
    state_keys = ["state.arm_1_pose", "state.digital_input"]
    # Separate contiguous slices: raw DO indices 8 and 11: DO14 and DO17 only.
    action_keys = ["action.arm_1_pose", "action.left_middle_suction_do", "action.left_middle_cylinder_do"]
    state_key_dims = {"state.arm_1_pose": 7, "state.digital_input": 17}
    action_key_dims = {"action.arm_1_pose": 7, "action.left_middle_suction_do": 1, "action.left_middle_cylinder_do": 1}

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.arm_1_pose": "q99", "state.digital_input": "binary",
            }),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.arm_1_pose": "q99", "action.left_middle_suction_do": "binary",
                "action.left_middle_cylinder_do": "binary",
            }),
        ])

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return DualArmLTPushPullDataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {"dualarm_lt_push_pull_v2_h50_2cam": DualArmLTPushPullV2H50Config()}
DATASET_NAMED_MIXTURES = {
    "dualarm_lt_push_pull_v2_h50_2cam": [("push_pull_lt_v2", 1.0, "dualarm_lt_push_pull_v2_h50_2cam")],
    "dualarm_lt_push_v2_h50_2cam": [("push_lt_v2", 1.0, "dualarm_lt_push_pull_v2_h50_2cam")],
    "dualarm_lt_pull_v2_h50_2cam": [("pull_lt_v2", 1.0, "dualarm_lt_push_pull_v2_h50_2cam")],
}
