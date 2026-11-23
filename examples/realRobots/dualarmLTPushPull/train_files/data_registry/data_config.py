"""StarVLA auto-discovered push_pull_lt registration; shares the existing V2 loader."""
from examples.realRobots.dualarmLT.train_files.data_registry.data_config import DualArmLTV2H50Config
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


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


ROBOT_TYPE_CONFIG_MAP = {"dualarm_lt_push_pull_v2_h50_2cam": DualArmLTPushPullV2H50Config()}
DATASET_NAMED_MIXTURES = {
    "dualarm_lt_push_pull_v2_h50_2cam": [("push_pull_lt_v2", 1.0, "dualarm_lt_push_pull_v2_h50_2cam")],
}
