"""StarVLA auto-discovered place_lt registration; shares the existing V2 loader."""
from examples.realRobots.dualarmLT.train_files.data_registry.data_config import DualArmLTV2H50Config
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class DualArmLTPlaceV2H50Config(DualArmLTV2H50Config):
    video_keys = ["video.left_wrist", "video.right_side"]
    state_keys = ["state.arm_1_pose", "state.digital_input"]
    # Separate contiguous slices: raw DO indices 6,7 and 9,10. DO14 is skipped.
    action_keys = ["action.arm_1_pose", "action.left_suction_do", "action.left_cylinder_do"]
    state_key_dims = {"state.arm_1_pose": 7, "state.digital_input": 17}
    action_key_dims = {"action.arm_1_pose": 7, "action.left_suction_do": 2, "action.left_cylinder_do": 2}

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={
                "state.arm_1_pose": "q99", "state.digital_input": "binary",
            }),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={
                "action.arm_1_pose": "q99", "action.left_suction_do": "binary",
                "action.left_cylinder_do": "binary",
            }),
        ])


ROBOT_TYPE_CONFIG_MAP = {"dualarm_lt_place_v2_h50_2cam": DualArmLTPlaceV2H50Config()}
DATASET_NAMED_MIXTURES = {
    "dualarm_lt_place_v2_h50_2cam": [("place_lt_v2", 1.0, "dualarm_lt_place_v2_h50_2cam")],
}
