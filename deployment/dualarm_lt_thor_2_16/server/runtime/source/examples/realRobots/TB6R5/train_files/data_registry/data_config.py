"""Opt-in TB6-R5 1019-1 recorded-model LeRobot v3 contract; no hardware mapping."""
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class TB6R5Abs8Config:
    direct_episode_split_dataset = True
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ['video.front', 'video.wrist']
    state_keys = ['state.xyz', 'state.quaternion_xyzw', 'state.gripper']
    action_keys = ['action.xyz', 'action.quaternion_xyzw', 'action.gripper']
    state_key_dims = dict(zip(state_keys, [3, 4, 1]))
    action_key_dims = dict(zip(action_keys, [3, 4, 1]))
    language_keys = ['annotation.language']

    def modality_config(self):
        return {name: ModalityConfig(delta_indices=offsets, modality_keys=keys) for name, offsets, keys in
                [('video', [0], self.video_keys), ('state', [0], self.state_keys),
                 ('action', list(range(8)), self.action_keys), ('language', [0], self.language_keys)]}

    def transform(self):
        keys = self.state_keys + self.action_keys
        return ComposedModalityTransform(transforms=[StateActionToTensor(apply_to=keys),
            StateActionTransform(apply_to=keys, normalization_modes={k: 'min_max' for k in keys})])

    def make_dataset(self, **kwargs):
        from starVLA.dataloader.tb6r5_v3 import TB6R5V3Dataset
        kwargs.pop('dataset_name', None)
        return TB6R5V3Dataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {'tb6r5_recorded_abs8': TB6R5Abs8Config()}
DATASET_NAMED_MIXTURES = {'tb6r5_1019_v3_abs8': [('1019-1_lerobot_v3_480p_no_source', 1.0, 'tb6r5_recorded_abs8')]}
