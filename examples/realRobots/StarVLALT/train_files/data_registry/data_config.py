"""StarVLA-LT: five-view, dual-arm Cartesian datasets.

The Suck dataset only has four physical videos. Its fifth, fixed camera slot is
filled with a black 224x224 image while packing a sample; source data remains
untouched.
"""

from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


TARGET_NUM_VIEWS = 5


def pad_sample_to_five_views(sample, image_size=(224, 224)):
    """Append black RGB images until a sample has five ordered camera slots."""

    images = sample["image"]
    if len(images) > TARGET_NUM_VIEWS:
        raise ValueError(f"Expected at most {TARGET_NUM_VIEWS} views, got {len(images)}")
    while len(images) < TARGET_NUM_VIEWS:
        images.append(Image.new("RGB", tuple(image_size), color=(0, 0, 0)))
    return sample


class FiveViewLeRobotDataset(LeRobotSingleDataset):
    def _pack_sample(self, data):
        sample = super()._pack_sample(data)
        image_size = self.data_cfg.get("image_size", 224) if self.data_cfg is not None else 224
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return pad_sample_to_five_views(sample, image_size=image_size)


class StarVLALTCartesianConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    state_keys = ["state.left_pose", "state.right_pose"]
    action_keys = ["action.left_pose", "action.right_pose"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(16))

    def __init__(self, num_real_views):
        self.video_keys = [f"video.camera{i}" for i in range(num_real_views)]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )

    def make_dataset(self, **kwargs):
        kwargs.pop("dataset_name", None)
        return FiveViewLeRobotDataset(**kwargs)


ROBOT_TYPE_CONFIG_MAP = {
    "starvlalt_cartesian14_five_view": StarVLALTCartesianConfig(num_real_views=5),
    "starvlalt_cartesian14_suck_four_plus_black": StarVLALTCartesianConfig(num_real_views=4),
}


DATASET_NAMED_MIXTURES = {
    "starvla_push_pull": [
        ("push_pull", 1.0, "starvlalt_cartesian14_five_view"),
    ],
    "starvla_insert": [
        ("insert", 1.0, "starvlalt_cartesian14_five_view"),
    ],
    "starvla_snap": [
        ("snap", 1.0, "starvlalt_cartesian14_five_view"),
    ],
    "starvla_suck": [
        ("suck", 1.0, "starvlalt_cartesian14_suck_four_plus_black"),
    ],
    "starvla_place": [
        ("place", 1.0, "starvlalt_cartesian14_five_view"),
    ],
}
