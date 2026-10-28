import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


RECIPE = (
    Path(__file__).resolve().parents[1]
    / "examples/realRobots/StarVLALT/train_files/data_registry/data_config.py"
)


def load_recipe_module():
    spec = importlib.util.spec_from_file_location("starvlalt_data_config_test", RECIPE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class StarVLALTRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = load_recipe_module()

    def test_suck_uses_four_real_views_and_all_samples_have_five_slots(self):
        config = self.recipe.ROBOT_TYPE_CONFIG_MAP[
            "starvlalt_cartesian14_suck_four_plus_black"
        ]
        self.assertEqual(config.video_keys, [f"video.camera{i}" for i in range(4)])

        sample = {"image": [Image.new("RGB", (224, 224), color="white") for _ in range(4)]}
        padded = self.recipe.pad_sample_to_five_views(sample)
        self.assertEqual(len(padded["image"]), 5)
        black = np.asarray(padded["image"][4])
        self.assertEqual(black.shape, (224, 224, 3))
        self.assertTrue(np.all(black == 0))

    def test_five_task_mixtures_are_registered(self):
        expected = {
            "starvla_push_pull",
            "starvla_insert",
            "starvla_snap",
            "starvla_suck",
            "starvla_place",
        }
        self.assertEqual(set(self.recipe.DATASET_NAMED_MIXTURES), expected)


if __name__ == "__main__":
    unittest.main()
