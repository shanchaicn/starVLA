"""Separate snap/insert mixtures using the tested left-arm V2 mapping."""
from examples.realRobots.dualarmLTPlace.train_files.data_registry.data_config import DualArmLTPlaceV2H50Config

ROBOT_TYPE_CONFIG_MAP = {
    f"dualarm_lt_{task}_v2_h50_2cam": DualArmLTPlaceV2H50Config()
    for task in ("snap", "insert")
}
DATASET_NAMED_MIXTURES = {
    f"dualarm_lt_{task}_v2_h50_2cam": [
        (f"{task}_lt_v2", 1.0, f"dualarm_lt_{task}_v2_h50_2cam")
    ]
    for task in ("snap", "insert")
}
