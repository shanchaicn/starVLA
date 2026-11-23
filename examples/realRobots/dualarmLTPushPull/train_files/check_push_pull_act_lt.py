#!/usr/bin/env python3
"""Validate one task-filtered Push/Pull dataset and its ACT H50 recipe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.realRobots.dualarmLTPushPull.robot_mapping_v2 import (
    ACTIVE_OUTPUT_INDICES, DIGITAL_INPUT_NAMES, DIGITAL_OUTPUT_NAMES,
    pack_action, pack_observation,
)
from examples.realRobots.dualarmLTPushPull.train_files.prepare_push_pull_act_lt import TASKS


EXPECTED_EPISODES = {"push": 193, "pull": 184}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/datasets"))
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dataset = args.data_root / f"{args.task}_lt_v2"
    info = json.loads((dataset / "meta/info.json").read_text())
    manifest = json.loads((dataset / "meta/filter_manifest.json").read_text())
    require(manifest["task"] == args.task and manifest["prompt"] == TASKS[args.task], "Wrong filter manifest")
    require(info["total_episodes"] == EXPECTED_EPISODES[args.task], "Unexpected filtered episode count")
    require(info["total_frames"] == manifest["frames"] > 0, "Filtered frame count mismatch")
    require(json.loads((dataset / "meta/modality.json").read_text()) ==
            json.loads(Path(__file__).with_name("modality.json").read_text()), "Wrong modality mapping")

    model = cfg["framework"]
    require(model["name"] == "ACT", "Expected ACT")
    require((model["state_dim"], model["action_dim"], model["chunk_size"]) == (24, 9, 50), "Wrong ACT dimensions")
    require(model["image_keys"] == ["left_wrist", "left_side"], "Wrong ACT camera order")
    require(cfg["trainer"]["pretrained_checkpoint"] is None, "Fresh ACT must not load a policy checkpoint")
    expected_mix = f"dualarm_lt_{args.task}_v2_h50_2cam"
    require(cfg["datasets"]["vla_data"]["data_mix"] == expected_mix, "Wrong task data mixture")

    info_features = info["features"]
    require(info_features["observation.state.digital_input"]["names"] == list(DIGITAL_INPUT_NAMES), "DI order mismatch")
    do_names = info_features["action.digital_output"]["names"]
    require([do_names[i] for i in ACTIVE_OUTPUT_INDICES] == list(DIGITAL_OUTPUT_NAMES), "DO order mismatch")

    frame_count = 0
    episode_ids = set()
    task_indices = set()
    active_do_values = set()
    columns = ["episode_index", "task_index", "observation.state.cartesian_position",
               "observation.state.digital_input", "action.cartesian_position", "action.digital_output"]
    for path in sorted((dataset / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
            values = {key: np.asarray(batch.column(i).to_pylist()) for i, key in enumerate(columns)}
            episode_ids.update(int(value) for value in values["episode_index"])
            task_indices.update(int(value) for value in values["task_index"])
            obs = values["observation.state.cartesian_position"].astype(np.float32)
            di = values["observation.state.digital_input"].astype(np.float32)
            action_pose = values["action.cartesian_position"].astype(np.float32)
            dos = values["action.digital_output"].astype(np.float32)
            require(np.isfinite(obs).all() and np.isfinite(di).all() and
                    np.isfinite(action_pose).all() and np.isfinite(dos).all(), f"Non-finite data: {path}")
            require(np.isin(di, [0, 1]).all() and np.isin(dos, [0, 1]).all(), f"Non-binary IO: {path}")
            require(np.all((np.linalg.norm(obs[:, 10:14], axis=1) > .9) &
                           (np.linalg.norm(obs[:, 10:14], axis=1) < 1.1)), f"Invalid state quaternion: {path}")
            require(np.all((np.linalg.norm(action_pose[:, 10:14], axis=1) > .9) &
                           (np.linalg.norm(action_pose[:, 10:14], axis=1) < 1.1)), f"Invalid action quaternion: {path}")
            require(pack_observation(obs[0], di[0]).shape == (24,), "Wrong packed state")
            packed = pack_action(action_pose[0], dos[0])
            require(packed.shape == (9,), "Wrong packed action")
            require(np.array_equal(packed[7:], dos[0, list(ACTIVE_OUTPUT_INDICES)]), "Wrong active DO mapping")
            active_do_values.update(
                tuple(int(value) for value in row)
                for row in dos[:, list(ACTIVE_OUTPUT_INDICES)].astype(int)
            )
            frame_count += len(obs)

    require(frame_count == info["total_frames"], "Read frame count differs from filtered metadata")
    require(len(episode_ids) == info["total_episodes"], "Read episode count differs from filtered metadata")
    require(len(task_indices) == 1, f"Filtered data still contains multiple task indices: {task_indices}")
    tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pydict()
    require(tasks["task"] == [TASKS[args.task]], "Filtered tasks.parquet has wrong prompt")
    report = {
        "task": args.task, "prompt": TASKS[args.task], "episodes": len(episode_ids),
        "frames": frame_count, "state_dim": 24, "action_dim": 9, "horizon": 50,
        "camera_order": ["left_wrist", "left_side"], "do_order": list(DIGITAL_OUTPUT_NAMES),
        "active_do_patterns": sorted([list(values) for values in active_do_values]),
    }
    print("ACT_TASK_PREFLIGHT_OK", json.dumps(report, ensure_ascii=False), flush=True)

    if args.sample:
        from omegaconf import OmegaConf
        from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
        dc = ROBOT_TYPE_CONFIG_MAP["dualarm_lt_push_pull_v2_h50_2cam"]
        data = OmegaConf.create(cfg["datasets"]["vla_data"])
        data.load_all_data_for_training = False
        data.state_action_dtype = "float32"
        ds = dc.make_dataset(dataset_path=dataset, modality_configs=dc.modality_config(),
                             transforms=dc.transform(), embodiment_tag=dc.embodiment_tag,
                             video_backend="pyav", data_cfg=data)
        require(len(ds.trajectory_ids) == info["total_episodes"], "Loader did not retain exactly the filtered episodes")
        sample = ds[0]
        require(np.asarray(sample["state"]).shape == (1, 24), "Wrong sample state")
        require(np.asarray(sample["action"]).shape == (50, 9), "Wrong sample action")
        require(len(sample["image"]) == 2, "Wrong sample camera count")
        require(sample["lang"] == TASKS[args.task], "Wrong sample language")
        print(f"ACT_TASK_SAMPLE_OK task={args.task} state=(1,24) action=(50,9) cameras=2")


if __name__ == "__main__":
    main()
