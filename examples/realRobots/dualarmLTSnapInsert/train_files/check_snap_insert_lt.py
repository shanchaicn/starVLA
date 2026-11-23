#!/usr/bin/env python3
"""Validate metadata, full raw vectors and a real StarVLA training sample."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.realRobots.dualarmLTSnapInsert.robot_mapping_v2 import (
    ACTIVE_OUTPUT_INDICES, DIGITAL_INPUT_NAMES, DIGITAL_OUTPUT_NAMES,
    pack_action, pack_observation, unpack_policy_action,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/datasets"))
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--task", choices=("snap", "insert"), required=True)
    args = parser.parse_args()
    sentinel = pack_action(np.arange(14), np.arange(100, 112))
    require(np.array_equal(sentinel, [7, 8, 9, 10, 11, 12, 13, 106, 107, 109, 110]),
            "Synthetic mapping test failed: wrong arm or noncontiguous DO order")
    cfg = yaml.safe_load(args.config.read_text())
    dataset = args.data_root / f"{args.task}_lt_v2"
    info = json.loads((dataset / "meta/info.json").read_text())
    stats = json.loads((dataset / "meta/stats.json").read_text())
    modality = json.loads((dataset / "meta/modality.json").read_text())
    expected = json.loads(Path(__file__).with_name("modality.json").read_text())
    require(modality == expected, "Dataset modality differs from the left-arm V2 recipe")
    require(info["codebase_version"] == "v3.0", "Expected LeRobot v3.0")
    require(info["fps"] > 0 and info["total_episodes"] > 0, "Invalid episode count/fps")
    features = info["features"]
    for key, width in {"observation.state.cartesian_position": 14,
                       "action.cartesian_position": 14,
                       "observation.state.digital_input": 17,
                       "action.digital_output": 12}.items():
        require(features[key]["shape"] == [width], f"Wrong vector width: {key}")
        require(len(stats[key]["min"]) == width, f"Missing/wrong statistics: {key}")
    require(features["observation.state.digital_input"]["names"] == list(DIGITAL_INPUT_NAMES), "DI order mismatch")
    do_names = features["action.digital_output"]["names"]
    require([do_names[i] for i in ACTIVE_OUTPUT_INDICES] == list(DIGITAL_OUTPUT_NAMES), "DO index/name mismatch")
    for key in ("observation.state.cartesian_position", "action.cartesian_position"):
        require(features[key]["names"][7:14] == [f"arm_1.{k}" for k in ("x", "y", "z", "qx", "qy", "qz", "qw")], "arm_1 pose layout mismatch")
    for camera in ("left_wrist", "right_side"):
        require(features[f"observation.images.{camera}"]["dtype"] == "video", f"Missing camera: {camera}")
        require(any((dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4")), f"No video files: {camera}")
    require(cfg["datasets"]["vla_data"]["data_mix"] == f"dualarm_lt_{args.task}_v2_h50_2cam", "Wrong data mixture")
    require(cfg["datasets"]["vla_data"]["action_mode"] == "abs", "Expected absolute action mode")
    model = cfg["framework"]
    dims = model if model["name"] == "ACT" else model["action_model"]
    require((dims["state_dim"], dims["action_dim"]) == (24, 11), "Wrong policy dimensions")
    require(dims.get("chunk_size", dims.get("action_horizon")) == 50, "Expected H=50")
    if model["name"] == "ACT":
        require(model["image_keys"] == ["left_wrist", "right_side"], "ACT camera order mismatch")
    else:
        require(model["name"] == "QwenPI_v3", "Expected ACT or QwenPI_v3")
        require(Path(model["qwenvl"]["base_vlm"]).is_dir(), "Base VLM missing")
        require(Path(cfg["trainer"]["pretrained_checkpoint"]).is_file(), "VLAct checkpoint missing")

    columns = ["observation.state.cartesian_position", "observation.state.digital_input",
               "action.cartesian_position", "action.digital_output"]
    count = 0
    positives = np.zeros(12, dtype=np.int64)
    for path in sorted((dataset / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096):
            vectors = {key: np.asarray(batch.column(i).to_pylist(), dtype=np.float32) for i, key in enumerate(columns)}
            for key, values in vectors.items():
                require(np.isfinite(values).all(), f"Non-finite {key} in {path}")
            for key in columns[0::2]:
                norms = np.linalg.norm(vectors[key][:, 10:14], axis=1)
                require(np.all((norms > 0.9) & (norms < 1.1)), f"Invalid arm_1 quaternion in {path}: {key}")
            for key in (columns[1], columns[3]):
                require(np.isin(vectors[key], [0, 1]).all(), f"Non-binary I/O in {path}: {key}")
            obs, di, poses, dos = [vectors[key] for key in columns]
            action = pack_action(poses[0], dos[0])
            state = pack_observation(obs[0], di[0])
            require(state.shape == (24,) and action.shape == (11,), "Packing shape mismatch")
            require(np.array_equal(action[7:], dos[0, list(ACTIVE_OUTPUT_INDICES)]), "Wrong noncontiguous DO packing")
            restored = unpack_policy_action(action, obs[0], dos[0])
            require(np.array_equal(restored["action.cartesian_position"][:7], obs[0, :7]), "Other arm modified")
            require(np.array_equal(restored["action.digital_output"], dos[0]), "DO restoration mismatch")
            positives += (dos == 1).sum(axis=0)
            count += len(obs)
    require(count == info["total_frames"], f"Frame count mismatch: read {count}, metadata {info['total_frames']}")
    tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pydict()
    report = {"episodes": info["total_episodes"], "frames": count, "fps": info["fps"],
              "tasks": tasks, "state_dim": 24, "action_dim": 11, "horizon": 50,
              "camera_order": ["left_wrist", "right_side"], "do_order": list(DIGITAL_OUTPUT_NAMES),
              "do_high_samples": dict(zip(do_names, positives.tolist())),
              "excluded_active_do": [name for i, name in enumerate(do_names) if positives[i] and i not in ACTIVE_OUTPUT_INDICES]}
    print("PREFLIGHT_OK", json.dumps(report, ensure_ascii=False), flush=True)
    if args.sample:
        from omegaconf import OmegaConf
        from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
        dc = ROBOT_TYPE_CONFIG_MAP[f"dualarm_lt_{args.task}_v2_h50_2cam"]
        data = OmegaConf.create(cfg["datasets"]["vla_data"])
        data.load_all_data_for_training = False  # Decode only sample frames on CPU.
        data.state_action_dtype = "float32"
        ds = dc.make_dataset(dataset_path=dataset, modality_configs=dc.modality_config(),
                             transforms=dc.transform(), embodiment_tag=dc.embodiment_tag,
                             video_backend="pyav", data_cfg=data)
        sample = ds[0]
        require(np.asarray(sample["state"]).shape == (1, 24), "StarVLA state sample shape mismatch")
        require(np.asarray(sample["action"]).shape == (50, 11), "StarVLA action sample shape mismatch")
        require(len(sample["image"]) == 2, "StarVLA camera sample count mismatch")
        require(np.isfinite(sample["state"]).all() and np.isfinite(sample["action"]).all(), "Nonfinite normalized sample")
        raw = ds.get_step_data(int(ds.trajectory_ids[0]), 0)
        expected_do = np.concatenate([raw["action.left_suction_do"], raw["action.left_cylinder_do"]], axis=1)
        require(np.array_equal(sample["action"][:, 7:], expected_do), "Loaded policy DO order differs from raw slices")
        print("DATALOADER_SAMPLE_OK state=(1,24) action=(50,11) cameras=2 lang=" + sample["lang"])


if __name__ == "__main__":
    main()
