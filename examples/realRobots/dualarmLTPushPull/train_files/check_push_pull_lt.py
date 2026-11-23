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
from examples.realRobots.dualarmLTPushPull.robot_mapping_v2 import (
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
    args = parser.parse_args()
    sentinel = pack_action(np.arange(14), np.arange(100, 112))
    require(np.array_equal(sentinel, [7, 8, 9, 10, 11, 12, 13, 108, 111]),
            "Synthetic mapping test failed: wrong arm or noncontiguous DO order")
    cfg = yaml.safe_load(args.config.read_text())
    dataset = args.data_root / "push_pull_lt_v2"
    info = json.loads((dataset / "meta/info.json").read_text())
    stats = json.loads((dataset / "meta/stats.json").read_text())
    modality = json.loads((dataset / "meta/modality.json").read_text())
    expected = json.loads(Path(__file__).with_name("modality.json").read_text())
    require(modality == expected, "Dataset modality differs from the push/pull V2 recipe")
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
    for camera in ("left_wrist", "left_side"):
        require(features[f"observation.images.{camera}"]["dtype"] == "video", f"Missing camera: {camera}")
        require(any((dataset / "videos" / f"observation.images.{camera}").rglob("*.mp4")), f"No video files: {camera}")
    require(cfg["datasets"]["vla_data"]["data_mix"] == "dualarm_lt_push_pull_v2_h50_2cam", "Wrong data mixture")
    require(cfg["datasets"]["vla_data"]["action_mode"] == "abs", "Expected absolute action mode")
    model = cfg["framework"]
    require(model["name"] == "QwenPI_v3", "Expected QwenPI_v3")
    dims = model["action_model"]
    require((dims["state_dim"], dims["action_dim"]) == (24, 9), "Wrong policy dimensions")
    require(dims.get("chunk_size", dims.get("action_horizon")) == 50, "Expected H=50")
    base_vlm = Path(model["qwenvl"]["base_vlm"])
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"):
        require((base_vlm / name).is_file(), f"Base VLM missing: {name}")
    index = json.loads((base_vlm / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        require((base_vlm / shard).is_file() and (base_vlm / shard).stat().st_size > 0, f"Base VLM shard missing: {shard}")
    require(Path(cfg["trainer"]["pretrained_checkpoint"]).is_file(), "VLAct checkpoint missing")
    require(cfg["trainer"].get("random_init_action_model") is True, "New 9D action head must be randomly initialized")

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
            require(state.shape == (24,) and action.shape == (9,), "Packing shape mismatch")
            require(np.array_equal(action[7:], dos[0, list(ACTIVE_OUTPUT_INDICES)]), "Wrong noncontiguous DO packing")
            restored = unpack_policy_action(action, obs[0], dos[0])
            require(np.array_equal(restored["action.cartesian_position"][:7], obs[0, :7]), "Other arm modified")
            require(np.array_equal(restored["action.digital_output"], dos[0]), "DO restoration mismatch")
            positives += (dos == 1).sum(axis=0)
            count += len(obs)
    require(count == info["total_frames"], f"Frame count mismatch: read {count}, metadata {info['total_frames']}")
    tasks = pq.read_table(dataset / "meta/tasks.parquet").to_pydict()
    report = {"episodes": info["total_episodes"], "frames": count, "fps": info["fps"],
              "tasks": tasks, "state_dim": 24, "action_dim": 9, "horizon": 50,
              "camera_order": ["left_wrist", "left_side"], "do_order": list(DIGITAL_OUTPUT_NAMES),
              "do_high_samples": dict(zip(do_names, positives.tolist())),
              "excluded_active_do": [name for i, name in enumerate(do_names) if positives[i] and i not in ACTIVE_OUTPUT_INDICES]}
    print("PREFLIGHT_OK", json.dumps(report, ensure_ascii=False), flush=True)
    if args.sample:
        from omegaconf import OmegaConf
        from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
        dc = ROBOT_TYPE_CONFIG_MAP["dualarm_lt_push_pull_v2_h50_2cam"]
        require(dc.video_keys == ["video.left_wrist", "video.left_side"], "Camera registry order mismatch")
        data = OmegaConf.create(cfg["datasets"]["vla_data"])
        data.load_all_data_for_training = False  # Decode only sample frames on CPU.
        data.state_action_dtype = "float32"
        ds = dc.make_dataset(dataset_path=dataset, modality_configs=dc.modality_config(),
                             transforms=dc.transform(), embodiment_tag=dc.embodiment_tag,
                             video_backend="pyav", data_cfg=data)
        sample = ds[0]
        require(np.asarray(sample["state"]).shape == (1, 24), "StarVLA state sample shape mismatch")
        require(np.asarray(sample["action"]).shape == (50, 9), "StarVLA action sample shape mismatch")
        require(len(sample["image"]) == 2, "StarVLA camera sample count mismatch")
        require(np.isfinite(sample["state"]).all() and np.isfinite(sample["action"]).all(), "Nonfinite normalized sample")
        raw = ds.get_step_data(int(ds.trajectory_ids[0]), 0)
        expected_do = np.concatenate([raw["action.left_middle_suction_do"], raw["action.left_middle_cylinder_do"]], axis=1)
        require(np.array_equal(sample["action"][:, 7:], expected_do), "Loaded policy DO order differs from raw slices")
        print("DATALOADER_SAMPLE_OK state=(1,24) action=(50,9) cameras=2 lang=" + sample["lang"])


if __name__ == "__main__":
    main()
