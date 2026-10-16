"""Verify LIBERO-3D against the official LIBERO HDF5 trajectories.

Run this with the simulation environment. It replays a few saved MuJoCo
states and compares RGB, metric depth, calibration, robot state, and actions.
It never starts training.
"""
import argparse
import inspect
import json
from pathlib import Path

import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.models.grippers.panda_gripper import PandaGripper
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
from robosuite.utils.transform_utils import quat2axisangle


ROOT = Path(__file__).resolve().parent


def robot_state(obs):
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    )


def errors(actual, expected):
    delta = np.abs(np.asarray(actual, np.float64) - np.asarray(expected, np.float64)).reshape(-1)
    return {
        "mae": float(delta.mean()),
        "median": float(np.median(delta)),
        "p99": float(np.quantile(delta, 0.99)),
        "max": float(delta.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", default="/root/data/datasets/LIBERO-original-selected")
    parser.add_argument("--processed-root", default="/root/data/datasets/LIBERO-3D-selected")
    parser.add_argument("--output", default=str(ROOT / "reports" / "contract_verification.json"))
    args = parser.parse_args()

    selected = json.loads((ROOT / "selected_tasks.json").read_text())
    original_root, processed_root = Path(args.original_root), Path(args.processed_root)
    report = {
        "method": "official HDF5 arrays plus replay of saved MuJoCo states",
        "gripper_source": inspect.getsource(PandaGripper.format_action),
        "tasks": [],
    }

    for entry in selected["tasks"]:
        suite = benchmark.get_benchmark_dict()[entry["suite"]]()
        ids = [i for i in range(suite.n_tasks) if suite.get_task(i).name == entry["task_name"]]
        if len(ids) != 1:
            raise ValueError(f"Missing or ambiguous simulator task: {entry['task_name']}")
        task = suite.get_task(ids[0])
        hdf5_path = original_root / entry["suite"] / f"{entry['task_name']}_demo.hdf5"
        task_root = processed_root / entry["relative_dir"]
        npz_path = task_root / "episode_000" / "data.npz"
        metadata = json.loads((task_root / "metadata.json").read_text())
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
            camera_depths=True,
            camera_segmentations="instance",
            control_freq=20,
        )
        try:
            env.reset()
            with h5py.File(hdf5_path, "r") as source, np.load(npz_path, allow_pickle=False) as data:
                demo = source["data/demo_0"]
                source_state = np.concatenate(
                    [demo["obs/ee_states"][()], demo["obs/gripper_states"][()]], axis=1
                )
                task_report = {
                    "suite": entry["suite"],
                    "task": entry["task_name"],
                    "episode": 0,
                    "length": int(len(data["actions"])),
                    "actions_same_index": errors(data["actions"], demo["actions"][()]),
                    "copied_state_same_index": errors(data["state"], source_state),
                    "samples": [],
                }
                full_states = demo["states"]
                indices = sorted({1, len(data["actions"]) // 2, len(data["actions"]) - 2})
                for index in indices:
                    obs = env.set_init_state(full_states[index])
                    sample = {
                        "index": index,
                        # Processed state[index-1] is the robot state at full_states[index].
                        "sim_state_vs_processed_previous": errors(robot_state(obs), data["state"][index - 1]),
                        "sim_state_vs_processed_same": errors(robot_state(obs), data["state"][index]),
                        "cameras": {},
                    }
                    for camera, key in (("agentview", "agentview"), ("robot0_eye_in_hand", "wrist")):
                        rgb = obs[camera + "_image"][::-1]
                        depth = get_real_depth_map(env.env.sim, obs[camera + "_depth"].squeeze(-1))[::-1]
                        intrinsic = get_camera_intrinsic_matrix(env.env.sim, camera, 256, 256)
                        extrinsic = get_camera_extrinsic_matrix(env.env.sim, camera)
                        expected_intrinsic = (
                            np.asarray(metadata["cameras"]["agentview"]["intrinsic_static"])
                            if key == "agentview"
                            else data["wrist_intrinsic"][index]
                        )
                        expected_extrinsic = (
                            np.asarray(metadata["cameras"]["agentview"]["extrinsic_static"])
                            if key == "agentview"
                            else data["wrist_extrinsic"][index]
                        )
                        sample["cameras"][key] = {
                            "transform": "flip_y",
                            "rgb_uint8": errors(rgb, data[key + "_rgb"][index]),
                            "metric_depth_m": errors(depth, data[key + "_depth"][index]),
                            "intrinsic": errors(intrinsic, expected_intrinsic),
                            "extrinsic": errors(extrinsic, expected_extrinsic),
                            "render_depth_range_m": [float(depth.min()), float(depth.max())],
                            "dataset_depth_range": [
                                float(data[key + "_depth"][index].min()),
                                float(data[key + "_depth"][index].max()),
                            ],
                        }
                    task_report["samples"].append(sample)
                report["tasks"].append(task_report)
        finally:
            env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
