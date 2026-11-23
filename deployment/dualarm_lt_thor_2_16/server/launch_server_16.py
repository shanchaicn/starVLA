#!/usr/bin/env python3
"""Launch the migrated, inference-only server using paths from one YAML file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import yaml


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_contract_errors(model, profile):
    errors = []
    config_file = model / "config.yaml"
    stats_file = model / "dataset_statistics.json"
    if not config_file.is_file():
        errors.append("config.yaml missing")
    else:
        try:
            saved = yaml.safe_load(config_file.read_text())["framework"]
            if profile["script"] == "dualarm_lt_rtc_server_16.py":
                found = saved["action_model"]
                dimensions = (int(found["state_dim"]), int(found["action_dim"]), int(found["action_horizon"]))
                expected_name = "QwenPI_v3"
            else:
                dimensions = (int(saved["state_dim"]), int(saved["action_dim"]), int(saved["chunk_size"]))
                expected_name = "ACT"
                if list(saved["image_keys"]) != list(profile["camera_order"]):
                    errors.append("config.yaml camera order differs from profile")
            if saved["name"] != expected_name or dimensions != (
                int(profile["state_dim"]), int(profile["action_dim"]), 50
            ):
                errors.append(f"config.yaml architecture/dimensions mismatch: {saved['name']} {dimensions}")
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"config.yaml unreadable: {exc}")
    if not stats_file.is_file():
        errors.append("dataset_statistics.json missing")
    else:
        try:
            stats = json.loads(stats_file.read_text())["new_embodiment"]
            dimensions = (len(stats["state"]["q01"]), len(stats["action"]["q01"]))
            if dimensions != (int(profile["state_dim"]), int(profile["action_dim"])):
                errors.append(f"dataset statistics dimensions mismatch: {dimensions}")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"dataset_statistics.json unreadable: {exc}")
    if profile.get("modelscope_repo"):
        modality_file = model / "modality.json"
        if modality_file.is_file():
            try:
                modality = json.loads(modality_file.read_text())
                cameras = list(modality["video"])
                if cameras != list(profile["camera_order"]):
                    errors.append(f"modality.json camera order mismatch: {cameras}")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"modality.json unreadable: {exc}")
        manifest_file = model / "upload_manifest.json"
        if not manifest_file.is_file():
            errors.append("upload_manifest.json missing")
        else:
            try:
                manifest = json.loads(manifest_file.read_text())
                found_repo = manifest["model"]["repo_id"]
                if found_repo != profile["modelscope_repo"]:
                    errors.append(f"wrong ModelScope repo: {found_repo}")
                if list(manifest["model"]["do"]) != list(profile["action_do_nums"]):
                    errors.append("ModelScope manifest DO mapping differs from profile")
                for name in ("config.yaml", "dataset_statistics.json", "modality.json", "final_model/pytorch_model.pt"):
                    file = model / name
                    if not file.is_file():
                        errors.append(f"{name} missing")
                    elif file.stat().st_size != int(manifest["files"][name]["bytes"]):
                        errors.append(f"{name} size differs from manifest")
                    elif sha256_file(file) != manifest["files"][name]["sha256"]:
                        errors.append(f"{name} SHA-256 differs from manifest")
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"upload_manifest.json unreadable: {exc}")
    return errors


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(here / "server_16.yaml"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    ports = [int(item["port"]) for item in config["profiles"].values()]
    if len(ports) != len(set(ports)):
        parser.error("server_16.yaml contains duplicate profile ports")
    if args.profile not in config["profiles"]:
        parser.error(f"unknown profile: {args.profile}; choose from {', '.join(sorted(config['profiles']))}")
    paths = {key: (config_path.parent / value).resolve() for key, value in config["paths"].items()}
    profile = config["profiles"][args.profile]
    if profile["script"] not in ("dualarm_lt_rtc_server_16.py", "dualarm_lt_act_server_16.py"):
        parser.error("unknown model server script")
    if int(profile["state_dim"]) not in (7, 24) or int(profile["action_dim"]) not in (9, 11):
        parser.error("unsupported state/action dimension")
    if len(set(profile["camera_order"])) != len(profile["camera_order"]):
        parser.error("duplicate cameras in model profile")
    if len(profile["action_do_nums"]) != int(profile["action_dim"]) - 7:
        parser.error("action DO mapping length differs from action dimension")
    model = paths["models_root"] / profile["model_dir"]
    checkpoint = model / "final_model/pytorch_model.pt"
    port = args.port if args.port is not None else int(profile["port"])
    if not 1024 <= port <= 65535:
        parser.error("port must be between 1024 and 65535")
    expected = int(profile["checkpoint_size"])
    status = {
        "config": str(config_path), "profile": args.profile,
        "model_root": str(model), "port": port, "bind": "127.0.0.1",
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        "expected_checkpoint_bytes": expected,
        "source_exists": paths["source_root"].is_dir(),
        "motion_authorized": False,
    }
    status["checkpoint_ready"] = status["checkpoint_exists"] and status["checkpoint_bytes"] == expected
    status["contract_errors"] = model_contract_errors(model, profile)
    status["ready"] = status["checkpoint_ready"] and not status["contract_errors"] and status["source_exists"]
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    if args.check_config:
        return 0 if status["ready"] else 2
    if not args.self_test and not status["ready"]:
        raise RuntimeError("Model transfer incomplete or model files do not match profile")
    if not args.self_test:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    runtime = config["runtime"]
    os.environ.update({
        "STARVLA_DEPLOY_CONFIG": str(config_path),
        "STARVLA_MODELS_ROOT": str(paths["models_root"]),
        "STARVLA_MODEL_ROOT": str(model),
        "STARVLA_SOURCE_ROOT": str(paths["source_root"]),
        "STARVLA_EVAL_DEPS": str(paths["eval_deps"]),
        "STARVLA_VENDOR_ACT": str(paths["vendor_act"]),
        "STARVLA_BASE_VLM": str(paths["base_vlm"]),
        "STARVLA_UNIFIED_MEMORY": "1" if runtime["unified_memory"] else "0",
        "STARVLA_MIN_MEMORY_GIB": str(runtime["min_available_memory_gib"]),
        "CUDA_VISIBLE_DEVICES": str(runtime["cuda_visible_devices"]),
        "OMP_NUM_THREADS": str(runtime["omp_num_threads"]),
    })
    script = here / profile["script"]
    command = [sys.executable, str(script)]
    command.extend(["--profile", args.profile])
    command.extend(["--self-test"] if args.self_test else ["--serve", "--port", str(port)])
    os.execv(sys.executable, command)


if __name__ == "__main__":
    sys.exit(main())
