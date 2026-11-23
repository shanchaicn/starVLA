#!/usr/bin/env python3
"""Synthetic HTTP inference check; never subscribes or opens robot RPC."""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import yaml


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--instruction", help="Override task text, e.g. push versus pull")
    args = parser.parse_args()
    cfg = yaml.safe_load((here / "server_16.yaml").read_text())
    if args.profile not in cfg["profiles"]:
        parser.error(f"unknown profile: {args.profile}")
    profile = cfg["profiles"][args.profile]
    model = (here / cfg["paths"]["models_root"] / profile["model_dir"]).resolve()
    stats = json.loads((model / "dataset_statistics.json").read_text())["new_embodiment"]["state"]
    state = (np.asarray(stats["q01"]) + np.asarray(stats["q99"])) / 2
    state[3:7] /= np.linalg.norm(state[3:7])
    state[7:] = (state[7:] > 0.5).astype(float)
    cameras = profile["camera_order"]
    action_dim = int(profile["action_dim"])
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (127, 127, 127)).save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    body = {"protocol": "dualarm-lt-rtc-v1", "instruction": args.instruction or profile["instruction"],
            "state": state.tolist(), "images": {cam: encoded for cam in cameras}, "rtc": {"num_steps": 4}}

    def request():
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/predict", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=180) as response:
            result = json.load(response)
        actions = np.asarray(result["candidate_actions"])
        assert actions.shape == (50, action_dim) and np.isfinite(actions).all()
        assert result["model_identity"] == profile["model_id"]
        assert result["camera_order"] == cameras
        assert result["actions_executed"] == 0 and result["motion_authorized"] is False
        print(json.dumps({k: result[k] for k in ["model_identity", "action_shape", "rtc_mode", "rtc_guided", "inference_elapsed_s", "actions_executed"]}), flush=True)
        return result

    first = request()
    body["rtc"].update(previous_actions=first["candidate_actions"][10:], executed_steps=10, delay_steps=4, guidance_beta=5.0)
    request()


if __name__ == "__main__":
    main()
