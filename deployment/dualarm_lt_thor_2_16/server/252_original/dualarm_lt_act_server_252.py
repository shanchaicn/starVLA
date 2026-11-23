#!/usr/bin/env python3
"""Fresh-replan RTC server for dualarm_lt_suck_act_h50_v2.

ACT has no guided sampler. Requests containing previous_actions trigger a fresh
50-step prediction from the newest observation. This process has no robot SDK.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
STARVLA_ROOT = Path("/home/codeit/Public/zpf/StarVLA")
MODEL_ROOT = STARVLA_ROOT / "staging/dualarm_lt_suck_act_h50_v2"
SOURCE_ROOT = STARVLA_ROOT / "inference/isolated_a800_step10000_20261102/project_root/source"
VENDOR_ROOT = ROOT / "vendor_act"
MODEL_ID = "dualarm_lt_suck_act_h50_v2-RTC-FRESH-REPLAN-DEV"
PORT = 10201
STATE_DIM = 24
ACTION_DIM = 11
HORIZON = 50
CAMERAS = ("left_side", "left_wrist", "right_wrist", "right_side")
CHECKPOINT_SIZE = 103_360_178

sys.path[:0] = [str(ROOT), str(VENDOR_ROOT), str(SOURCE_ROOT)]
from rtc_contract import ContractError, array, parse_rtc

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    WANDB_DISABLED="true",
    TOKENIZERS_PARALLELISM="false",
)


def stats_arrays():
    import numpy as np

    stats = json.loads((MODEL_ROOT / "dataset_statistics.json").read_text())["new_embodiment"]
    state, action = stats["state"], stats["action"]
    return (
        np.asarray(state["q01"], dtype=np.float32),
        np.asarray(state["q99"], dtype=np.float32),
        np.asarray(action["q01"], dtype=np.float32),
        np.asarray(action["q99"], dtype=np.float32),
    )


def normalize_state(raw, q01, q99):
    import numpy as np

    raw = array(raw, (STATE_DIM,), "state")
    out = raw.copy()
    out[:7] = np.clip(2 * (raw[:7] - q01[:7]) / (q99[:7] - q01[:7]) - 1, -2.2, 2.2)
    out[7:] = (raw[7:] > 0.5).astype(np.float32)
    return out


def unnormalize_actions(norm, q01, q99):
    import numpy as np

    norm = array(norm, (1, HORIZON, ACTION_DIM), "normalized_actions")[0]
    out = norm.copy()
    out[:, :7] = (norm[:, :7] + 1) * 0.5 * (q99[:7] - q01[:7]) + q01[:7]
    out[:, 7:] = (norm[:, 7:] > 0.5).astype(np.float32)
    qn = np.linalg.norm(out[:, 3:7], axis=1)
    if np.any(qn < 0.25):
        raise ContractError("predicted quaternion invalid")
    out[:, 3:7] /= qn[:, None]
    return out


def decode_images(body):
    from PIL import Image
    import numpy as np

    images = body.get("images")
    if not isinstance(images, dict) or set(images) != set(CAMERAS):
        raise ContractError(f"images must contain exactly {CAMERAS}")
    result = []
    for name in CAMERAS:
        value = images[name]
        if not isinstance(value, str) or not 0 < len(value) <= 4_000_000:
            raise ContractError(f"invalid {name} JPEG")
        try:
            raw = base64.b64decode(value, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise ContractError(f"cannot decode {name} JPEG") from exc
        result.append(Image.fromarray(np.asarray(image)))
    return result


class Predictor:
    def __init__(self):
        import gc
        import typing
        # The locally available LeRobot 0.6 package was installed under
        # Python 3.12, while the CUDA runtime is Python 3.10. LeRobot imports
        # typing.Self (3.11+); expose the standard backport before importing it.
        if not hasattr(typing, "Self"):
            from typing_extensions import Self
            typing.Self = Self
        import torch
        from omegaconf import OmegaConf
        from starVLA.model.framework.VM4A.ACT import ACT

        # The source snapshot on 252 predates the LeRobot 0.4 compatibility
        # guard now present on the GitHub branch. LeRobot 0.4 has no policy
        # normalizer modules; StarVLA already owns state/action normalization.
        def set_identity_normalizers_compat(instance):
            with torch.no_grad():
                for name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
                    normalizer = getattr(instance.action_model, name, None)
                    if normalizer is None:
                        continue
                    for buffer_name, value in list(normalizer.named_buffers()) + list(normalizer.named_parameters()):
                        if buffer_name.endswith(("mean", "min")):
                            value.zero_()
                        elif buffer_name.endswith(("std", "max")):
                            value.fill_(1.0)
        ACT._set_identity_normalizers = set_identity_normalizers_compat

        checkpoint = MODEL_ROOT / "final_model/pytorch_model.pt"
        if checkpoint.stat().st_size != CHECKPOINT_SIZE:
            raise RuntimeError("ACT checkpoint size mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        cfg = OmegaConf.load(str(MODEL_ROOT / "config.yaml"))
        framework = cfg.framework
        if (
            framework.name != "ACT"
            or framework.action_dim != ACTION_DIM
            or framework.state_dim != STATE_DIM
            or framework.chunk_size != HORIZON
            or tuple(framework.image_keys) != CAMERAS
        ):
            raise RuntimeError("wrong ACT architecture/config contract")
        # Every backbone weight is restored from the final checkpoint; do not
        # attempt an offline torchvision ImageNet download during construction.
        framework.pretrained_backbone_weights = None
        model = ACT(cfg)
        weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        result = model.load_state_dict(weights, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("ACT checkpoint strict restore failed")
        del weights
        gc.collect()
        self.model = model.eval().to("cuda:0")
        torch.cuda.synchronize()
        self.torch = torch
        self.state_q01, self.state_q99, self.action_q01, self.action_q99 = stats_arrays()
        self.requests = 0
        self.replan_requests = 0
        self.latencies = []
        print("DUALARM_LT_ACT_GPU_READY shadow_only=True rtc_mode=fresh_replan", flush=True)

    def predict(self, body):
        import numpy as np

        if body.get("protocol") != "dualarm-lt-rtc-v1":
            raise ContractError("wrong protocol")
        instruction = body.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ContractError("invalid instruction")
        rtc = parse_rtc(body)
        state = normalize_state(body.get("state"), self.state_q01, self.state_q99)
        example = {"image": decode_images(body), "state": state}
        started = time.perf_counter()
        with self.torch.inference_mode():
            norm = np.asarray(
                self.model.predict_action(examples=[example])["normalized_actions"],
                dtype=np.float32,
            )
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        physical = unnormalize_actions(norm, self.action_q01, self.action_q99)
        self.requests += 1
        if rtc.guided:
            self.replan_requests += 1
            self.latencies = (self.latencies + [elapsed])[-200:]
        return {
            "protocol": "dualarm-lt-rtc-v1",
            "model_identity": MODEL_ID,
            "action_shape": [HORIZON, ACTION_DIM],
            "candidate_actions": physical.tolist(),
            "normalized_actions": norm[0].tolist(),
            "rtc_guided": False,
            "rtc_request_had_history": rtc.guided,
            "rtc_mode": "fresh_replan",
            "rtc_executed_steps": rtc.executed_steps,
            "rtc_delay_steps": rtc.delay_steps,
            "inference_elapsed_s": elapsed,
            "camera_order": list(CAMERAS),
            "state_contract": "arm_0_dataset_pose_xyzw + DI4,DI6,DI8..DI22",
            "instruction_conditioned": False,
            "motion_authorized": False,
            "actions_executed": 0,
        }

    def health(self):
        values = sorted(self.latencies)
        p95 = values[min(len(values) - 1, int(0.95 * (len(values) - 1)))] if values else None
        return {
            "status": "DEV_SHADOW_ONLY",
            "model": MODEL_ID,
            "rtc_supported": True,
            "guided_sampler_implemented": False,
            "rtc_mode": "fresh_replan",
            "instruction_conditioned": False,
            "rtc_hardware_validated": False,
            "motion_authorized": False,
            "action_shape": [HORIZON, ACTION_DIM],
            "state_dim": STATE_DIM,
            "camera_order": list(CAMERAS),
            "requests": self.requests,
            "replan_requests": self.replan_requests,
            "replan_p95_latency_s": p95,
        }


class Handler(BaseHTTPRequestHandler):
    predictor = None

    def json_response(self, status, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self.json_response(200, self.predictor.health())
        self.json_response(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/predict":
            return self.json_response(404, {"error": "not_found"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 18_000_000:
                raise ContractError("invalid request size")
            body = json.loads(self.rfile.read(size))
            result = self.predictor.predict(body)
        except (ContractError, ValueError, TypeError, KeyError) as exc:
            return self.json_response(
                422, {"status": "INVALID_REQUEST", "error": str(exc), "motion_authorized": False}
            )
        except Exception as exc:
            return self.json_response(
                500, {"status": "MODEL_ERROR", "error": str(exc), "motion_authorized": False}
            )
        self.json_response(200, result)


def self_test():
    import numpy as np

    if (MODEL_ROOT / "dataset_statistics.json").is_file():
        sq01, sq99, aq01, aq99 = stats_arrays()
    else:
        sq01 = np.zeros(STATE_DIM, dtype=np.float32)
        sq99 = np.ones(STATE_DIM, dtype=np.float32)
        aq01 = np.r_[np.zeros(3), [-1, -1, -1, -1], np.zeros(4)].astype(np.float32)
        aq99 = np.ones(ACTION_DIM, dtype=np.float32)
    assert sq01.shape == sq99.shape == (STATE_DIM,)
    assert aq01.shape == aq99.shape == (ACTION_DIM,)
    raw = np.r_[0.5 * (sq01[:7] + sq99[:7]), np.ones(17)]
    normalized = normalize_state(raw, sq01, sq99)
    assert normalized.shape == (STATE_DIM,) and np.allclose(normalized[:7], 0, atol=1e-5)
    fake = np.zeros((1, HORIZON, ACTION_DIM), dtype=np.float32)
    fake[:, :, 3] = 1
    physical = unnormalize_actions(fake, aq01, aq99)
    assert physical.shape == (HORIZON, ACTION_DIM)
    print("DUALARM_LT_ACT_SERVER_SELF_TEST_PASS; NO GPU/NETWORK/ROBOT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.serve:
        parser.error("--serve required")
    Handler.predictor = Predictor()
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        print(
            f"DUALARM_LT_ACT_HTTP_READY loopback:{args.port} "
            "shadow_only=True rtc_mode=fresh_replan",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
