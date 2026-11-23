#!/usr/bin/env python3
"""Guided RTC inference server for dualarm LT QwenPI-v3 V2/V3 models.

This process has no robot SDK and cannot command motion. It binds loopback only.
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
import yaml

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = Path(os.environ.get("STARVLA_MODELS_ROOT", str(ROOT.parent / "models")))
BASE_ROOT = Path(os.environ.get("STARVLA_BASE_VLM", str(MODELS_ROOT / "base/Qwen3-VL-4B-Instruct-Action")))
SOURCE_ROOT = Path(os.environ.get("STARVLA_SOURCE_ROOT", str(ROOT / "runtime/source")))
EVAL_DEPS = Path(os.environ.get("STARVLA_EVAL_DEPS", str(ROOT / "runtime/eval_deps")))
CONFIG_PATH = Path(os.environ.get("STARVLA_DEPLOY_CONFIG", str(ROOT / "server_16.yaml")))
PROFILES = {
    name: profile for name, profile in yaml.safe_load(CONFIG_PATH.read_text())["profiles"].items()
    if profile["script"] == Path(__file__).name
}
PROFILE_NAME = "qwenpi_v3_v2"
PROFILE = PROFILES[PROFILE_NAME]
MODEL_ROOT = Path(os.environ.get("STARVLA_MODEL_ROOT", str(MODELS_ROOT / PROFILE["model_dir"])))
MODEL_ID = PROFILE["model_id"]
STATE_DIM = PROFILE["state_dim"]
PORT = PROFILE["port"]
CHECKPOINT_SIZE = PROFILE["checkpoint_size"]
ACTION_DIM = int(PROFILE["action_dim"])
CAMERAS = tuple(PROFILE["camera_order"])

sys.path[:0] = [str(ROOT), str(EVAL_DEPS), str(SOURCE_ROOT)]
from rtc_contract import ContractError, HORIZON, array, parse_rtc

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    WANDB_DISABLED="true",
    TOKENIZERS_PARALLELISM="false",
)


def stats_arrays():
    import numpy as np

    stats = json.loads((MODEL_ROOT / "dataset_statistics.json").read_text())["new_embodiment"]
    state = stats["state"]
    action = stats["action"]
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
    if STATE_DIM > 7:
        out[7:] = (raw[7:] > 0.5).astype(np.float32)
    return out


def normalize_actions(raw, q01, q99):
    import numpy as np

    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != ACTION_DIM or not np.isfinite(raw).all():
        raise ContractError(f"previous actions must be finite Nx{ACTION_DIM}")
    out = raw.copy()
    out[:, :7] = np.clip(2 * (raw[:, :7] - q01[:7]) / (q99[:7] - q01[:7]) - 1, -2.2, 2.2)
    out[:, 7:] = (raw[:, 7:] > 0.5).astype(np.float32)
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
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3

        checkpoint = MODEL_ROOT / "final_model/pytorch_model.pt"
        if checkpoint.stat().st_size != CHECKPOINT_SIZE:
            raise RuntimeError("dualarm checkpoint size mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        free, _ = torch.cuda.mem_get_info()
        available = free
        if os.environ.get("STARVLA_UNIFIED_MEMORY") == "1":
            import psutil
            available = max(available, psutil.virtual_memory().available)
        minimum = float(os.environ.get("STARVLA_MIN_MEMORY_GIB", "18")) * 1024**3
        if available < minimum:
            raise RuntimeError(f"DUALARM_RTC_GUARD: available={available / 1024**3:.1f}GiB below configured minimum")
        cfg = OmegaConf.load(str(MODEL_ROOT / "config.yaml"))
        action_cfg = cfg.framework.action_model
        if cfg.framework.name != "QwenPI_v3" or action_cfg.action_dim != ACTION_DIM or action_cfg.state_dim != STATE_DIM or action_cfg.action_horizon != HORIZON:
            raise RuntimeError("wrong dualarm architecture")
        cfg.framework.qwenvl.base_vlm = str(BASE_ROOT)
        self.state_q01, self.state_q99, self.action_q01, self.action_q99 = stats_arrays()

        from transformers import AutoConfig, Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, Qwen3VLVisionRotaryEmbedding
        from torch.distributions import Beta as OriginalBeta
        import starVLA.model.modules.action_model.LayerwiseFM_ActionHeader as head_module

        original_pretrained = Qwen3VLForConditionalGeneration.from_pretrained
        original_beta = head_module.Beta

        def meta_base(*_args, **_kwargs):
            config = AutoConfig.from_pretrained(str(BASE_ROOT), local_files_only=True, trust_remote_code=False)
            model = Qwen3VLForConditionalGeneration(config)
            return model, dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])

        Qwen3VLForConditionalGeneration.from_pretrained = meta_base
        # Model construction happens on the meta device so the 10.9 GB checkpoint
        # can be assigned without first allocating a duplicate initialized model.
        # PyTorch distribution validation calls Tensor.item(), which is undefined
        # for meta tensors; the two positive scalar parameters are config-validated.
        head_module.Beta = lambda a, b: OriginalBeta(
            torch.tensor(float(a)), torch.tensor(float(b)), validate_args=False
        )
        try:
            with torch.device("meta"):
                model = Qwen_PI_v3(cfg)
        finally:
            Qwen3VLForConditionalGeneration.from_pretrained = original_pretrained
            head_module.Beta = original_beta
        weights = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        result = model.load_state_dict(weights, strict=True, assign=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("checkpoint strict restore failed")
        del weights
        gc.collect()
        vl = model.qwen_vl_interface.model
        vision_rope = vl.model.visual.rotary_pos_emb
        vision = Qwen3VLVisionRotaryEmbedding(int(vision_rope.inv_freq.numel()) * 2)
        vision_rope.register_buffer("inv_freq", vision.inv_freq, persistent=False)
        text_rope = vl.model.language_model.rotary_emb
        text = Qwen3VLTextRotaryEmbedding(vl.config.text_config, device="cpu")
        text_rope.register_buffer("inv_freq", text.inv_freq, persistent=False)
        text_rope.original_inv_freq = text_rope.inv_freq
        if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
            raise RuntimeError("meta tensor remains after restore")
        self.model = model.eval().to("cuda:0")
        torch.cuda.synchronize()
        self.torch = torch
        self.requests = 0
        self.guided_requests = 0
        self.latencies = []
        if not hasattr(self.model.action_model, "predict_action_realtime"):
            raise RuntimeError("predict_action_realtime missing")
        print("DUALARM_LT_RTC_GPU_READY shadow_only=True", flush=True)

    def context(self, example):
        from starVLA.model.framework.VLM4A.QwenPI_v3 import resize_images, to_pil_preserve

        images = [to_pil_preserve(example["image"])]
        instruction = self.model.add_discretized_state_to_instruction([example["lang"]], [example["state"]])
        size = getattr(self.model.config.datasets.vla_data, "obs_image_size", None)
        if size:
            images = resize_images(images, target_size=size)
        with self.torch.no_grad():
            embeddings, mask = self.model._encode_vl_hidden_states(images, instruction)
        return [x.detach() for x in embeddings], None if mask is None else mask.to(dtype=self.torch.bool)

    def infer(self, example, rtc):
        import numpy as np

        head = self.model.action_model
        old_steps = head.num_inference_timesteps
        head.num_inference_timesteps = rtc.num_steps
        try:
            if not rtc.guided:
                with self.torch.inference_mode():
                    return np.asarray(self.model.predict_action(examples=[example])["normalized_actions"], dtype=np.float32)
            embeddings, mask = self.context(example)
            previous = normalize_actions(rtc.previous_actions, self.action_q01, self.action_q99)
            previous = self.torch.from_numpy(previous[None]).to(embeddings[0].device, dtype=embeddings[0].dtype)
            with self.torch.enable_grad():
                prediction = head.predict_action_realtime(
                    vl_embs_list=embeddings,
                    state=None,
                    prev_action_chunk=previous,
                    inference_delay=rtc.delay_steps,
                    mode="pigdm",
                    suffix_length=rtc.delay_steps,
                    max_guidance_weight=rtc.guidance_beta,
                    encoder_attention_mask=mask,
                )
            # Guided sampling follows the VLM embedding dtype (bfloat16 on this
            # host). NumPy has no bfloat16 scalar type, so convert only the
            # returned action tensor to float32 at the API boundary.
            return prediction.detach().float().cpu().numpy()
        finally:
            head.num_inference_timesteps = old_steps

    def predict(self, body):
        import numpy as np

        if body.get("protocol") != "dualarm-lt-rtc-v1":
            raise ContractError("wrong protocol")
        instruction = body.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1000:
            raise ContractError("invalid instruction")
        rtc = parse_rtc(body, ACTION_DIM)
        state = normalize_state(body.get("state"), self.state_q01, self.state_q99)
        example = {"image": decode_images(body), "lang": instruction, "state": state.reshape(1, STATE_DIM)}
        started = time.perf_counter()
        norm = self.infer(example, rtc)
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        physical = unnormalize_actions(norm, self.action_q01, self.action_q99)
        self.requests += 1
        if rtc.guided:
            self.guided_requests += 1
            self.latencies = (self.latencies + [elapsed])[-200:]
        return {
            "protocol": "dualarm-lt-rtc-v1",
            "model_identity": MODEL_ID,
            "action_shape": [HORIZON, ACTION_DIM],
            "candidate_actions": physical.tolist(),
            "normalized_actions": np.asarray(norm)[0].tolist(),
            "rtc_guided": rtc.guided,
            "rtc_executed_steps": rtc.executed_steps,
            "rtc_delay_steps": rtc.delay_steps,
            "inference_elapsed_s": elapsed,
            "camera_order": list(CAMERAS),
            "state_contract": f"{PROFILE['arm']}_dataset_pose_xyzw" + (" + DI4,DI6,DI8..DI22" if STATE_DIM > 7 else ""),
            "rtc_mode": "guided",
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
            "guided_sampler_implemented": True,
            "rtc_mode": "guided",
            "rtc_hardware_validated": False,
            "motion_authorized": False,
            "action_shape": [HORIZON, ACTION_DIM],
            "state_dim": STATE_DIM,
            "camera_order": list(CAMERAS),
            "requests": self.requests,
            "guided_requests": self.guided_requests,
            "guided_p95_latency_s": p95,
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
            if not 0 < size <= 14_000_000:
                raise ContractError("invalid request size")
            body = json.loads(self.rfile.read(size))
            result = self.predictor.predict(body)
        except (ContractError, ValueError, TypeError, KeyError) as exc:
            return self.json_response(422, {"status": "INVALID_REQUEST", "error": str(exc), "motion_authorized": False})
        except Exception as exc:
            return self.json_response(500, {"status": "MODEL_ERROR", "error": str(exc), "motion_authorized": False})
        self.json_response(200, result)


def self_test():
    import numpy as np

    if (MODEL_ROOT / "dataset_statistics.json").is_file():
        sq01, sq99, aq01, aq99 = stats_arrays()
    else:
        # Contract-only self-test for development hosts that do not carry the
        # 10+ GB model bundle. Deployment tests still read the real statistics.
        sq01 = np.zeros(STATE_DIM, dtype=np.float32)
        sq99 = np.ones(STATE_DIM, dtype=np.float32)
        aq01 = np.r_[np.zeros(3), [-1, -1, -1, -1], np.zeros(ACTION_DIM - 7)].astype(np.float32)
        aq99 = np.ones(ACTION_DIM, dtype=np.float32)
    assert sq01.shape == sq99.shape == (STATE_DIM,)
    assert aq01.shape == aq99.shape == (ACTION_DIM,)
    raw = 0.5 * (sq01 + sq99)
    if STATE_DIM > 7:
        raw[7:] = 1
    normalized = normalize_state(raw, sq01, sq99)
    assert normalized.shape == (STATE_DIM,) and np.allclose(normalized[:7], 0, atol=1e-5)
    fake = np.zeros((1, HORIZON, ACTION_DIM), dtype=np.float32)
    fake[:, :, 3] = 1
    physical = unnormalize_actions(fake, aq01, aq99)
    assert physical.shape == (HORIZON, ACTION_DIM)
    print("DUALARM_LT_RTC_SERVER_SELF_TEST_PASS; NO GPU/NETWORK/ROBOT")


def main():
    global PROFILE_NAME, PROFILE, MODEL_ROOT, MODEL_ID, STATE_DIM, ACTION_DIM, CAMERAS, PORT, CHECKPOINT_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="qwenpi_v3_v2")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    PROFILE_NAME = args.profile
    PROFILE = PROFILES[PROFILE_NAME]
    MODEL_ROOT = Path(os.environ.get("STARVLA_MODEL_ROOT", str(MODELS_ROOT / PROFILE["model_dir"])))
    MODEL_ID = PROFILE["model_id"]
    STATE_DIM = int(PROFILE["state_dim"])
    ACTION_DIM = int(PROFILE["action_dim"])
    CAMERAS = tuple(PROFILE["camera_order"])
    CHECKPOINT_SIZE = int(PROFILE["checkpoint_size"])
    PORT = int(PROFILE["port"] if args.port is None else args.port)
    if args.self_test:
        return self_test()
    if not args.serve:
        parser.error("--serve required")
    if PORT in (10196, 10197) or not 1024 <= PORT <= 65535:
        parser.error("use a dedicated port; 10196/10197 are reserved")
    Handler.predictor = Predictor()
    with HTTPServer(("127.0.0.1", PORT), Handler) as server:
        print(
            f"DUALARM_LT_RTC_HTTP_READY profile={PROFILE_NAME} "
            f"loopback:{PORT} shadow_only=True",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()

