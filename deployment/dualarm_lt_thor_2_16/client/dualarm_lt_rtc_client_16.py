#!/usr/bin/env python3
"""2.16 dual-arm LT observation/inference support for all Docker profiles.

This module reads Topic/cameras and calls a loopback Docker model service. The
separate executor owns dry-run/real RPC behavior and imports the command SDK.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np
import yaml

KNOWN_CAMERAS = ("left_side", "left_wrist", "right_wrist", "right_side")
HORIZON = 50
FULL_STATE_DIM = 24
FT_COMPONENTS = ("fx", "fy", "fz", "mx", "my", "mz")
NO_PROXY_OPENER = build_opener(ProxyHandler({}))


class ClientError(RuntimeError):
    pass


class PersistentTopicState:
    """Keep one StatusReader connection and serve thread-safe RT/NRT snapshots.

    The control loop and the background inference thread only copy references to
    complete messages under the condition lock.  StatusReader.read() is called
    by this class's single receiver thread.
    """

    NAMES = ("SystemRtState", "SystemNrtState")

    def __init__(self, cfg):
        robot = cfg["robot"]
        topic = robot.get("topic", {})
        self.ip = str(robot["topic_ip"])
        self.port = int(robot["topic_port"])
        self.initial_timeout_s = float(topic.get("initial_timeout_s", 4.0))
        self.rt_stale_s = float(topic.get("rt_stale_s", 0.20))
        self.nrt_stale_s = float(topic.get("nrt_stale_s", 0.30))
        self.frame_stale_s = float(topic.get("frame_stale_s", 1.0))
        self.max_skew_s = float(topic.get("max_rt_nrt_rx_skew_s", 0.30))
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._connected = False
        self._ever_connected = False
        self._faulted = False
        self._generation = 0
        self._last_error = None
        self._messages = {name: None for name in self.NAMES}
        self._last_rx = {name: None for name in self.NAMES}
        self._frame_id = {name: None for name in self.NAMES}
        self._timestamp = {name: None for name in self.NAMES}
        self._last_frame_change = {name: None for name in self.NAMES}
        self._last_timestamp_change = {name: None for name in self.NAMES}

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def start(self):
        if self._thread is not None:
            raise ClientError("persistent Topic reader was already started")
        self._thread = threading.Thread(
            target=self._receive_loop, name="dualarm-topic-reader", daemon=True
        )
        self._thread.start()
        try:
            self.snapshot(timeout_s=self.initial_timeout_s)
        except Exception:
            self.close()
            raise
        print(
            f"[TOPIC] persistent reader ready tcp://{self.ip}:{self.port} "
            f"generation={self._generation}", flush=True,
        )
        return self

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    def _receive_loop(self):
        from get_status import StatusReader

        try:
            with StatusReader(self.ip, self.port) as reader:
                with self._condition:
                    self._connected = True
                    self._ever_connected = True
                    self._generation += 1
                    self._condition.notify_all()
                while not self._stop.is_set():
                    message = reader.read(250)
                    if message is None:
                        continue
                    name = type(message).__name__
                    if name not in self._messages:
                        continue
                    now = time.monotonic()
                    frame_id = int(getattr(message.head, "frame_id"))
                    timestamp = int(getattr(message.head, "timestamp"))
                    with self._condition:
                        if self._frame_id[name] != frame_id:
                            self._last_frame_change[name] = now
                        if self._timestamp[name] != timestamp:
                            self._last_timestamp_change[name] = now
                        self._messages[name] = message
                        self._last_rx[name] = now
                        self._frame_id[name] = frame_id
                        self._timestamp[name] = timestamp
                        self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    # Once a live connection is lost, keep the fault sticky.
                    # A real client must be restarted instead of silently
                    # resuming motion after a transport reconnect.
                    self._faulted = self._ever_connected
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._connected = False
                self._condition.notify_all()

    def diagnostics(self):
        now = time.monotonic()
        with self._condition:
            result = {
                "connected": self._connected,
                "faulted": self._faulted,
                "connection_generation": self._generation,
                "last_error": self._last_error,
            }
            for name, prefix in (("SystemRtState", "rt"), ("SystemNrtState", "nrt")):
                result[f"{prefix}_age_s"] = (
                    None if self._last_rx[name] is None else now - self._last_rx[name]
                )
                result[f"{prefix}_frame_change_age_s"] = (
                    None if self._last_frame_change[name] is None
                    else now - self._last_frame_change[name]
                )
                result[f"{prefix}_timestamp_change_age_s"] = (
                    None if self._last_timestamp_change[name] is None
                    else now - self._last_timestamp_change[name]
                )
                result[f"{prefix}_frame_id"] = self._frame_id[name]
                result[f"{prefix}_timestamp"] = self._timestamp[name]
            rt_rx = self._last_rx["SystemRtState"]
            nrt_rx = self._last_rx["SystemNrtState"]
            result["rt_nrt_rx_skew_s"] = (
                None if rt_rx is None or nrt_rx is None else abs(rt_rx - nrt_rx)
            )
            return result

    def snapshot(self, timeout_s=None):
        timeout_s = self.initial_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while any(self._messages[name] is None for name in self.NAMES):
                if self._faulted:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            messages = tuple(self._messages[name] for name in self.NAMES)
        diag = self.diagnostics()
        if self._faulted or not self._connected:
            raise ClientError(
                "persistent Topic connection lost; restart client; "
                f"diag={json.dumps(diag, sort_keys=True)}"
            )
        if any(message is None for message in messages):
            raise ClientError(
                "Topic initial timeout waiting for both RT and NRT state; "
                f"diag={json.dumps(diag, sort_keys=True)}"
            )
        checks = (
            ("rt", self.rt_stale_s),
            ("nrt", self.nrt_stale_s),
        )
        for prefix, stale_s in checks:
            if diag[f"{prefix}_age_s"] > stale_s:
                raise ClientError(
                    f"Topic {prefix.upper()} stale; diag={json.dumps(diag, sort_keys=True)}"
                )
            if diag[f"{prefix}_frame_change_age_s"] > self.frame_stale_s:
                raise ClientError(
                    f"Topic {prefix.upper()} frame frozen; diag={json.dumps(diag, sort_keys=True)}"
                )
            if diag[f"{prefix}_timestamp_change_age_s"] > self.frame_stale_s:
                raise ClientError(
                    f"Topic {prefix.upper()} timestamp frozen; diag={json.dumps(diag, sort_keys=True)}"
                )
        if diag["rt_nrt_rx_skew_s"] > self.max_skew_s:
            raise ClientError(
                "Topic RT/NRT receive skew exceeds limit; "
                f"diag={json.dumps(diag, sort_keys=True)}"
            )
        return messages[0], messages[1], diag


def load_config(path: str, profile_name: str | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or cfg.get("version") != "dualarm-lt-rtc-v1":
        raise ClientError("wrong or missing config version")
    robot, cameras, infer, rtc = (cfg.get(k, {}) for k in ("robot", "cameras", "inference", "rtc"))
    expected_di = ["DI4", "DI6"] + [f"DI{i}" for i in range(8, 23)]
    if robot.get("digital_inputs") != expected_di:
        raise ClientError("digital_inputs must be DI4,DI6,DI8..DI22 in training order")
    topic = robot.get("topic", {})
    topic_limits = {
        "initial_timeout_s": (0.5, 30.0),
        "rt_stale_s": (0.02, 2.0),
        "nrt_stale_s": (0.05, 2.0),
        "frame_stale_s": (0.10, 5.0),
        "max_rt_nrt_rx_skew_s": (0.05, 2.0),
    }
    for name, (minimum, maximum) in topic_limits.items():
        value = float(topic.get(name, {
            "initial_timeout_s": 4.0,
            "rt_stale_s": 0.20,
            "nrt_stale_s": 0.30,
            "frame_stale_s": 1.0,
            "max_rt_nrt_rx_skew_s": 0.30,
        }[name]))
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ClientError(f"robot.topic.{name} must be in [{minimum},{maximum}]")
    profiles = infer.get("profiles", {})
    selected = profile_name or infer.get("default_profile")
    if selected not in profiles:
        raise ClientError(f"unknown inference profile: {selected}")
    profile = dict(profiles[selected])
    profile["name"] = selected
    if int(profile.get("state_dim", 0)) not in (7, 24):
        raise ClientError("inference profile state_dim must be 7 or 24")
    action_dim = int(profile.get("action_dim", 0))
    do_names = list(profile.get("action_do_names", []))
    do_nums = list(profile.get("action_do_nums", []))
    if action_dim not in (9, 11) or action_dim != 7 + len(do_nums):
        raise ClientError("profile action_dim must equal pose 7D plus configured DO outputs")
    if len(do_names) != len(do_nums) or len(set(do_names)) != len(do_names):
        raise ClientError("profile action DO names/numbers are inconsistent")
    if any(name != f"DO{number}" for name, number in zip(do_names, do_nums)):
        raise ClientError("profile action DO name does not match its number")
    if profile.get("moving_model") not in ("HRC-dh-L-tool", "HRC-dh-R2-fix-tool"):
        raise ClientError("profile moving_model is not a supported arm")
    if int(profile.get("joganyc_slot", -1)) not in (0, 1):
        raise ClientError("profile joganyc_slot must be 0 for left or 1 for right")
    if not str(profile.get("instruction", "")).strip():
        raise ClientError("profile instruction is empty")
    camera_order = tuple(profile.get("camera_order", []))
    if not camera_order or len(set(camera_order)) != len(camera_order):
        raise ClientError("inference profile camera_order is empty or duplicated")
    if any(name not in KNOWN_CAMERAS or not cameras.get(name) for name in camera_order):
        raise ClientError("inference profile contains an unknown/unconfigured camera")
    if profile.get("rtc_mode") not in ("guided", "fresh_replan"):
        raise ClientError("inference profile rtc_mode must be guided or fresh_replan")
    upload_size = profile.get("upload_image_size")
    if upload_size is not None:
        if upload_size != [224, 224]:
            raise ClientError("selected profile does not support 224x224 image upload")
        quality = profile.get("upload_jpeg_quality", 90)
        if isinstance(quality, bool) or not isinstance(quality, int) or not 80 <= quality <= 100:
            raise ClientError("upload_jpeg_quality must be an integer in [80,100]")
    te = profile.get("temporal_ensemble", {})
    if te.get("enabled", False):
        if profile.get("rtc_mode") != "fresh_replan":
            raise ClientError("temporal ensemble requires fresh_replan RTC mode")
        if not 0 < float(te.get("decay_per_plan", 0)) <= 5:
            raise ClientError("temporal_ensemble.decay_per_plan must be in (0,5]")
        if not 2 <= int(te.get("max_plans", 0)) <= 20:
            raise ClientError("temporal_ensemble.max_plans must be in [2,20]")
    if not str(profile.get("url", "")).startswith("http://127.0.0.1:"):
        raise ClientError("inference URL must use a loopback Docker service")
    frames = profile.get("frames", {})
    for key in ("topic_pose_frame_without_setwobj", "topic_pose_frame_after_setwobj"):
        if frames.get(key) not in ("auto", "dataset", "controller"):
            raise ClientError(f"profile.frames.{key} must be auto, dataset or controller")
    if frames.get("command_pose_frame_with_setwobj") not in ("dataset", "controller"):
        raise ClientError("profile command pose frame must be dataset or controller")
    needs_transform = any(frames.get(key) in ("auto", "controller") for key in (
        "topic_pose_frame_without_setwobj", "topic_pose_frame_after_setwobj",
    )) or frames.get("command_pose_frame_with_setwobj") == "controller"
    frame = frames.get("dataset_from_controller", {})
    if needs_transform and (
        len(frame.get("translation_m", [])) != 3
        or len(frame.get("quaternion_xyzw", [])) != 4
    ):
        raise ClientError("selected profile requires dataset_from_controller transform")
    safety = profile.get("safety", {})
    xyz_min = np.asarray(safety.get("dataset_xyz_min", []), dtype=np.float64)
    xyz_max = np.asarray(safety.get("dataset_xyz_max", []), dtype=np.float64)
    if xyz_min.shape != (3,) or xyz_max.shape != (3,) or not np.all(xyz_min < xyz_max):
        raise ClientError("profile safety workspace must contain finite ordered XYZ bounds")
    for key in ("max_live_translation_m", "max_live_rotation_rad"):
        value = float(safety.get(key, 0))
        if not math.isfinite(value) or value <= 0:
            raise ClientError(f"profile safety {key} must be positive")
    real = profile.get("real", {})
    if any(not isinstance(real.get(key), bool) for key in (
        "enabled", "wobj_dataset_frame_verified", "workspace_verified",
    )):
        raise ClientError("profile real authorization flags must be explicit booleans")
    robot["moving_model"] = profile["moving_model"]
    cfg["frames"] = frames
    execution = cfg.get("execution", {})
    execution["action_do_names"] = do_names
    execution["action_do_nums"] = do_nums
    execution["joganyc_slot"] = int(profile["joganyc_slot"])
    execution["safety"] = safety
    cfg["execution"] = execution
    cfg["inference"]["active_profile"] = profile
    hz = float(rtc.get("shadow_hz", 0))
    if not 0 < hz <= 10:
        raise ClientError("rtc.shadow_hz must be in (0,10]")
    return cfg


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def quat_rotate(q, p):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    v = q[:3]
    return np.asarray(p, dtype=np.float64) + 2 * np.cross(v, np.cross(v, p) + q[3] * np.asarray(p))


def controller_to_dataset(pose, cfg):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ClientError("controller pose is not finite 7D xyzw")
    tf = cfg.get("frames", {}).get("dataset_from_controller")
    if not isinstance(tf, dict):
        raise ClientError("selected profile has no controller-to-dataset calibration")
    t = np.asarray(tf["translation_m"], dtype=np.float64)
    q = np.asarray(tf["quaternion_xyzw"], dtype=np.float64)
    q /= np.linalg.norm(q)
    out_q = quat_mul(q, pose[3:] / np.linalg.norm(pose[3:]))
    out_q /= np.linalg.norm(out_q)
    return np.r_[quat_rotate(q, pose[:3]) + t, out_q].astype(np.float32)


def extract_compensated_ft(rt):
    """Return the operator-confirmed compensated arm wrenches from RT Topic."""
    ftvalues = list(getattr(getattr(rt, "controller", None), "ftvalues", []))

    def vector(index):
        if index >= len(ftvalues):
            return None
        result = [float(getattr(ftvalues[index], name)) for name in FT_COMPONENTS]
        if not np.isfinite(result).all():
            raise ClientError(f"controller.ftvalues[{index}] contains NaN or infinity")
        return result

    return {
        "components": list(FT_COMPONENTS),
        "left_arm1": {"index": 0, "values": vector(0)},
        "right_arm0": {"index": 2, "values": vector(2)},
        "available_count": len(ftvalues),
    }


def capture_state(
    cfg, timeout_s=4.0, return_metadata=False, topic_pose_frame=None,
    allow_missing_io=False,
):
    robot = cfg["robot"]
    topic_state = cfg.get("_topic_state")
    if topic_state is None:
        raise ClientError("persistent Topic reader is not initialized")
    rt, nrt, topic_diag = topic_state.snapshot(timeout_s=timeout_s)
    models = {m.model_name: m for m in rt.model}
    moving = models.get(robot["moving_model"])
    if moving is None or len(moving.current_point.robottarget) != 7:
        raise ClientError("moving-arm RT pose unavailable")
    io_models = {m.model_name: m for m in nrt.model}
    io_model = io_models.get(robot["io_model"])
    values = {} if io_model is None else {
        item.io_name.rsplit(".", 1)[-1]: float(item.io_data) for item in io_model.io
    }
    missing = [name for name in robot["digital_inputs"] if name not in values]
    if missing:
        if not allow_missing_io:
            raise ClientError(
                f"missing DI: {missing}; available NRT models={sorted(io_models)}"
            )
        values.update({name: 0.0 for name in missing})
    topic_pose = np.asarray(list(moving.current_point.robottarget), dtype=np.float64)
    if topic_pose.shape != (7,) or not np.isfinite(topic_pose).all():
        raise ClientError("Topic robottarget is not finite 7D xyzw")
    topic_pose[3:] /= np.linalg.norm(topic_pose[3:])
    if topic_pose_frame is None:
        topic_pose_frame = cfg["frames"]["topic_pose_frame_without_setwobj"]
    topic_frame = str(topic_pose_frame)
    if topic_frame not in ("auto", "dataset", "controller"):
        raise ClientError("topic_pose_frame override must be auto, dataset or controller")
    transformed_pose = None
    if topic_frame in ("auto", "controller"):
        transformed_pose = controller_to_dataset(topic_pose, cfg)
    if topic_frame == "auto":
        safety = cfg.get("execution", {}).get("safety", {})
        xyz_min = np.asarray(safety.get("dataset_xyz_min", []), dtype=np.float64)
        xyz_max = np.asarray(safety.get("dataset_xyz_max", []), dtype=np.float64)
        if xyz_min.shape != (3,) or xyz_max.shape != (3,):
            raise ClientError("auto Topic frame detection requires dataset XYZ bounds")
        raw_inside = bool(np.all(topic_pose[:3] >= xyz_min) and np.all(topic_pose[:3] <= xyz_max))
        transformed_inside = bool(
            np.all(transformed_pose[:3] >= xyz_min) and np.all(transformed_pose[:3] <= xyz_max)
        )
        if raw_inside == transformed_inside:
            raise ClientError(
                "Topic pose frame is ambiguous: "
                f"raw_inside={raw_inside} transformed_inside={transformed_inside}"
            )
        topic_frame = "dataset" if raw_inside else "controller"
    if topic_frame == "dataset":
        pose = topic_pose.astype(np.float32)
    else:
        pose = transformed_pose
    state = np.r_[pose, [values[name] for name in robot["digital_inputs"]]].astype(np.float32)
    if state.shape != (FULL_STATE_DIM,) or not np.isfinite(state).all():
        raise ClientError("packed state is not finite 24D")
    if return_metadata:
        return state, {
            "arm_label": cfg["inference"]["active_profile"].get("arm_label"),
            "moving_model": robot["moving_model"],
            "system_running_state": bool(rt.system_running_state),
            "all_moving_joints_enabled": all(bool(joint.is_enabled) for joint in moving.joint),
            "topic_pose_xyzw": topic_pose.tolist(),
            "topic_pose_frame": topic_frame,
            "io_values": values,
            "io_available": not missing,
            "compensated_ft": extract_compensated_ft(rt),
            "topic_diagnostics": topic_diag,
        }
    return state


def capture_images(cfg, camera_order):
    ccfg = cfg["cameras"]
    profile = cfg["inference"]["active_profile"]
    upload_size = profile.get("upload_image_size")
    result = {}
    for name in camera_order:
        url = str(ccfg[name])
        if not url.startswith("http://192.168.2.16:6888/ObCameraSensor/"):
            raise ClientError(f"camera must use the local ObCameraSensor endpoint: {name}")
        try:
            with NO_PROXY_OPENER.open(url, timeout=5.0) as stream:
                data = bytearray()
                jpeg = None
                while len(data) < 8_000_000:
                    chunk = stream.read(64 * 1024)
                    if not chunk:
                        break
                    data.extend(chunk)
                    start = data.find(b"\xff\xd8")
                    end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start >= 0 and end >= 0:
                        jpeg = bytes(data[start:end + 2])
                        break
        except OSError as exc:
            raise ClientError(f"camera HTTP read failed: {name}={url}: {exc}") from exc
        if jpeg is None:
            raise ClientError(f"no complete JPEG frame: {name}={url}")
        if upload_size is not None:
            from PIL import Image
            try:
                with Image.open(io.BytesIO(jpeg)) as image:
                    resized = image.convert("RGB").resize(tuple(upload_size))
                output = io.BytesIO()
                resized.save(
                    output, format="JPEG", quality=int(profile.get("upload_jpeg_quality", 90))
                )
                jpeg = output.getvalue()
            except (OSError, ValueError) as exc:
                raise ClientError(f"camera resize/encode failed: {name}: {exc}") from exc
        result[name] = base64.b64encode(jpeg).decode("ascii")
    return result


def post_json(url, path, body=None, timeout=30.0):
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    req = Request(url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with NO_PROXY_OPENER.open(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read(4096).decode(errors="replace")
        raise ClientError(f"HTTP {exc.code}: {detail}") from exc


def check_server_health(cfg):
    infer = cfg["inference"]
    profile = infer["active_profile"]
    health = post_json(profile["url"], "/health", timeout=5.0)
    expected = {
        "status": "DEV_SHADOW_ONLY",
        "model": profile["model_identity"],
        "action_shape": [HORIZON, int(profile["action_dim"])],
        "state_dim": int(profile["state_dim"]),
        "camera_order": list(profile["camera_order"]),
        "rtc_mode": profile["rtc_mode"],
        "motion_authorized": False,
    }
    mismatches = {key: (expected_value, health.get(key)) for key, expected_value in expected.items()
                  if health.get(key) != expected_value}
    if mismatches:
        raise ClientError(f"inference server/profile mismatch: {mismatches}")
    return health


def make_request(cfg, previous=None, executed=None, delay=None, state=None):
    started = time.monotonic()
    profile = cfg["inference"]["active_profile"]
    if state is None:
        state = capture_state(
            cfg, allow_missing_io=int(profile["state_dim"]) == 7
        )
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (FULL_STATE_DIM,) or not np.isfinite(state).all():
        raise ClientError("inference state is not finite 24D")
    model_state = state[:int(profile["state_dim"])]
    state_ready = time.monotonic()
    images = capture_images(cfg, profile["camera_order"])
    images_ready = time.monotonic()
    body = {
        "protocol": "dualarm-lt-rtc-v1",
        "instruction": profile["instruction"],
        "state": model_state.tolist(),
        "images": images,
        "rtc": {"num_steps": int(cfg["rtc"]["num_inference_steps"])},
    }
    if previous is not None:
        body["rtc"].update(
            previous_actions=np.asarray(previous, dtype=np.float32).tolist(),
            executed_steps=int(executed),
            delay_steps=int(delay),
            guidance_beta=float(cfg["rtc"]["guidance_beta"]),
        )
    result = post_json(profile["url"], "/predict", body, float(cfg["inference"]["timeout_s"]))
    response_ready = time.monotonic()
    actions = np.asarray(result.get("candidate_actions"), dtype=np.float32)
    action_dim = int(profile["action_dim"])
    if actions.shape != (HORIZON, action_dim) or not np.isfinite(actions).all():
        raise ClientError(f"server returned invalid 50x{action_dim} actions")
    if result.get("motion_authorized") is not False or result.get("actions_executed") != 0:
        raise ClientError("server shadow-only contract violated")
    if result.get("model_identity") != profile["model_identity"]:
        raise ClientError("prediction came from the wrong model identity")
    if result.get("rtc_mode") != profile["rtc_mode"]:
        raise ClientError("prediction RTC mode does not match selected profile")
    result["_client_timing"] = {
        "state_s": state_ready - started,
        "cameras_s": images_ready - state_ready,
        "http_s": response_ready - images_ready,
        "total_s": response_ready - started,
    }
    return actions, result


def schedule(cfg):
    rtc = cfg["rtc"]
    hz = float(rtc["shadow_hz"])
    delay = math.ceil(float(rtc["expected_guided_latency_s"]) * hz) + 1 + int(rtc["schedule_margin_steps"])
    if delay < 1 or delay >= HORIZON - 1:
        raise ClientError(f"RTC schedule infeasible: delay={delay}, horizon={HORIZON}")
    # Request early enough to keep the guided prediction within 50 steps.
    request_at = min(delay + 2, HORIZON - delay)
    return delay, request_at


def log_line(handle, kind, **values):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **values}
    handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def run_shadow(cfg, once=False):
    if not cfg["cameras"].get("mapping_verified", False):
        raise ClientError("camera mapping is unverified; run camera_probe.py and set mapping_verified: true")
    health = check_server_health(cfg)
    actions, meta = make_request(cfg)
    print(f"[RTC][initial] profile={cfg['inference']['active_profile']['name']} "
          f"mode={health['rtc_mode']} shape={actions.shape} "
          f"latency={meta['inference_elapsed_s']:.3f}s shadow-only")
    if once:
        return
    delay, request_at = schedule(cfg)
    period = 1.0 / float(cfg["rtc"]["shadow_hz"])
    limit = int(cfg["rtc"]["max_shadow_actions"])
    log_dir = Path(cfg["logging"]["directory"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("dualarm_lt_rtc_shadow_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".jsonl")
    total = index = cycle = 0
    pending = None
    request_start = 0.0
    with log_path.open("x") as log, ThreadPoolExecutor(max_workers=1) as pool:
        while total < limit:
            tick = time.monotonic()
            if index == request_at and pending is None:
                suffix = actions[index:].copy()
                pending = pool.submit(make_request, cfg, suffix, index, delay)
                request_start = tick
                print(f"[RTC] {health['rtc_mode']} request cycle={cycle} index={index} delay={delay}")
            candidate = actions[index]
            log_line(log, "candidate_not_sent", total=total, cycle=cycle, index=index, action=candidate.tolist())
            total += 1
            index += 1
            if pending is not None and pending.done():
                replacement, info = pending.result()
                elapsed_steps = max(1, math.ceil((time.monotonic() - request_start) / period))
                actions = replacement
                index = min(elapsed_steps, HORIZON - 1)
                cycle += 1
                pending = None
                print(f"[RTC][replacement] cycle={cycle} latency={info['inference_elapsed_s']:.3f}s skipped={index}")
            elif index >= HORIZON:
                if pending is None:
                    actions, info = make_request(cfg)
                    cycle += 1
                    index = 0
                    print(f"[RTC][fresh] cycle={cycle} latency={info['inference_elapsed_s']:.3f}s")
                else:
                    replacement, info = pending.result()
                    actions, index, pending = replacement, 0, None
                    cycle += 1
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    print(f"DUALARM_LT_RTC_SHADOW_DONE candidates={total} commands_sent=0 log={log_path}")


def self_test():
    cfg = {
        "frames": {"dataset_from_controller": {
            "translation_m": [0.11193867415485362, -0.47926710205246703, -0.20341395009452984],
            "quaternion_xyzw": [-0.5242465118463747, 0.3707491517186816, -0.49148424278267977, 0.5883484515260688],
        }}
    }
    controller = [0.1637749059213684, 0.07758614898564292, 0.07697811245458305,
                  0.9911178111928656, 0.13226829373843216, -0.009351784367984565, 0.010155143371058932]
    expected = [0.2395293265581131, -0.6207573413848877, -0.2530989348888397,
                0.6393394470214844, -0.41043657064437866, -0.44729048013687134, 0.47193020582199097]
    actual = controller_to_dataset(controller, cfg)
    if not np.allclose(actual, expected, atol=2e-6):
        raise AssertionError((actual, expected))
    print("DUALARM_LT_RTC_CLIENT_SELF_TEST_PASS; SHADOW_ONLY; NO NETWORK/CAMERA/ROBOT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("dualarm_lt_rtc_16.yaml")))
    parser.add_argument("--profile")
    parser.add_argument("--instruction", help="Override an instruction-conditioned profile task")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--check-config", action="store_true")
    group.add_argument("--once", action="store_true")
    group.add_argument("--shadow-rtc", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    cfg = load_config(args.config, args.profile)
    if args.instruction is not None:
        profile = cfg["inference"]["active_profile"]
        if not profile.get("instruction_conditioned", False):
            parser.error("--instruction requires an instruction-conditioned profile")
        if not args.instruction.strip():
            parser.error("--instruction must not be empty")
        profile["instruction"] = args.instruction.strip()
    if args.check_config:
        delay, request_at = schedule(cfg)
        profile = cfg["inference"]["active_profile"]
        print(f"CONFIG_OK profile={profile['name']} rtc_mode={profile['rtc_mode']} "
              f"state_dim={profile['state_dim']} action_dim={profile['action_dim']} "
              f"arm={profile['arm_label']} DO={profile['action_do_names']} "
              f"cameras={profile['camera_order']} real_enabled={profile['real']['enabled']} "
              f"shadow_only=True camera_mapping_verified={cfg['cameras']['mapping_verified']} "
              f"delay={delay} request_at={request_at}")
        return
    with PersistentTopicState(cfg) as topic_state:
        cfg["_topic_state"] = topic_state
        try:
            run_shadow(cfg, once=args.once)
        finally:
            cfg.pop("_topic_state", None)


if __name__ == "__main__":
    try:
        main()
    except (ClientError, OSError, ValueError) as exc:
        print(f"DUALARM_LT_RTC_CLIENT_FAIL_CLOSED {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

