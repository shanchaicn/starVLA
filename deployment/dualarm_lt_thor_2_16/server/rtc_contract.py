"""CPU-testable RTC wire contract for the dualarm LT QwenPI-v3 model."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

HORIZON = 50
STATE_DIM = 24
ACTION_DIM = 11
CAMERAS = ("left_side", "right_wrist", "right_side")


class ContractError(ValueError):
    pass


def array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}: numeric array required") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ContractError(f"{name}: expected finite {shape}, got {result.shape}")
    return result


def integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ContractError(f"{name}: expected integer [{low}, {high}]")
    return int(value)


@dataclass(frozen=True)
class RtcRequest:
    guided: bool
    executed_steps: int
    delay_steps: int
    num_steps: int
    guidance_beta: float
    previous_actions: np.ndarray | None


def parse_rtc(body: dict[str, Any], action_dim: int = ACTION_DIM) -> RtcRequest:
    rtc = body.get("rtc", {})
    if not isinstance(rtc, dict):
        raise ContractError("rtc must be an object")
    num_steps = integer(rtc.get("num_steps", 4), "rtc.num_steps", 2, 8)
    if "previous_actions" not in rtc:
        if "executed_steps" in rtc or "delay_steps" in rtc:
            raise ContractError("guided scheduling fields require previous_actions")
        return RtcRequest(False, 0, 0, num_steps, 0.0, None)
    executed = integer(rtc.get("executed_steps"), "rtc.executed_steps", 1, HORIZON - 1)
    delay = integer(rtc.get("delay_steps"), "rtc.delay_steps", 1, HORIZON - 1)
    previous = array(
        rtc["previous_actions"],
        (HORIZON - executed, action_dim),
        "rtc.previous_actions",
    )
    if delay > len(previous):
        raise ContractError("RTC delay exceeds previous-action suffix length")
    beta = rtc.get("guidance_beta", 5.0)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 < beta <= 10:
        raise ContractError("guidance_beta must be finite in (0, 10]")
    return RtcRequest(True, executed, delay, num_steps, float(beta), previous)


def schedule(latency_s: float, fps: float, margin: int = 2) -> tuple[int, int]:
    if not math.isfinite(latency_s) or latency_s <= 0 or not math.isfinite(fps) or not 0 < fps <= 10:
        raise ContractError("latency/fps invalid")
    delay = math.ceil(latency_s * fps) + 1 + margin
    request_at = delay + 2
    if request_at >= HORIZON or delay > HORIZON - request_at:
        raise ContractError(
            f"RTC schedule infeasible: fps={fps}, latency={latency_s}, delay={delay}, request={request_at}"
        )
    return delay, request_at

