"""Shared train/serve depth and state semantics; no inferred uint16 scaling."""
import numpy as np

CAMERAS = ("agentview", "wrist")

def metric_depth(raw, *, encoding, near=None, far=None):
    x = np.asarray(raw, dtype=np.float32)
    valid = np.isfinite(x)
    if encoding == "meters":
        valid &= x > 0
        out = x
    elif encoding == "millimeters":
        valid &= x > 0
        out = x * .001
    elif encoding == "mujoco_zbuffer":
        if near is None or far is None or not 0 < near < far:
            raise ValueError("Z-buffer requires actual model near/far in meters")
        valid &= (x >= 0) & (x < 1)
        out = near / (1.0 - np.clip(x, 0, 1) * (1.0 - near / far))
    else:
        raise ValueError("Depth units unverified: supply an audited encoding")
    return np.where(valid, out, 0).astype(np.float32), valid

def state8(obs):
    """Exactly the metadata order; never append camera calibration to state."""
    x = np.asarray(obs, np.float32)
    if x.shape[-1] != 8 or not np.isfinite(x).all():
        raise ValueError("Expected finite xyz + axis-angle + two signed finger qpos")
    return x

def normalize_state(x, stats):
    x = state8(x)
    lo, hi = np.asarray(stats["min"]), np.asarray(stats["max"])
    span = hi-lo
    return np.where(span > 1e-6, np.clip(2*(x-lo)/np.maximum(span, 1e-6)-1, -1, 1), 0).astype(np.float32)

def normalize_actions(x):
    """LIBERO native OSC_POSE [-1,1], including -1 open / +1 close. Identity."""
    x = np.asarray(x, np.float32)
    if x.shape[-1] != 7 or not np.isfinite(x).all() or np.max(np.abs(x)) > 1.0001:
        raise ValueError("Expected finite native LIBERO 7-D actions in [-1,1]")
    return x

def validate_contract(contract, *, lingbot=False):
    for field in ("depth_units_verified", "temporal_alignment_verified", "action_semantics_verified"):
        if not contract.get(field, False):
            raise ValueError(f"Dataset contract not verified: {field}; see alignment report")
    if lingbot and contract.get("alignment_status") != "aligned":
        raise ValueError("LingBot requires verified RGB-D pixel alignment")
    if lingbot and not contract.get("intrinsics_verified", False):
        raise ValueError("LingBot requires verified camera intrinsics")
