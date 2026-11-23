"""snap_lt / insert_lt V2: arm_1 pose + all DI; arm_1 target + DO12/13/15/16."""
from __future__ import annotations

import numpy as np

CAMERA_KEYS = ("observation.images.left_wrist", "observation.images.right_side")
MOVING_ARM_SLICE = slice(7, 14)
DIGITAL_INPUT_NAMES = ("DI4", "DI6", *[f"DI{i}" for i in range(8, 23)])
DIGITAL_OUTPUT_NAMES = ("DO12", "DO13", "DO15", "DO16")
ACTIVE_OUTPUT_INDICES = (6, 7, 9, 10)
POLICY_STATE_DIM = 24
POLICY_ACTION_DIM = 11


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({size},) vector; got {array.shape}")
    return array


def pack_observation(cartesian_position, digital_input):
    pose = _vector(cartesian_position, 14, "cartesian_position")
    di = _vector(digital_input, 17, "digital_input")
    return np.concatenate((pose[MOVING_ARM_SLICE], di))


def pack_action(cartesian_position, digital_output):
    pose = _vector(cartesian_position, 14, "cartesian_position")
    do = _vector(digital_output, 12, "digital_output")
    return np.concatenate((pose[MOVING_ARM_SLICE], do[list(ACTIVE_OUTPUT_INDICES)]))


def unpack_policy_action(policy_action, current_cartesian_position, current_digital_output):
    """Restore full vectors, preserving the other arm and unmodelled outputs.

    This only maps values. A real executor must separately validate poses and
    threshold the four model outputs before issuing robot commands.
    """
    action = _vector(policy_action, POLICY_ACTION_DIM, "policy_action")
    poses = _vector(current_cartesian_position, 14, "current_cartesian_position").copy()
    outputs = _vector(current_digital_output, 12, "current_digital_output").copy()
    poses[MOVING_ARM_SLICE] = action[:7]
    outputs[list(ACTIVE_OUTPUT_INDICES)] = action[7:]
    return {"action.cartesian_position": poses, "action.digital_output": outputs}
