"""V2 policy I/O mapping for the dual-arm LT suction task.

Only arm_0 moves in the recorded task.  V2 models that arm plus suction-related
digital I/O, while preserving arm_1 and all unmodelled outputs at execution.
Force/torque is deliberately excluded.
"""

from __future__ import annotations

import numpy as np


CAMERA_KEYS = (
    "observation.images.left_side",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.right_side",
)

MOVING_ARM_SLICE = slice(0, 7)
ACTIVE_OUTPUT_SLICE = slice(2, 6)  # DO8..DO11
POLICY_POSE_DIM = 7
POLICY_DIGITAL_INPUT_DIM = 17
POLICY_DIGITAL_OUTPUT_DIM = 4
POLICY_STATE_DIM = 24
POLICY_ACTION_DIM = 11


def _vector(value, expected: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {array.shape}")
    return array


def pack_observation(cartesian_position, digital_input) -> np.ndarray:
    """Pack arm_0 pose and all digital inputs into the 24-D V2 state."""

    poses = _vector(cartesian_position, 14, "cartesian_position")
    inputs = _vector(digital_input, 17, "digital_input")
    return np.concatenate((poses[MOVING_ARM_SLICE], inputs))


def pack_action(cartesian_position, digital_output) -> np.ndarray:
    """Pack arm_0 target and DO8--DO11 into the 11-D V2 action."""

    poses = _vector(cartesian_position, 14, "cartesian_position")
    outputs = _vector(digital_output, 12, "digital_output")
    return np.concatenate((poses[MOVING_ARM_SLICE], outputs[ACTIVE_OUTPUT_SLICE]))


def unpack_policy_action(
    policy_action,
    current_cartesian_position,
    current_digital_output,
) -> dict[str, np.ndarray]:
    """Restore an 11-D prediction to safe full robot command vectors."""

    action = _vector(policy_action, POLICY_ACTION_DIM, "policy_action")
    poses = _vector(current_cartesian_position, 14, "current_cartesian_position").copy()
    outputs = _vector(current_digital_output, 12, "current_digital_output").copy()
    poses[MOVING_ARM_SLICE] = action[:POLICY_POSE_DIM]
    outputs[ACTIVE_OUTPUT_SLICE] = action[POLICY_POSE_DIM:]
    return {
        "action.cartesian_position": poses,
        "action.digital_output": outputs,
    }
