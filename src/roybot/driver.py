"""Closed-form differential drive: body velocity command -> wheel angular speeds.

No learning here — this is solved kinematics. The policy outputs (v_fwd, v_yaw);
this turns it into the two wheel speeds MuJoCo's velocity actuators consume.
"""
from .config import WHEEL_RADIUS, WHEEL_BASE, MAX_WHEEL_RAD_S


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def differential_drive(
    v_fwd: float,
    v_yaw: float,
    *,
    calib_left: float = 1.0,
    calib_right: float = 1.0,
) -> tuple[float, float]:
    """(forward m/s, yaw rad/s) -> (left_rad_s, right_rad_s), clamped to motor cap."""
    half = WHEEL_BASE / 2.0
    v_left = (v_fwd - v_yaw * half) / WHEEL_RADIUS
    v_right = (v_fwd + v_yaw * half) / WHEEL_RADIUS
    v_left *= calib_left
    v_right *= calib_right
    return (
        _clamp(v_left, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S),
        _clamp(v_right, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S),
    )
