import math
from roybot.driver import differential_drive
from roybot import config

def test_straight_is_equal_wheels():
    l, r = differential_drive(0.3, 0.0)
    assert math.isclose(l, r, rel_tol=1e-9)
    assert l > 0

def test_pure_yaw_is_opposite_wheels():
    l, r = differential_drive(0.0, 2.0)
    assert math.isclose(l, -r, rel_tol=1e-9)
    assert r > 0  # +yaw (CCW) spins right wheel forward, left backward

def test_zero_command_is_zero():
    assert differential_drive(0.0, 0.0) == (0.0, 0.0)

def test_clamped_to_max():
    l, r = differential_drive(100.0, 0.0)
    assert abs(l) <= config.MAX_WHEEL_RAD_S + 1e-9
    assert math.isclose(l, config.MAX_WHEEL_RAD_S)

def test_calibration_scales_each_wheel():
    l, r = differential_drive(0.3, 0.0, calib_left=0.5, calib_right=1.0)
    assert l < r

def test_kinematics_value():
    # wheel angular speed = linear wheel speed / radius
    v = 0.25
    l, r = differential_drive(v, 0.0)
    assert math.isclose(l, v / config.WHEEL_RADIUS, rel_tol=1e-9)
