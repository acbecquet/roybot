# tests/test_model.py
import os
import numpy as np
import mujoco
from roybot import config

MODEL = os.path.join("models", "roybot.xml")

def load():
    m = mujoco.MjModel.from_xml_path(MODEL)
    return m, mujoco.MjData(m)

def test_model_loads_with_two_motors_and_sensors():
    m, _ = load()
    assert m.nu == 2
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)}
    assert names == {"left_motor", "right_motor"}
    for s in ("chassis_pos", "chassis_quat", "chassis_linvel", "chassis_angvel"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0
    assert abs(m.opt.timestep - config.SIM_TIMESTEP) < 1e-12

def test_zero_control_stays_upright_and_finite():
    m, d = load()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    quat = d.sensor("chassis_quat").data  # w,x,y,z
    up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    assert up_z > 0.9  # sitting flat

def test_equal_forward_control_drives_plus_x():
    m, d = load()
    d.ctrl[:] = [20.0, 20.0]
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert d.sensor("chassis_pos").data[0] > 0.05  # moved forward in +x

def test_opposite_control_yaws_in_place():
    m, d = load()
    d.ctrl[:] = [-15.0, 15.0]
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert abs(d.sensor("chassis_angvel").data[2]) > 0.5  # yaw rate present
