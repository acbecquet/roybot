# tests/test_env_core.py
import numpy as np
from roybot.env import RoybotChaseEnv
from roybot import config

def test_spaces():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    assert env.observation_space.shape == (12 * config.N_STACK,)
    assert env.action_space.shape == (2,)

def test_reset_returns_obs_and_info():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (12 * config.N_STACK,)
    assert np.all(np.isfinite(obs))

def test_step_returns_5_tuple_and_drives_forward():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    x0 = env._robot_state()["pos"][0]
    for _ in range(40):
        obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0]))  # full forward
    assert obs.shape == (12 * config.N_STACK,) and isinstance(rew, float)
    assert env._robot_state()["pos"][0] > x0

def test_time_limit_truncates():
    # zero action => robot sits still, never tips, so it truncates at the time limit
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    trunc = False
    for steps in range(1, 1002):
        _, _, term, trunc, _ = env.step(np.zeros(2))
        if trunc:
            break
    assert trunc and steps == 1000  # EPISODE_SECONDS(20) * CONTROL_HZ(50)

def test_difficulty_sets_cat_speed_scale_and_varies():
    from roybot import config
    scales = set()
    for s in range(8):
        env = RoybotChaseEnv(domain_randomize=True, seed=s)
        env.reset(seed=s)
        assert 1.0 <= env.cat.speed_scale <= config.CAT_SPEED_SCALE_AT_MAX + 1e-9
        scales.add(round(env.cat.speed_scale, 3))
    assert len(scales) >= 3   # difficulty actually varies across episodes

def test_no_dr_keeps_baseline_difficulty():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    assert env.cat.speed_scale == 1.0

def test_stationary_robot_earns_no_anticipation():
    # C1 regression: a still robot must not farm anticipation from the cat's motion.
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    for _ in range(15):
        env.cat.engagement = 0.9              # willing -> anticipate term active
        _, _, _, _, info = env.step(np.zeros(2))  # robot holds still
        assert abs(info["anticipate_rate"]) < 0.05
