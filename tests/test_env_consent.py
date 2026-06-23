# tests/test_env_consent.py
import numpy as np
from roybot.env import RoybotChaseEnv

def _run(env, policy, steps=400):
    obs, _ = env.reset(seed=1)
    total = 0.0
    last_info = {}
    for _ in range(steps):
        obs, rew, term, trunc, info = env.step(policy(obs))
        total += rew
        last_info = info
        if term or trunc:
            break
    return total, last_info

def test_step_populates_consent_info():
    env = RoybotChaseEnv(domain_randomize=False, seed=1)
    _, info = _run(env, lambda o: np.zeros(2), steps=5)
    assert "reward_terms" in info and "willing" in info and "dist" in info

def test_pestering_scores_worse_than_giving_space_when_disinterested():
    # Pin a disinterested cat directly in front (+x) and frozen, so "drive forward"
    # is unambiguously approaching and "drive backward" is retreating.
    def make():
        e = RoybotChaseEnv(domain_randomize=False, seed=2)
        e.reset(seed=2)
        e.cat.pos = np.array([0.4, 0.0])     # in front of the robot (starts at origin, yaw 0)
        e.cat.vel = np.zeros(2)
        e.cat.step = lambda dt, robot_xy: None  # freeze the cat
        e.cat.engagement = 0.0                  # disinterested
        e._sync_cat()
        e._prev_dist = e._distance()
        return e

    charge_env, retreat_env = make(), make()
    charge_total = retreat_total = 0.0
    for _ in range(30):  # 30 steps @0.6 m/s < 0.4 m: charger stays approaching, never overshoots
        _, rc, *_ = charge_env.step(np.array([1.0, 0.0]))   # toward the cat
        _, rr, *_ = retreat_env.step(np.array([-1.0, 0.0]))  # away from the cat
        charge_total += rc
        retreat_total += rr
    assert retreat_total > charge_total
