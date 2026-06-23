# scripts/demo.py
"""Evaluate / visualize the trained chase policy against the moody cat."""
import argparse
import numpy as np

from roybot.env import RoybotChaseEnv
from roybot.infer import NumpyPolicy
from roybot.metrics import episode_metrics


def _rollout(env, policy, record, view_handle=None):
    obs, _ = env.reset()
    prev_dist = env._distance()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, term, trunc, info = env.step(action)
        record.append({"dist": info["dist"], "willing": info["willing"],
                       "prev_dist": prev_dist, "upright": env._robot_state()["upright"]})
        prev_dist = info["dist"]
        if view_handle is not None:
            view_handle.sync()
        done = term or trunc


def evaluate(policy, episodes=10, seed=0):
    records = []
    for i in range(episodes):
        env = RoybotChaseEnv(domain_randomize=False, seed=seed + i)
        _rollout(env, policy, records)
    return episode_metrics(records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("policy_npz")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--view", action="store_true")
    a = p.parse_args()
    policy = NumpyPolicy.from_npz(a.policy_npz)
    if a.view:
        import mujoco.viewer
        env = RoybotChaseEnv(domain_randomize=False, seed=0)
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            _rollout(env, policy, [], view_handle=v)
    else:
        print(evaluate(policy, a.episodes))
