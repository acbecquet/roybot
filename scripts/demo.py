# scripts/demo.py
"""Evaluate / visualize the trained chase policy against the moody cat."""
import argparse

from roybot.env import RoybotChaseEnv
from roybot.infer import NumpyPolicy
from roybot.metrics import episode_metrics


def _rollout(env, policy, record, view_handle=None):
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, term, trunc, info = env.step(action)
        record.append({"dist": info["dist"], "willing": info["willing"],
                       "approach_rate": info["approach_rate"],
                       "upright": env._robot_state()["upright"]})
        if view_handle is not None:
            view_handle.sync()
            if not view_handle.is_running():   # window closed -> stop promptly
                return
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
            while v.is_running():   # keep the window open: replay episodes (fresh cat each round) until you close it
                _rollout(env, policy, [], view_handle=v)
    else:
        print(evaluate(policy, a.episodes))
