# scripts/train.py
"""Train the consent-based chase policy with PPO (stable-baselines3)."""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from roybot.env import RoybotChaseEnv


def build_env(seed=0, domain_randomize=True):
    return RoybotChaseEnv(domain_randomize=domain_randomize, seed=seed)


def train(total_timesteps=1_000_000, save_path="runs/chase_policy", n_envs=8, seed=0):
    # make_vec_env resets each worker with seed+rank, so they get distinct RNG
    # streams (cat + domain randomization). Don't pin a constructor seed here, or
    # all workers would start as clones.
    venv = make_vec_env(
        lambda: RoybotChaseEnv(domain_randomize=True),
        n_envs=n_envs, seed=seed,
    )
    model = PPO(
        "MlpPolicy", venv, seed=seed, verbose=1,
        n_steps=2048, batch_size=512, gae_lambda=0.95, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),  # tanh MLP; mirrored in infer.py
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    venv.close()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--out", default="runs/chase_policy")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a.timesteps, a.out, a.n_envs, a.seed)
