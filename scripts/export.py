"""Extract the deterministic actor MLP from an SB3 PPO model into a .npz."""
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution


def export(model_path, npz_path):
    model = PPO.load(model_path, device="cpu")
    # infer.py replicates an unsquashed tanh-MLP -> linear -> clip. Fail loudly if a
    # future config (SDE / squashed dist) would silently make that path wrong on the Pi.
    assert not getattr(model.policy, "use_sde", False), "infer.py does not support SDE policies"
    assert isinstance(model.policy.action_dist, DiagGaussianDistribution), \
        "infer.py only matches a non-squashed DiagGaussian PPO policy"
    layers = []
    # policy_net: alternating Linear/Tanh -> grab the Linear layers in order
    for mod in model.policy.mlp_extractor.policy_net:
        if mod.__class__.__name__ == "Linear":
            layers.append(mod)
    layers.append(model.policy.action_net)  # final mean head (Linear)

    arrays = {}
    for i, lin in enumerate(layers):
        arrays[f"w{i}"] = lin.weight.detach().cpu().numpy()  # (out, in)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy()    # (out,)
    arrays["n_layers"] = np.array(len(layers))
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(npz_path, **arrays)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("npz")
    a = p.parse_args()
    export(a.model, a.npz)
