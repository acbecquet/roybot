"""Torch-free deterministic policy inference. numpy only — runs on the Pi.

Deployment note: the caller must frame-stack the last config.N_STACK observations
(concatenated into a single flat array) before passing them to act(), mirroring the env.
"""
import numpy as np


class NumpyPolicy:
    def __init__(self, layers):
        self.layers = layers  # list of (W (out,in), b (out,))

    @classmethod
    def from_npz(cls, path):
        d = np.load(path)
        n = int(d["n_layers"])
        layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(n)]
        return cls(layers)

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:   # tanh on hidden layers only
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0).astype(np.float32)
