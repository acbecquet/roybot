# tests/test_infer.py
import sys
import numpy as np
sys.path.insert(0, "scripts")
from stable_baselines3 import PPO
from roybot.env import RoybotChaseEnv
from roybot.infer import NumpyPolicy
import export as export_mod

def test_numpy_policy_matches_sb3(tmp_path):
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    model = PPO("MlpPolicy", env, seed=0, policy_kwargs=dict(net_arch=[64, 64]))
    zip_path = str(tmp_path / "m")
    model.save(zip_path)
    npz_path = str(tmp_path / "m.npz")
    export_mod.export(zip_path, npz_path)

    pol = NumpyPolicy.from_npz(npz_path)
    rng = np.random.default_rng(0)
    n = env.observation_space.shape[0]
    for _ in range(20):
        obs = rng.standard_normal(n).astype(np.float32)
        sb3_action, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(pol.act(obs), sb3_action, rtol=1e-4, atol=1e-4)

def test_infer_imports_only_numpy():
    import roybot.infer as m
    src = open(m.__file__).read()
    assert "torch" not in src and "stable_baselines3" not in src

def test_inference_is_fast_enough_for_50hz(tmp_path):
    # operationalizes spec success-criterion #5: tiny MLP must run well within the
    # 50 Hz (20 ms) control budget. Desktop proxy; the Pi target is checked in Phase 2.
    import time
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    model = PPO("MlpPolicy", env, seed=0, policy_kwargs=dict(net_arch=[64, 64]))
    model.save(str(tmp_path / "m"))
    npz_path = str(tmp_path / "m.npz")
    export_mod.export(str(tmp_path / "m"), npz_path)
    pol = NumpyPolicy.from_npz(npz_path)
    n = env.observation_space.shape[0]
    obs = np.zeros(n, dtype=np.float32)
    pol.act(obs)  # warm up
    t0 = time.perf_counter()
    for _ in range(1000):
        pol.act(obs)
    mean_ms = (time.perf_counter() - t0) / 1000.0 * 1000.0
    assert mean_ms < 1.0  # generous; real inference is microseconds
