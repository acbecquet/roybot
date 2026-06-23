# tests/test_train_smoke.py
import os, sys
sys.path.insert(0, "scripts")
import train as train_mod

def test_short_training_produces_a_model(tmp_path):
    out = tmp_path / "smoke"
    model = train_mod.train(total_timesteps=512, save_path=str(out), n_envs=1, seed=0)
    assert os.path.exists(str(out) + ".zip")
    # a couple of deterministic predictions of the right shape
    env = train_mod.build_env(seed=0)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (2,)
