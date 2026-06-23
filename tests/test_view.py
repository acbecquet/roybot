# tests/test_view.py
# Covers the GL-free helpers in scripts/view.py. The window/MjrContext path
# needs a live display and isn't unit-tested here.
import sys
sys.path.insert(0, "scripts")

import numpy as np
import mujoco
import view  # scripts/view.py
from roybot.env import RoybotChaseEnv


def _scene(env, n=100):
    return mujoco.MjvScene(env.model, n)  # CPU-side scene; no GL context needed


def test_add_cat_marker_appends_one_sphere_at_cat():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    scene = _scene(env)
    n0 = scene.ngeom
    view.add_cat_marker(scene, env)
    assert scene.ngeom == n0 + 1
    g = scene.geoms[scene.ngeom - 1]
    assert g.type == mujoco.mjtGeom.mjGEOM_SPHERE.value
    assert np.allclose(g.pos[:2], [env.cat.pos[0], env.cat.pos[1]], atol=1e-5)


def test_add_cat_marker_color_tracks_willingness():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)

    env.cat.engagement = 0.9
    assert env.cat.willing                       # guard: actually willing
    s1 = _scene(env); view.add_cat_marker(s1, env)
    willing_rgba = np.array(s1.geoms[s1.ngeom - 1].rgba)

    env.cat.engagement = 0.0
    assert not env.cat.willing                    # guard: actually disinterested
    s2 = _scene(env); view.add_cat_marker(s2, env)
    grey_rgba = np.array(s2.geoms[s2.ngeom - 1].rgba)

    assert not np.allclose(willing_rgba, grey_rgba)


def test_add_cat_marker_respects_maxgeom():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    scene = _scene(env)
    scene.ngeom = scene.maxgeom                    # scene full
    view.add_cat_marker(scene, env)               # must be a no-op, not an overflow
    assert scene.ngeom == scene.maxgeom


def test_overlay_text_reflects_state():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    env.cat.engagement = 0.9
    txt = view.overlay_text(env, ep_reward=12.5, paused=True)
    assert "ROYBOT" in txt and "[PAUSED]" in txt
    assert "episode reward: 12.5" in txt
    assert env.cat.mode in txt
    assert "[PLAYING]" in view.overlay_text(env, 0.0, paused=False)
