# scripts/view.py
"""Large-font interactive viewer for the trained chase policy.

Why this exists: mujoco.viewer.launch_passive's UI font size is baked into its
C++ render context and is NOT settable from Python (MuJoCo 3.x). This builds its
own MjrContext at a large mjtFontScale, so the on-screen text is readable, and
draws the (otherwise invisible) cat as a marker so you can see what Roybot chases.

Run:  .venv/Scripts/python scripts/view.py runs/smart_policy.npz
      (--fontscale 100/150/200/250/300, default 250; --width/--height)
Controls: drag L = rotate, drag R = pan, scroll = zoom, SPACE = pause,
          R = new episode, ESC = quit. The window loops episodes until closed.
"""
import argparse

import numpy as np
import mujoco

from roybot.env import RoybotChaseEnv
from roybot.infer import NumpyPolicy


# ---- headless-testable helpers (no GL needed) -------------------------------

def add_cat_marker(scene, env):
    """Append a sphere geom at the cat's position (colored by willingness)."""
    if scene.ngeom >= scene.maxgeom:
        return
    cat = env.cat
    rgba = (np.array([0.95, 0.35, 0.2, 1.0], dtype=np.float32) if cat.willing
            else np.array([0.55, 0.55, 0.55, 1.0], dtype=np.float32))
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE.value,
        np.array([0.03, 0.0, 0.0]),                       # size (radius)
        np.array([float(cat.pos[0]), float(cat.pos[1]), 0.03]),  # pos
        np.eye(3).flatten(),                              # orientation (identity)
        rgba,
    )
    scene.ngeom += 1


def overlay_text(env, ep_reward, paused):
    """Return the big-font overlay string for the current state."""
    cat = env.cat
    status = "[PAUSED]" if paused else "[PLAYING]"
    return "\n".join([
        f"ROYBOT  {status}",
        f"cat mode:   {cat.mode}",
        f"willing:    {cat.willing}  (engagement {cat.engagement:.2f})",
        f"difficulty: {getattr(env, 'difficulty', 0.0):.2f}   cat speed x{cat.speed_scale:.2f}",
        f"episode reward: {ep_reward:.1f}",
        "",
        "drag L: rotate   drag R: pan   scroll: zoom",
        "SPACE: pause   R: new episode   ESC: quit",
    ])


# ---- interactive GL viewer --------------------------------------------------

def run(policy_npz, fontscale=250, width=1200, height=900):
    import glfw  # PyGLFW (same binding mujoco.viewer uses)

    policy = NumpyPolicy.from_npz(policy_npz)
    env = RoybotChaseEnv(domain_randomize=True, seed=0)
    obs, _ = env.reset(seed=0)
    model, data = env.model, env.data

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")
    window = glfw.create_window(width, height, "Roybot - chase viewer", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    cam.azimuth, cam.elevation, cam.distance = 90.0, -35.0, 2.5
    cam.lookat[:] = [0.0, 0.0, 0.0]

    scene = mujoco.MjvScene(model, 10000)
    font = getattr(mujoco.mjtFontScale, f"mjFONTSCALE_{fontscale}")
    context = mujoco.MjrContext(model, font.value)

    mouse = {"x": 0.0, "y": 0.0, "L": False, "R": False, "M": False}
    state = {"paused": False}

    def on_key(win, key, scancode, act, mods):
        if act != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_R:
            env.reset()

    def on_mouse_button(win, button, act, mods):
        mouse["L"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        mouse["R"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        mouse["M"] = glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)

    def on_cursor(win, xpos, ypos):
        dx, dy = xpos - mouse["x"], ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if not (mouse["L"] or mouse["R"] or mouse["M"]):
            return
        h = max(1, glfw.get_window_size(win)[1])
        if mouse["L"]:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif mouse["R"]:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_H
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(model, act.value, dx / h, dy / h, scene, cam)

    def on_scroll(win, xoff, yoff):
        mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM.value,
                              0.0, -0.05 * yoff, scene, cam)

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)

    ep_reward = 0.0
    while not glfw.window_should_close(window):
        if not state["paused"]:
            obs, rew, term, trunc, _ = env.step(policy.act(obs))
            ep_reward += rew
            if term or trunc:
                obs, _ = env.reset()
                ep_reward = 0.0

        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        add_cat_marker(scene, env)
        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG.value,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                           viewport, overlay_text(env, ep_reward, state["paused"]),
                           "", context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_npz")
    ap.add_argument("--fontscale", type=int, default=250,
                    choices=[100, 150, 200, 250, 300])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()
    run(a.policy_npz, a.fontscale, a.width, a.height)
