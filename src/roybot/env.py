# src/roybot/env.py
"""Gymnasium env wrapping the MuJoCo twin + closed-form driver (+ moody cat in Task 7)."""
import math
from collections import deque

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from . import config
from .driver import differential_drive
from .cat import Cat
from .reward import compute_reward


def _quat_to_rpy(q):
    """MuJoCo quat (w,x,y,z) -> (roll, pitch, yaw)."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _world_to_robot(vec_xy, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * vec_xy[0] - s * vec_xy[1], s * vec_xy[0] + c * vec_xy[1]])


class RoybotChaseEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, model_path="models/roybot.xml", domain_randomize=True, seed=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.domain_randomize = domain_randomize
        self._base_mass = self.model.body_mass.copy()
        self._base_friction = self.model.geom_friction.copy()  # baseline so DR doesn't drift
        self.rng = np.random.default_rng(seed)  # all env/cat randomness; gym np_random intentionally unused
        self.cat = Cat(self.rng)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self._stack = deque(maxlen=config.N_STACK)
        high = np.full(12 * config.N_STACK, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._max_steps = int(round(config.EPISODE_SECONDS * config.CONTROL_HZ))
        self.cat_xy = np.array([config.BAND_CENTER, 0.0])  # default; overwritten in reset() via _sync_cat()
        self._prev_action = np.zeros(2)
        self._latency = 0
        self._action_buf = []
        self._motor_gain = 1.0
        self._steps = 0
        self.difficulty = 0.0

    # --- helpers ---
    def _robot_state(self):
        pos = np.array(self.data.sensor("chassis_pos").data[:2])
        q = self.data.sensor("chassis_quat").data
        roll, pitch, yaw = _quat_to_rpy(q)
        linvel_w = np.array(self.data.sensor("chassis_linvel").data[:2])
        vbody = _world_to_robot(linvel_w, yaw)
        yaw_rate = float(self.data.sensor("chassis_angvel").data[2])
        up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        return {
            "pos": pos, "yaw": yaw, "vfwd": float(vbody[0]), "vlat": float(vbody[1]),
            "yaw_rate": yaw_rate, "roll": roll, "pitch": pitch,
            "upright": up_z > config.TIP_UPRIGHT_MIN,
        }

    def _base_obs(self):
        st = self._robot_state()
        rel = _world_to_robot(self.cat_xy - st["pos"], st["yaw"])
        cat_vel_rel = _world_to_robot(getattr(self, "cat_vel", np.zeros(2)), st["yaw"])
        engagement = getattr(self, "cat_engagement", 0.5)
        return np.array([
            rel[0], rel[1], cat_vel_rel[0], cat_vel_rel[1], engagement,
            st["vfwd"], st["vlat"], st["yaw_rate"], st["roll"], st["pitch"],
            self._prev_action[0], self._prev_action[1],
        ], dtype=np.float32)

    def _get_obs(self):
        base = self._base_obs()
        if not self._stack:
            for _ in range(config.N_STACK):
                self._stack.append(base)
        else:
            self._stack.append(base)
        return np.concatenate(list(self._stack)).astype(np.float32)

    def _apply_domain_randomization(self):
        if not self.domain_randomize:
            self._motor_gain, self._latency = 1.0, 0
            self.difficulty = 0.0
            return
        self.model.body_mass[:] = self._base_mass * self.rng.uniform(*config.DR_MASS)
        self.model.geom_friction[:, 0] = np.clip(
            self._base_friction[:, 0] * self.rng.uniform(*config.DR_FRICTION), 0.01, None)
        self._motor_gain = float(self.rng.uniform(*config.DR_MOTOR_GAIN))
        self._latency = int(self.rng.integers(*config.DR_LATENCY_STEPS))
        self.difficulty = float(self.rng.uniform(*config.DIFFICULTY_RANGE))

    # --- gym API ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._apply_domain_randomization()
        mujoco.mj_forward(self.model, self.data)
        self._prev_action = np.zeros(2)
        self._action_buf = [np.zeros(2)] * self._latency  # latency=0 => apply immediately
        self._steps = 0
        self.cat.rng = self.rng
        self.cat.reset()
        self.cat.speed_scale = (1.0 + self.difficulty * (config.CAT_SPEED_SCALE_AT_MAX - 1.0)) if self.domain_randomize else 1.0
        self._sync_cat()
        self._prev_dist = self._distance()
        self._stack.clear()
        return self._get_obs(), {}

    def _sync_cat(self):
        self.cat_xy = self.cat.pos.copy()
        self.cat_vel = self.cat.vel.copy()
        self.cat_engagement = self.cat.engagement

    def _distance(self):
        return float(np.linalg.norm(self.cat_xy - self._robot_state()["pos"]))

    def _drive(self, action):
        self._action_buf.append(np.asarray(action, dtype=float))
        delayed = self._action_buf.pop(0)  # apply latency-delayed action
        v_fwd = float(delayed[0]) * config.ACTION_VFWD_MAX
        v_yaw = float(delayed[1]) * config.ACTION_VYAW_MAX
        l, r = differential_drive(v_fwd, v_yaw)
        self.data.ctrl[0] = l * self._motor_gain
        self.data.ctrl[1] = r * self._motor_gain

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self._drive(action)
        for _ in range(config.N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        st = self._robot_state()
        cat_prev = self.cat.pos.copy()
        self.cat.step(config.CONTROL_DT, st["pos"])
        self._sync_cat()
        dist = float(np.linalg.norm(self.cat_xy - st["pos"]))          # current dist (for band)
        # robot-attributed approach: change in distance to where the cat WAS, due to robot motion only
        approach_rate = self._prev_dist - float(np.linalg.norm(cat_prev - st["pos"]))
        predicted = cat_prev + self.cat_vel * config.ANTICIPATE_HORIZON
        anticipate_rate = self._prev_dist - float(np.linalg.norm(predicted - st["pos"]))
        reward, terms = compute_reward(
            dist=dist, approach_rate=approach_rate, anticipate_rate=anticipate_rate,
            willing=self.cat.willing,
            action=action, prev_action=self._prev_action, upright=st["upright"],
        )
        self._prev_dist = dist
        terminated = not st["upright"]
        truncated = self._steps >= self._max_steps
        self._prev_action = action
        info = {"reward_terms": terms, "willing": self.cat.willing,
                "dist": dist, "approach_rate": approach_rate, "anticipate_rate": anticipate_rate}
        return self._get_obs(), float(reward), terminated, truncated, info
