# Roybot Phase 1 — Sim Consent-Based Chase Policy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, in simulation only, a tiny neural policy that drives a differential-drive robot to play with a cat *on the cat's terms* — engaging when she's willing, giving her space when she's not — and ship it as a torch-free NumPy inference module ready for the Pi.

**Architecture:** A MuJoCo model of a 2-wheel + caster robot is wrapped in a Gymnasium env. A closed-form differential driver turns `(v_fwd, v_yaw)` commands into wheel speeds, so the policy only outputs a 2-D velocity command (it never learns trivial motor math). A scripted "moody cat" with an engagement state moves around; the consent-based reward pays the policy to engage a willing cat and to back off from a disinterested one. PPO (stable-baselines3) trains it; the trained MLP is exported to a `.npz` and re-implemented in pure NumPy so the same policy runs on the Pi later with no PyTorch.

**Tech Stack:** Python 3.11+, `mujoco` (official bindings, includes MJCF + viewer), `gymnasium`, `stable-baselines3` (PPO, PyTorch), `numpy`, `pytest`. GPU optional (PPO here is small; MJX/Brax is an optional later accelerator, not in this plan).

## Global Constraints

- Python **3.11+**. All randomness goes through a seeded `numpy.random.Generator` — **no global `random`/`np.random`** (reproducible episodes, per karpathy verifiable-criteria).
- **No PyTorch on the Pi path:** `src/roybot/infer.py` must import only `numpy`. PyTorch/SB3 may appear only under `scripts/` and `train`-side code.
- Shared physical/behavioral constants live in **`src/roybot/config.py`** — never hard-code a magic number that already has a name there (DRY).
- Calibration knobs (`calib_left`, `calib_right`, motor gain, latency) stay in even though sim doesn't strictly need them — Phase 2 hardware will (ponytail hardware rule).
- TDD: every behavior change is a failing test first. Commit after each task. Conventional-commit messages (`feat:`, `test:`, `chore:`).
- Run tests with `pytest -q`. Author the code to run from the repo root with `src/` on the path (set in Task 1).

---

## File Structure

```
roybot/
  requirements.txt              # Task 1
  pyproject.toml                # Task 1 (package config + pytest pythonpath)
  .gitignore                    # Task 1
  src/roybot/__init__.py        # Task 1
  src/roybot/config.py          # Task 1 — all shared constants
  src/roybot/driver.py          # Task 2 — closed-form differential drive
  src/roybot/cat.py             # Task 3 — moody simulated cat
  src/roybot/reward.py          # Task 4 — consent-based per-step reward
  models/roybot.xml             # Task 5 — MuJoCo MJCF wheeled twin
  src/roybot/env.py             # Tasks 6 & 7 — Gymnasium env
  scripts/train.py              # Task 8 — PPO training entrypoint
  src/roybot/infer.py           # Task 9 — torch-free NumPy MLP inference
  scripts/export.py             # Task 9 — SB3 policy -> .npz
  src/roybot/metrics.py         # Task 10 — episode metrics (success criteria)
  scripts/demo.py               # Task 10 — viewer demo + consent/play eval
  tests/test_*.py               # one per module
```

---

### Task 1: Project scaffolding & toolchain

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `.gitignore`, `src/roybot/__init__.py`, `src/roybot/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: the importable package `roybot` and `roybot.config` exposing the constants below.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from roybot import config

def test_constants_present_and_sane():
    assert config.CONTROL_HZ == 50
    assert config.SIM_TIMESTEP == 0.002
    # 50 Hz control over a 2 ms sim step = 10 substeps, exactly
    assert config.N_SUBSTEPS == 10
    assert config.BAND_MIN < config.BAND_MAX
    assert 0.0 < config.WILLING_THRESHOLD < 1.0
    assert config.WHEEL_RADIUS > 0 and config.WHEEL_BASE > 0
    assert set(config.REWARD_WEIGHTS) >= {
        "engage", "juke", "give_space", "pester", "energy", "action_rate", "tip"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot'`

- [ ] **Step 3: Create the project files**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "roybot"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
addopts = "-q"
```

```text
# requirements.txt
mujoco>=3.1
gymnasium>=0.29
stable-baselines3>=2.3
numpy>=1.26
pytest>=8.0
```

```text
# .gitignore
__pycache__/
*.pyc
.venv/
venv/
runs/
*.zip
*.npz
*.mp4
.pytest_cache/
.superpowers/
```

```python
# src/roybot/__init__.py
"""Roybot — sim-side training package for the consent-based chase policy."""
__version__ = "0.1.0"
```

```python
# src/roybot/config.py
"""All shared physical and behavioral constants. DRY: import from here."""

# --- control / sim timing ---
CONTROL_HZ = 50
SIM_TIMESTEP = 0.002            # must match models/roybot.xml <option timestep>
N_SUBSTEPS = round((1.0 / CONTROL_HZ) / SIM_TIMESTEP)  # = 10
EPISODE_SECONDS = 20.0
CONTROL_DT = 1.0 / CONTROL_HZ

# --- drivetrain geometry ---
WHEEL_RADIUS = 0.025            # m
WHEEL_BASE = 0.09               # m (left-right wheel separation)
MAX_WHEEL_RAD_S = 30.0          # rad/s motor speed cap

# --- action scaling (policy outputs in [-1, 1]) ---
ACTION_VFWD_MAX = 0.6           # m/s
ACTION_VYAW_MAX = 6.0           # rad/s

# --- moody cat ---
BAND_MIN = 0.15                 # m — closer than this = crowding/pestering
BAND_MAX = 0.50                 # m — the "interesting play band" outer edge
BAND_CENTER = (BAND_MIN + BAND_MAX) / 2
WILLING_THRESHOLD = 0.40        # engagement >= this => cat wants to play
ENGAGE_GAIN = 0.8               # /s engagement gained during good in-band play
PESTER_RATE = 1.2               # /s engagement lost while being crowded
ENGAGE_DECAY = 0.05             # /s natural waning
CAT_FLEE_SPEED = 0.7            # m/s
CAT_WANDER_SPEED = 0.1          # m/s
CAT_DART_PROB = 0.04            # per control step, when willing

# --- termination ---
TIP_UPRIGHT_MIN = 0.5           # body +z world-component below this => tipped over

# --- reward weights ---
REWARD_WEIGHTS = {
    "engage": 1.0,       # willing + in band
    "juke": 0.3,         # willing + dynamic distance (anti-static)
    "give_space": 1.0,   # disinterested + retreating
    "pester": 1.5,       # disinterested + approaching (penalty)
    "energy": 0.02,      # ||action||^2 penalty
    "action_rate": 0.05, # ||action - prev_action||^2 penalty
    "tip": 10.0,         # one-time penalty on tip-over
}
```

- [ ] **Step 4: Install and verify the toolchain + test passes**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   bash: source .venv/Scripts/activate
pip install -r requirements.txt
python -c "import mujoco, gymnasium, stable_baselines3, numpy; print('toolchain ok')"
pytest tests/test_config.py -q
```
Expected: `toolchain ok` then `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore src/roybot/__init__.py src/roybot/config.py tests/test_config.py
git commit -m "chore: scaffold roybot sim package + shared config"
```

---

### Task 2: Closed-form differential driver

**Files:**
- Create: `src/roybot/driver.py`
- Test: `tests/test_driver.py`

**Interfaces:**
- Consumes: `roybot.config` (WHEEL_RADIUS, WHEEL_BASE, MAX_WHEEL_RAD_S).
- Produces: `differential_drive(v_fwd, v_yaw, *, calib_left=1.0, calib_right=1.0) -> tuple[float, float]` returning `(left_rad_s, right_rad_s)`, each clamped to `±MAX_WHEEL_RAD_S`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_driver.py
import math
from roybot.driver import differential_drive
from roybot import config

def test_straight_is_equal_wheels():
    l, r = differential_drive(0.3, 0.0)
    assert math.isclose(l, r, rel_tol=1e-9)
    assert l > 0

def test_pure_yaw_is_opposite_wheels():
    l, r = differential_drive(0.0, 2.0)
    assert math.isclose(l, -r, rel_tol=1e-9)
    assert r > 0  # +yaw (CCW) spins right wheel forward, left backward

def test_zero_command_is_zero():
    assert differential_drive(0.0, 0.0) == (0.0, 0.0)

def test_clamped_to_max():
    l, r = differential_drive(100.0, 0.0)
    assert abs(l) <= config.MAX_WHEEL_RAD_S + 1e-9
    assert math.isclose(l, config.MAX_WHEEL_RAD_S)

def test_calibration_scales_each_wheel():
    l, r = differential_drive(0.3, 0.0, calib_left=0.5, calib_right=1.0)
    assert l < r

def test_kinematics_value():
    # wheel angular speed = linear wheel speed / radius
    v = 0.25
    l, r = differential_drive(v, 0.0)
    assert math.isclose(l, v / config.WHEEL_RADIUS, rel_tol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_driver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.driver'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/roybot/driver.py
"""Closed-form differential drive: body velocity command -> wheel angular speeds.

No learning here — this is solved kinematics. The policy outputs (v_fwd, v_yaw);
this turns it into the two wheel speeds MuJoCo's velocity actuators consume.
"""
from .config import WHEEL_RADIUS, WHEEL_BASE, MAX_WHEEL_RAD_S


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def differential_drive(
    v_fwd: float,
    v_yaw: float,
    *,
    calib_left: float = 1.0,
    calib_right: float = 1.0,
) -> tuple[float, float]:
    """(forward m/s, yaw rad/s) -> (left_rad_s, right_rad_s), clamped to motor cap."""
    half = WHEEL_BASE / 2.0
    v_left = (v_fwd - v_yaw * half) / WHEEL_RADIUS
    v_right = (v_fwd + v_yaw * half) / WHEEL_RADIUS
    v_left *= calib_left
    v_right *= calib_right
    return (
        _clamp(v_left, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S),
        _clamp(v_right, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_driver.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/driver.py tests/test_driver.py
git commit -m "feat: closed-form differential drive kinematics"
```

---

### Task 3: Moody simulated cat

**Files:**
- Create: `src/roybot/cat.py`
- Test: `tests/test_cat.py`

**Interfaces:**
- Consumes: `roybot.config` (BAND_MIN/MAX, WILLING_THRESHOLD, ENGAGE_GAIN, PESTER_RATE, ENGAGE_DECAY, CAT_FLEE_SPEED, CAT_WANDER_SPEED, CAT_DART_PROB), `numpy`.
- Produces: `class Cat` with `__init__(self, rng: numpy.random.Generator)`, `reset(self) -> None`, `step(self, dt: float, robot_xy: numpy.ndarray) -> None`, attributes `pos` (shape (2,)), `vel` (shape (2,)), `engagement` (float), and property `willing -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cat.py
import numpy as np
from roybot.cat import Cat
from roybot import config

def make_cat(seed=0):
    return Cat(np.random.default_rng(seed))

def test_reset_places_cat_and_seeds_engagement():
    c = make_cat(); c.reset()
    assert c.pos.shape == (2,) and c.vel.shape == (2,)
    assert 0.0 <= c.engagement <= 1.0

def test_in_band_play_raises_engagement():
    c = make_cat(); c.reset()
    c.engagement = 0.5
    c.pos = np.array([0.0, 0.0])
    robot = np.array([config.BAND_CENTER, 0.0])  # right in the play band
    before = c.engagement
    for _ in range(20):
        c.step(0.02, robot)
    assert c.engagement > before

def test_crowding_a_cat_lowers_engagement_and_makes_it_flee():
    c = make_cat(); c.reset()
    c.engagement = 0.5
    c.pos = np.array([0.0, 0.0])
    robot = np.array([0.02, 0.0])  # way inside BAND_MIN = pestering
    before_eng = c.engagement
    c.step(0.02, robot)
    assert c.engagement < before_eng
    # fleeing = moving away from the robot (negative x here)
    assert c.vel[0] <= 0.0

def test_engagement_naturally_decays_when_left_alone():
    c = make_cat(); c.reset()
    c.engagement = 0.5
    c.pos = np.array([0.0, 0.0])
    robot = np.array([5.0, 5.0])  # far away, no interaction
    for _ in range(50):
        c.step(0.02, robot)
    assert c.engagement < 0.5

def test_willing_property_tracks_threshold():
    c = make_cat(); c.reset()
    c.engagement = config.WILLING_THRESHOLD + 0.1
    assert c.willing is True
    c.engagement = config.WILLING_THRESHOLD - 0.1
    assert c.willing is False

def test_deterministic_under_fixed_seed():
    a, b = make_cat(7), make_cat(7)
    a.reset(); b.reset()
    robot = np.array([0.3, 0.0])
    for _ in range(30):
        a.step(0.02, robot); b.step(0.02, robot)
    assert np.allclose(a.pos, b.pos) and a.engagement == b.engagement
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cat.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.cat'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/roybot/cat.py
"""A scripted, moody cat: position dynamics + an engagement state.

Engagement is what makes "consent-based play" learnable: it rises with good
in-band play, falls when the robot crowds/pesters or over time, and gates
whether the cat is `willing`. A disinterested cat flees crowding and wanders off.
"""
import numpy as np

from . import config


class Cat:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.engagement = 0.5
        self._playfulness = 1.0
        self.reset()

    def reset(self) -> None:
        # personality: some sessions friskier than others
        self._playfulness = float(self.rng.uniform(0.6, 1.4))
        self.pos = self.rng.uniform(-0.6, 0.6, size=2)
        self.vel = np.zeros(2)
        self.engagement = float(self.rng.uniform(0.3, 0.7))

    @property
    def willing(self) -> bool:
        return self.engagement >= config.WILLING_THRESHOLD

    def step(self, dt: float, robot_xy: np.ndarray) -> None:
        to_robot = robot_xy - self.pos
        dist = float(np.linalg.norm(to_robot))
        unit = to_robot / dist if dist > 1e-6 else np.zeros(2)

        # --- engagement dynamics ---
        if dist < config.BAND_MIN:                       # crowded / pestered
            self.engagement -= config.PESTER_RATE * dt
        elif config.BAND_MIN <= dist <= config.BAND_MAX: # good play band
            self.engagement += config.ENGAGE_GAIN * self._playfulness * dt
        self.engagement -= config.ENGAGE_DECAY * dt      # natural waning
        self.engagement = float(np.clip(self.engagement, 0.0, 1.0))

        # --- movement ---
        if dist < config.BAND_MIN:
            vel = -unit * config.CAT_FLEE_SPEED          # flee the crowding
        elif self.willing and dist <= config.BAND_MAX:
            # playful: occasional dart in a random direction
            if self.rng.random() < config.CAT_DART_PROB:
                ang = self.rng.uniform(0, 2 * np.pi)
                vel = np.array([np.cos(ang), np.sin(ang)]) * config.CAT_FLEE_SPEED
            else:
                vel = self.vel * 0.8                      # coast
        else:
            # disinterested: slow wander, ignore the robot
            ang = self.rng.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(ang), np.sin(ang)]) * config.CAT_WANDER_SPEED

        self.vel = vel
        self.pos = self.pos + vel * dt
        self.pos = np.clip(self.pos, -2.0, 2.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cat.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/cat.py tests/test_cat.py
git commit -m "feat: moody simulated cat with engagement dynamics"
```

---

### Task 4: Consent-based per-step reward

**Files:**
- Create: `src/roybot/reward.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: `roybot.config` (BAND_MIN/MAX/CENTER, REWARD_WEIGHTS), `numpy`.
- Produces: `compute_reward(*, dist, prev_dist, willing, action, prev_action, upright) -> tuple[float, dict]`. `dist`/`prev_dist` are floats (m), `willing`/`upright` bools, `action`/`prev_action` are length-2 sequences. Returns `(total_reward, terms)` where `terms` maps each weight key to its (signed, weighted) contribution.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py
import numpy as np
from roybot.reward import compute_reward
from roybot import config

A0 = [0.0, 0.0]

def test_willing_in_band_is_positive():
    r, terms = compute_reward(dist=config.BAND_CENTER, prev_dist=config.BAND_CENTER,
                              willing=True, action=A0, prev_action=A0, upright=True)
    assert terms["engage"] > 0
    assert r > 0

def test_disinterested_and_approaching_is_penalized():
    # cat not willing, robot closing distance => pestering
    r, terms = compute_reward(dist=0.3, prev_dist=0.5, willing=False,
                              action=A0, prev_action=A0, upright=True)
    assert terms["pester"] < 0
    assert r < 0

def test_disinterested_and_retreating_is_rewarded():
    r, terms = compute_reward(dist=0.6, prev_dist=0.4, willing=False,
                              action=A0, prev_action=A0, upright=True)
    assert terms["give_space"] > 0

def test_tip_is_a_large_penalty():
    r, terms = compute_reward(dist=config.BAND_CENTER, prev_dist=config.BAND_CENTER,
                              willing=True, action=A0, prev_action=A0, upright=False)
    assert terms["tip"] == -config.REWARD_WEIGHTS["tip"]
    assert r < 0

def test_action_rate_and_energy_penalize_big_jerky_actions():
    r_calm, _ = compute_reward(dist=config.BAND_CENTER, prev_dist=config.BAND_CENTER,
                               willing=True, action=A0, prev_action=A0, upright=True)
    r_jerky, _ = compute_reward(dist=config.BAND_CENTER, prev_dist=config.BAND_CENTER,
                                willing=True, action=[1.0, -1.0], prev_action=[-1.0, 1.0],
                                upright=True)
    assert r_jerky < r_calm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.reward'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/roybot/reward.py
"""Consent-based per-step reward.

Willing cat  -> pay for engaging (in-band) and for dynamic, juking play.
Disinterested cat -> pay for giving space (retreating), penalize approaching.
Always       -> small energy/jerk penalties; a big one-time tip penalty.
"""
import numpy as np

from . import config


def _in_band_score(dist: float) -> float:
    """1.0 at the band center, decaying to ~0 outside the band."""
    half_width = (config.BAND_MAX - config.BAND_MIN) / 2.0
    z = (dist - config.BAND_CENTER) / half_width
    return float(np.exp(-(z ** 2)))


def compute_reward(*, dist, prev_dist, willing, action, prev_action, upright):
    w = config.REWARD_WEIGHTS
    a = np.asarray(action, dtype=float)
    pa = np.asarray(prev_action, dtype=float)
    approach_rate = prev_dist - dist  # >0 means getting closer

    terms = {k: 0.0 for k in w}

    if willing:
        terms["engage"] = w["engage"] * _in_band_score(dist)
        # juking: reward changing the distance (dynamic play), only while in band
        if config.BAND_MIN <= dist <= config.BAND_MAX:
            terms["juke"] = w["juke"] * min(abs(approach_rate), 0.5)
    else:
        # retreating (approach_rate < 0) is good; approaching is pestering
        terms["give_space"] = w["give_space"] * max(0.0, -approach_rate)
        terms["pester"] = -w["pester"] * max(0.0, approach_rate)

    terms["energy"] = -w["energy"] * float(np.dot(a, a))
    terms["action_rate"] = -w["action_rate"] * float(np.dot(a - pa, a - pa))
    if not upright:
        terms["tip"] = -w["tip"]

    return float(sum(terms.values())), terms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/reward.py tests/test_reward.py
git commit -m "feat: consent-based per-step reward"
```

---

### Task 5: MuJoCo wheeled twin (MJCF)

**Files:**
- Create: `models/roybot.xml`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: a loadable MuJoCo model with exactly 2 actuators named `left_motor`, `right_motor` (velocity actuators on the wheel hinges), and sensors named `chassis_pos`, `chassis_quat`, `chassis_linvel`, `chassis_angvel`. Body `chassis` has a free joint. `SIM_TIMESTEP` matches `config.SIM_TIMESTEP`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
import os
import numpy as np
import mujoco
from roybot import config

MODEL = os.path.join("models", "roybot.xml")

def load():
    m = mujoco.MjModel.from_xml_path(MODEL)
    return m, mujoco.MjData(m)

def test_model_loads_with_two_motors_and_sensors():
    m, _ = load()
    assert m.nu == 2
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)}
    assert names == {"left_motor", "right_motor"}
    for s in ("chassis_pos", "chassis_quat", "chassis_linvel", "chassis_angvel"):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0
    assert abs(m.opt.timestep - config.SIM_TIMESTEP) < 1e-12

def test_zero_control_stays_upright_and_finite():
    m, d = load()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
    quat = d.sensor("chassis_quat").data  # w,x,y,z
    up_z = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    assert up_z > 0.9  # sitting flat

def test_equal_forward_control_drives_plus_x():
    m, d = load()
    d.ctrl[:] = [20.0, 20.0]
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert d.sensor("chassis_pos").data[0] > 0.05  # moved forward in +x

def test_opposite_control_yaws_in_place():
    m, d = load()
    d.ctrl[:] = [-15.0, 15.0]
    for _ in range(300):
        mujoco.mj_step(m, d)
    assert abs(d.sensor("chassis_angvel").data[2]) > 0.5  # yaw rate present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -q`
Expected: FAIL — `FileNotFoundError`/`could not open file` for `models/roybot.xml`

- [ ] **Step 3: Write the model**

```xml
<!-- models/roybot.xml : differential-drive robot, 2 wheels + caster -->
<mujoco model="roybot">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.01"/>
    <geom friction="1 0.1 0.01"/>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.85 0.9 0.85 1"
          friction="1 0.1 0.01"/>

    <body name="chassis" pos="0 0 0.03">
      <freejoint name="root"/>
      <geom name="chassis_geom" type="box" size="0.05 0.04 0.015" mass="0.30"
            rgba="0.3 0.45 0.8 1"/>
      <site name="imu" pos="0 0 0" size="0.005"/>

      <body name="left_wheel" pos="0 0.045 -0.005">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.025 0.008" quat="0.7071 0.7071 0 0"
              mass="0.02" rgba="0.15 0.15 0.15 1" friction="1.2 0.1 0.01"/>
      </body>

      <body name="right_wheel" pos="0 -0.045 -0.005">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.025 0.008" quat="0.7071 0.7071 0 0"
              mass="0.02" rgba="0.15 0.15 0.15 1" friction="1.2 0.1 0.01"/>
      </body>

      <!-- caster: low-friction ball so the rear slides freely -->
      <body name="caster" pos="0.04 0 -0.018">
        <geom type="sphere" size="0.012" mass="0.005" rgba="0.6 0.6 0.6 1"
              friction="0.05 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <velocity name="left_motor"  joint="left_wheel_joint"  kv="0.05"/>
    <velocity name="right_motor" joint="right_wheel_joint" kv="0.05"/>
  </actuator>

  <sensor>
    <framepos    name="chassis_pos"    objtype="site" objname="imu"/>
    <framequat   name="chassis_quat"   objtype="site" objname="imu"/>
    <framelinvel name="chassis_linvel" objtype="site" objname="imu"/>
    <frameangvel name="chassis_angvel" objtype="site" objname="imu"/>
  </sensor>
</mujoco>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -q`
Expected: PASS (4 passed).
> If the body sinks/explodes, raise `chassis pos z` or wheel positions so wheels+caster touch the floor; if it won't drive, raise actuator `kv` or the `d.ctrl` values. Tune until the 4 tests pass — this is the calibration the spec calls for.

- [ ] **Step 5: Commit**

```bash
git add models/roybot.xml tests/test_model.py
git commit -m "feat: MuJoCo differential-drive twin (2 wheels + caster)"
```

---

### Task 6: Gymnasium env — physics core

**Files:**
- Create: `src/roybot/env.py`
- Test: `tests/test_env_core.py`

**Interfaces:**
- Consumes: `roybot.config`, `roybot.driver.differential_drive`, `models/roybot.xml`, `mujoco`, `gymnasium`, `numpy`.
- Produces: `class RoybotChaseEnv(gymnasium.Env)` with `__init__(self, model_path="models/roybot.xml", domain_randomize=True, seed=None)`, `observation_space` = `Box` shape (12,), `action_space` = `Box(-1, 1, (2,))`. In this task `reset`/`step` work with a **placeholder still cat** at a fixed point and reward `0.0`; Task 7 adds the moody cat + real reward. Produces helpers used by Task 7: `self.cat_xy` (np (2,)), `self._robot_state()` -> dict with keys `pos`(2), `yaw`(float), `vfwd`(float), `vlat`(float), `yaw_rate`(float), `roll`(float), `pitch`(float), `upright`(bool).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_core.py
import numpy as np
from roybot.env import RoybotChaseEnv

def test_spaces():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    assert env.observation_space.shape == (12,)
    assert env.action_space.shape == (2,)

def test_reset_returns_obs_and_info():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (12,)
    assert np.all(np.isfinite(obs))

def test_step_returns_5_tuple_and_drives_forward():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    x0 = env._robot_state()["pos"][0]
    for _ in range(40):
        obs, rew, term, trunc, info = env.step(np.array([1.0, 0.0]))  # full forward
    assert obs.shape == (12,) and isinstance(rew, float)
    assert env._robot_state()["pos"][0] > x0

def test_time_limit_truncates():
    # zero action => robot sits still, never tips, so it truncates at the time limit
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    trunc = False
    for steps in range(1, 1002):
        _, _, term, trunc, _ = env.step(np.zeros(2))
        if trunc:
            break
    assert trunc and steps == 1000  # EPISODE_SECONDS(20) * CONTROL_HZ(50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_core.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.env'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/roybot/env.py
"""Gymnasium env wrapping the MuJoCo twin + closed-form driver (+ moody cat in Task 7)."""
import math

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from . import config
from .driver import differential_drive


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
        self._base_gain = self.model.actuator_gainprm.copy()
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        high = np.full(12, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._max_steps = int(round(config.EPISODE_SECONDS * config.CONTROL_HZ))
        self.cat_xy = np.array([config.BAND_CENTER, 0.0])  # placeholder (Task 7 replaces)
        self._prev_action = np.zeros(2)
        self._latency = 0
        self._action_buf = []
        self._motor_gain = 1.0
        self._steps = 0

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

    def _get_obs(self):
        st = self._robot_state()
        rel = _world_to_robot(self.cat_xy - st["pos"], st["yaw"])
        cat_vel_rel = _world_to_robot(getattr(self, "cat_vel", np.zeros(2)), st["yaw"])
        engagement = getattr(self, "cat_engagement", 0.5)
        return np.array([
            rel[0], rel[1], cat_vel_rel[0], cat_vel_rel[1], engagement,
            st["vfwd"], st["vlat"], st["yaw_rate"], st["roll"], st["pitch"],
            self._prev_action[0], self._prev_action[1],
        ], dtype=np.float32)

    def _apply_domain_randomization(self):
        if not self.domain_randomize:
            self._motor_gain, self._latency = 1.0, 0
            return
        self.model.body_mass[:] = self._base_mass * self.rng.uniform(0.8, 1.2)
        self.model.geom_friction[:, 0] = np.clip(
            self.model.geom_friction[:, 0] * self.rng.uniform(0.7, 1.3), 0.01, None)
        self._motor_gain = float(self.rng.uniform(0.85, 1.15))
        self._latency = int(self.rng.integers(0, 3))  # 0..2 control steps

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
        self.cat_xy = np.array([config.BAND_CENTER, 0.0])
        self.cat_vel = np.zeros(2)
        self.cat_engagement = 0.5
        return self._get_obs(), {}

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
        terminated = not st["upright"]
        truncated = self._steps >= self._max_steps
        reward = 0.0  # Task 7 fills this in
        self._prev_action = action
        return self._get_obs(), float(reward), terminated, truncated, {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_env_core.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/env.py tests/test_env_core.py
git commit -m "feat: gymnasium env physics core (driver + mujoco, placeholder reward)"
```

---

### Task 7: Wire the moody cat + consent reward into the env

**Files:**
- Modify: `src/roybot/env.py`
- Test: `tests/test_env_consent.py`

**Interfaces:**
- Consumes: `roybot.cat.Cat`, `roybot.reward.compute_reward`, the Task-6 env.
- Produces: env `step` now advances the `Cat`, fills `self.cat_xy/cat_vel/cat_engagement`, and returns the consent-based reward. `info` gains `{"reward_terms": dict, "willing": bool, "dist": float}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_consent.py
import numpy as np
from roybot.env import RoybotChaseEnv

def _run(env, policy, steps=400):
    obs, _ = env.reset(seed=1)
    total = 0.0
    last_info = {}
    for _ in range(steps):
        obs, rew, term, trunc, info = env.step(policy(obs))
        total += rew
        last_info = info
        if term or trunc:
            break
    return total, last_info

def test_step_populates_consent_info():
    env = RoybotChaseEnv(domain_randomize=False, seed=1)
    _, info = _run(env, lambda o: np.zeros(2), steps=5)
    assert "reward_terms" in info and "willing" in info and "dist" in info

def test_pestering_scores_worse_than_giving_space_when_disinterested():
    # Pin a disinterested cat directly in front (+x) and frozen, so "drive forward"
    # is unambiguously approaching and "drive backward" is retreating.
    def make():
        e = RoybotChaseEnv(domain_randomize=False, seed=2)
        e.reset(seed=2)
        e.cat.pos = np.array([0.4, 0.0])     # in front of the robot (starts at origin, yaw 0)
        e.cat.vel = np.zeros(2)
        e.cat.step = lambda dt, robot_xy: None  # freeze the cat
        e.cat.engagement = 0.0                  # disinterested
        e._sync_cat()
        e._prev_dist = e._distance()
        return e

    charge_env, retreat_env = make(), make()
    charge_total = retreat_total = 0.0
    for _ in range(60):
        _, rc, *_ = charge_env.step(np.array([1.0, 0.0]))   # toward the cat
        _, rr, *_ = retreat_env.step(np.array([-1.0, 0.0]))  # away from the cat
        charge_total += rc
        retreat_total += rr
    assert retreat_total > charge_total
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_env_consent.py -q`
Expected: FAIL — `AttributeError: 'RoybotChaseEnv' object has no attribute 'cat'` (and missing info keys)

- [ ] **Step 3: Update the env**

In `src/roybot/env.py`, add imports and a `Cat`, and replace the placeholder cat/reward.

```python
# add to imports at top of src/roybot/env.py
from .cat import Cat
from .reward import compute_reward
```

In `__init__`, after `self.rng = np.random.default_rng(seed)` add:

```python
        self.cat = Cat(self.rng)
```

In `reset`, replace the three placeholder cat lines
(`self.cat_xy = ...`, `self.cat_vel = ...`, `self.cat_engagement = ...`) with:

```python
        self.cat.rng = self.rng
        self.cat.reset()
        self._sync_cat()
        self._prev_dist = self._distance()
```

Add two helpers to the class:

```python
    def _sync_cat(self):
        self.cat_xy = self.cat.pos.copy()
        self.cat_vel = self.cat.vel.copy()
        self.cat_engagement = self.cat.engagement

    def _distance(self):
        return float(np.linalg.norm(self.cat_xy - self._robot_state()["pos"]))
```

Replace the body of `step` after `self._steps += 1` with:

```python
        st = self._robot_state()
        self.cat.step(config.CONTROL_DT, st["pos"])
        self._sync_cat()
        dist = self._distance()
        reward, terms = compute_reward(
            dist=dist, prev_dist=self._prev_dist, willing=self.cat.willing,
            action=action, prev_action=self._prev_action, upright=st["upright"],
        )
        self._prev_dist = dist
        terminated = not st["upright"]
        truncated = self._steps >= self._max_steps
        self._prev_action = action
        info = {"reward_terms": terms, "willing": self.cat.willing, "dist": dist}
        return self._get_obs(), float(reward), terminated, truncated, info
```

- [ ] **Step 4: Run tests to verify they pass (and Task 6 still passes)**

Run: `pytest tests/test_env_core.py tests/test_env_consent.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/env.py tests/test_env_consent.py
git commit -m "feat: wire moody cat + consent reward into env"
```

---

### Task 8: PPO training entrypoint

**Files:**
- Create: `scripts/train.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `roybot.env.RoybotChaseEnv`, `stable_baselines3.PPO`.
- Produces: `scripts/train.py` with `build_env()`, `train(total_timesteps, save_path, n_envs, seed) -> PPO`, runnable as `python scripts/train.py`. Saves an SB3 model `.zip`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_smoke.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Write the trainer**

```python
# scripts/train.py
"""Train the consent-based chase policy with PPO (stable-baselines3)."""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from roybot.env import RoybotChaseEnv


def build_env(seed=0, domain_randomize=True):
    return RoybotChaseEnv(domain_randomize=domain_randomize, seed=seed)


def train(total_timesteps=1_000_000, save_path="runs/chase_policy", n_envs=8, seed=0):
    venv = make_vec_env(
        lambda: RoybotChaseEnv(domain_randomize=True, seed=seed),
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
```

- [ ] **Step 4: Run the smoke test, then a real training run**

Run: `pytest tests/test_train_smoke.py -q`
Expected: PASS (1 passed)

Then train for real and watch `ep_rew_mean` climb:
```bash
python scripts/train.py --timesteps 1000000 --out runs/chase_policy
```
Expected: SB3 logs print; `rollout/ep_rew_mean` trends upward. On the RTX 3070 this is **minutes**, not hours. (If it's slow on CPU, lower `--timesteps` to 300k for a first look.)

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py tests/test_train_smoke.py
git commit -m "feat: PPO training entrypoint for the chase policy"
```

---

### Task 9: Export to torch-free NumPy inference

**Files:**
- Create: `src/roybot/infer.py`, `scripts/export.py`
- Test: `tests/test_infer.py`

**Interfaces:**
- Consumes: a trained SB3 `PPO` model (`policy.mlp_extractor.policy_net` = tanh MLP, `policy.action_net` = final Linear).
- Produces:
  - `scripts/export.py`: `export(model_path, npz_path) -> None` writing weight arrays `w0,b0,w1,b1,...,w_out,b_out`.
  - `src/roybot/infer.py`: `class NumpyPolicy` with `@classmethod from_npz(path)` and `act(self, obs) -> np.ndarray` (deterministic, tanh MLP -> linear -> clip to [-1, 1]); imports only `numpy`.

- [ ] **Step 1: Write the failing test**

```python
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
    for _ in range(20):
        obs = rng.standard_normal(12).astype(np.float32)
        sb3_action, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(pol.act(obs), sb3_action, rtol=1e-4, atol=1e-4)

def test_infer_imports_only_numpy():
    import roybot.infer as m
    src = open(m.__file__).read()
    assert "import torch" not in src and "stable_baselines3" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_infer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.infer'` / `export`

- [ ] **Step 3: Write export + inference**

```python
# scripts/export.py
"""Extract the deterministic actor MLP from an SB3 PPO model into a .npz."""
import numpy as np
from stable_baselines3 import PPO


def export(model_path, npz_path):
    model = PPO.load(model_path, device="cpu")
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
    np.savez(npz_path, **arrays)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("npz")
    a = p.parse_args()
    export(a.model, a.npz)
```

```python
# src/roybot/infer.py
"""Torch-free deterministic policy inference. numpy only — runs on the Pi."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_infer.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/roybot/infer.py scripts/export.py tests/test_infer.py
git commit -m "feat: torch-free numpy policy inference + SB3 export"
```

---

### Task 10: Episode metrics + sim demo (success criteria)

**Files:**
- Create: `src/roybot/metrics.py`, `scripts/demo.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `roybot.config`, `numpy`; the env + a policy (SB3 or `NumpyPolicy`).
- Produces:
  - `src/roybot/metrics.py`: `episode_metrics(records) -> dict` where each record is `{"dist": float, "willing": bool, "prev_dist": float, "upright": bool}`. Returns `{"time_in_band": float, "distance_variation": float, "approach_rate_when_disinterested": float, "tip_count": int}`.
  - `scripts/demo.py`: `evaluate(policy, episodes, seed) -> dict` (aggregated metrics) and a `--view` flag that runs the MuJoCo viewer. Runnable as `python scripts/demo.py runs/chase_policy.npz`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
from roybot.metrics import episode_metrics
from roybot import config

def rec(dist, willing, prev_dist, upright=True):
    return {"dist": dist, "willing": willing, "prev_dist": prev_dist, "upright": upright}

def test_time_in_band_fraction():
    recs = [rec(config.BAND_CENTER, True, config.BAND_CENTER),  # in band
            rec(2.0, True, 2.0)]                                # out of band
    m = episode_metrics(recs)
    assert abs(m["time_in_band"] - 0.5) < 1e-9

def test_approach_rate_when_disinterested_counts_only_disinterested_steps():
    recs = [rec(0.3, False, 0.5),   # disinterested, approaching (+0.2)
            rec(0.6, True, 0.4)]    # willing -> ignored
    m = episode_metrics(recs)
    assert abs(m["approach_rate_when_disinterested"] - 0.2) < 1e-9

def test_tip_count():
    recs = [rec(0.3, True, 0.3, upright=False), rec(0.3, True, 0.3, upright=True)]
    assert episode_metrics(recs)["tip_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'roybot.metrics'`

- [ ] **Step 3: Write metrics + demo**

```python
# src/roybot/metrics.py
"""Episode metrics that operationalize the Phase-1 success criteria."""
import numpy as np

from . import config


def episode_metrics(records):
    if not records:
        return {"time_in_band": 0.0, "distance_variation": 0.0,
                "approach_rate_when_disinterested": 0.0, "tip_count": 0}
    in_band = [config.BAND_MIN <= r["dist"] <= config.BAND_MAX for r in records]
    dists = np.array([r["dist"] for r in records])
    disint_approach = [max(0.0, r["prev_dist"] - r["dist"])
                       for r in records if not r["willing"]]
    return {
        "time_in_band": float(np.mean(in_band)),
        "distance_variation": float(np.std(dists)),
        "approach_rate_when_disinterested":
            float(np.mean(disint_approach)) if disint_approach else 0.0,
        "tip_count": int(sum(not r["upright"] for r in records)),
    }
```

```python
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
```

- [ ] **Step 4: Run unit tests, then the full success-criteria evaluation**

Run: `pytest tests/test_metrics.py -q`
Expected: PASS (3 passed)

Then export the trained policy and evaluate (this is the Phase-1 acceptance gate):
```bash
python scripts/export.py runs/chase_policy runs/chase_policy.npz
python scripts/demo.py runs/chase_policy.npz --episodes 20
```
Expected (a well-trained policy): `time_in_band` high (≳0.5), `tip_count` 0, and
`approach_rate_when_disinterested` near 0 (**the consent test** — it doesn't pester).
Watch it live with `--view`. If the consent metric is high, raise `REWARD_WEIGHTS["pester"]`
in `config.py` and retrain (the tuning loop the spec calls for).

- [ ] **Step 5: Commit**

```bash
git add src/roybot/metrics.py scripts/demo.py tests/test_metrics.py
git commit -m "feat: episode metrics + sim demo with consent/play eval"
```

---

## Final verification (whole-suite gate)

- [ ] Run the full suite: `pytest -q` → all green.
- [ ] Confirm Phase-1 success criteria from the spec:
  1. ✅ twin loads + `drive()` moves it (Tasks 2, 5).
  2. ✅ cat moves + engagement rises/falls (Task 3).
  3. ✅ policy plays a willing cat without tipping (Tasks 7–8, demo).
  4. ✅ consent test: backs off a disinterested cat (Task 7 test + demo metric).
  5. ✅ policy exports + runs torch-free (Task 9) — Pi-ready (deploy is Phase 2).
  6/7. Real-base drive + safety floor → **Phase 2** (hardware), out of this plan.
- [ ] `git push`

---

## Self-review notes (author)

- **Spec coverage:** §4B driver→Task 2; §4C twin+moody cat→Tasks 3,5; §4D obs/action/reward+consent→Tasks 4,6,7; training stack/DR→Tasks 6,8; export/@50 Hz-ready→Task 9; §9 success criteria→Task 10 + final gate. Hardware (§4A/E real reflexes), perception, dock = later sub-projects, intentionally out of Phase 1.
- **Type consistency:** `differential_drive` signature, `Cat.step(dt, robot_xy)`, `compute_reward(**kwargs)` keys, env helper names (`_robot_state`, `_distance`, `_get_obs`, `cat_xy/cat_vel/cat_engagement`), and `NumpyPolicy.act/from_npz` are used identically across tasks.
- **Optional later accelerator (not in this plan):** port the env to MJX/Brax for minute-scale GPU training if SB3 iteration becomes the bottleneck — deferred per YAGNI.
