# Roybot Phase 1.5 — Smartest-Play Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the chase/play policy genuinely *smart* — anticipatory, lifelike, robust — by upgrading the four behavior levers, all on the existing SB3/CPU stack (no GPU/MJX).

**Design (settled):** (1) a **rich parameterized cat** behavior state machine (stalk/pounce/bat/dart/flee/rest) with randomized personality; (2) **frame-stacking** the last N observations so the policy anticipates motion; (3) a **reward upgrade** adding an *anticipation* term (close on where the cat is heading); (4) **wider domain randomization + per-episode difficulty** (implicit curriculum) for sim2real robustness. The policy stays a tiny MLP (Pi-deployable); the torch-free `infer.py` is architecture-agnostic and needs no change.

**Tech Stack:** Python 3.11+, mujoco, gymnasium, stable-baselines3 (PPO, CPU), numpy, pytest. Builds on the Phase-1 package (`src/roybot/`, `models/`, `scripts/`, `tests/`).

## Global Constraints

- All randomness through a seeded `numpy.random.Generator` — no global `random`/`np.random`.
- `infer.py` imports only numpy (unchanged). New behavior must keep the policy a plain MLP.
- All tunables live in `src/roybot/config.py` (DRY) — no new magic numbers elsewhere.
- **Preserve Phase-1 behavioral contracts:** the cat's engagement dynamics (rises in-band, falls when crowded, decays) and `willing` semantics must still hold; the consent reward (pester < give_space when disinterested) must still hold.
- TDD; commit per task; conventional commits. Run tests: `.venv/Scripts/python -m pytest -q`.

---

### Task 1: Config additions

**Files:** Modify `src/roybot/config.py`; Test: `tests/test_config.py` (extend)

**Interfaces — Produces (new constants):** `N_STACK`, cat mode speeds/durations + personality ranges, `ANTICIPATE_HORIZON`, `REWARD_WEIGHTS["anticipate"]`, DR-range constants, difficulty range.

- [ ] **Step 1: Write the failing test** (append to `tests/test_config.py`)

```python
def test_phase15_constants_present():
    assert config.N_STACK >= 2
    assert "anticipate" in config.REWARD_WEIGHTS
    assert config.ANTICIPATE_HORIZON > 0
    for k in ("CAT_STALK_SPEED", "CAT_POUNCE_SPEED", "CAT_BAT_SPEED",
              "CAT_DART_SPEED", "CAT_MODE_MIN_S", "CAT_MODE_MAX_S"):
        assert hasattr(config, k)
    assert config.CAT_MODE_MIN_S < config.CAT_MODE_MAX_S
    assert config.DR_MASS[0] < 1.0 < config.DR_MASS[1]
    assert config.DIFFICULTY_RANGE[0] <= config.DIFFICULTY_RANGE[1]
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py::test_phase15_constants_present -q`
Expected: FAIL (`AttributeError`).

- [ ] **Step 3: Append to `src/roybot/config.py`**

```python
# --- Phase 1.5: frame stacking ---
N_STACK = 6                      # observations stacked for motion/anticipation context

# --- Phase 1.5: rich cat behavior modes ---
CAT_STALK_SPEED = 0.12          # m/s slow circling creep
CAT_POUNCE_SPEED = 0.9          # m/s burst toward the robot
CAT_BAT_SPEED = 0.45            # m/s quick swipe
CAT_DART_SPEED = 0.7            # m/s erratic burst
CAT_MODE_MIN_S = 0.4            # min seconds a play mode is held
CAT_MODE_MAX_S = 1.6            # max seconds a play mode is held
# personality is sampled per episode in Cat.reset() from these ranges:
CAT_PLAYFULNESS_RANGE = (0.6, 1.4)
CAT_SKITTISHNESS_RANGE = (0.5, 1.5)
CAT_ATTENTION_RANGE = (0.5, 1.5)

# --- Phase 1.5: anticipation reward ---
ANTICIPATE_HORIZON = 0.4        # s ahead to predict the cat's position
REWARD_WEIGHTS["anticipate"] = 0.6   # reward closing on the predicted (lead) position

# --- Phase 1.5: domain randomization ranges (widened) + difficulty ---
DR_MASS = (0.7, 1.3)            # body-mass scale
DR_FRICTION = (0.6, 1.4)        # geom-friction scale
DR_MOTOR_GAIN = (0.8, 1.2)      # commanded-speed scale
DR_LATENCY_STEPS = (0, 4)       # control-step latency (exclusive high -> 0..3)
DIFFICULTY_RANGE = (0.0, 1.0)   # per-episode difficulty; scales cat speed/evasiveness
CAT_SPEED_SCALE_AT_MAX = 1.6    # cat speed multiplier at difficulty=1.0 (1.0 at difficulty=0)
```

- [ ] **Step 4: Run, verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/roybot/config.py tests/test_config.py
git commit -m "feat: phase 1.5 config (frame stack, cat modes, anticipation, DR/difficulty)"
```

---

### Task 2: Rich parameterized cat (behavior state machine)

**Files:** Modify `src/roybot/cat.py`; Test: `tests/test_cat.py` (extend, keep existing tests passing)

**Interfaces:**
- Consumes: new `config` constants from Task 1.
- Produces: `Cat` gains attribute `mode` (str in `{"flee","rest","stalk","pounce","bat","dart"}`) and `speed_scale` (float, default 1.0, set by the env per episode to apply difficulty). `reset()` randomizes `_playfulness/_skittishness/_attention`. Engagement dynamics and `willing` unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_cat.py`)

```python
def test_modes_appear_during_willing_play():
    c = make_cat(3); c.reset()
    seen = set()
    for _ in range(800):
        c.engagement = 0.9                 # keep her willing
        c.pos = np.array([0.0, 0.0])        # re-pin mid-band each step
        c.step(0.02, np.array([config.BAND_CENTER, 0.0]))
        seen.add(c.mode)
    assert {"flee", "rest"}.isdisjoint(seen)          # not crowded, willing -> play modes only
    assert len(seen & {"stalk", "pounce", "bat", "dart"}) >= 2  # variety

def test_pounce_moves_toward_robot():
    c = make_cat(0); c.reset()
    c.pos = np.array([0.0, 0.0]); c.engagement = 0.9
    c.mode = "pounce"; c._mode_timer = 5.0            # hold pounce
    robot = np.array([0.3, 0.0])
    c.step(0.02, robot)
    assert np.dot(c.vel, robot - np.array([0.0, 0.0])) > 0   # velocity has a toward-robot component

def test_speed_scale_scales_motion():
    a, b = make_cat(5), make_cat(5)
    a.reset(); b.reset()
    a.speed_scale = 1.0; b.speed_scale = 2.0
    robot = np.array([0.3, 0.0])
    for _ in range(5):
        a.engagement = 0.9; b.engagement = 0.9
        a.step(0.02, robot); b.step(0.02, robot)
    assert np.linalg.norm(b.vel) > np.linalg.norm(a.vel)

def test_personality_sampled_in_reset():
    c = make_cat(0); c.reset()
    assert config.CAT_SKITTISHNESS_RANGE[0] <= c._skittishness <= config.CAT_SKITTISHNESS_RANGE[1]
    assert config.CAT_ATTENTION_RANGE[0] <= c._attention <= config.CAT_ATTENTION_RANGE[1]
```

(The existing Task-3 tests — engagement rise/fall/decay, flee, willing, determinism — MUST still pass unchanged.)

- [ ] **Step 2: Run, verify new tests fail**

Run: `.venv/Scripts/python -m pytest tests/test_cat.py -q`
Expected: the four new tests FAIL (no `mode`/`speed_scale`), existing ones still pass.

- [ ] **Step 3: Rewrite `src/roybot/cat.py`** (preserve engagement dynamics exactly; add the mode machine)

```python
"""A rich, moody cat: engagement state + a behavior state machine.

Engagement (unchanged from Phase 1) gates willingness; on top of it a play-mode
machine (stalk/pounce/bat/dart) driven by randomized personality makes the cat a
varied, lifelike opponent. `speed_scale` lets the env apply per-episode difficulty.
"""
import numpy as np

from . import config

PLAY_MODES = ("stalk", "pounce", "bat", "dart")


class Cat:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.engagement = 0.5
        self.speed_scale = 1.0
        self.mode = "rest"
        self._playfulness = 1.0
        self._skittishness = 1.0
        self._attention = 1.0
        self._mode_timer = 0.0
        self._heading = np.array([1.0, 0.0])
        # no reset() here: the env calls reset() explicitly

    def reset(self) -> None:
        self._playfulness = float(self.rng.uniform(*config.CAT_PLAYFULNESS_RANGE))
        self._skittishness = float(self.rng.uniform(*config.CAT_SKITTISHNESS_RANGE))
        self._attention = float(self.rng.uniform(*config.CAT_ATTENTION_RANGE))
        self.speed_scale = 1.0
        self.pos = self.rng.uniform(-0.6, 0.6, size=2)
        self.vel = np.zeros(2)
        self.engagement = float(self.rng.uniform(0.3, 0.7))
        self.mode = "rest"
        self._mode_timer = 0.0
        self._heading = self._random_unit()

    @property
    def willing(self) -> bool:
        return self.engagement >= config.WILLING_THRESHOLD

    def _random_unit(self) -> np.ndarray:
        ang = self.rng.uniform(0, 2 * np.pi)
        return np.array([np.cos(ang), np.sin(ang)])

    def _pick_play_mode(self) -> str:
        # playful cats pounce/dart more; weights normalized
        w = np.array([0.3, 0.2 * self._playfulness, 0.2, 0.2 * self._playfulness])
        w = w / w.sum()
        return str(self.rng.choice(PLAY_MODES, p=w))

    def _play_velocity(self, unit_to_robot: np.ndarray) -> np.ndarray:
        if self.mode == "pounce":
            return unit_to_robot * config.CAT_POUNCE_SPEED
        if self.mode == "dart":
            return self._heading * config.CAT_DART_SPEED
        if self.mode == "bat":
            return self._heading * config.CAT_BAT_SPEED
        # stalk: slow creep perpendicular to the robot direction (circling)
        perp = np.array([-unit_to_robot[1], unit_to_robot[0]])
        return perp * config.CAT_STALK_SPEED

    def step(self, dt: float, robot_xy: np.ndarray) -> None:
        to_robot = robot_xy - self.pos
        dist = float(np.linalg.norm(to_robot))
        unit = to_robot / dist if dist > 1e-6 else np.zeros(2)

        # --- engagement dynamics (UNCHANGED from Phase 1) ---
        if dist < config.BAND_MIN:
            self.engagement -= config.PESTER_RATE * dt
        elif config.BAND_MIN <= dist <= config.BAND_MAX:
            self.engagement += config.ENGAGE_GAIN * self._playfulness * dt
        self.engagement -= config.ENGAGE_DECAY * dt
        self.engagement = float(np.clip(self.engagement, 0.0, 1.0))

        # --- behavior mode + velocity ---
        self._mode_timer -= dt
        if dist < config.BAND_MIN:                       # crowded -> flee
            self.mode = "flee"
            vel = -unit * config.CAT_FLEE_SPEED * (1.0 + 0.3 * self._skittishness)
        elif not self.willing:                           # disinterested -> wander/rest
            self.mode = "rest"
            if self._mode_timer <= 0:
                self._heading = self._random_unit()
                self._mode_timer = float(self.rng.uniform(0.5, 1.5))
            vel = self._heading * config.CAT_WANDER_SPEED
        else:                                            # willing -> play
            if self._mode_timer <= 0:
                self.mode = self._pick_play_mode()
                self._heading = self._random_unit()
                self._mode_timer = float(
                    self.rng.uniform(config.CAT_MODE_MIN_S, config.CAT_MODE_MAX_S) * self._attention)
            vel = self._play_velocity(unit)

        self.vel = vel * self.speed_scale
        self.pos = np.clip(self.pos + self.vel * dt,
                           -config.CAT_ARENA_HALF, config.CAT_ARENA_HALF)
```

- [ ] **Step 4: Run, verify all cat tests pass**

Run: `.venv/Scripts/python -m pytest tests/test_cat.py -q`
Expected: PASS (old + 4 new).
> Engagement-block code is byte-identical to Phase 1, so the engagement/flee/willing/determinism tests are preserved.

- [ ] **Step 5: Commit**

```bash
git add src/roybot/cat.py tests/test_cat.py
git commit -m "feat: rich parameterized cat behavior state machine"
```

---

### Task 3: Frame-stacking in the env

**Files:** Modify `src/roybot/env.py`; Modify `tests/test_env_core.py`, `tests/test_infer.py` (obs dim); Test: `tests/test_env_core.py`

**Interfaces:** `observation_space` becomes shape `(12 * config.N_STACK,)`. Internally `_base_obs()` returns the 12-D vector (the old `_get_obs` body); `_get_obs()` returns the concatenation of the last `N_STACK` base obs. `reset` fills the stack with `N_STACK` copies of the first base obs.

- [ ] **Step 1: Write the failing test** (edit `tests/test_env_core.py` — update `test_spaces` and `test_reset_returns_obs_and_info`)

```python
def test_spaces():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    from roybot import config
    assert env.observation_space.shape == (12 * config.N_STACK,)
    assert env.action_space.shape == (2,)

def test_reset_returns_obs_and_info():
    from roybot import config
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (12 * config.N_STACK,)
    assert np.all(np.isfinite(obs))
```

(Also update the `test_step_returns_5_tuple_and_drives_forward` assertion `obs.shape == (12,)` → `obs.shape == (12 * config.N_STACK,)`, and add `from roybot import config` at top of the test file.)

- [ ] **Step 2: Run, verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_env_core.py -q`
Expected: FAIL (shape (12,) != (72,)).

- [ ] **Step 3: Edit `src/roybot/env.py`**

In `__init__`, replace the observation-space line and add a stack buffer:
```python
        from collections import deque
        self._stack = deque(maxlen=config.N_STACK)
        high = np.full(12 * config.N_STACK, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
```
(Keep `self.action_space` as is. Remove the old `high = np.full(12, ...)` / observation_space lines.)

Rename the existing `_get_obs` method to `_base_obs` (its body is unchanged — it returns the 12-D vector). Then add:
```python
    def _get_obs(self):
        base = self._base_obs()
        if not self._stack:
            for _ in range(config.N_STACK):
                self._stack.append(base)
        else:
            self._stack.append(base)
        return np.concatenate(list(self._stack)).astype(np.float32)
```
In `reset`, clear the stack before the first obs: add `self._stack.clear()` just before `return self._get_obs(), {}`.

- [ ] **Step 4: Run, verify env + infer tests pass**

`tests/test_infer.py` builds a PPO from the env, so its obs auto-sizes; but its hand-built `obs` vectors are hardcoded to 12. Edit `tests/test_infer.py`: replace each `np.zeros(12)` / `rng.standard_normal(12)` with the env's real width, e.g. add at the top of each test that needs it:
```python
    n = env.observation_space.shape[0]
```
and use `rng.standard_normal(n)` / `np.zeros(n, dtype=np.float32)`.

Run: `.venv/Scripts/python -m pytest tests/test_env_core.py tests/test_env_consent.py tests/test_infer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/roybot/env.py tests/test_env_core.py tests/test_infer.py
git commit -m "feat: frame-stack observations for motion/anticipation context"
```

---

### Task 4: Reward upgrade — anticipation term

**Files:** Modify `src/roybot/reward.py`, `src/roybot/env.py`; Test: `tests/test_reward.py`, `tests/test_env_consent.py`

**Interfaces:** `compute_reward` gains a keyword `anticipate_rate` (robot-attributed closing rate toward the cat's *predicted* position). New term `terms["anticipate"]` rewards positive anticipate_rate while willing. The env computes `anticipate_rate` and passes it; `info` gains `"anticipate_rate"`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_reward.py`; update existing calls to pass `anticipate_rate=0.0`)

```python
def test_anticipation_rewards_leading_a_willing_cat():
    r_lead, terms = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                                   anticipate_rate=0.3, willing=True,
                                   action=A0, prev_action=A0, upright=True)
    assert terms["anticipate"] > 0
    r_none, _ = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                               anticipate_rate=0.0, willing=True,
                               action=A0, prev_action=A0, upright=True)
    assert r_lead > r_none

def test_anticipation_inactive_when_disinterested():
    _, terms = compute_reward(dist=0.6, approach_rate=-0.1, anticipate_rate=0.3,
                              willing=False, action=A0, prev_action=A0, upright=True)
    assert terms["anticipate"] == 0.0
```

(Update the five existing `compute_reward(...)` calls in this file to add `anticipate_rate=0.0`.)

- [ ] **Step 2: Run, verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_reward.py -q`
Expected: FAIL (`TypeError: unexpected keyword 'anticipate_rate'` once updated, or missing term).

- [ ] **Step 3: Edit `src/roybot/reward.py`**

Change the signature and add the term:
```python
def compute_reward(*, dist, approach_rate, anticipate_rate, willing, action, prev_action, upright):
```
Inside the `if willing:` branch, after the `juke` term, add:
```python
        terms["anticipate"] = config.REWARD_WEIGHTS["anticipate"] * max(0.0, anticipate_rate)
```
(`terms` is already seeded from `REWARD_WEIGHTS` keys, so `"anticipate"` exists and defaults to 0.0 in the disinterested branch — no other change needed.)

- [ ] **Step 4: Edit `src/roybot/env.py` `step()`** to compute and pass `anticipate_rate`

After `cat_prev = self.cat.pos.copy()` and the cat step, where `approach_rate` is computed, add a predicted-position closing rate (robot-attributed, same pattern):
```python
        predicted = cat_prev + self.cat_vel * config.ANTICIPATE_HORIZON
        anticipate_rate = self._prev_dist - float(np.linalg.norm(predicted - st["pos"]))
```
Pass it into `compute_reward(..., anticipate_rate=anticipate_rate, ...)` and add `"anticipate_rate": anticipate_rate` to the `info` dict.

- [ ] **Step 5: Run, verify reward + env tests pass**

Run: `.venv/Scripts/python -m pytest tests/test_reward.py tests/test_env_core.py tests/test_env_consent.py -q`
Expected: PASS. (Consent tests freeze the cat ⇒ `cat_vel≈0` ⇒ `anticipate_rate≈approach_rate`; the pester<give_space ordering still holds.)

- [ ] **Step 6: Commit**

```bash
git add src/roybot/reward.py src/roybot/env.py tests/test_reward.py
git commit -m "feat: anticipation reward (lead the cat's predicted position)"
```

---

### Task 5: Wider domain randomization + per-episode difficulty

**Files:** Modify `src/roybot/env.py`; Test: `tests/test_env_core.py`

**Interfaces:** `_apply_domain_randomization` uses the `config.DR_*` ranges. On reset (when `domain_randomize`), sample a `self.difficulty ∈ DIFFICULTY_RANGE` and set `self.cat.speed_scale` accordingly (1.0 at difficulty 0 → `CAT_SPEED_SCALE_AT_MAX` at 1.0). `info`/attribute `difficulty` exposed for inspection.

- [ ] **Step 1: Write the failing test** (append to `tests/test_env_core.py`)

```python
def test_difficulty_sets_cat_speed_scale_and_varies():
    from roybot import config
    scales = set()
    for s in range(8):
        env = RoybotChaseEnv(domain_randomize=True, seed=s)
        env.reset(seed=s)
        assert 1.0 <= env.cat.speed_scale <= config.CAT_SPEED_SCALE_AT_MAX + 1e-9
        scales.add(round(env.cat.speed_scale, 3))
    assert len(scales) >= 3   # difficulty actually varies across episodes

def test_no_dr_keeps_baseline_difficulty():
    env = RoybotChaseEnv(domain_randomize=False, seed=0)
    env.reset(seed=0)
    assert env.cat.speed_scale == 1.0
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_env_core.py::test_difficulty_sets_cat_speed_scale_and_varies -q`
Expected: FAIL.

- [ ] **Step 3: Edit `src/roybot/env.py`**

Replace the hardcoded ranges in `_apply_domain_randomization` with the config ones, and add difficulty:
```python
    def _apply_domain_randomization(self):
        if not self.domain_randomize:
            self._motor_gain, self._latency = 1.0, 0
            self.difficulty = 0.0
            self.cat.speed_scale = 1.0
            return
        self.model.body_mass[:] = self._base_mass * self.rng.uniform(*config.DR_MASS)
        self.model.geom_friction[:, 0] = np.clip(
            self._base_friction[:, 0] * self.rng.uniform(*config.DR_FRICTION), 0.01, None)
        self._motor_gain = float(self.rng.uniform(*config.DR_MOTOR_GAIN))
        self._latency = int(self.rng.integers(*config.DR_LATENCY_STEPS))
        self.difficulty = float(self.rng.uniform(*config.DIFFICULTY_RANGE))
        self.cat.speed_scale = 1.0 + self.difficulty * (config.CAT_SPEED_SCALE_AT_MAX - 1.0)
```
Note: `_apply_domain_randomization` runs in `reset()` AFTER `self.cat` exists; it also runs before `self.cat.reset()` currently — ensure `self.cat.speed_scale` is set after `self.cat.reset()` (which resets speed_scale to 1.0). **In `reset()`, move the `self.cat.speed_scale = ...` assignment to AFTER `self.cat.reset()`** by setting `self.difficulty` in DR but applying `self.cat.speed_scale` right after `self.cat.reset()`:
```python
        # in reset(), after self.cat.reset():
        self.cat.speed_scale = (1.0 + self.difficulty * (config.CAT_SPEED_SCALE_AT_MAX - 1.0)) if self.domain_randomize else 1.0
```
and in `_apply_domain_randomization` only set `self.difficulty` (not speed_scale). Keep it simple: set `self.difficulty` in DR; apply `speed_scale` after `cat.reset()` in `reset()`.

- [ ] **Step 4: Run, verify all tests pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: PASS (full suite).

- [ ] **Step 5: Commit**

```bash
git add src/roybot/env.py tests/test_env_core.py
git commit -m "feat: wider DR from config + per-episode difficulty (implicit curriculum)"
```

---

## Final verification + retrain

- [ ] Full suite green: `.venv/Scripts/python -m pytest -q`.
- [ ] Retrain on the upgraded env (CPU; ~minutes): `.venv/Scripts/python scripts/train.py --timesteps 1500000 --out runs/smart_policy`.
- [ ] `.venv/Scripts/python scripts/export.py runs/smart_policy runs/smart_policy.npz`
- [ ] `.venv/Scripts/python scripts/demo.py runs/smart_policy.npz --view` — confirm richer, anticipatory, consent-respecting play across cat modes.
- [ ] `git push`

## Self-review notes (author)

- **Scope:** four levers (rich cat, frame-stack, anticipation reward, DR/difficulty). Self-play and recurrence deliberately NOT included (chosen against — parameterized cat + frame-stack). A true progressive curriculum (callback that ramps difficulty with reward) is deferred; per-episode difficulty sampling is the lazy-effective stand-in.
- **Preserved contracts:** cat engagement block is byte-identical to Phase 1; consent reward ordering preserved (anticipation only adds a willing-branch term; consent tests freeze the cat so anticipate≈approach).
- **Pi path intact:** policy stays an MLP; `infer.py` unchanged (reads dims from weights). Phase-2 note: the Pi runtime must frame-stack the last `N_STACK` observations before calling `infer.act` — mirror the env's stack.
- **Type consistency:** `compute_reward` signature change (`anticipate_rate` added) is reflected in env call + all reward tests. Obs width `12*N_STACK` updated in env + the two tests that hardcoded 12.
