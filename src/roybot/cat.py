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
