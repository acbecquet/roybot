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
        # no reset() here: Gym-style, the env calls reset() explicitly

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
