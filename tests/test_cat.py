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
    before = c.engagement
    for _ in range(20):
        c.pos = np.array([0.0, 0.0])  # re-pin each step so distance stays exactly in-band
        c.step(0.02, np.array([config.BAND_CENTER, 0.0]))
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
