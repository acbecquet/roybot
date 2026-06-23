from roybot.reward import compute_reward
from roybot import config

A0 = [0.0, 0.0]

def test_willing_in_band_is_positive():
    r, terms = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                              anticipate_rate=0.0,
                              willing=True, action=A0, prev_action=A0, upright=True)
    assert terms["engage"] > 0
    assert r > 0

def test_disinterested_and_approaching_is_penalized():
    # cat not willing, robot closing distance => pestering
    r, terms = compute_reward(dist=0.3, approach_rate=0.2, anticipate_rate=0.0,
                              willing=False,
                              action=A0, prev_action=A0, upright=True)
    assert terms["pester"] < 0
    assert r < 0

def test_disinterested_and_retreating_is_rewarded():
    r, terms = compute_reward(dist=0.6, approach_rate=-0.2, anticipate_rate=0.0,
                              willing=False,
                              action=A0, prev_action=A0, upright=True)
    assert terms["give_space"] > 0

def test_tip_is_a_large_penalty():
    r, terms = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                              anticipate_rate=0.0,
                              willing=True, action=A0, prev_action=A0, upright=False)
    assert terms["tip"] == -config.REWARD_WEIGHTS["tip"]
    assert r < 0

def test_action_rate_and_energy_penalize_big_jerky_actions():
    r_calm, _ = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                               anticipate_rate=0.0,
                               willing=True, action=A0, prev_action=A0, upright=True)
    r_jerky, _ = compute_reward(dist=config.BAND_CENTER, approach_rate=0.0,
                                anticipate_rate=0.0,
                                willing=True, action=[1.0, -1.0], prev_action=[-1.0, 1.0],
                                upright=True)
    assert r_jerky < r_calm


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
