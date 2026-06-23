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


def compute_reward(*, dist, approach_rate, anticipate_rate, willing, action, prev_action, upright):
    w = config.REWARD_WEIGHTS
    a = np.asarray(action, dtype=float)
    pa = np.asarray(prev_action, dtype=float)

    terms = {k: 0.0 for k in w}

    if willing:
        terms["engage"] = w["engage"] * _in_band_score(dist)
        # juking: reward changing the distance (dynamic play), only while in band
        if config.BAND_MIN <= dist <= config.BAND_MAX:
            terms["juke"] = w["juke"] * min(abs(approach_rate), config.JUKE_RATE_CAP)
        terms["anticipate"] = w["anticipate"] * max(0.0, anticipate_rate)
    else:
        # retreating (approach_rate < 0) is good; approaching is pestering
        terms["give_space"] = w["give_space"] * max(0.0, -approach_rate)
        terms["pester"] = -w["pester"] * max(0.0, approach_rate)

    terms["energy"] = -w["energy"] * float(np.dot(a, a))
    terms["action_rate"] = -w["action_rate"] * float(np.dot(a - pa, a - pa))
    if not upright:
        terms["tip"] = -w["tip"]

    return float(sum(terms.values())), terms
