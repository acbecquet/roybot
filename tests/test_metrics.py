# tests/test_metrics.py
from roybot.metrics import episode_metrics
from roybot import config

def rec(dist, willing, approach_rate, upright=True):
    return {"dist": dist, "willing": willing, "approach_rate": approach_rate, "upright": upright}

def test_time_in_band_fraction():
    recs = [rec(config.BAND_CENTER, True, 0.0),  # in band
            rec(2.0, True, 0.0)]                 # out of band
    m = episode_metrics(recs)
    assert abs(m["time_in_band"] - 0.5) < 1e-9

def test_approach_rate_when_disinterested_counts_only_disinterested_steps():
    recs = [rec(0.3, False, 0.2),   # disinterested, approaching (+0.2)
            rec(0.6, True, -0.2)]   # willing -> ignored
    m = episode_metrics(recs)
    assert abs(m["approach_rate_when_disinterested"] - 0.2) < 1e-9

def test_tip_count():
    recs = [rec(0.3, True, 0.0, upright=False), rec(0.3, True, 0.0, upright=True)]
    assert episode_metrics(recs)["tip_count"] == 1
