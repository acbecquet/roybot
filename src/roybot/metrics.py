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
    disint_approach = [max(0.0, r["approach_rate"]) for r in records if not r["willing"]]
    return {
        "time_in_band": float(np.mean(in_band)),
        "distance_variation": float(np.std(dists)),
        "approach_rate_when_disinterested":
            float(np.mean(disint_approach)) if disint_approach else 0.0,
        "tip_count": int(sum(not r["upright"] for r in records)),
    }
