"""Feature-drift check: compare a window of candles against baseline statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_NAMES = ("ret_mean", "ret_std", "range_mean")


def window_stats(df: pd.DataFrame) -> dict:
    close = df["close"].to_numpy(dtype=float)
    rets = np.diff(np.log(close))
    rng = ((df["high"] - df["low"]) / df["close"]).to_numpy(dtype=float)
    return {
        "ret_mean": float(rets.mean()),
        "ret_std": float(rets.std()),
        "range_mean": float(rng.mean()),
    }


def drift_report(stats: dict, baseline: dict | None, z_limit: float = 3.0) -> dict | None:
    if not baseline:
        return None
    z = {}
    for name in STAT_NAMES:
        b = baseline["stats"][name]
        z[name] = round((stats[name] - b["mean"]) / b["std"], 3) if b["std"] > 0 else 0.0
    max_abs = max(abs(v) for v in z.values())
    return {
        "stats": {k: round(v, 8) for k, v in stats.items()},
        "z_scores": z,
        "max_abs_z": max_abs,
        "drift": bool(max_abs > z_limit),
    }
