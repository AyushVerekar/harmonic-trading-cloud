"""Build baseline window statistics from historical H1 candles."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import load_and_clean_csv  # noqa: E402
from service.drift import STAT_NAMES, window_stats  # noqa: E402

WINDOW, STEP = 200, 50
df = load_and_clean_csv(str(ROOT / "data" / "xauusd_h1.csv")).reset_index(drop=True)
rows = [window_stats(df.iloc[i:i + WINDOW]) for i in range(0, len(df) - WINDOW, STEP)]
baseline = {
    "source": "data/xauusd_h1.csv",
    "window": WINDOW,
    "step": STEP,
    "n_windows": len(rows),
    "stats": {
        n: {"mean": float(np.mean([r[n] for r in rows])), "std": float(np.std([r[n] for r in rows]))}
        for n in STAT_NAMES
    },
}
out = ROOT / "service" / "baseline_stats.json"
out.write_text(json.dumps(baseline, indent=2))
print("wrote", out, "from", len(rows), "windows")
