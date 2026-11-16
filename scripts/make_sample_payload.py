"""Create sample_payload.json (real window with a recent pattern) and drift_payload.json (5x volatility)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime import engine, load_and_clean_csv  # noqa: E402

WINDOW, STEP, RECENT, MAX_WINDOWS = 200, 25, 20, 400
df = load_and_clean_csv(str(ROOT / "data" / "xauusd_h1.csv")).reset_index(drop=True)


def to_candles(w):
    return [
        {"time": t.strftime("%Y-%m-%d %H:%M:%S"), "open": float(o), "high": float(h), "low": float(lo), "close": float(c)}
        for t, o, h, lo, c in zip(w["time"], w["open"], w["high"], w["low"], w["close"])
    ]


chosen, fallback = None, None
for k, end in enumerate(range(len(df), WINDOW, -STEP)):
    if k > MAX_WINDOWS:
        break
    w = df.iloc[end - WINDOW:end].reset_index(drop=True)
    pats = engine.detect(w)
    if pats and fallback is None:
        fallback = (w, pats)
    if any(p["index"] >= WINDOW - RECENT for p in pats):
        chosen = (w, pats)
        break
chosen = chosen or fallback
if chosen is None:
    sys.exit("No patterns found in the scanned windows. Paste this message to Claude.")

w, pats = chosen
(ROOT / "sample_payload.json").write_text(json.dumps({"candles": to_candles(w)}))

base, k = float(w["close"].iloc[0]), 5.0
s = w.copy()
for col in ("open", "high", "low", "close"):
    s[col] = base * (s[col] / base) ** k
(ROOT / "drift_payload.json").write_text(json.dumps({"candles": to_candles(s)}))

print("patterns in chosen window:", len(pats))
print("directions seen:", sorted({str(p['direction']) for p in pats}))
print("example:", pats[0])
