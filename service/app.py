"""Harmonic Signal API: serves the Harmonic-Trading detector over REST."""
from __future__ import annotations

import hmac
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from harmonic.pattern_engine import HarmonicEngine
from service.drift import drift_report, window_stats

ROOT = Path(__file__).resolve().parent.parent
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
API_KEY = os.getenv("API_KEY", "")
RECENT_BARS = int(os.getenv("RECENT_BARS", "20"))
DRIFT_Z_LIMIT = float(os.getenv("DRIFT_Z_LIMIT", "3.0"))
PATTERNS = ["gartley", "bat", "butterfly", "crab", "shark"]
STARTED = time.time()

engine = HarmonicEngine(PATTERNS)
_baseline_path = ROOT / "service" / "baseline_stats.json"
BASELINE = json.loads(_baseline_path.read_text()) if _baseline_path.exists() else None

app = FastAPI(title="Harmonic Signal API", version=MODEL_VERSION)

BUY = {"bull", "bullish", "buy", "long", "up"}
SELL = {"bear", "bearish", "sell", "short", "down"}


def log(event: str, severity: str = "INFO", **fields) -> None:
    """One JSON line per event; Cloud Run turns it into a structured Cloud Logging entry."""
    record = {"severity": severity, "event": event, "model_version": MODEL_VERSION, **fields}
    print(json.dumps(record, default=str), flush=True)


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


class Candle(BaseModel):
    time: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)


class CandleWindow(BaseModel):
    candles: List[Candle] = Field(min_length=50, max_length=5000)


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and not hmac.compare_digest((x_api_key or "").encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def to_frame(req: CandleWindow) -> pd.DataFrame:
    df = pd.DataFrame([c.model_dump() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise HTTPException(status_code=422, detail="unparseable 'time' value")
    if (df["high"] < df["low"]).any():
        raise HTTPException(status_code=422, detail="high must be >= low")
    return df.reset_index(drop=True)


def action_for(direction) -> str:
    d = str(direction).strip().lower()
    return "buy" if d in BUY else "sell" if d in SELL else "skip"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "patterns": PATTERNS,
        "baseline_loaded": BASELINE is not None,
        "uptime_s": round(time.time() - STARTED, 1),
    }


@app.post("/predict", dependencies=[Depends(require_key)])
def predict(req: CandleWindow):
    t0 = time.perf_counter()
    df = to_frame(req)
    try:
        patterns = jsonable(engine.detect(df))
    except Exception as exc:  # detector failure must not crash the service
        log("predict_error", severity="ERROR", error=str(exc))
        raise HTTPException(status_code=500, detail="detector failed")
    n = len(df)
    recent = [p for p in patterns if p.get("index", -1) >= n - RECENT_BARS]
    best = max(recent, key=lambda p: p.get("confidence", 0)) if recent else None
    signal = action_for(best.get("direction")) if best else "skip"
    drift = drift_report(window_stats(df), BASELINE, DRIFT_Z_LIMIT)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    log(
        "predict",
        n_candles=n,
        patterns_found=len(patterns),
        recent_patterns=len(recent),
        signal=signal,
        drift_flag=bool(drift and drift["drift"]),
        drift_max_abs_z=drift["max_abs_z"] if drift else None,
        latency_ms=latency_ms,
    )
    return {
        "model_version": MODEL_VERSION,
        "signal": signal,
        "best_pattern": best,
        "patterns_found": len(patterns),
        "recent_patterns": recent[:10],
        "n_candles": n,
        "recent_bars_window": RECENT_BARS,
        "drift": drift,
        "latency_ms": latency_ms,
    }


@app.post("/drift/check", dependencies=[Depends(require_key)])
def drift_check(req: CandleWindow):
    df = to_frame(req)
    report = drift_report(window_stats(df), BASELINE, DRIFT_Z_LIMIT)
    if report is None:
        raise HTTPException(status_code=503, detail="baseline not loaded")
    log("drift_check", drift_flag=report["drift"], drift_max_abs_z=report["max_abs_z"])
    return {"model_version": MODEL_VERSION, **report}


@lru_cache(maxsize=1)
def _lifetime_summary():
    path = ROOT / "harmonic_lifetime_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    g = (
        df.groupby(["timeframe", "pattern"])
        .agg(count=("pattern", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
        .round(4)
    )
    return {"total_patterns": int(len(df)), "by_timeframe_pattern": jsonable(g.to_dict(orient="records"))}


@app.get("/patterns/summary")
def patterns_summary():
    summary = _lifetime_summary()
    if summary is None:
        raise HTTPException(status_code=404, detail="harmonic_lifetime_data.csv not found")
    return summary


@app.get("/metrics/summary")
def metrics_summary():
    path = ROOT / "artifacts" / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no metrics.json yet; run the batch pipeline")
    return json.loads(path.read_text())
