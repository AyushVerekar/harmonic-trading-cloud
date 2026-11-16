import os

os.environ["API_KEY"] = "test-key"
os.environ["MODEL_VERSION"] = "test"

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from service.app import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def make_candles(n=150, seed=0, vol=0.001, start=2000.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(0, vol, n)))
    times = pd.date_range("2025-01-01", periods=n, freq="h")
    out = []
    for t, c in zip(times, close):
        o = c * (1 + rng.normal(0, vol / 2))
        h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
        lo = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
        out.append({"time": t.strftime("%Y-%m-%d %H:%M:%S"), "open": o, "high": h, "low": lo, "close": c})
    return out


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_version"] == "test"


def test_predict_requires_key():
    r = client.post("/predict", json={"candles": make_candles()})
    assert r.status_code == 401


def test_predict_ok():
    r = client.post("/predict", json={"candles": make_candles()}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["signal"] in {"buy", "sell", "skip"}
    assert "latency_ms" in body


def test_too_few_candles_rejected():
    r = client.post("/predict", json={"candles": make_candles(10)}, headers=HEADERS)
    assert r.status_code == 422


def test_drift_check_flags_shock():
    calm = client.post("/drift/check", json={"candles": make_candles(200)}, headers=HEADERS)
    assert calm.status_code in (200, 503)
    if calm.status_code == 200:
        wild = client.post("/drift/check", json={"candles": make_candles(200, vol=0.02)}, headers=HEADERS)
        assert wild.json()["max_abs_z"] >= calm.json()["max_abs_z"]
