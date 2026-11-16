from __future__ import annotations

import pandas as pd

from harmonic.pattern_engine import HarmonicEngine

from ..config import HarmonicConfig
from ..schemas import TradingSignal
from ..simulation import compute_risk_reward


class HarmonicSignalDetector:
    def __init__(
        self,
        config: HarmonicConfig,
        reliability_lookup: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.config = config
        self.engine = HarmonicEngine(list(config.patterns))
        self.reliability_lookup = reliability_lookup or {}

    def _calibrated_confidence(self, timeframe: str, pattern: str, raw_confidence: float) -> float:
        feedback_key = f"{timeframe}:{pattern}"
        feedback = self.reliability_lookup.get(feedback_key, {})
        win_rate = float(feedback.get("win_rate", 0.5))
        confidence = raw_confidence * (0.6 + 0.8 * win_rate)
        return max(self.config.confidence_floor, min(confidence, 0.99))

    def detect(self, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        signals: list[dict[str, object]] = []

        for timeframe, frame in frames.items():
            patterns = self.engine.detect(frame)

            for pattern in patterns:
                signal_index = int(pattern["index"])
                if signal_index >= len(frame):
                    continue

                timestamp = frame.iloc[signal_index]["time"]
                direction = "long" if pattern["direction"] == "bullish" else "short"
                entry = float(pattern["PRZ"]["mid"])
                stop_loss = float(pattern["risk"]["stop_loss"])
                target_1 = float(pattern["risk"]["tp1"])
                target_2 = float(pattern["risk"]["tp2"])
                risk_reward = float(compute_risk_reward(entry, stop_loss, target_1))

                if risk_reward <= 0.0:
                    continue

                confidence = self._calibrated_confidence(
                    timeframe=timeframe,
                    pattern=str(pattern["pattern"]),
                    raw_confidence=float(pattern["confidence"]),
                )

                signal = TradingSignal(
                    timestamp=timestamp,
                    source="harmonic_rule_engine",
                    timeframe=timeframe,
                    direction=direction,
                    confidence=confidence,
                    risk_reward=risk_reward,
                    entry=entry,
                    stop_loss=stop_loss,
                    target_1=target_1,
                    target_2=target_2,
                    metadata={
                        "pattern": pattern["pattern"],
                        "raw_confidence": float(pattern["confidence"]),
                        "d_price": float(pattern["D_price"]),
                        "signal_index": signal_index,
                    },
                )
                signals.append(signal.to_dict())

        if not signals:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "source",
                    "timeframe",
                    "direction",
                    "confidence",
                    "risk_reward",
                    "entry",
                    "stop_loss",
                    "target_1",
                    "target_2",
                    "pattern",
                    "raw_confidence",
                    "d_price",
                ]
            )

        signal_frame = pd.DataFrame(signals)
        return signal_frame.sort_values("timestamp").reset_index(drop=True)
