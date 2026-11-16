from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import NeuralConfig
from ..mlp import MLPClassifier
from ..schemas import TradingSignal
from ..simulation import build_atr_trade_plan, simulate_trade

LABEL_SKIP = 0
LABEL_LONG = 1
LABEL_SHORT = 2


class NonHarmonicNeuralDetector:
    def __init__(self, config: NeuralConfig, master_timeframe: str) -> None:
        self.config = config
        self.master_timeframe = master_timeframe
        self.atr_column = f"{master_timeframe}_atr"
        self.model: MLPClassifier | None = None
        self.feature_columns: list[str] = []
        self.training_summary: dict[str, object] = {}

    def _candidate_indices(
        self,
        total_rows: int,
        stride: int,
        max_samples: int | None = None,
    ) -> np.ndarray:
        indices = np.arange(
            self.config.window_size,
            total_rows - self.config.forecast_horizon,
            stride,
            dtype=np.int64,
        )

        if max_samples and len(indices) > max_samples:
            selector = np.linspace(0, len(indices) - 1, num=max_samples, dtype=np.int64)
            indices = indices[selector]

        return indices

    def _build_window_batch(
        self,
        feature_values: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        window_size = self.config.window_size
        feature_count = feature_values.shape[1]
        batch = np.empty((len(indices), window_size * feature_count), dtype=np.float32)

        for row_index, master_index in enumerate(indices):
            batch[row_index] = feature_values[master_index - window_size : master_index].reshape(-1)

        return batch

    def _label_index(self, feature_frame: pd.DataFrame, master_index: int) -> tuple[int, float, float]:
        row = feature_frame.iloc[master_index]
        entry = float(row["close"])
        atr = float(row[self.atr_column])
        timestamp = row["time"]

        long_plan = build_atr_trade_plan(
            entry=entry,
            atr=atr,
            direction="long",
            stop_atr_multiple=self.config.stop_atr_multiple,
            primary_target_atr_multiple=self.config.primary_target_atr_multiple,
            secondary_target_atr_multiple=self.config.secondary_target_atr_multiple,
            source="non_harmonic_neural_net",
            timeframe=self.master_timeframe,
            confidence=1.0,
            metadata={},
        )
        short_plan = build_atr_trade_plan(
            entry=entry,
            atr=atr,
            direction="short",
            stop_atr_multiple=self.config.stop_atr_multiple,
            primary_target_atr_multiple=self.config.primary_target_atr_multiple,
            secondary_target_atr_multiple=self.config.secondary_target_atr_multiple,
            source="non_harmonic_neural_net",
            timeframe=self.master_timeframe,
            confidence=1.0,
            metadata={},
        )

        long_outcome = simulate_trade(
            master_frame=feature_frame,
            entry_index=master_index,
            plan=long_plan,
            max_hold_bars=self.config.forecast_horizon,
            timestamp=timestamp,
        )
        short_outcome = simulate_trade(
            master_frame=feature_frame,
            entry_index=master_index,
            plan=short_plan,
            max_hold_bars=self.config.forecast_horizon,
            timestamp=timestamp,
        )

        if max(long_outcome.reward_r, short_outcome.reward_r) <= 0.25:
            return LABEL_SKIP, float(long_outcome.reward_r), float(short_outcome.reward_r)

        if long_outcome.reward_r >= short_outcome.reward_r:
            return LABEL_LONG, float(long_outcome.reward_r), float(short_outcome.reward_r)

        return LABEL_SHORT, float(long_outcome.reward_r), float(short_outcome.reward_r)

    def fit(self, feature_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, object]:
        self.feature_columns = feature_columns
        feature_values = feature_frame[feature_columns].to_numpy(dtype=np.float32)
        candidate_indices = self._candidate_indices(
            total_rows=len(feature_frame),
            stride=self.config.training_stride,
            max_samples=self.config.max_training_samples,
        )
        if len(candidate_indices) == 0:
            raise RuntimeError(
                "Not enough rows to build neural-network training windows. "
                "Increase the available history or lower the window/horizon settings."
            )

        training_matrix = self._build_window_batch(feature_values, candidate_indices)
        labels = np.empty(len(candidate_indices), dtype=np.int64)
        reward_snapshots = np.empty((len(candidate_indices), 2), dtype=np.float32)

        for row_index, master_index in enumerate(candidate_indices):
            label, long_reward, short_reward = self._label_index(feature_frame, int(master_index))
            labels[row_index] = label
            reward_snapshots[row_index] = np.array([long_reward, short_reward], dtype=np.float32)

        label_counts = Counter(labels.tolist())
        class_counts = np.array([label_counts.get(label, 1) for label in range(3)], dtype=np.float32)
        class_weights = class_counts.sum() / class_counts
        class_weights = class_weights / class_weights.mean()

        self.model = MLPClassifier(
            input_dim=training_matrix.shape[1],
            hidden_sizes=self.config.hidden_sizes,
            output_dim=3,
            learning_rate=self.config.learning_rate,
            l2=self.config.l2,
            seed=self.config.seed,
        )
        self.model.fit(
            features=training_matrix,
            labels=labels,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weights=class_weights,
        )

        self.training_summary = {
            "samples": int(len(candidate_indices)),
            "label_distribution": {str(label): int(count) for label, count in label_counts.items()},
            "avg_long_reward": float(reward_snapshots[:, 0].mean()),
            "avg_short_reward": float(reward_snapshots[:, 1].mean()),
            "final_loss": float(self.model.loss_history[-1]) if self.model.loss_history else 0.0,
        }
        return self.training_summary

    def generate_signals(self, feature_frame: pd.DataFrame, batch_size: int = 4096) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() must be called before generate_signals().")

        feature_values = feature_frame[self.feature_columns].to_numpy(dtype=np.float32)
        candidate_indices = self._candidate_indices(
            total_rows=len(feature_frame),
            stride=self.config.inference_stride,
            max_samples=None,
        )
        if len(candidate_indices) == 0:
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
                    "long_probability",
                    "short_probability",
                    "skip_probability",
                ]
            )

        signals: list[dict[str, object]] = []
        for batch_start in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[batch_start : batch_start + batch_size]
            window_batch = self._build_window_batch(feature_values, batch_indices)
            probabilities = self.model.predict_proba(window_batch)

            for row_position, master_index in enumerate(batch_indices):
                predicted_label = int(probabilities[row_position].argmax())
                confidence = float(probabilities[row_position, predicted_label])
                if predicted_label == LABEL_SKIP or confidence < self.config.min_signal_confidence:
                    continue

                row = feature_frame.iloc[int(master_index)]
                direction = "long" if predicted_label == LABEL_LONG else "short"
                atr = float(row[self.atr_column])
                plan = build_atr_trade_plan(
                    entry=float(row["close"]),
                    atr=atr,
                    direction=direction,
                    stop_atr_multiple=self.config.stop_atr_multiple,
                    primary_target_atr_multiple=self.config.primary_target_atr_multiple,
                    secondary_target_atr_multiple=self.config.secondary_target_atr_multiple,
                    source="non_harmonic_neural_net",
                    timeframe=self.master_timeframe,
                    confidence=confidence,
                    metadata={
                        "long_probability": float(probabilities[row_position, LABEL_LONG]),
                        "short_probability": float(probabilities[row_position, LABEL_SHORT]),
                        "skip_probability": float(probabilities[row_position, LABEL_SKIP]),
                        "signal_index": int(master_index),
                        "window_start_index": int(master_index - self.config.window_size),
                        "window_end_index": int(master_index),
                        "pattern_class": "long_setup" if direction == "long" else "short_setup",
                    },
                )

                signal = TradingSignal(
                    timestamp=row["time"],
                    source=plan.source,
                    timeframe=plan.timeframe,
                    direction=plan.direction,
                    confidence=plan.confidence,
                    risk_reward=plan.risk_reward,
                    entry=plan.entry,
                    stop_loss=plan.stop_loss,
                    target_1=plan.target_1,
                    target_2=plan.target_2,
                    metadata=plan.metadata,
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
                    "long_probability",
                    "short_probability",
                    "skip_probability",
                ]
            )

        return pd.DataFrame(signals).sort_values("timestamp").reset_index(drop=True)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("fit() must be called before save().")
        self.model.save(path)
