from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig, TIMEFRAME_MINUTES
from .data import load_market_data
from .detectors import HarmonicSignalDetector, NonHarmonicNeuralDetector
from .features import build_feature_frame
from .feedback import build_detector_feedback, compute_trade_metrics, ensure_output_dir, save_json
from .pattern_gallery import PatternGalleryBuilder
from .portfolio import PortfolioManager
from .rl import ContextualBanditAgent
from .schemas import TradeOutcome, TradePlan
from .simulation import make_skip_outcome, simulate_trade

EPSILON = 1e-9


class TradingResearchPipeline:
    def __init__(self, repo_root: Path, config: PipelineConfig | None = None) -> None:
        self.repo_root = Path(repo_root)
        self.config = config or PipelineConfig()
        self.output_dir = self.repo_root / self.config.artifacts.output_dir
        self.master_atr_column = f"{self.config.data.master_timeframe}_atr"
        self.signal_prefixes: list[str] = []

    def _split_cutoffs(self, feature_frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        train_end_index = int(len(feature_frame) * self.config.backtest.train_ratio)
        validation_end_index = int(
            len(feature_frame)
            * (self.config.backtest.train_ratio + self.config.backtest.validation_ratio)
        )
        train_end_index = min(max(train_end_index, 1), len(feature_frame) - 2)
        validation_end_index = min(max(validation_end_index, train_end_index + 1), len(feature_frame) - 1)
        return (
            pd.Timestamp(feature_frame.iloc[train_end_index]["time"]),
            pd.Timestamp(feature_frame.iloc[validation_end_index]["time"]),
        )

    def _load_feedback_lookup(self) -> dict[str, dict[str, float]]:
        feedback_path = self.output_dir / self.config.artifacts.detector_feedback_file
        if not feedback_path.exists():
            return {}
        with feedback_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _build_event_timeline(
        self,
        feature_frame: pd.DataFrame,
        harmonic_signals: pd.DataFrame,
        neural_signals: pd.DataFrame,
    ) -> pd.DataFrame:
        timestamp_series: list[pd.Series] = []
        for signal_frame in [harmonic_signals, neural_signals]:
            if not signal_frame.empty:
                timestamp_series.append(signal_frame["timestamp"])

        if not timestamp_series:
            return pd.DataFrame()

        unique_times = pd.Index(pd.concat(timestamp_series).drop_duplicates().sort_values())
        event_frame = pd.DataFrame({"time": unique_times})
        event_frame = pd.merge_asof(
            event_frame.sort_values("time"),
            feature_frame.sort_values("time"),
            on="time",
            direction="backward",
        )
        return event_frame.dropna(subset=["master_index"]).reset_index(drop=True)

    def build_state_frame(
        self,
        event_frame: pd.DataFrame,
        market_feature_columns: list[str],
        harmonic_signals: pd.DataFrame,
        neural_signals: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        event_frame = event_frame.copy()
        state_feature_columns = list(market_feature_columns)
        self.signal_prefixes = []

        for timeframe in self.config.data.timeframes:
            prefix = f"harmonic_{timeframe}"
            self.signal_prefixes.append(prefix)
            timeframe_signals = harmonic_signals.loc[harmonic_signals["timeframe"] == timeframe].copy()
            event_frame, slot_features = self._merge_signal_slot(
                event_frame=event_frame,
                signal_frame=timeframe_signals,
                prefix=prefix,
                active_minutes=self.config.harmonic.active_minutes[timeframe],
                extra_numeric_columns=["raw_confidence"],
                extra_text_columns=["pattern"],
            )
            state_feature_columns.extend(slot_features)

        nn_active_minutes = TIMEFRAME_MINUTES[self.config.data.master_timeframe] * max(
            self.config.neural.inference_stride,
            6,
        )
        self.signal_prefixes.append("nn")
        event_frame, nn_slot_features = self._merge_signal_slot(
            event_frame=event_frame,
            signal_frame=neural_signals,
            prefix="nn",
            active_minutes=nn_active_minutes,
            extra_numeric_columns=["long_probability", "short_probability", "skip_probability"],
            extra_text_columns=[],
        )
        state_feature_columns.extend(nn_slot_features)

        long_confidence_columns = []
        short_confidence_columns = []
        direction_strength_columns = []
        for prefix in self.signal_prefixes:
            long_column = f"{prefix}_long_confidence"
            short_column = f"{prefix}_short_confidence"
            strength_column = f"{prefix}_direction_strength"

            event_frame[long_column] = np.where(
                event_frame[f"{prefix}_direction_code"] > 0,
                event_frame[f"{prefix}_confidence"],
                0.0,
            )
            event_frame[short_column] = np.where(
                event_frame[f"{prefix}_direction_code"] < 0,
                event_frame[f"{prefix}_confidence"],
                0.0,
            )
            event_frame[strength_column] = (
                event_frame[f"{prefix}_direction_code"] * event_frame[f"{prefix}_confidence"]
            )
            long_confidence_columns.append(long_column)
            short_confidence_columns.append(short_column)
            direction_strength_columns.append(strength_column)

        event_frame["long_signal_count"] = sum(
            (event_frame[f"{prefix}_direction_code"] > 0).astype(np.float32) for prefix in self.signal_prefixes
        )
        event_frame["short_signal_count"] = sum(
            (event_frame[f"{prefix}_direction_code"] < 0).astype(np.float32) for prefix in self.signal_prefixes
        )
        event_frame["best_long_confidence"] = event_frame[long_confidence_columns].max(axis=1)
        event_frame["best_short_confidence"] = event_frame[short_confidence_columns].max(axis=1)
        event_frame["detector_bias"] = event_frame[direction_strength_columns].sum(axis=1)
        event_frame["consensus_strength"] = event_frame["best_long_confidence"] - event_frame["best_short_confidence"]

        state_feature_columns.extend(
            [
                "long_signal_count",
                "short_signal_count",
                "best_long_confidence",
                "best_short_confidence",
                "detector_bias",
                "consensus_strength",
            ]
        )
        return event_frame, state_feature_columns

    def _merge_signal_slot(
        self,
        event_frame: pd.DataFrame,
        signal_frame: pd.DataFrame,
        prefix: str,
        active_minutes: int,
        extra_numeric_columns: list[str] | None = None,
        extra_text_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        extra_numeric_columns = extra_numeric_columns or []
        extra_text_columns = extra_text_columns or []

        slot_state_columns = [
            f"{prefix}_direction_code",
            f"{prefix}_confidence",
            f"{prefix}_risk_reward",
            f"{prefix}_freshness",
            f"{prefix}_entry_distance_atr",
            f"{prefix}_stop_distance_atr",
            f"{prefix}_target_distance_atr",
        ]
        slot_state_columns.extend(f"{prefix}_{column}" for column in extra_numeric_columns)

        if signal_frame.empty:
            event_frame[f"{prefix}_signal_time"] = pd.NaT
            event_frame[f"{prefix}_direction"] = "none"
            event_frame[f"{prefix}_confidence"] = 0.0
            event_frame[f"{prefix}_risk_reward"] = 0.0
            event_frame[f"{prefix}_entry"] = np.nan
            event_frame[f"{prefix}_stop_loss"] = np.nan
            event_frame[f"{prefix}_target_1"] = np.nan
            event_frame[f"{prefix}_target_2"] = np.nan
            event_frame[f"{prefix}_source"] = ""
            event_frame[f"{prefix}_timeframe"] = ""
            event_frame[f"{prefix}_direction_code"] = 0.0
            event_frame[f"{prefix}_freshness"] = 0.0
            event_frame[f"{prefix}_entry_distance_atr"] = 0.0
            event_frame[f"{prefix}_stop_distance_atr"] = 0.0
            event_frame[f"{prefix}_target_distance_atr"] = 0.0
            for column in extra_numeric_columns:
                event_frame[f"{prefix}_{column}"] = 0.0
            for column in extra_text_columns:
                event_frame[f"{prefix}_{column}"] = ""
            return event_frame, slot_state_columns

        keep_columns = [
            "timestamp",
            "direction",
            "confidence",
            "risk_reward",
            "entry",
            "stop_loss",
            "target_1",
            "target_2",
            "source",
            "timeframe",
        ]
        keep_columns.extend(extra_numeric_columns)
        keep_columns.extend(extra_text_columns)

        slot_frame = (
            signal_frame[keep_columns]
            .sort_values(["timestamp", "confidence"])
            .groupby("timestamp", as_index=False)
            .tail(1)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        rename_map = {
            "timestamp": f"{prefix}_signal_time",
            "direction": f"{prefix}_direction",
            "confidence": f"{prefix}_confidence",
            "risk_reward": f"{prefix}_risk_reward",
            "entry": f"{prefix}_entry",
            "stop_loss": f"{prefix}_stop_loss",
            "target_1": f"{prefix}_target_1",
            "target_2": f"{prefix}_target_2",
            "source": f"{prefix}_source",
            "timeframe": f"{prefix}_timeframe",
        }
        rename_map.update({column: f"{prefix}_{column}" for column in extra_numeric_columns + extra_text_columns})
        slot_frame = slot_frame.rename(columns=rename_map)

        merged = pd.merge_asof(
            event_frame.sort_values("time"),
            slot_frame.sort_values(f"{prefix}_signal_time"),
            left_on="time",
            right_on=f"{prefix}_signal_time",
            direction="backward",
        )
        merged = merged.copy()

        age_minutes = (
            merged["time"] - merged[f"{prefix}_signal_time"]
        ).dt.total_seconds() / 60.0
        merged[f"{prefix}_freshness"] = 1.0 - (age_minutes / max(active_minutes, 1))
        inactive_mask = merged[f"{prefix}_signal_time"].isna() | (age_minutes > active_minutes)

        merged[f"{prefix}_direction"] = merged[f"{prefix}_direction"].where(~inactive_mask, other="none")
        merged[f"{prefix}_confidence"] = merged[f"{prefix}_confidence"].where(~inactive_mask, other=0.0)
        merged[f"{prefix}_risk_reward"] = merged[f"{prefix}_risk_reward"].where(~inactive_mask, other=0.0)
        merged[f"{prefix}_freshness"] = merged[f"{prefix}_freshness"].where(~inactive_mask, other=0.0)
        merged[f"{prefix}_source"] = merged[f"{prefix}_source"].where(~inactive_mask, other="")
        merged[f"{prefix}_timeframe"] = merged[f"{prefix}_timeframe"].where(~inactive_mask, other="")

        for column in ["entry", "stop_loss", "target_1", "target_2"]:
            merged[f"{prefix}_{column}"] = merged[f"{prefix}_{column}"].where(~inactive_mask, other=np.nan)

        for column in extra_numeric_columns:
            merged[f"{prefix}_{column}"] = merged[f"{prefix}_{column}"].where(~inactive_mask, other=0.0)
        for column in extra_text_columns:
            merged[f"{prefix}_{column}"] = merged[f"{prefix}_{column}"].where(~inactive_mask, other="")

        merged[f"{prefix}_direction_code"] = merged[f"{prefix}_direction"].map({"long": 1.0, "short": -1.0}).fillna(0.0)
        atr = merged[self.master_atr_column].abs() + EPSILON
        merged[f"{prefix}_entry_distance_atr"] = np.where(
            inactive_mask,
            0.0,
            (merged[f"{prefix}_entry"] - merged["close"]) / atr,
        )
        merged[f"{prefix}_stop_distance_atr"] = np.where(
            inactive_mask,
            0.0,
            (merged["close"] - merged[f"{prefix}_stop_loss"]).abs() / atr,
        )
        merged[f"{prefix}_target_distance_atr"] = np.where(
            inactive_mask,
            0.0,
            (merged[f"{prefix}_target_1"] - merged["close"]).abs() / atr,
        )

        return merged.reset_index(drop=True), slot_state_columns

    def _choose_plan(self, row: pd.Series, direction: str) -> TradePlan | None:
        candidates: list[TradePlan] = []

        for prefix in self.signal_prefixes:
            if row.get(f"{prefix}_direction") != direction:
                continue

            confidence = float(row.get(f"{prefix}_confidence", 0.0) or 0.0)
            entry = row.get(f"{prefix}_entry")
            if confidence <= 0.0 or pd.isna(entry):
                continue

            metadata: dict[str, Any] = {}
            pattern = row.get(f"{prefix}_pattern", "")
            if isinstance(pattern, str) and pattern:
                metadata["pattern"] = pattern

            for probability_column in ["long_probability", "short_probability", "skip_probability"]:
                value = row.get(f"{prefix}_{probability_column}")
                if value is not None and not pd.isna(value):
                    metadata[probability_column] = float(value)

            candidates.append(
                TradePlan(
                    direction=direction,
                    entry=float(entry),
                    stop_loss=float(row[f"{prefix}_stop_loss"]),
                    target_1=float(row[f"{prefix}_target_1"]),
                    target_2=float(row[f"{prefix}_target_2"]),
                    risk_reward=float(row[f"{prefix}_risk_reward"]),
                    source=str(row[f"{prefix}_source"]),
                    timeframe=str(row[f"{prefix}_timeframe"]),
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        if not candidates:
            return None

        return max(candidates, key=lambda plan: (plan.confidence, plan.risk_reward))

    def _plan_payload(self, action: str, plan: TradePlan | None, reward: float) -> dict[str, Any]:
        payload = {
            f"reward_{action}": float(reward),
            f"{action}_plan_source": "",
            f"{action}_plan_timeframe": "",
            f"{action}_plan_confidence": 0.0,
            f"{action}_plan_entry": np.nan,
            f"{action}_plan_stop_loss": np.nan,
            f"{action}_plan_target_1": np.nan,
            f"{action}_plan_target_2": np.nan,
            f"{action}_plan_risk_reward": 0.0,
            f"{action}_plan_pattern": "",
        }

        if plan is None:
            return payload

        payload.update(
            {
                f"{action}_plan_source": plan.source,
                f"{action}_plan_timeframe": plan.timeframe,
                f"{action}_plan_confidence": float(plan.confidence),
                f"{action}_plan_entry": float(plan.entry),
                f"{action}_plan_stop_loss": float(plan.stop_loss),
                f"{action}_plan_target_1": float(plan.target_1),
                f"{action}_plan_target_2": float(plan.target_2),
                f"{action}_plan_risk_reward": float(plan.risk_reward),
                f"{action}_plan_pattern": str(plan.metadata.get("pattern", "")),
            }
        )
        return payload

    def _score_event_frame(self, event_frame: pd.DataFrame, master_frame: pd.DataFrame) -> pd.DataFrame:
        reward_rows: list[dict[str, Any]] = []

        for _, row in event_frame.iterrows():
            master_index = int(row["master_index"])
            timestamp = row["time"]

            buy_plan = self._choose_plan(row, direction="long")
            sell_plan = self._choose_plan(row, direction="short")
            buy_valid = buy_plan is not None
            sell_valid = sell_plan is not None

            if not buy_valid:
                buy_reward = -self.config.rl.invalid_action_penalty
            else:
                buy_reward = simulate_trade(
                    master_frame=master_frame,
                    entry_index=master_index,
                    plan=buy_plan,
                    max_hold_bars=self.config.backtest.max_hold_bars,
                    timestamp=timestamp,
                ).reward_r

            if not sell_valid:
                sell_reward = -self.config.rl.invalid_action_penalty
            else:
                sell_reward = simulate_trade(
                    master_frame=master_frame,
                    entry_index=master_index,
                    plan=sell_plan,
                    max_hold_bars=self.config.backtest.max_hold_bars,
                    timestamp=timestamp,
                ).reward_r

            reward_rows.append(
                {
                    "reward_skip": 0.0,
                    "mask_skip": 1.0,
                    "mask_buy": float(buy_valid),
                    "mask_sell": float(sell_valid),
                    **self._plan_payload("buy", buy_plan, buy_reward),
                    **self._plan_payload("sell", sell_plan, sell_reward),
                }
            )

        return pd.concat([event_frame.reset_index(drop=True), pd.DataFrame(reward_rows)], axis=1)

    def _row_action_mask(self, row: pd.Series, portfolio: PortfolioManager | None = None) -> np.ndarray:
        can_trade = True if portfolio is None else portfolio.can_trade()
        return np.array(
            [
                [
                    True,
                    bool(row.get("mask_buy", 0.0)) and can_trade,
                    bool(row.get("mask_sell", 0.0)) and can_trade,
                ]
            ],
            dtype=bool,
        )

    def _plan_from_row(self, row: pd.Series, action: str) -> TradePlan | None:
        source = row.get(f"{action}_plan_source", "")
        if not isinstance(source, str) or not source:
            return None

        metadata: dict[str, Any] = {}
        pattern = row.get(f"{action}_plan_pattern", "")
        if isinstance(pattern, str) and pattern:
            metadata["pattern"] = pattern

        return TradePlan(
            direction="long" if action == "buy" else "short",
            entry=float(row[f"{action}_plan_entry"]),
            stop_loss=float(row[f"{action}_plan_stop_loss"]),
            target_1=float(row[f"{action}_plan_target_1"]),
            target_2=float(row[f"{action}_plan_target_2"]),
            risk_reward=float(row[f"{action}_plan_risk_reward"]),
            source=source,
            timeframe=str(row[f"{action}_plan_timeframe"]),
            confidence=float(row[f"{action}_plan_confidence"]),
            metadata=metadata,
        )

    def _invalid_action_outcome(self, row: pd.Series, action: str) -> TradeOutcome:
        return TradeOutcome(
            timestamp=row["time"],
            action=action,
            status="invalid_action",
            reward_r=-self.config.rl.invalid_action_penalty,
            bars_held=0,
            entry=np.nan,
            exit_price=np.nan,
            stop_loss=np.nan,
            target_1=np.nan,
            target_2=np.nan,
            source="rl_policy",
            timeframe="event",
            confidence=0.0,
            metadata={"exit_index": int(row["master_index"])},
        )

    def _run_backtest(
        self,
        agent: ContextualBanditAgent,
        test_event_frame: pd.DataFrame,
        state_feature_columns: list[str],
        master_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if test_event_frame.empty:
            return pd.DataFrame()

        test_event_frame = test_event_frame.sort_values("time").reset_index(drop=True)
        event_master_indices = test_event_frame["master_index"].to_numpy(dtype=np.int64)
        trades: list[dict[str, Any]] = []
        pointer = 0
        portfolio = PortfolioManager(self.config.portfolio)

        while pointer < len(test_event_frame):
            row = test_event_frame.iloc[pointer]
            state_matrix = test_event_frame.iloc[[pointer]][state_feature_columns].to_numpy(dtype=np.float32)
            action_mask = self._row_action_mask(row, portfolio=portfolio)
            if self.config.rl.hard_mask_actions:
                scores, raw_indices, chosen_indices = agent.choose_action_with_metadata(
                    state_matrix,
                    action_mask=action_mask,
                )
            else:
                scores, raw_indices, chosen_indices = agent.choose_action_with_metadata(state_matrix)

            scores = scores[0]
            raw_action = self.config.rl.action_names[int(raw_indices[0])]
            action = self.config.rl.action_names[int(chosen_indices[0])]
            action_was_masked = raw_action != action

            if action == "skip":
                outcome = make_skip_outcome(row["time"])
                snapshot = portfolio.settle(outcome, plan=None)
                payload = outcome.to_dict()
                payload.update(
                    {
                        "event_master_index": int(row["master_index"]),
                        "policy_skip_score": float(scores[0]),
                        "policy_buy_score": float(scores[1]),
                        "policy_sell_score": float(scores[2]),
                        "raw_policy_action": raw_action,
                        "executed_policy_action": action,
                        "action_was_masked": action_was_masked,
                        "valid_buy_plan": bool(row.get("mask_buy", 0.0)),
                        "valid_sell_plan": bool(row.get("mask_sell", 0.0)),
                        "pattern": "",
                    }
                )
                payload.update(snapshot.to_dict())
                trades.append(payload)
                pointer += 1
                continue

            plan = self._plan_from_row(row, action)
            if plan is None:
                outcome = self._invalid_action_outcome(row, action)
                snapshot = portfolio.settle(outcome, plan=None)
                payload = outcome.to_dict()
                payload.update(
                    {
                        "event_master_index": int(row["master_index"]),
                        "policy_skip_score": float(scores[0]),
                        "policy_buy_score": float(scores[1]),
                        "policy_sell_score": float(scores[2]),
                        "raw_policy_action": raw_action,
                        "executed_policy_action": action,
                        "action_was_masked": action_was_masked,
                        "valid_buy_plan": bool(row.get("mask_buy", 0.0)),
                        "valid_sell_plan": bool(row.get("mask_sell", 0.0)),
                        "pattern": "",
                    }
                )
                payload.update(snapshot.to_dict())
                trades.append(payload)
                pointer += 1
                continue

            outcome = simulate_trade(
                master_frame=master_frame,
                entry_index=int(row["master_index"]),
                plan=plan,
                max_hold_bars=self.config.backtest.max_hold_bars,
                timestamp=row["time"],
            )
            snapshot = portfolio.settle(outcome, plan=plan)
            payload = outcome.to_dict()
            payload.update(
                {
                    "event_master_index": int(row["master_index"]),
                    "policy_skip_score": float(scores[0]),
                    "policy_buy_score": float(scores[1]),
                    "policy_sell_score": float(scores[2]),
                    "raw_policy_action": raw_action,
                    "executed_policy_action": action,
                    "action_was_masked": action_was_masked,
                    "valid_buy_plan": bool(row.get("mask_buy", 0.0)),
                    "valid_sell_plan": bool(row.get("mask_sell", 0.0)),
                    "pattern": str(plan.metadata.get("pattern", "")),
                }
            )
            payload.update(snapshot.to_dict())
            trades.append(payload)

            exit_index = int(outcome.metadata.get("exit_index", row["master_index"]))
            next_pointer = int(np.searchsorted(event_master_indices, exit_index + 1, side="left"))
            pointer = max(pointer + 1, next_pointer)

        return pd.DataFrame(trades)

    def run(self) -> dict[str, Any]:
        ensure_output_dir(self.output_dir)

        market_frames = load_market_data(self.config.data, repo_root=self.repo_root)
        feature_frame, market_feature_columns = build_feature_frame(
            market_frames,
            config=self.config.features,
            master_timeframe=self.config.data.master_timeframe,
        )
        train_cutoff, validation_cutoff = self._split_cutoffs(feature_frame)

        train_frame = feature_frame.loc[feature_frame["time"] <= train_cutoff].reset_index(drop=True)
        feedback_lookup = self._load_feedback_lookup()

        harmonic_detector = HarmonicSignalDetector(
            config=self.config.harmonic,
            reliability_lookup=feedback_lookup,
        )
        harmonic_signals = harmonic_detector.detect(market_frames)

        neural_detector = NonHarmonicNeuralDetector(
            config=self.config.neural,
            master_timeframe=self.config.data.master_timeframe,
        )
        neural_summary = neural_detector.fit(train_frame, feature_columns=market_feature_columns)
        neural_signals = neural_detector.generate_signals(feature_frame)

        event_frame = self._build_event_timeline(feature_frame, harmonic_signals, neural_signals)
        if event_frame.empty:
            raise RuntimeError("No detector signals were produced, so the RL state set is empty.")

        event_frame, state_feature_columns = self.build_state_frame(
            event_frame=event_frame,
            market_feature_columns=market_feature_columns,
            harmonic_signals=harmonic_signals,
            neural_signals=neural_signals,
        )

        event_frame = self._score_event_frame(event_frame, master_frame=feature_frame)
        state_feature_columns.extend(["mask_skip", "mask_buy", "mask_sell"])

        train_events = event_frame.loc[event_frame["time"] <= train_cutoff].reset_index(drop=True)
        test_events = event_frame.loc[event_frame["time"] > validation_cutoff].reset_index(drop=True)
        if train_events.empty or test_events.empty:
            raise RuntimeError("The train/test split left no RL events in one side of the timeline.")

        rl_agent = ContextualBanditAgent(
            input_dim=len(state_feature_columns),
            config=self.config.rl,
        )
        rl_summary = rl_agent.fit(
            state_matrix=train_events[state_feature_columns].to_numpy(dtype=np.float32),
            reward_table=train_events[["reward_skip", "reward_buy", "reward_sell"]].to_numpy(dtype=np.float32),
            action_mask=train_events[["mask_skip", "mask_buy", "mask_sell"]].to_numpy(dtype=bool),
        )

        backtest_frame = self._run_backtest(
            agent=rl_agent,
            test_event_frame=test_events,
            state_feature_columns=state_feature_columns,
            master_frame=feature_frame,
        )
        detector_feedback = build_detector_feedback(backtest_frame)
        trade_metrics = compute_trade_metrics(backtest_frame)

        combined_signals = pd.concat([harmonic_signals, neural_signals], ignore_index=True).sort_values(
            "timestamp"
        )
        combined_signals.to_csv(self.output_dir / self.config.artifacts.signals_file, index=False)
        event_frame.to_csv(self.output_dir / self.config.artifacts.state_file, index=False)
        backtest_frame.to_csv(self.output_dir / self.config.artifacts.trade_file, index=False)
        neural_detector.save(self.output_dir / self.config.artifacts.neural_model_file)
        rl_agent.save(self.output_dir / self.config.artifacts.rl_model_file)
        gallery_summary = PatternGalleryBuilder(
            repo_root=self.repo_root,
            pipeline_config=self.config,
            artifact_root=self.output_dir,
        ).build(
            market_frames=market_frames,
            harmonic_signals=harmonic_signals,
            neural_signals=neural_signals,
        )

        metrics = {
            "symbol": self.config.data.symbol,
            "market_rows": int(len(feature_frame)),
            "harmonic_signal_count": int(len(harmonic_signals)),
            "neural_signal_count": int(len(neural_signals)),
            "event_count": int(len(event_frame)),
            "train_event_count": int(len(train_events)),
            "test_event_count": int(len(test_events)),
            "state_feature_count": int(len(state_feature_columns)),
            "train_cutoff": str(train_cutoff),
            "validation_cutoff": str(validation_cutoff),
            "portfolio": {
                "starting_balance": float(self.config.portfolio.starting_balance),
                "base_risk_per_trade": float(self.config.portfolio.base_risk_per_trade),
                "min_risk_per_trade": float(self.config.portfolio.min_risk_per_trade),
                "max_risk_per_trade": float(self.config.portfolio.max_risk_per_trade),
            },
            "hard_mask_actions": bool(self.config.rl.hard_mask_actions),
            "neural_training": neural_summary,
            "rl_training": rl_summary,
            "backtest": trade_metrics,
            "pattern_gallery": gallery_summary,
        }
        save_json(self.output_dir / self.config.artifacts.metrics_file, metrics)
        save_json(
            self.output_dir / self.config.artifacts.detector_feedback_file,
            detector_feedback,
        )

        return metrics
