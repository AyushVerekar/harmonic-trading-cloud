from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schemas import TradeOutcome, TradePlan

EPSILON = 1e-9


def compute_risk_reward(entry: float, stop_loss: float, target: float) -> float:
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    if risk <= EPSILON:
        return 0.0
    return reward / risk


def build_atr_trade_plan(
    entry: float,
    atr: float,
    direction: str,
    stop_atr_multiple: float,
    primary_target_atr_multiple: float,
    secondary_target_atr_multiple: float,
    source: str,
    timeframe: str,
    confidence: float,
    metadata: dict[str, Any] | None = None,
) -> TradePlan:
    atr = max(float(atr), EPSILON)
    stop_distance = atr * stop_atr_multiple
    target_1_distance = atr * primary_target_atr_multiple
    target_2_distance = atr * secondary_target_atr_multiple

    if direction == "long":
        stop_loss = entry - stop_distance
        target_1 = entry + target_1_distance
        target_2 = entry + target_2_distance
    else:
        stop_loss = entry + stop_distance
        target_1 = entry - target_1_distance
        target_2 = entry - target_2_distance

    return TradePlan(
        direction=direction,
        entry=float(entry),
        stop_loss=float(stop_loss),
        target_1=float(target_1),
        target_2=float(target_2),
        risk_reward=float(compute_risk_reward(entry, stop_loss, target_1)),
        source=source,
        timeframe=timeframe,
        confidence=float(confidence),
        metadata=metadata or {},
    )


def simulate_trade(
    master_frame: pd.DataFrame,
    entry_index: int,
    plan: TradePlan,
    max_hold_bars: int,
    timestamp: Any,
) -> TradeOutcome:
    entry = float(plan.entry)
    stop_loss = float(plan.stop_loss)
    target_1 = float(plan.target_1)
    target_2 = float(plan.target_2)
    risk_unit = abs(entry - stop_loss) + EPSILON
    max_index = min(int(entry_index + max_hold_bars), len(master_frame) - 1)

    for bars_held, frame_index in enumerate(range(entry_index + 1, max_index + 1), start=1):
        candle = master_frame.iloc[frame_index]
        high = float(candle["high"])
        low = float(candle["low"])

        if plan.direction == "long":
            stop_hit = low <= stop_loss
            target_1_hit = high >= target_1
            target_2_hit = high >= target_2
        else:
            stop_hit = high >= stop_loss
            target_1_hit = low <= target_1
            target_2_hit = low <= target_2

        if stop_hit and (target_1_hit or target_2_hit):
            return TradeOutcome(
                timestamp=timestamp,
                action="buy" if plan.direction == "long" else "sell",
                status="sl_hit",
                reward_r=-1.0,
                bars_held=bars_held,
                entry=entry,
                exit_price=stop_loss,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                source=plan.source,
                timeframe=plan.timeframe,
                confidence=plan.confidence,
                metadata={**plan.metadata, "exit_index": frame_index},
            )

        if stop_hit:
            return TradeOutcome(
                timestamp=timestamp,
                action="buy" if plan.direction == "long" else "sell",
                status="sl_hit",
                reward_r=-1.0,
                bars_held=bars_held,
                entry=entry,
                exit_price=stop_loss,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                source=plan.source,
                timeframe=plan.timeframe,
                confidence=plan.confidence,
                metadata={**plan.metadata, "exit_index": frame_index},
            )

        if target_2_hit:
            return TradeOutcome(
                timestamp=timestamp,
                action="buy" if plan.direction == "long" else "sell",
                status="tp2_hit",
                reward_r=float(compute_risk_reward(entry, stop_loss, target_2)),
                bars_held=bars_held,
                entry=entry,
                exit_price=target_2,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                source=plan.source,
                timeframe=plan.timeframe,
                confidence=plan.confidence,
                metadata={**plan.metadata, "exit_index": frame_index},
            )

        if target_1_hit:
            return TradeOutcome(
                timestamp=timestamp,
                action="buy" if plan.direction == "long" else "sell",
                status="tp1_hit",
                reward_r=float(compute_risk_reward(entry, stop_loss, target_1)),
                bars_held=bars_held,
                entry=entry,
                exit_price=target_1,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                source=plan.source,
                timeframe=plan.timeframe,
                confidence=plan.confidence,
                metadata={**plan.metadata, "exit_index": frame_index},
            )

    final_candle = master_frame.iloc[max_index]
    exit_price = float(final_candle["close"])
    directional_move = exit_price - entry if plan.direction == "long" else entry - exit_price
    reward_r = directional_move / risk_unit

    return TradeOutcome(
        timestamp=timestamp,
        action="buy" if plan.direction == "long" else "sell",
        status="timeout",
        reward_r=float(reward_r),
        bars_held=int(max_index - entry_index),
        entry=entry,
        exit_price=exit_price,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        source=plan.source,
        timeframe=plan.timeframe,
        confidence=plan.confidence,
        metadata={**plan.metadata, "exit_index": int(max_index)},
    )


def make_skip_outcome(timestamp: Any) -> TradeOutcome:
    return TradeOutcome(
        timestamp=timestamp,
        action="skip",
        status="skip",
        reward_r=0.0,
        bars_held=0,
        entry=np.nan,
        exit_price=np.nan,
        stop_loss=np.nan,
        target_1=np.nan,
        target_2=np.nan,
        source="rl",
        timeframe="event",
        confidence=0.0,
        metadata={},
    )
