from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Unsupported value type for JSON serialization: {type(value)!r}")


def compute_trade_metrics(trade_frame: pd.DataFrame) -> dict[str, float | int]:
    if trade_frame.empty:
        return {
            "events": 0,
            "executed_trades": 0,
            "skip_count": 0,
            "win_rate": 0.0,
            "avg_reward_r": 0.0,
            "total_reward_r": 0.0,
            "profit_factor": 0.0,
            "starting_balance": 0.0,
            "ending_balance": 0.0,
            "total_pnl_amount": 0.0,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "masked_decisions": 0,
        }

    executed = trade_frame[
        trade_frame["action"].isin(["buy", "sell"]) & trade_frame["status"].ne("invalid_action")
    ].copy()
    positive_reward = executed.loc[executed["reward_r"] > 0.0, "reward_r"].sum()
    negative_reward = -executed.loc[executed["reward_r"] < 0.0, "reward_r"].sum()
    win_rate = float((executed["reward_r"] > 0.0).mean()) if len(executed) else 0.0
    starting_balance = float(trade_frame["balance_before"].iloc[0]) if "balance_before" in trade_frame.columns else 0.0
    ending_balance = (
        float(trade_frame["balance_after"].iloc[-1])
        if "balance_after" in trade_frame.columns and len(trade_frame)
        else starting_balance
    )
    total_pnl_amount = float(executed["pnl_amount"].sum()) if "pnl_amount" in executed.columns else 0.0
    max_drawdown_pct = float(trade_frame["drawdown_pct"].max()) if "drawdown_pct" in trade_frame.columns else 0.0
    masked_decisions = (
        int(trade_frame["action_was_masked"].astype(bool).sum())
        if "action_was_masked" in trade_frame.columns
        else 0
    )

    return {
        "events": int(len(trade_frame)),
        "executed_trades": int(len(executed)),
        "skip_count": int((trade_frame["action"] == "skip").sum()),
        "win_rate": win_rate,
        "avg_reward_r": float(executed["reward_r"].mean()) if len(executed) else 0.0,
        "total_reward_r": float(executed["reward_r"].sum()) if len(executed) else 0.0,
        "profit_factor": float(positive_reward / negative_reward) if negative_reward > 0 else float(positive_reward),
        "starting_balance": starting_balance,
        "ending_balance": ending_balance,
        "total_pnl_amount": total_pnl_amount,
        "return_pct": ((ending_balance - starting_balance) / starting_balance) if starting_balance > 0 else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "masked_decisions": masked_decisions,
    }


def build_detector_feedback(trade_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    if trade_frame.empty:
        return {}

    executed = trade_frame[
        trade_frame["action"].isin(["buy", "sell"]) & trade_frame["status"].ne("invalid_action")
    ].copy()
    feedback: dict[str, dict[str, float]] = {}

    if "pattern" in executed.columns:
        harmonic = executed[
            (executed["source"] == "harmonic_rule_engine") & executed["pattern"].fillna("").ne("")
        ]
        grouped = harmonic.groupby(["timeframe", "pattern"], dropna=False)
        for (timeframe, pattern), group in grouped:
            key = f"{timeframe}:{pattern}"
            feedback[key] = {
                "trades": float(len(group)),
                "win_rate": float((group["reward_r"] > 0.0).mean()),
                "avg_reward": float(group["reward_r"].mean()),
            }

    neural = executed[executed["source"] == "non_harmonic_neural_net"]
    grouped = neural.groupby(["source", "timeframe"], dropna=False)
    for (_, timeframe), group in grouped:
        key = f"non_harmonic_neural_net:{timeframe}"
        feedback[key] = {
            "trades": float(len(group)),
            "win_rate": float((group["reward_r"] > 0.0).mean()),
            "avg_reward": float(group["reward_r"].mean()),
        }

    return feedback
