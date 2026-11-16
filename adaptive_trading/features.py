from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig

EPSILON = 1e-9


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    relative_strength = avg_gain / (avg_loss + EPSILON)
    return 100.0 - (100.0 / (1.0 + relative_strength))


def compute_market_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    prefix: str,
    include_raw_columns: bool = False,
) -> pd.DataFrame:
    frame = df.copy()
    feature_frame = pd.DataFrame({"time": frame["time"]})

    if include_raw_columns:
        for column in ["open", "high", "low", "close", "tickvol", "vol", "spread", "timeframe"]:
            if column in frame.columns:
                feature_frame[column] = frame[column]

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    open_ = frame["open"].astype(float)

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(config.atr_period).mean()

    candle_range = (high - low).replace(0.0, EPSILON)
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    feature_frame[f"{prefix}return_1"] = close.pct_change(1)
    feature_frame[f"{prefix}return_3"] = close.pct_change(3)
    feature_frame[f"{prefix}return_12"] = close.pct_change(12)
    feature_frame[f"{prefix}body_pct"] = (close - open_) / (open_.abs() + EPSILON)
    feature_frame[f"{prefix}range_pct"] = candle_range / (close.abs() + EPSILON)
    feature_frame[f"{prefix}upper_wick_pct"] = upper_wick / candle_range
    feature_frame[f"{prefix}lower_wick_pct"] = lower_wick / candle_range
    feature_frame[f"{prefix}atr"] = atr
    feature_frame[f"{prefix}atr_pct"] = atr / (close.abs() + EPSILON)
    feature_frame[f"{prefix}rsi"] = _compute_rsi(close, config.rsi_period) / 100.0
    feature_frame[f"{prefix}hour_sin"] = np.sin(2.0 * np.pi * frame["time"].dt.hour / 24.0)
    feature_frame[f"{prefix}hour_cos"] = np.cos(2.0 * np.pi * frame["time"].dt.hour / 24.0)
    feature_frame[f"{prefix}dow_sin"] = np.sin(2.0 * np.pi * frame["time"].dt.dayofweek / 7.0)
    feature_frame[f"{prefix}dow_cos"] = np.cos(2.0 * np.pi * frame["time"].dt.dayofweek / 7.0)

    if "tickvol" in frame.columns:
        tickvol = frame["tickvol"].fillna(0.0).astype(float)
        rolling_mean = tickvol.rolling(48).mean()
        rolling_std = tickvol.rolling(48).std().replace(0.0, 1.0)
        feature_frame[f"{prefix}tickvol_zscore"] = (tickvol - rolling_mean) / rolling_std

    for window in config.ema_windows:
        ema = close.ewm(span=window, adjust=False).mean()
        feature_frame[f"{prefix}close_vs_ema_{window}"] = (close - ema) / (atr + EPSILON)

    for window in config.volatility_windows:
        feature_frame[f"{prefix}volatility_{window}"] = feature_frame[f"{prefix}return_1"].rolling(window).std()

    for window in config.trend_windows:
        feature_frame[f"{prefix}trend_{window}"] = (close - close.shift(window)) / (
            atr * np.sqrt(window) + EPSILON
        )

    return feature_frame


def build_feature_frame(
    frames: dict[str, pd.DataFrame],
    config: FeatureConfig,
    master_timeframe: str,
) -> tuple[pd.DataFrame, list[str]]:
    master_features = compute_market_features(
        frames[master_timeframe],
        config=config,
        prefix=f"{master_timeframe}_",
        include_raw_columns=True,
    )

    merged = master_features.sort_values("time").reset_index(drop=True)

    for timeframe, frame in frames.items():
        if timeframe == master_timeframe:
            continue

        higher_features = compute_market_features(
            frame,
            config=config,
            prefix=f"{timeframe}_",
            include_raw_columns=False,
        )
        merged = pd.merge_asof(
            merged.sort_values("time"),
            higher_features.sort_values("time"),
            on="time",
            direction="backward",
        )

    merged["master_index"] = np.arange(len(merged), dtype=np.int64)
    merged = merged.dropna().reset_index(drop=True)
    merged["master_index"] = np.arange(len(merged), dtype=np.int64)

    excluded_columns = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "tickvol",
        "vol",
        "spread",
        "timeframe",
        "master_index",
    }
    feature_columns = [column for column in merged.columns if column not in excluded_columns]
    return merged, feature_columns
