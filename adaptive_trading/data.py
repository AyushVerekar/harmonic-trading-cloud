from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig


def _normalize_columns(columns: list[str]) -> list[str]:
    return [column.strip().lower().replace("<", "").replace(">", "") for column in columns]


def load_mt5_csv(path: Path, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = _normalize_columns(df.columns.tolist())

    if "date" in df.columns and "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format="%Y.%m.%d %H:%M:%S",
            errors="coerce",
        )
        df = df.drop(columns=["date"])

    required_columns = ["time", "open", "high", "low", "close"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    numeric_columns = [
        column
        for column in ["open", "high", "low", "close", "tickvol", "vol", "spread"]
        if column in df.columns
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = (
        df.dropna(subset=required_columns)
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    df["timeframe"] = timeframe
    return df


def load_market_data(config: DataConfig, repo_root: Path) -> dict[str, pd.DataFrame]:
    repo_root = Path(repo_root)
    frames: dict[str, pd.DataFrame] = {}

    for timeframe in config.timeframes:
        filename = config.file_map[timeframe]
        path = repo_root / config.data_dir / filename
        frames[timeframe] = load_mt5_csv(path, timeframe=timeframe)

    if config.max_master_rows:
        master_df = frames[config.master_timeframe].tail(config.max_master_rows).copy()
        master_start = master_df["time"].iloc[0] - pd.Timedelta(days=3)
        master_end = master_df["time"].iloc[-1]

        frames[config.master_timeframe] = master_df.reset_index(drop=True)
        for timeframe, frame in list(frames.items()):
            if timeframe == config.master_timeframe:
                continue
            mask = (frame["time"] >= master_start) & (frame["time"] <= master_end)
            frames[timeframe] = frame.loc[mask].reset_index(drop=True)

    return frames
