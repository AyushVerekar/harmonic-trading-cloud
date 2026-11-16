from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60}


@dataclass(slots=True)
class DataConfig:
    symbol: str = "XAUUSD"
    data_dir: Path = Path("data")
    file_map: dict[str, str] = field(
        default_factory=lambda: {
            "5m": "xauusd_m5.csv",
            "15m": "xauusd_m15.csv",
            "1h": "xauusd_h1.csv",
        }
    )
    master_timeframe: str = "5m"
    timeframes: tuple[str, ...] = ("5m", "15m", "1h")
    max_master_rows: int | None = None


@dataclass(slots=True)
class FeatureConfig:
    atr_period: int = 14
    rsi_period: int = 14
    ema_windows: tuple[int, ...] = (12, 48)
    volatility_windows: tuple[int, ...] = (12, 48)
    trend_windows: tuple[int, ...] = (12, 48)


@dataclass(slots=True)
class HarmonicConfig:
    patterns: tuple[str, ...] = ("gartley", "bat", "butterfly", "crab", "shark")
    active_minutes: dict[str, int] = field(
        default_factory=lambda: {"5m": 60, "15m": 180, "1h": 480}
    )
    confidence_floor: float = 0.35


@dataclass(slots=True)
class NeuralConfig:
    window_size: int = 48
    forecast_horizon: int = 36
    training_stride: int = 12
    inference_stride: int = 3
    max_training_samples: int = 80000
    min_signal_confidence: float = 0.58
    stop_atr_multiple: float = 1.0
    primary_target_atr_multiple: float = 2.0
    secondary_target_atr_multiple: float = 3.25
    hidden_sizes: tuple[int, ...] = (128, 64)
    epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 1e-3
    l2: float = 1e-5
    seed: int = 42


@dataclass(slots=True)
class RLConfig:
    hidden_sizes: tuple[int, ...] = (96, 48)
    epochs: int = 24
    batch_size: int = 256
    learning_rate: float = 1e-3
    l2: float = 1e-5
    invalid_action_penalty: float = 0.25
    reward_clip: float = 3.0
    action_names: tuple[str, ...] = ("skip", "buy", "sell")
    hard_mask_actions: bool = True
    seed: int = 42


@dataclass(slots=True)
class BacktestConfig:
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    max_hold_bars: int = 48


@dataclass(slots=True)
class PortfolioConfig:
    starting_balance: float = 100.0
    base_risk_per_trade: float = 0.02
    min_risk_per_trade: float = 0.005
    max_risk_per_trade: float = 0.03
    target_risk_reward: float = 2.0
    confidence_weight: float = 0.5
    risk_reward_weight: float = 0.5
    commission_per_trade: float = 0.0
    min_balance: float = 1.0


@dataclass(slots=True)
class ArtifactConfig:
    output_dir: Path = Path("artifacts")
    signals_file: str = "signals.csv"
    state_file: str = "rl_state_dataset.csv"
    trade_file: str = "trade_log.csv"
    metrics_file: str = "metrics.json"
    detector_feedback_file: str = "detector_feedback.json"
    neural_model_file: str = "non_harmonic_nn.npz"
    rl_model_file: str = "rl_policy.npz"


@dataclass(slots=True)
class PatternGalleryConfig:
    enabled: bool = True
    output_dir: Path = Path("pattern_gallery")
    manifest_file: str = "gallery_manifest.csv"
    class_members_file: str = "neural_class_members.csv"
    summary_file: str = "summary.json"
    harmonic_context_bars: dict[str, int] = field(
        default_factory=lambda: {"5m": 72, "15m": 72, "1h": 72}
    )
    harmonic_forward_bars: dict[str, int] = field(
        default_factory=lambda: {"5m": 12, "15m": 8, "1h": 4}
    )
    neural_context_bars: int = 64
    neural_forward_bars: int = 0
    neural_cluster_count: int = 6
    neural_examples_per_class: int = 12
    chart_width: int = 1280
    chart_height: int = 720


@dataclass(slots=True)
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    gallery: PatternGalleryConfig = field(default_factory=PatternGalleryConfig)
