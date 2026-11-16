from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TradingSignal:
    timestamp: Any
    source: str
    timeframe: str
    direction: str
    confidence: float
    risk_reward: float
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(self.metadata)
        payload.pop("metadata", None)
        return payload


@dataclass(slots=True)
class TradePlan:
    direction: str
    entry: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    source: str
    timeframe: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(self.metadata)
        payload.pop("metadata", None)
        return payload


@dataclass(slots=True)
class TradeOutcome:
    timestamp: Any
    action: str
    status: str
    reward_r: float
    bars_held: int
    entry: float
    exit_price: float
    stop_loss: float
    target_1: float
    target_2: float
    source: str
    timeframe: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(self.metadata)
        payload.pop("metadata", None)
        return payload
