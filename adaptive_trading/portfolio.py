from __future__ import annotations

from dataclasses import dataclass

from .config import PortfolioConfig
from .schemas import TradeOutcome, TradePlan

EPSILON = 1e-9


@dataclass(slots=True)
class PortfolioSnapshot:
    balance_before: float
    balance_after: float
    peak_balance: float
    drawdown_pct: float
    risk_fraction: float
    risk_amount: float
    position_units: float
    pnl_amount: float
    return_pct: float
    commission_paid: float

    def to_dict(self) -> dict[str, float]:
        return {
            "balance_before": float(self.balance_before),
            "balance_after": float(self.balance_after),
            "peak_balance": float(self.peak_balance),
            "drawdown_pct": float(self.drawdown_pct),
            "risk_fraction": float(self.risk_fraction),
            "risk_amount": float(self.risk_amount),
            "position_units": float(self.position_units),
            "pnl_amount": float(self.pnl_amount),
            "return_pct": float(self.return_pct),
            "commission_paid": float(self.commission_paid),
        }


class PortfolioManager:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        self.balance = float(config.starting_balance)
        self.peak_balance = float(config.starting_balance)

    def can_trade(self) -> bool:
        return self.balance >= self.config.min_balance

    def _risk_fraction(self, plan: TradePlan) -> float:
        confidence = min(max(float(plan.confidence), 0.0), 1.0)
        confidence_scale = 1.0 + self.config.confidence_weight * (confidence - 0.5)

        rr_ratio = float(plan.risk_reward) / max(self.config.target_risk_reward, EPSILON)
        rr_scale = 1.0 + self.config.risk_reward_weight * (rr_ratio - 1.0)

        risk_fraction = self.config.base_risk_per_trade * confidence_scale * rr_scale
        risk_fraction = max(self.config.min_risk_per_trade, risk_fraction)
        risk_fraction = min(self.config.max_risk_per_trade, risk_fraction)
        return float(risk_fraction)

    def settle(self, outcome: TradeOutcome, plan: TradePlan | None) -> PortfolioSnapshot:
        balance_before = self.balance

        if outcome.action == "skip" or plan is None or not self.can_trade():
            drawdown_pct = (
                (self.peak_balance - balance_before) / self.peak_balance if self.peak_balance > EPSILON else 0.0
            )
            return PortfolioSnapshot(
                balance_before=balance_before,
                balance_after=balance_before,
                peak_balance=self.peak_balance,
                drawdown_pct=drawdown_pct,
                risk_fraction=0.0,
                risk_amount=0.0,
                position_units=0.0,
                pnl_amount=0.0,
                return_pct=0.0,
                commission_paid=0.0,
            )

        risk_fraction = self._risk_fraction(plan)
        risk_amount = balance_before * risk_fraction
        stop_distance = abs(float(plan.entry) - float(plan.stop_loss)) + EPSILON
        position_units = risk_amount / stop_distance
        gross_pnl = risk_amount * float(outcome.reward_r)
        commission_paid = float(self.config.commission_per_trade)
        pnl_amount = gross_pnl - commission_paid

        self.balance = max(0.0, balance_before + pnl_amount)
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown_pct = (
            (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > EPSILON else 0.0
        )

        return PortfolioSnapshot(
            balance_before=balance_before,
            balance_after=self.balance,
            peak_balance=self.peak_balance,
            drawdown_pct=drawdown_pct,
            risk_fraction=risk_fraction,
            risk_amount=risk_amount,
            position_units=position_units,
            pnl_amount=pnl_amount,
            return_pct=(pnl_amount / balance_before) if balance_before > EPSILON else 0.0,
            commission_paid=commission_paid,
        )
