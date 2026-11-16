from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_trading import PipelineConfig, TradingResearchPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train harmonic + non-harmonic detectors and an RL policy on historical XAUUSD data.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root containing the data directory.",
    )
    parser.add_argument(
        "--max-master-rows",
        type=int,
        default=None,
        help="Optional cap on the master timeframe rows for faster iteration.",
    )
    parser.add_argument(
        "--nn-max-samples",
        type=int,
        default=None,
        help="Optional cap on neural-network training samples.",
    )
    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=None,
        help="Override the neural-network epoch count.",
    )
    parser.add_argument(
        "--rl-epochs",
        type=int,
        default=None,
        help="Override the RL policy epoch count.",
    )
    parser.add_argument(
        "--hold-bars",
        type=int,
        default=None,
        help="Override the maximum bars a trade may stay open during backtests.",
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=None,
        help="Override the starting portfolio balance for the sequential backtest.",
    )
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=None,
        help="Override the base fraction of balance risked on each executed trade.",
    )
    parser.add_argument(
        "--max-risk-per-trade",
        type=float,
        default=None,
        help="Cap the maximum balance fraction risked on a single trade.",
    )
    parser.add_argument(
        "--commission-per-trade",
        type=float,
        default=None,
        help="Optional flat commission deducted from every executed trade.",
    )
    parser.add_argument(
        "--disable-hard-mask",
        action="store_true",
        help="Allow RL to choose invalid buy or sell actions for debugging.",
    )
    parser.add_argument(
        "--disable-pattern-gallery",
        action="store_true",
        help="Skip generating the pattern gallery SVGs after the offline run.",
    )
    parser.add_argument(
        "--gallery-neural-classes",
        type=int,
        default=None,
        help="Override how many learned non-harmonic visual classes are created.",
    )
    parser.add_argument(
        "--gallery-examples-per-class",
        type=int,
        default=None,
        help="Override how many example candle charts are saved for each neural visual class.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig()

    if args.max_master_rows is not None:
        config.data.max_master_rows = args.max_master_rows
    if args.nn_max_samples is not None:
        config.neural.max_training_samples = args.nn_max_samples
    if args.nn_epochs is not None:
        config.neural.epochs = args.nn_epochs
    if args.rl_epochs is not None:
        config.rl.epochs = args.rl_epochs
    if args.hold_bars is not None:
        config.backtest.max_hold_bars = args.hold_bars
    if args.starting_balance is not None:
        config.portfolio.starting_balance = args.starting_balance
    if args.risk_per_trade is not None:
        config.portfolio.base_risk_per_trade = args.risk_per_trade
    if args.max_risk_per_trade is not None:
        config.portfolio.max_risk_per_trade = args.max_risk_per_trade
    if args.commission_per_trade is not None:
        config.portfolio.commission_per_trade = args.commission_per_trade
    if args.disable_hard_mask:
        config.rl.hard_mask_actions = False
    if args.disable_pattern_gallery:
        config.gallery.enabled = False
    if args.gallery_neural_classes is not None:
        config.gallery.neural_cluster_count = max(1, args.gallery_neural_classes)
    if args.gallery_examples_per_class is not None:
        config.gallery.neural_examples_per_class = max(1, args.gallery_examples_per_class)

    pipeline = TradingResearchPipeline(repo_root=args.repo_root, config=config)
    metrics = pipeline.run()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
