# Adaptive XAUUSD Pattern Research Stack

This repository now contains a practical offline baseline for the full learning loop you described:

1. Load 5m, 15m, and 1h MT5 XAUUSD history.
2. Detect harmonic patterns with rules on all three timeframes.
3. Train a non-harmonic neural model that learns directional price-action windows from historical TP/SL outcomes.
4. Convert both detector families into one common signal schema with direction, confidence, targets, stop loss, and risk/reward.
5. Feed those signals into an RL-style contextual bandit policy that decides `buy`, `sell`, or `skip`.
6. Hard-mask the RL so it can only execute detector-backed actions.
7. Simulate what happens next with TP/SL/timeout logic and a live portfolio balance.
8. Size every trade from the current account balance using risk-based position sizing.
9. Save the resulting outcomes so the harmonic engine can be recalibrated and the neural model can be retrained on fresh feedback.
10. Render a pattern gallery with harmonic examples and learned non-harmonic visual classes as candle-chart SVGs.

## What This Baseline Is

- Harmonic detector: rule-based geometric scanner using the existing harmonic engine.
- Non-harmonic detector: a NumPy MLP sequence classifier trained on rolling multi-timeframe feature windows.
- RL layer: a contextual bandit policy trained to estimate the reward of `skip`, `buy`, and `sell` from detector signals plus market context.
- Execution layer: offline trade simulator using target/stop hit order on future candles.
- Risk layer: hard-masked action selection plus balance-aware portfolio sizing from a configurable starting capital.
- Feedback layer: artifact files that store trades, metrics, and detector reliability stats for the next run.

This is intentionally a strong baseline, not a promise of profitability. The architecture is designed so we can later replace pieces with deeper sequence models, richer harmonic scoring, and a true sequential RL agent.

## Project Layout

- `adaptive_trading/data.py`: loads and cleans MT5 exports.
- `adaptive_trading/features.py`: creates aligned multi-timeframe market features.
- `adaptive_trading/detectors/harmonic_detector.py`: wraps the harmonic engine into normalized trade signals.
- `adaptive_trading/detectors/non_harmonic_nn.py`: trains and runs the learned non-harmonic detector.
- `adaptive_trading/rl/bandit_agent.py`: trains the RL decision policy.
- `adaptive_trading/simulation.py`: TP/SL/timeout simulator and ATR trade plans.
- `adaptive_trading/pattern_gallery.py`: renders harmonic and non-harmonic pattern galleries as SVG candle charts.
- `adaptive_trading/pipeline.py`: full orchestration from training to backtest artifacts.
- `main.py`: CLI entrypoint.

## Run It

Install dependencies:

```powershell
.\.venv\Scripts\pip.exe install -r .\Harmonic-Trading\requirements.txt
```

Run the full cycle:

```powershell
.\.venv\Scripts\python.exe .\Harmonic-Trading\main.py
```

Quick iteration on a smaller slice:

```powershell
.\.venv\Scripts\python.exe .\Harmonic-Trading\main.py --max-master-rows 250000 --nn-max-samples 30000 --nn-epochs 8 --rl-epochs 10
```

Run with a custom portfolio setup:

```powershell
.\.venv\Scripts\python.exe .\Harmonic-Trading\main.py --starting-balance 100 --risk-per-trade 0.02 --max-risk-per-trade 0.03
```

Tune the visual gallery:

```powershell
.\.venv\Scripts\python.exe .\Harmonic-Trading\main.py --gallery-neural-classes 8 --gallery-examples-per-class 10
```

One-click wrappers from the workspace root:

```powershell
.\run_quick.ps1
.\run_full.ps1
```

## Output Artifacts

All outputs are written to `artifacts/`:

- `signals.csv`: harmonic and neural detector outputs.
- `rl_state_dataset.csv`: RL state table with reward labels for `skip`, `buy`, and `sell`.
- `trade_log.csv`: sequential backtest decisions, raw vs executed RL action, and balance/equity updates.
- `metrics.json`: run summary.
- `detector_feedback.json`: per-detector reliability stats for the next run.
- `non_harmonic_nn.npz`: saved neural detector weights.
- `rl_policy.npz`: saved RL policy weights.
- `pattern_gallery/`: harmonic folders plus learned non-harmonic class folders with SVG candle charts.
- `pattern_gallery/gallery_manifest.csv`: every saved chart with its class, timestamp, and path.
- `pattern_gallery/neural_class_members.csv`: every non-harmonic neural signal and the learned visual class it was assigned to.

## Practical Next Steps

1. Replace the NumPy MLP with a richer CNN/LSTM/Transformer once you want a stronger sequence model.
2. Add spread and slippage models that vary by session and volatility.
3. Turn the contextual bandit into a true multi-step RL environment with inventory state and trade management.
4. Add live MT5 execution only after the offline backtest metrics are stable and believable.
