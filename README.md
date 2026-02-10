# ICT Trading Bot

AI-controlled forex trading bot based on ICT (Inner Circle Trading) methodology.

## Architecture

- **Signal Detection** — ICT primitives (FVG, OB, MSS, liquidity sweeps) with confluence scoring
- **Backtesting** — Walk-forward validated backtesting with vectorbt
- **Risk Management** — Position sizing, circuit breakers, drawdown limits
- **Execution** — OANDA v20 API (paper + live via factory pattern)
- **Communication** — Telegram bot for alerts and control

## Tech Stack

Python 3.11+ | OANDA | Telegram | Claude API

## Status

Early development — building data pipeline and backtesting framework first.
