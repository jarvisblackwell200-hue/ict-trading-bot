# ICT Trading Bot

## Project Structure

- `src/ict_bot/signals/detector.py` — ICT signal generation (BOS/CHoCH detection, FVG/OB zones, confluence scoring)
- `src/ict_bot/signals/trend_adaptive.py` — Adaptive Trend Continuation signal generator (BOS + EMA + ATR)
- `src/ict_bot/backtest/engine.py` — Backtest engine with trade simulation, trailing stops, BE, partial TP
- `src/ict_bot/backtest/metrics.py` — Performance metrics (win rate, expectancy, Sharpe, drawdown)
- `src/ict_bot/backtest/walk_forward.py` — Walk-forward validation, Monte Carlo, parameter sensitivity
- `src/ict_bot/risk/manager.py` — Risk manager with circuit breakers, position sizing, exposure limits
- `src/ict_bot/data/loader.py` — Loads OHLC from `data/processed/*.parquet`
- `scripts/` — Backtest and analysis scripts
- `tests/` — Unit tests (pytest)

## Commands

- Run tests: `PYTHONPATH=src venv/bin/python -m pytest tests/ -v`
- Run backtest: `PYTHONPATH=src venv/bin/python scripts/run_backtest.py --pair EUR_USD`
- Run filter analysis: `PYTHONPATH=src venv/bin/python scripts/filter_isolation.py`

## Data

7 pairs: EUR_USD, GBP_USD, USD_JPY, AUD_USD, USD_CAD, NZD_USD, EUR_GBP
- H1 + Daily parquet files (~17K H1 bars each, 2.8 years)
- M15 parquet files (~28K bars each, 423 days) — downloaded from IB Gateway
- M5 parquet files (~42K bars each, 211 days) — downloaded from IB Gateway

## Timeframe Comparison (Feb 2026)

| Metric | M5 (211d) | M15 (423d) | H1 (2.8yr) |
|--------|-----------|------------|------------|
| Best config | sw=5 t=3 | **sw=5 t=3** | sw=10 t=3 |
| Per-trade expectancy | +0.111R | +0.220R | +0.265R |
| Trades (per-pair, unfiltered) | 10,111 | 7,396 | 1,744 |
| Estimated R/year (portfolio) | ~1,000+ | **~245R** | ~80-100R |
| Walk-forward (true OOS) | Weak (7 windows, 57-100%) | **7/7 windows profitable** | 4 windows, 75-100% |
| Variable spread tested | No | **Yes** | No |
| Portfolio stress-tested | No | **Yes (max 3 pos, corr limits)** | No |
| All pairs profitable | 6/6 (no JPY data) | **7/7** | 7/7 |
| Confidence | Low (short sample) | **Highest** | Medium |

**M15 is the recommended timeframe.** Best trade-off of frequency, expectancy, and validation depth. M5 has higher volume but lower quality and only 211 days of data. H1 has best per-trade quality but ~3x fewer portfolio-level R/year.

---

## Validated Strategy Configs

### M15 Strategy (PRIMARY — most profitable, best validated)

```python
# M15 — "sw=5 t=3" config
# Robust walk-forward: 7/7 windows profitable, 7/7 pairs positive
# Portfolio (max 3 pos): 915 trades, +0.311R, +284.3R over 423 days
# Max DD: 11.8R | Variable spread: 0.8-3.75 pips by session
timeframe = "M15"
swing_length = 5
confluence_threshold = 3
use_displacement = False
use_breakeven = False
use_partial_tp = False
min_rr = 2.0
sl_buffer_pips = 10.0
skip_days = []               # No day filtering on M15
fvg_lookback = 16
pullback_window = 40
compute_ob = False
max_sl_pips = 100.0
# All ICT extras OFF: sweep, IFVG, breaker, CE = False
```

**M15 higher-quality variant:**
```python
# M15 — "sw=5 t=4" config (fewer trades, better per-trade quality)
# Portfolio: 1,006 trades, +0.288R, +289.7R, max DD 8.6R
confluence_threshold = 4     # only change from above
```

### M5 Strategy (EXPERIMENTAL — high frequency, less validated)

```python
# M5 — "sw=5 t=3" config
# Full-sample: 10,111 trades, +0.111R, +21,081 pips over 211 days
# Walk-forward: 7 windows, GBP_USD 100%, others 71-86%
# WARNING: Only 211 days of data, not robustly validated
timeframe = "M5"
swing_length = 5
confluence_threshold = 3
use_displacement = False
use_breakeven = False
use_partial_tp = False
min_rr = 2.0
sl_buffer_pips = 10.0
skip_days = []
fvg_lookback = 16
pullback_window = 40
compute_ob = False
max_sl_pips = 100.0
```

### H1 Strategy (PROVEN — longest track record)

```python
# H1 — best validated on 2.8 years of data
# In-sample: 1,744 trades, +0.265R, +24,604 pips, 7/7 pairs
# Walk-forward: 5/7 pairs 100% profitable windows
timeframe = "H1"
swing_length = 10
confluence_threshold = 3
use_displacement = False
use_breakeven = False
use_partial_tp = False
min_rr = 2.0
sl_buffer_pips = 10.0
skip_days = [0, 4]           # Skip Mon/Fri (important on H1)
fvg_lookback = 16
pullback_window = 40
compute_ob = False
max_sl_pips = 100.0
```

### Key Differences Between Timeframes

| Parameter | M5/M15 | H1 |
|-----------|--------|-----|
| `swing_length` | **5** | **10** |
| `skip_days` | `[]` (none) | `[0, 4]` (Mon/Fri) |
| Everything else | Same | Same |

The core edge is **BOS + FVG pullback entry** — only `swing_length` and `skip_days` differ.

### Portfolio Risk Limits (for live trading)

```python
max_positions = 3              # max concurrent across all pairs
max_correlated_same_dir = 2    # max same-direction USD pairs
risk_per_trade = 0.01          # 1% of account
```

### Robust Backtest Results — M15 sw=5 t=3 (Feb 2026)

**Full-sample with variable spread (423 days, 7 pairs):**

| Pair | Trades | WR | Expect | PF | Pips | AvgSpread |
|------|--------|-----|--------|------|------|-----------|
| EUR_USD | 1,069 | 50.9% | +0.237R | 1.48 | +6,624 | 1.5p |
| GBP_USD | 1,074 | 48.4% | +0.217R | 1.48 | +7,430 | 1.9p |
| USD_JPY | 982 | 51.1% | +0.273R | 1.57 | +10,096 | 1.6p |
| AUD_USD | 1,106 | 53.7% | +0.229R | 1.47 | +5,519 | 1.9p |
| USD_CAD | 1,082 | 51.4% | +0.227R | 1.48 | +6,222 | 2.0p |
| NZD_USD | 1,152 | 52.2% | +0.198R | 1.45 | +5,436 | 2.3p |
| EUR_GBP | 931 | 53.7% | +0.159R | 1.41 | +3,190 | 2.2p |

**Portfolio-filtered (max 3 positions, max 2 correlated same-direction):**
- 915 accepted / 6,481 rejected
- 55.1% WR, +0.311R expectancy, +8,461 pips
- Max drawdown: 11.8R | Final: +284.3R

**True out-of-sample walk-forward (signals on truncated data, no look-ahead):**

| Window | Pairs | Trades | AvgWR | AvgExp | TotPnL |
|--------|-------|--------|-------|--------|--------|
| Mar→Apr 2025 | 7 | 513 | 55.4% | +0.303R | +6,190 |
| May→Jun 2025 | 7 | 540 | 52.0% | +0.215R | +3,215 |
| Jun→Jul 2025 | 7 | 476 | 53.7% | +0.200R | +2,480 |
| Aug→Sep 2025 | 7 | 569 | 46.3% | +0.064R | +804 |
| Sep→Oct 2025 | 7 | 520 | 49.3% | +0.182R | +2,719 |
| Nov→Dec 2025 | 7 | 582 | 49.5% | +0.188R | +2,196 |
| Dec→Jan 2026 | 7 | 444 | 55.9% | +0.255R | +2,635 |

**7/7 windows profitable (100%). All 7 pairs positive total PnL.**

| Pair | Windows | Profitable | AvgExp | TotPnL |
|------|---------|-----------|--------|--------|
| GBP_USD | 7 | 100% | +0.261R | +4,608 |
| USD_CAD | 7 | 100% | +0.208R | +3,050 |
| NZD_USD | 7 | 100% | +0.225R | +2,988 |
| USD_JPY | 7 | 86% | +0.218R | +3,584 |
| EUR_USD | 7 | 86% | +0.162R | +2,214 |
| AUD_USD | 7 | 86% | +0.155R | +1,965 |
| EUR_GBP | 7 | 71% | +0.179R | +1,828 |

## Backtesting Notes

- Always use a permissive risk manager for research (disable circuit breakers) — the default `RiskManager` has a 3-consecutive-loss pause that silently kills 80%+ of trades and confounds results.
- `compute_ob=True` is extremely slow (~3+ min/pair). Use `compute_ob=False` for iteration.
- `detect_primitives` caches by `(id(ohlc), len, swing_length, ...)` — changing swing_length forces full recomputation.

## Filter Isolation Findings (Feb 2026)

Systematic ablation study across all 7 pairs, permissive risk manager, `compute_ob=False`.

### What actually drives edge

| Factor | Effect | Evidence |
|--------|--------|----------|
| `swing_length=10` (vs default 20) | **Dominant factor** | +0.373R exp, +12,199 pips, 60.8% WR, 735 trades, **7/7 pairs profitable** |
| `skip_days=[0,4]` (Mon/Fri) | **Strong positive** | Removing skip days drops expectancy from +0.068R to -0.044R |
| `threshold=4` | Better expectancy but fewer trades | +0.155R exp, 407 trades, 6/7 pairs |

### What doesn't matter (removing has <0.02R impact)

| Factor | Effect |
|--------|--------|
| Premium/discount filter | No measurable effect |
| Liquidity targets (vs fixed R:R TP) | No measurable effect |
| Confluence sizing | No measurable effect |
| FVG lookback (8 vs 16 vs 32) | Marginal |
| SL buffer (5 vs 10 vs 15) | Marginal |
| min_rr (1.5 vs 2.0 vs 3.0) | Marginal |

### What actively hurts

| Factor | Effect | Evidence |
|--------|--------|----------|
| Displacement validation | **Destroys good trades** | Removing it: +5,568 pips (vs +2,769 with it). Adding it to minimal config: flips from +1,744 to -1,042 pips |
| Breakeven stop | **Slightly harmful** | Locks in tiny wins that would've run further. Removing BE: +2,933 vs +2,769 with BE |
| `swing_length=30` | Too coarse | -0.161R exp, 2/7 pairs profitable |
| BE at 1.0R | **Most harmful BE level** | -0.127R exp with BE at 1.0R vs -0.047R without |

### Per-pair returns: config comparison (1% risk, $10K start, 2.8 years, no OB)

**FULL baseline (sw=20, displacement, BE, partial, skip Mon/Fri, thresh=3)**

| Pair | Trades | Trd/Yr | WR | Expect | PF | Pips | Sharpe | MaxDD% | Return | Ann% | Final$ |
|------|--------|--------|-----|--------|------|------|--------|--------|--------|------|--------|
| EUR_USD | 75 | 26.8 | 46.7% | +0.095R | 1.39 | +580 | 2.15 | 60.5% | +9.0% | +3.1% | $10,897 |
| GBP_USD | 77 | 27.5 | 48.1% | +0.088R | 1.44 | +924 | 2.53 | 35.6% | +4.0% | +1.4% | $10,400 |
| USD_JPY | 28 | 10.0 | 25.0% | -0.419R | 0.47 | -623 | -5.39 | 368.0% | -10.1% | -3.7% | $8,992 |
| AUD_USD | 84 | 30.0 | 50.0% | +0.080R | 1.30 | +524 | 1.83 | 46.3% | +8.4% | +2.9% | $10,840 |
| USD_CAD | 15 | 5.4 | 66.7% | +0.262R | 1.25 | +95 | 1.56 | 54.1% | +3.6% | +1.3% | $10,355 |
| NZD_USD | 114 | 40.8 | 47.4% | +0.028R | 1.15 | +376 | 0.95 | 71.7% | +3.1% | +1.1% | $10,313 |
| EUR_GBP | 59 | 21.1 | 61.0% | +0.344R | 2.69 | +892 | 6.44 | 16.1% | +17.6% | +6.0% | $11,756 |
| **TOTAL** | **452** | | | | | **+2,769** | | | **+5.1%** | | |

**sw=10 (same filters as baseline, only swing_length changed)**

| Pair | Trades | Trd/Yr | WR | Expect | PF | Pips | Sharpe | MaxDD% | Return | Ann% | Final$ |
|------|--------|--------|-----|--------|------|------|--------|--------|--------|------|--------|
| EUR_USD | 133 | 47.6 | 58.6% | +0.305R | 2.14 | +2,153 | 5.18 | 12.9% | +52.0% | +16.2% | $15,205 |
| GBP_USD | 122 | 43.6 | 57.4% | +0.285R | 1.94 | +2,237 | 4.52 | 14.7% | +42.9% | +13.6% | $14,294 |
| USD_JPY | 59 | 21.1 | 55.9% | +0.366R | 2.17 | +1,270 | 4.87 | 22.7% | +24.7% | +8.2% | $12,470 |
| AUD_USD | 115 | 41.1 | 57.4% | +0.290R | 1.96 | +1,670 | 4.41 | 17.9% | +37.9% | +12.2% | $13,793 |
| USD_CAD | 33 | 11.8 | 78.8% | +0.738R | 12.54 | +1,412 | 16.61 | 6.1% | +26.9% | +8.9% | $12,691 |
| NZD_USD | 177 | 63.3 | 55.9% | +0.276R | 1.75 | +2,201 | 3.75 | 13.0% | +71.2% | +21.2% | $17,117 |
| EUR_GBP | 96 | 34.3 | 61.5% | +0.348R | 2.25 | +1,257 | 5.45 | 11.7% | +43.8% | +13.9% | $14,378 |
| **TOTAL** | **735** | | | | | **+12,200** | | | **+42.8%** | | |

**sw=10, no displacement (more trades, still profitable)**

| Pair | Trades | Trd/Yr | WR | Expect | PF | Pips | Sharpe | MaxDD% | Return | Ann% | Final$ |
|------|--------|--------|-----|--------|------|------|--------|--------|--------|------|--------|
| EUR_USD | 294 | 105.2 | 57.1% | +0.255R | 2.00 | +4,511 | 4.80 | 5.2% | +107.7% | +29.9% | $20,765 |
| GBP_USD | 261 | 93.4 | 55.6% | +0.240R | 1.73 | +3,723 | 3.74 | 13.1% | +89.6% | +25.7% | $18,958 |
| USD_JPY | 161 | 57.6 | 54.7% | +0.248R | 1.70 | +2,541 | 3.54 | 20.0% | +60.8% | +18.5% | $16,082 |
| AUD_USD | 270 | 96.6 | 55.6% | +0.265R | 1.85 | +3,396 | 4.06 | 10.6% | +99.7% | +28.1% | $19,973 |
| USD_CAD | 169 | 60.5 | 65.1% | +0.448R | 2.90 | +4,068 | 7.24 | 4.5% | +111.7% | +30.8% | $21,167 |
| NZD_USD | 336 | 120.2 | 52.1% | +0.173R | 1.49 | +2,728 | 2.63 | 14.7% | +91.5% | +26.2% | $19,154 |
| EUR_GBP | 253 | 90.5 | 62.8% | +0.344R | 2.22 | +2,973 | 5.45 | 5.5% | +163.9% | +41.5% | $26,385 |
| **TOTAL** | **1,744** | | | | | **+23,939** | | | **+103.5%** | | |

**sw=10, no displacement, no BE (best total PnL)**

| Pair | Trades | Trd/Yr | WR | Expect | PF | Pips | Sharpe | MaxDD% | Return | Ann% | Final$ |
|------|--------|--------|-----|--------|------|------|--------|--------|--------|------|--------|
| EUR_USD | 294 | 105.2 | 52.7% | +0.257R | 2.01 | +4,542 | 4.76 | 5.1% | +109.1% | +30.2% | $20,911 |
| GBP_USD | 261 | 93.4 | 52.5% | +0.290R | 1.84 | +4,278 | 4.16 | 11.4% | +119.0% | +32.4% | $21,902 |
| USD_JPY | 161 | 57.6 | 50.3% | +0.273R | 1.75 | +2,706 | 3.68 | 18.4% | +68.3% | +20.5% | $16,833 |
| AUD_USD | 270 | 96.6 | 51.9% | +0.274R | 1.86 | +3,422 | 4.04 | 10.6% | +106.1% | +29.5% | $20,608 |
| USD_CAD | 169 | 60.5 | 58.6% | +0.451R | 2.89 | +4,041 | 7.07 | 4.6% | +112.6% | +31.0% | $21,261 |
| NZD_USD | 336 | 120.2 | 48.5% | +0.164R | 1.47 | +2,631 | 2.53 | 15.2% | +85.9% | +24.8% | $18,590 |
| EUR_GBP | 253 | 90.5 | 59.7% | +0.346R | 2.23 | +2,985 | 5.42 | 6.2% | +164.7% | +41.6% | $26,466 |
| **TOTAL** | **1,744** | | | | | **+24,604** | | | **+109.4%** | | |

### Cross-config comparison summary

| Config | Trades | Avg WR | Avg Expect | Total Pips | Avg Ann% | Pairs +/- |
|--------|--------|--------|------------|------------|----------|-----------|
| Full baseline (sw=20) | 452 | 49.3% | +0.068R | +2,769 | +1.7% | 6+ / 1- |
| **sw=10** | **735** | **60.8%** | **+0.373R** | **+12,200** | **+13.5%** | **7+ / 0-** |
| sw=10, no disp | 1,744 | 57.1% | +0.255R | +23,939 | +28.7% | 7+ / 0- |
| sw=10, no disp, no BE | 1,744 | 53.5% | +0.265R | +24,604 | +30.0% | 7+ / 0- |

### Key takeaway

The edge lives in **entry precision** (shorter swing structure = tighter FVG zones) and **timing** (skip Mon/Fri). Confluence scoring, premium/discount, displacement validation, and most trade management features are either noise or actively harmful.

Best config candidates:
- **Conservative:** `sw=10`, displacement on, skip Mon/Fri, thresh=3 — 735 trades, +0.373R, 7/7 pairs, ~13.5% ann
- **Aggressive (walk-forward validated):** `sw=10`, no disp, no BE, skip Mon/Fri, thresh=3 — 1,733 trades, +0.326R, 7/7 pairs. Walk-forward: 100% profitable windows on 5/7 pairs, all 7 positive total PnL.
- **High-quality:** `sw=10`, no disp, no BE, skip Mon/Fri, thresh=4 — 1,341 trades, +0.413R, 7/7 pairs (best per-trade quality)

---

## ICT Concepts Backtest (Feb 2026)

Implemented and tested 4 additional ICT concepts from the guide against the proven sw=10 baseline:
- **Liquidity Sweep filter** (`use_sweep_filter`) — require wick-beyond-level-then-close-back sweep before BOS
- **Inverse FVG zones** (`use_ifvg`) — invalidated FVGs that flip role as entry zones
- **Breaker Blocks** (`use_breaker_blocks`) — structural S/R zones that were broken and flipped
- **CE Entry** (`use_ce_entry`) — Consequent Encroachment: 50% midpoint entry instead of 25% edge

### ICT Concepts: Impact on Edge

| Config | Trades | AvgWR | AvgExp | AvgPF | TotalPnL | Pairs+/- |
|--------|--------|-------|--------|-------|----------|----------|
| **BASELINE (sw=10 no-disp no-BE)** | **1,733** | **51.3%** | **+0.326R** | **2.08** | **+26,546** | **7+/0-** |
| + sweep filter | 65 | 45.4% | +0.315R | 2.90 | +852 | 6+/1- |
| + IFVG zones | 1,733 | 51.3% | +0.326R | 2.08 | +26,546 | 7+/0- |
| + breaker blocks | 1,733 | 51.3% | +0.326R | 2.08 | +26,546 | 7+/0- |
| + CE entry (50% midpoint) | 1,720 | 49.9% | +0.279R | 1.91 | +23,893 | 7+/0- |
| + all four | 65 | 42.0% | +0.230R | 2.45 | +739 | 6+/1- |

### ICT Concepts: Analysis

| Concept | Verdict | Explanation |
|---------|---------|-------------|
| **Liquidity Sweep** | Too aggressive on H1 | Kills 96% of trades (1,733 → 65). Higher per-trade PF (2.90) but statistically unreliable. May work on M5/M15 with more swing points. |
| **IFVG zones** | Zero effect | Identical results — regular FVGs already capture all pullback entries before price reaches any IFVG zone. |
| **Breaker Blocks** | Zero effect | Same — existing FVG zones dominate entry. Breaker zones exist but aren't reached during pullback window. |
| **CE Entry** | Slightly worse | 50% midpoint gives worse fill rate and R:R than 25% edge entry. Expectancy drops +0.326R → +0.279R. |

### Threshold Sensitivity

| Config | Trades | AvgWR | AvgExp | TotalPnL | Pairs+/- |
|--------|--------|-------|--------|----------|----------|
| thresh=1 | 1,752 | 50.9% | +0.315R | +26,146 | 7+/0- |
| thresh=2 | 1,752 | 50.9% | +0.315R | +26,146 | 7+/0- |
| **thresh=3** | **1,733** | **51.3%** | **+0.326R** | **+26,546** | **7+/0-** |
| thresh=4 | 1,341 | 54.5% | +0.413R | +24,066 | 7+/0- |

thresh=4 is the best per-trade quality (+0.413R) with fewer trades; thresh=3 has the best total PnL.

---

## Walk-Forward Validation (Feb 2026)

Rolling 18-month train / 6-month test / 3-month step. Tests if the strategy works on unseen data.

### Baseline walk-forward: sw=10, no disp, no BE, skip Mon/Fri, thresh=3

| Pair | Windows | Win% | AvgTrades | AvgExp | TotPnL |
|------|---------|------|-----------|--------|--------|
| EUR_USD | 4 | **100%** | 45.2 | +0.451R | +4,422 |
| GBP_USD | 4 | 75% | 44.2 | +0.199R | +2,282 |
| USD_JPY | 4 | 75% | 28.0 | +0.327R | +2,896 |
| AUD_USD | 4 | **100%** | 46.0 | +0.242R | +1,670 |
| USD_CAD | 4 | **100%** | 23.0 | +0.672R | +3,067 |
| NZD_USD | 4 | **100%** | 61.5 | +0.133R | +1,236 |
| EUR_GBP | 4 | **100%** | 46.5 | +0.337R | +2,184 |

**Walk-forward is extremely strong:**
- 5/7 pairs profitable in 100% of test windows
- 2/7 pairs profitable in 75% of test windows
- All 7 pairs have positive total PnL across all windows
- Average expectancy remains positive (+0.133R to +0.672R) even on unseen data
- This confirms the strategy is **not overfit** — the sw=10 edge is genuine

### Best Validated Strategy

```
swing_length = 10
confluence_threshold = 3
use_displacement = False
use_breakeven = False
use_partial_tp = True
skip_days = [0, 4]  # Mon/Fri
fvg_lookback = 16
pullback_window = 40
compute_ob = False
# All new ICT concepts (sweep, IFVG, breaker, CE) = OFF
```

**In-sample (2.8 years):** 1,733 trades, 51.3% WR, +0.326R, +26,546 pips, 7/7 pairs profitable
**Walk-forward (6-month windows):** 100% of windows profitable on 5/7 pairs, positive on all 7

### Adaptive Trend Continuation (failed)

BOS + EMA(50) + ATR-based stops, no FVG/OB pullback entry. Generated 1,089 trades across 7 pairs but **negative expectancy on all 7** (-0.184R to -0.518R, -11,221 total pips). The FVG pullback entry model is doing real work — entering at next-bar-open after BOS without waiting for a zone retest is essentially random.

---

## ICT Concepts Cross-Reference (Feb 2026)

Comprehensive analysis comparing the codebase against the ICT Concepts Guide PDF (`docs/ICT_Concepts_Guide.pdf`).

### ✅ ICT Concepts Already Implemented

| Concept | Implementation | Location | Notes |
|---------|---------------|----------|-------|
| **BOS (Break of Structure)** | Via smartmoneyconcepts lib | `detector.py` | Trend continuation signal |
| **CHoCH (Change of Character)** | Via smartmoneyconcepts lib | `detector.py` | Early reversal warning |
| **Fair Value Gaps (FVG)** | 3-candle imbalance detection | `detector.py` | Primary entry zones |
| **Order Blocks (OB)** | Via smartmoneyconcepts lib | `detector.py` | Optional (`compute_ob=True`) |
| **Kill Zones** | Asian, London, NY, London Close | `kill_zones.py` | Time-based filtering |
| **Premium/Discount Zones** | Equilibrium-based zone check | `_is_premium_discount_valid()` | Longs in discount, shorts in premium |
| **OTE (Optimal Trade Entry)** | 62-79% Fib retracement | `_compute_ote_zone()` | Entry refinement |
| **Displacement Validation** | Large-body candle sequences | `_validate_displacement()` | Confirms genuine breaks |
| **BSL/SSL Liquidity** | Equal highs/lows detection | `smc.liquidity()` | Target identification |
| **PDH/PDL, PWH/PWL** | Previous day/week highs/lows | `_find_liquidity_target()` | TP placement |
| **Dealing Range** | Swing-based range calculation | `_get_dealing_range()` | Context for P/D zones |
| **HTF Bias Alignment** | Daily bias check | `_get_htf_bias_at()` | Confluence scoring |
| **FVG Stacking** | Multiple overlapping FVGs | `_count_fvg_stack()` | Stronger zones |
| **Multi-Timeframe Analysis** | Structure TF + Entry TF | `generate_signals_mtf()` | H1 structure, M15/M5 entry |

### Implemented & Tested (No Edge on H1)

| Concept | Status | Backtest Result |
|---------|--------|----------------|
| **Liquidity Sweeps** | `_detect_liquidity_sweep()` | Too aggressive on H1 — kills 96% of trades. May help on M5/M15. |
| **Breaker Blocks** | `_find_breaker_blocks()` | Zero additional entries — FVGs already capture all pullback zones. |
| **Inverse FVG (IFVG)** | `_find_ifvg_zones()` | Zero additional entries — same as breaker blocks. |
| **CE Entry** | `use_ce_entry` param | Slightly worse — 50% midpoint gives lower WR and expectancy than 25% edge. |

### ❌ ICT Concepts Not Yet Implemented

#### High Priority — Core ICT Setups

| Concept | Description | Notes |
|---------|-------------|-------|
| **Market Structure Shift (MSS)** | Sweep → displacement → FVG → structure break | Sweep filter tested; needs M5 data for proper MSS |
| **ICT Unicorn Model** | FVG overlapping with Breaker Block | Breaker blocks produce no entries on H1; needs lower TF |
| **Balanced Price Range (BPR)** | Overlapping opposing FVGs — price magnet | Untested; could work as target identification |
| **Silver Bullet Strategy** | Time-window scalping: 3-4am, 10-11am, 2-3pm EST | Needs M5 data; only EUR_USD and GBP_USD have M5 |

#### Medium Priority — Advanced Structures

| Concept | Description | Recommended Implementation |
|---------|-------------|---------------------------|
| **ICT Macros** | 20-min algorithmic windows (8:50-9:10am, 9:50-10:10am, etc.). | Add macro window detection alongside kill zones; highest confluence during these windows |
| **Mitigation Blocks** | Failed OB that didn't sweep liquidity — lower probability than breakers. | Track OBs that fail without liquidity sweep; lower confluence scoring |
| **Propulsion Blocks** | Single candle within an OB that drives price away — tightest stops. | Refine OB detection to find propulsion candles; use for precision entry |
| **Rejection Blocks** | Long wicks at swing points after liquidity sweep — 80-90% retracement entry. | Detect long wicks at structural levels after sweeps |
| **Volume Imbalance** | Gap between candle bodies (not wicks) — micro-level inefficiencies. | Add two-candle body gap detection for micro entries |
| **Liquidity Voids** | Multiple consecutive FVGs forming large inefficiency zones. | Group consecutive FVGs; calculate composite CE (50%) |
| **Consequent Encroachment (CE)** | 50% midpoint of any imbalance as precision entry. | Add CE calculation to all FVG/OB/breaker zones for refined entries |

#### Lower Priority — Session/Cycle Analysis

| Concept | Description | Recommended Implementation |
|---------|-------------|---------------------------|
| **CBDR (Central Bank Dealers Range)** | 2pm-8pm EST range for next-day projections. | Calculate CBDR; project ±1, ±2, ±2.5 standard deviations for targets |
| **NWOG/NDOG** | Weekly/daily opening gaps (Friday close to Sunday open, daily close to next open). | Detect gaps; use as FVG-equivalent support/resistance |
| **Asian Range** | 7pm-12am EST range — accumulation phase. | Track Asian range high/low for London session sweep targets |
| **Power of 3 (AMD)** | Accumulation → Manipulation → Distribution daily cycle. | Classify time segments; only trade Distribution phase entries |
| **Judas Swing** | Manipulation phase false move (midnight-5am EST). | Detect pre-London false breakout for reversal entries |
| **True Day Open** | Midnight EST as algorithmic start; reference for daily bias. | Add midnight open price tracking for bias determination |
| **Market Maker Models (MMBM/MMSM)** | Full institutional cycle from consolidation to target. | Add model state tracker: Original Consolidation → Sell/Buy Side → SMR → Target |
| **Standard Deviation Projections** | Fib-based targets from manipulation leg (-0.5, -1, -2, -2.5, -4). | Add std dev projection targets alongside liquidity targets |
| **SMT Divergence** | Correlated asset divergence (ES/NQ, EUR/GBP) for reversal confirmation. | Add correlation check between related instruments |
| **IPDA Lookbacks** | 20/40/60 day ranges for premium/discount context. | Calculate IPDA ranges; use for higher-timeframe P/D analysis |
| **Quarterly Theory** | Time is fractal — Accumulation/Manipulation/Distribution at all scales. | Add quarterly cycle analysis for seasonal bias |

#### Newer ICT Concepts (Sept 2025+)

| Concept | Description | Recommended Implementation |
|---------|-------------|---------------------------|
| **Suspension Block** | Single candle with volume imbalance at both ends, body overlapped by prior wick. | Add specialized pattern detector |
| **Hidden Order Block** | OB invisible on single timeframe, visible via multi-TF wick overlap. | Enhance MTF analysis to detect hidden zones |
| **Opening Range Gap (ORG)** | RTH close to next RTH open gap for futures. | Add ORG tracking with std dev projections |

### Implemented Code (New ICT Concepts)

Functions added to `detector.py`:
- `_detect_liquidity_sweep()` — sweep detection before BOS (too aggressive on H1)
- `_find_ifvg_zones()` — invalidated FVGs as entry zones (no effect on H1)
- `_find_breaker_blocks()` — structural zones that flipped (no effect on H1)
- `use_ce_entry` parameter — 50% midpoint entry (slightly worse than 25% edge)

Parameters in `BacktestConfig`:
- `use_sweep_filter: bool = False`
- `use_ifvg: bool = False`
- `use_breaker_blocks: bool = False`
- `use_ce_entry: bool = False`

### Priority Matrix (Updated with Backtest Evidence)

| Priority | Concept | Status | Impact on H1 Edge |
|----------|---------|--------|-------------------|
| ~~P0~~ | ~~Breaker Blocks~~ | **Tested — no effect** | Zero additional entries on H1 |
| ~~P1~~ | ~~IFVG~~ | **Tested — no effect** | Zero additional entries on H1 |
| ~~P1~~ | ~~CE Entry~~ | **Tested — slightly worse** | -0.047R vs 25% edge entry |
| ~~P2~~ | ~~Liquidity Sweeps~~ | **Tested — too aggressive** | Kills 96% of H1 trades |
| **P0** | Silver Bullet (M5) | Needs M5 data | Time-window + FVG, different strategy |
| **P0** | MSS on M5 | Needs M5 data | Sweep + disp + BOS on lower TF |
| **P1** | BPR Detection | Untested | Potential as target identification |
| **P1** | ICT Macros | Untested | May not help on H1 (1-hour bars too coarse) |
| **P2** | CBDR/NWOG | Untested | Session context for daily bias |

### Key Insight

**Most advanced ICT concepts don't improve H1 performance.** The edge on H1 comes from simple, well-calibrated structure detection (sw=10) plus FVG pullback entry plus day-of-week filtering. Adding more zone types (IFVG, breaker blocks) doesn't generate new entries because regular FVGs already capture all pullback opportunities. The concepts that might add value (Silver Bullet, MSS, ICT Macros) require M5/M15 data to be meaningful — they're designed for sub-hourly timeframes.

### Reference

See `docs/ICT_Concepts_Guide.pdf` for detailed explanations of all ICT concepts.
