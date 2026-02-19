#!/usr/bin/env python3
"""A/B comparison: previous defaults vs new strategy (displacement + partials + day filter)."""
from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
logging.basicConfig(level=logging.WARNING)

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest
from src.ict_bot.data.loader import load_pair

pair = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"

print(f"Loading {pair}...", flush=True)
ohlc = load_pair(pair, "H1")
htf = load_pair(pair, "D")
print(f"  {len(ohlc)} H1 candles, {len(htf)} D candles", flush=True)

# Previous default: all features on, but NO displacement/partial/day-filter
# New default: adds displacement, partial TP, and Mon/Fri skip
CONFIGS = [
    ("Previous (no new)", dict(
        use_liquidity_targets=True, use_premium_discount=True,
        use_partial_tp=False, skip_days=None, use_displacement=False,
    )),
    ("+ Displacement", dict(
        use_liquidity_targets=True, use_premium_discount=True,
        use_partial_tp=False, skip_days=None, use_displacement=True,
    )),
    ("+ Partial TP", dict(
        use_liquidity_targets=True, use_premium_discount=True,
        use_partial_tp=True, skip_days=None, use_displacement=False,
    )),
    ("+ Day filter", dict(
        use_liquidity_targets=True, use_premium_discount=True,
        use_partial_tp=False, skip_days=[0, 4], use_displacement=False,
    )),
    ("ALL NEW (default)", dict(
        use_liquidity_targets=True, use_premium_discount=True,
        use_partial_tp=True, skip_days=[0, 4], use_displacement=True,
    )),
]

print(f"\n{'='*120}", flush=True)
print(f"  {pair} — Strategy Comparison", flush=True)
print(f"{'='*120}", flush=True)
print(f"{'Config':<24} {'Trades':>6} {'WR':>6} {'AvgRR':>6} {'Expect':>8} {'PF':>6} "
      f"{'PnL':>10} {'MaxDD':>8} {'Sharpe':>7} {'BE':>4} {'TP':>4} {'SL':>4} "
      f"{'Part':>4} {'L/S':>6}", flush=True)
print("-" * 120, flush=True)

all_results = {}
for label, overrides in CONFIGS:
    config = BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=3,
        min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        **overrides,
    )
    trades, m = run_backtest(ohlc, htf_ohlc=htf, config=config)

    be = sum(1 for t in trades if t["exit_reason"] == "BE")
    tp = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl = sum(1 for t in trades if t["exit_reason"] == "SL")
    part = sum(1 for t in trades if t.get("partial_taken", False))
    longs = sum(1 for t in trades if t["direction"] == "long")
    shorts = sum(1 for t in trades if t["direction"] == "short")

    print(f"  {label:<22} {m.total_trades:>6} {m.win_rate:>5.1%} {m.avg_rr:>6.2f} "
          f"{m.expectancy:>+7.3f}R {m.profit_factor:>6.2f} {m.total_pnl_pips:>+9.1f} "
          f"{m.max_drawdown_pips:>8.1f} {m.sharpe_ratio:>7.2f} {be:>4} {tp:>4} {sl:>4} "
          f"{part:>4} {longs:>2}L/{shorts:<2}S",
          flush=True)
    all_results[label] = (trades, m)

# Show full metrics for new default
print(f"\n--- Full metrics: ALL NEW ---", flush=True)
_, m = all_results["ALL NEW (default)"]
print(m.summary(), flush=True)

# Sample trades
trades, _ = all_results["ALL NEW (default)"]
if trades:
    print(f"\nSample trades (first 15):", flush=True)
    print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'TP':>9} {'Exit':>9} "
          f"{'PnL':>8} {'Out':>4} {'P':>1} {'S':>2} {'Day':>3} {'TP Method'}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for t in trades[:15]:
        day = t['entry_time'].strftime('%a')
        print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} "
              f"{t['entry_price']:>9.5f} {t['take_profit']:>9.5f} "
              f"{t['exit_price']:>9.5f} {t['pnl_pips']:>+7.1f} "
              f"{t['exit_reason']:>4} {'Y' if t.get('partial_taken') else 'N':>1} "
              f"{t['confluence_score']:>2} {day:>3} "
              f"{t.get('tp_method', 'rr_based')}", flush=True)
