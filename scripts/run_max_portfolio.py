#!/usr/bin/env python3
"""Simulate maximum portfolio: all 7 pairs, lower swing_length, higher risk."""
from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
logging.basicConfig(level=logging.WARNING)

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest
from src.ict_bot.data.loader import load_pair

ALL_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "EUR_GBP", "NZD_USD", "USD_CAD"]

# Preload all data once (same DataFrame objects reused across scenarios for caching)
print("Loading data...", flush=True)
DATA = {}
for pair in ALL_PAIRS:
    try:
        DATA[pair] = (load_pair(pair, "H1"), load_pair(pair, "D"))
        print(f"  {pair}: {len(DATA[pair][0])} H1 candles", flush=True)
    except FileNotFoundError:
        print(f"  {pair}: data not found", flush=True)

SCENARIOS = [
    ("Current (4 pairs, sw=20, 1%)", dict(
        pairs=["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"],
        swing_length=20, risk_per_trade=0.01, pullback_window=40,
        fvg_lookback=16,
    )),
    ("7 pairs, sw=20, 1%", dict(
        pairs=ALL_PAIRS,
        swing_length=20, risk_per_trade=0.01, pullback_window=40,
        fvg_lookback=16,
    )),
    ("7 pairs, sw=12, 1%", dict(
        pairs=ALL_PAIRS,
        swing_length=12, risk_per_trade=0.01, pullback_window=40,
        fvg_lookback=16,
    )),
    ("7 pairs, sw=12, 2%", dict(
        pairs=ALL_PAIRS,
        swing_length=12, risk_per_trade=0.02, pullback_window=40,
        fvg_lookback=16,
    )),
    ("7 pairs, sw=10, 2%", dict(
        pairs=ALL_PAIRS,
        swing_length=10, risk_per_trade=0.02, pullback_window=60,
        fvg_lookback=20,
    )),
]

print(f"\n{'Scenario':<34} {'Trd':>5} {'Pairs+':>6} {'WR':>6} {'Exp':>7} "
      f"{'PnL':>9} {'Start$':>8} {'End$':>8} {'Ret%':>7} {'Ann%':>7} "
      f"{'MaxDD%':>7} {'Sharpe':>7}", flush=True)
print("=" * 130, flush=True)

for label, params in SCENARIOS:
    pairs = params.pop("pairs")
    total_trades = 0
    total_wins = 0
    total_pnl_pips = 0
    combined_start = len(pairs) * 10000
    combined_end = 0
    all_trades = []
    pairs_positive = 0
    worst_dd = 0
    sharpe_sum = 0
    pair_count = 0

    for pair in pairs:
        if pair not in DATA:
            continue
        h1, htf = DATA[pair]

        config = BacktestConfig(
            pair=pair, confluence_threshold=3,
            min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
            use_displacement=True, use_partial_tp=True, skip_days=None,
            use_confluence_sizing=True,
            use_liquidity_targets=False, use_premium_discount=False, compute_ob=False,
            **params,
        )
        trades, m = run_backtest(h1, htf_ohlc=htf, config=config)

        end_bal = trades[-1]["balance"] if trades else 10000.0
        combined_end += end_bal
        total_trades += m.total_trades
        total_wins += int(m.win_rate * m.total_trades)
        total_pnl_pips += m.total_pnl_pips
        if m.total_pnl_pips > 0:
            pairs_positive += 1
        if m.max_drawdown_pips > worst_dd:
            worst_dd = m.max_drawdown_pips
        sharpe_sum += m.sharpe_ratio
        pair_count += 1
        all_trades.extend(trades)

    # Restore pairs for next scenario
    params["pairs"] = pairs

    if total_trades > 0:
        wr = total_wins / total_trades
        portfolio_ret = (combined_end - combined_start) / combined_start
        if all_trades:
            data_years = (h1.index[-1] - h1.index[0]).days / 365.25
            ann_honest = (1 + portfolio_ret) ** (1 / data_years) - 1
        else:
            ann_honest = 0

        avg_sharpe = sharpe_sum / pair_count if pair_count > 0 else 0
        exp_approx = total_pnl_pips / total_trades / 50

        print(f"  {label:<32} {total_trades:>5} {pairs_positive:>3}/{len(pairs):<2} "
              f"{wr:>5.1%} {exp_approx:>+6.2f}R "
              f"{total_pnl_pips:>+8.1f} ${combined_start:>7,} ${combined_end:>7,.0f} "
              f"{portfolio_ret:>+6.1%} {ann_honest:>+6.1%} "
              f"{worst_dd:>7.1f} {avg_sharpe:>7.2f}",
              flush=True)
    else:
        print(f"  {label:<32} {'no trades':>5}", flush=True)

# Detailed per-pair breakdown for best scenario
print(f"\n{'='*130}", flush=True)
print(f"  Per-pair breakdown: 7 pairs, sw=12, 2% risk", flush=True)
print(f"{'='*130}", flush=True)
print(f"  {'Pair':>8}  {'Trd':>4}  {'WR':>6}  {'Exp':>7}  {'PnL':>9}  "
      f"{'Start$':>8}  {'End$':>8}  {'Ret%':>7}  {'Period':>25}  {'Months':>6}", flush=True)
print(f"  {'-'*110}", flush=True)

best_params = dict(
    swing_length=12, risk_per_trade=0.02, pullback_window=40,
    fvg_lookback=16,
)
total_start = 0
total_end = 0
for pair in ALL_PAIRS:
    if pair not in DATA:
        print(f"  {pair:>8}  data not found", flush=True)
        continue
    h1, htf = DATA[pair]

    config = BacktestConfig(
        pair=pair, confluence_threshold=3,
        min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        use_displacement=True, use_partial_tp=True, skip_days=None,
        use_confluence_sizing=True,
        use_liquidity_targets=False, use_premium_discount=False,
        **best_params,
    )
    trades, m = run_backtest(h1, htf_ohlc=htf, config=config)
    end_bal = trades[-1]["balance"] if trades else 10000.0
    total_start += 10000
    total_end += end_bal

    if trades:
        first = trades[0]["entry_time"]
        last = trades[-1]["exit_time"]
        months = (last - first).days / 30.44
        period = f"{str(first)[:10]}→{str(last)[:10]}"
        ret = (end_bal - 10000) / 10000
        exp = m.expectancy
        print(f"  {pair:>8}  {m.total_trades:>4}  {m.win_rate:>5.1%}  {exp:>+6.3f}R  "
              f"{m.total_pnl_pips:>+8.1f}  $  10000  ${end_bal:>7.0f}  {ret:>+6.1%}  "
              f"{period:>25}  {months:>5.1f}m", flush=True)
    else:
        print(f"  {pair:>8}  {'—':>4}  no trades", flush=True)

data_years = 2.8
portfolio_ret = (total_end - total_start) / total_start
ann = (1 + portfolio_ret) ** (1 / data_years) - 1
print(f"\n  Portfolio: ${total_start:,} → ${total_end:,.0f} ({portfolio_ret:+.1%})", flush=True)
print(f"  Annualized (over ~{data_years}yr data): {ann:+.1%}", flush=True)
