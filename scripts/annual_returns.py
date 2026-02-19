#!/usr/bin/env python3
"""Compute annualized returns across pairs and configs."""
from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
logging.basicConfig(level=logging.WARNING)

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest
from src.ict_bot.data.loader import load_pair

pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]

header = (
    f"  {'Pair':>8}  {'Config':<26}  {'Period':>23}  {'Yrs':>4}  "
    f"{'Start$':>8}  {'End$':>8}  {'Total%':>7}  {'Annual%':>8}  "
    f"{'PnL pips':>9}  {'Trades':>6}"
)
print(header, flush=True)
print("-" * 130, flush=True)

for pair in pairs:
    ohlc = load_pair(pair, "H1")
    htf = load_pair(pair, "D")

    configs = [
        ("Baseline (off/off)",    False, False),
        ("Both on (new default)", True,  True),
    ]

    for label, use_liq, use_pd in configs:
        config = BacktestConfig(
            pair=pair, swing_length=20, confluence_threshold=3,
            min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
            use_liquidity_targets=use_liq, use_premium_discount=use_pd,
        )
        trades, m = run_backtest(ohlc, htf_ohlc=htf, config=config)

        if trades:
            start_date = trades[0]["entry_time"]
            end_date = trades[-1]["exit_time"]
            years = (end_date - start_date).days / 365.25
            if years < 0.01:
                years = 0.01

            start_bal = 10000.0
            end_bal = trades[-1]["balance"]
            total_ret = (end_bal - start_bal) / start_bal
            annual_ret = (1 + total_ret) ** (1 / years) - 1

            period = f"{str(start_date)[:10]} → {str(end_date)[:10]}"
            print(
                f"  {pair:>8}  {label:<26}  {period:>23}  {years:>4.1f}  "
                f"${start_bal:>7.0f}  ${end_bal:>7.0f}  {total_ret:>+6.1%}  {annual_ret:>+7.1%}  "
                f"{m.total_pnl_pips:>+8.1f}  {m.total_trades:>6}",
                flush=True,
            )
        else:
            print(f"  {pair:>8}  {label:<26}  {'no trades':>23}", flush=True)

    print(flush=True)

# Portfolio summary
print("=" * 130, flush=True)
print("  PORTFOLIO SUMMARY (all 4 pairs, Both on, 1% risk per trade, $10k start each)", flush=True)
print("=" * 130, flush=True)

total_pnl = 0
total_trades = 0
total_start = 0
total_end = 0
pair_results = []

for pair in pairs:
    ohlc = load_pair(pair, "H1")
    htf = load_pair(pair, "D")
    config = BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=3,
        min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        use_liquidity_targets=True, use_premium_discount=True,
    )
    trades, m = run_backtest(ohlc, htf_ohlc=htf, config=config)
    if trades:
        end_bal = trades[-1]["balance"]
        total_start += 10000
        total_end += end_bal
        total_pnl += m.total_pnl_pips
        total_trades += m.total_trades
        pair_results.append((pair, m, end_bal))

portfolio_ret = (total_end - total_start) / total_start if total_start > 0 else 0
# Approximate period: ~2.5 years
portfolio_annual = (1 + portfolio_ret) ** (1 / 2.5) - 1

print(f"  Combined starting capital: ${total_start:,.0f}", flush=True)
print(f"  Combined ending capital:   ${total_end:,.2f}", flush=True)
print(f"  Total return:              {portfolio_ret:+.1%}", flush=True)
print(f"  Annualized return:         {portfolio_annual:+.1%}", flush=True)
print(f"  Total trades:              {total_trades}", flush=True)
print(f"  Total PnL:                 {total_pnl:+.1f} pips", flush=True)
print(flush=True)
for pair, m, end_bal in pair_results:
    ret = (end_bal - 10000) / 10000
    print(f"    {pair}: ${end_bal:>10,.2f}  ({ret:>+6.1%})  {m.total_trades} trades  WR={m.win_rate:.1%}  Exp={m.expectancy:+.3f}R", flush=True)
