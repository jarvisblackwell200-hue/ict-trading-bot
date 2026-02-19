#!/usr/bin/env python3
"""A/B comparison: 4 configurations across pairs to evaluate liquidity targets & premium/discount."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest
from src.ict_bot.data.loader import load_pair

PAIRS = ["EUR_USD", "GBP_USD"]
CONFIGS = {
    "Baseline (both off)":       {"use_liquidity_targets": False, "use_premium_discount": False},
    "Liquidity targets only":    {"use_liquidity_targets": True,  "use_premium_discount": False},
    "Premium/discount only":     {"use_liquidity_targets": False, "use_premium_discount": True},
    "Both on (new default)":     {"use_liquidity_targets": True,  "use_premium_discount": True},
}


def main():
    for pair in PAIRS:
        print(f"\n{'#'*70}")
        print(f"  PAIR: {pair}")
        print(f"{'#'*70}")

        ohlc = load_pair(pair, "H1")
        htf_ohlc = load_pair(pair, "D")

        results = {}
        for label, flags in CONFIGS.items():
            config = BacktestConfig(
                pair=pair,
                swing_length=20,
                confluence_threshold=3,
                min_rr=2.0,
                sl_buffer_pips=10.0,
                spread_pips=1.5,
                slippage_pips=0.5,
                **flags,
            )
            trades, metrics = run_backtest(ohlc, htf_ohlc=htf_ohlc, config=config)

            # Count TP methods
            tp_methods = {}
            for t in trades:
                m = t.get("tp_method", "rr_based")
                tp_methods[m] = tp_methods.get(m, 0) + 1

            results[label] = {"trades": trades, "metrics": metrics, "tp_methods": tp_methods}

        # Print comparison table
        print(f"\n{'Configuration':<28} {'Trades':>6} {'WinR':>6} {'AvgRR':>6} "
              f"{'Expect':>8} {'PF':>6} {'PnL':>10} {'MaxDD':>8} {'Sharpe':>7} {'TP Methods'}")
        print("-" * 120)

        for label, r in results.items():
            m = r["metrics"]
            tp_str = ", ".join(f"{k}:{v}" for k, v in sorted(r["tp_methods"].items()))
            print(f"  {label:<26} {m.total_trades:>6} {m.win_rate:>5.1%} {m.avg_rr:>6.2f} "
                  f"{m.expectancy:>+7.3f}R {m.profit_factor:>6.2f} {m.total_pnl_pips:>+9.1f} "
                  f"{m.max_drawdown_pips:>8.1f} {m.sharpe_ratio:>7.2f}  {tp_str}")

        # Full metrics for the new default config
        print(f"\n--- Full metrics: Both on (new default) ---")
        print(results["Both on (new default)"]["metrics"].summary())

        # Show sample trades with TP method
        trades = results["Both on (new default)"]["trades"]
        if trades:
            print(f"\nSample trades (first 10):")
            print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'TP':>9} {'Exit':>9} "
                  f"{'PnL':>8} {'Reason':>7} {'Score':>5} {'TP Method'}")
            print(f"  {'-'*100}")
            for t in trades[:10]:
                print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} "
                      f"{t['entry_price']:>9.5f} {t['take_profit']:>9.5f} "
                      f"{t['exit_price']:>9.5f} {t['pnl_pips']:>+7.1f} "
                      f"{t['exit_reason']:>7} {t['confluence_score']:>5} "
                      f"{t.get('tp_method', 'rr_based')}")


if __name__ == "__main__":
    main()
