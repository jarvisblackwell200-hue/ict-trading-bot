#!/usr/bin/env python3
"""
Run a full backtest with walk-forward validation and robustness testing.

Usage:
  python scripts/run_backtest.py                           # Quick backtest
  python scripts/run_backtest.py --walk-forward            # With walk-forward
  python scripts/run_backtest.py --full                    # Full analysis (WF + MC + sensitivity)
  python scripts/run_backtest.py --pair GBP_USD            # Different pair
  python scripts/run_backtest.py --threshold 4 --rr 2.5    # Adjust parameters
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest
from src.ict_bot.backtest.walk_forward import (
    format_monte_carlo_report,
    format_walk_forward_report,
    monte_carlo,
    parameter_sensitivity,
    walk_forward,
)
from src.ict_bot.data.loader import load_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ICT strategy backtest")
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--timeframe", default="H1", help="Entry timeframe")
    parser.add_argument("--htf", default="D", help="Higher timeframe for bias")
    parser.add_argument("--swing-length", type=int, default=20)
    parser.add_argument("--threshold", type=int, default=3, help="Confluence threshold")
    parser.add_argument("--rr", type=float, default=2.0, help="Minimum R:R")
    parser.add_argument("--sl-buffer", type=float, default=10.0, help="SL buffer in pips")
    parser.add_argument("--spread", type=float, default=1.5, help="Spread in pips")
    parser.add_argument("--slippage", type=float, default=0.5, help="Slippage in pips")
    parser.add_argument("--kill-zones", nargs="*", default=None,
                        help="Kill zones to trade (e.g., new_york london)")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward analysis")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sensitivity")
    parser.add_argument("--monte-carlo", action="store_true", help="Run Monte Carlo simulation")
    parser.add_argument("--full", action="store_true", help="Run all analyses")
    parser.add_argument("--no-liquidity-targets", action="store_true",
                        help="Disable liquidity-based TP targets (use R:R only)")
    parser.add_argument("--no-premium-discount", action="store_true",
                        help="Disable premium/discount confluence filter")
    args = parser.parse_args()

    # Load data
    logger.info(f"Loading {args.pair} data...")
    ohlc = load_pair(args.pair, args.timeframe)
    htf_ohlc = load_pair(args.pair, args.htf)
    logger.info(f"  Entry TF: {len(ohlc)} candles ({ohlc.index.min().date()} to {ohlc.index.max().date()})")
    logger.info(f"  HTF:      {len(htf_ohlc)} candles")

    config = BacktestConfig(
        pair=args.pair,
        swing_length=args.swing_length,
        confluence_threshold=args.threshold,
        min_rr=args.rr,
        sl_buffer_pips=args.sl_buffer,
        spread_pips=args.spread,
        slippage_pips=args.slippage,
        target_kill_zones=args.kill_zones,
        use_liquidity_targets=not args.no_liquidity_targets,
        use_premium_discount=not args.no_premium_discount,
    )

    # --- Main backtest ---
    print(f"\nRunning backtest: {args.pair} {args.timeframe} | threshold={args.threshold} | R:R={args.rr}")
    trades, metrics = run_backtest(ohlc, htf_ohlc=htf_ohlc, config=config)
    print(metrics.summary())

    # Print sample trades
    if trades:
        print(f"\nSample trades (first 10):")
        print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'Exit':>9} {'PnL':>8} {'Reason':>7} {'Score':>5}")
        print(f"  {'-'*70}")
        for t in trades[:10]:
            print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} {t['entry_price']:>9.5f} "
                  f"{t['exit_price']:>9.5f} {t['pnl_pips']:>+7.1f} {t['exit_reason']:>7} {t['confluence_score']:>5}")

    # --- Walk-forward ---
    if args.walk_forward or args.full:
        print("\n\nRunning walk-forward analysis...")
        wf_results = walk_forward(ohlc, htf_ohlc=htf_ohlc, config=config)
        print(format_walk_forward_report(wf_results))

    # --- Monte Carlo ---
    if args.monte_carlo or args.full:
        print("\n\nRunning Monte Carlo simulation...")
        mc = monte_carlo(trades, n_simulations=1000)
        print(format_monte_carlo_report(mc))

    # --- Parameter sensitivity ---
    if args.sensitivity or args.full:
        print("\n\nRunning parameter sensitivity analysis...")
        sens = parameter_sensitivity(ohlc, htf_ohlc=htf_ohlc, base_config=config)
        for param_name, param_results in sens.items():
            print(f"\n  {param_name}:")
            for r in param_results:
                status = "+" if r["expectancy"] > 0 else "-"
                print(f"    [{status}] {param_name}={r['value']:<6} -> "
                      f"{r['trades']} trades | WR={r['win_rate']:.1%} | "
                      f"Exp={r['expectancy']:+.3f}R | PnL={r['pnl_pips']:+.1f}")


if __name__ == "__main__":
    main()
