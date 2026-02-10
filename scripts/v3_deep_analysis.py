#!/usr/bin/env python3
"""
Deep analysis of v3 signal detector results.

Tests:
  1. Threshold 3 vs 4 vs 5 across all pairs
  2. OTE-only filter (require OTE in confluences)
  3. Per-pair viability assessment
  4. Walk-forward on best configurations
  5. Long-only vs all-trades comparison
  6. Monte Carlo on best config
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest, simulate_trades
from src.ict_bot.backtest.metrics import calculate_metrics
from src.ict_bot.backtest.walk_forward import (
    format_monte_carlo_report,
    format_walk_forward_report,
    monte_carlo,
    walk_forward,
)
from src.ict_bot.data.loader import load_pair
from src.ict_bot.signals.detector import generate_signals

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

PAIRS = ["EUR_USD", "GBP_USD", "EUR_GBP", "AUD_USD", "NZD_USD", "USD_CAD", "USD_JPY"]


def load_data(pair: str) -> tuple:
    try:
        ohlc = load_pair(pair, "H1")
        htf = load_pair(pair, "D")
        return ohlc, htf
    except Exception:
        return None, None


def run_pair_backtest(pair: str, ohlc, htf, threshold: int = 3, min_rr: float = 2.0,
                      use_breakeven: bool = False, direction_filter: str | None = None) -> tuple:
    config = BacktestConfig(
        pair=pair,
        swing_length=20,
        confluence_threshold=threshold,
        min_rr=min_rr,
        sl_buffer_pips=10.0,
        spread_pips=1.5,
        slippage_pips=0.5,
        use_breakeven=use_breakeven,
        require_htf_bias=True,
        pullback_window=20,
    )
    trades, metrics = run_backtest(ohlc, htf_ohlc=htf, config=config)

    if direction_filter and trades:
        filtered = [t for t in trades if t["direction"] == direction_filter]
        metrics = calculate_metrics(filtered, pip_size=config.pip_size)
        trades = filtered

    return trades, metrics, config


def print_header(title: str):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def print_pair_row(pair: str, metrics, extra: str = ""):
    status = "+" if metrics.expectancy > 0 else "-"
    print(f"  [{status}] {pair:<10} {metrics.total_trades:>5} trades | "
          f"WR={metrics.win_rate:>5.1%} | Exp={metrics.expectancy:>+7.3f}R | "
          f"PnL={metrics.total_pnl_pips:>+8.1f} | PF={metrics.profit_factor:>5.2f} | "
          f"DD={metrics.max_drawdown_pips:>6.1f}{extra}")


def main():
    # Load all data first
    print("Loading data for all pairs...")
    data = {}
    for pair in PAIRS:
        ohlc, htf = load_data(pair)
        if ohlc is not None:
            data[pair] = (ohlc, htf)
            print(f"  {pair}: {len(ohlc)} candles")
        else:
            print(f"  {pair}: SKIP (no data)")

    # =========================================================================
    # TEST 1: Threshold comparison (3 vs 4 vs 5) across all pairs
    # =========================================================================
    for threshold in [3, 4, 5]:
        print_header(f"THRESHOLD = {threshold} (all pairs combined)")
        all_trades = []
        for pair, (ohlc, htf) in data.items():
            trades, metrics, _ = run_pair_backtest(pair, ohlc, htf, threshold=threshold)
            print_pair_row(pair, metrics)
            all_trades.extend(trades)

        combined = calculate_metrics(all_trades)
        print(f"  {'-'*68}")
        print_pair_row("COMBINED", combined)

    # =========================================================================
    # TEST 2: Threshold 4 — Long-only vs All
    # =========================================================================
    print_header("THRESHOLD 4: LONG-ONLY vs ALL TRADES")
    all_trades_t4 = []
    all_trades_t4_long = []
    for pair, (ohlc, htf) in data.items():
        trades_all, m_all, _ = run_pair_backtest(pair, ohlc, htf, threshold=4)
        trades_long, m_long, _ = run_pair_backtest(pair, ohlc, htf, threshold=4, direction_filter="long")
        all_trades_t4.extend(trades_all)
        all_trades_t4_long.extend(trades_long)
        print(f"  {pair:<10} ALL: {m_all.total_trades:>4}t WR={m_all.win_rate:>5.1%} Exp={m_all.expectancy:>+.3f}R | "
              f"LONG: {m_long.total_trades:>4}t WR={m_long.win_rate:>5.1%} Exp={m_long.expectancy:>+.3f}R")

    m_combined = calculate_metrics(all_trades_t4)
    m_combined_long = calculate_metrics(all_trades_t4_long)
    print(f"  {'-'*68}")
    print(f"  {'COMBINED':<10} ALL: {m_combined.total_trades:>4}t WR={m_combined.win_rate:>5.1%} Exp={m_combined.expectancy:>+.3f}R | "
          f"LONG: {m_combined_long.total_trades:>4}t WR={m_combined_long.win_rate:>5.1%} Exp={m_combined_long.expectancy:>+.3f}R")

    # =========================================================================
    # TEST 3: OTE filter — only keep trades where OTE is in confluences
    # =========================================================================
    print_header("OTE FILTER: TRADES WITH OTE vs WITHOUT (threshold=3)")
    all_ote = []
    all_no_ote = []
    for pair, (ohlc, htf) in data.items():
        trades, _, _ = run_pair_backtest(pair, ohlc, htf, threshold=3)
        ote_trades = [t for t in trades if t.get("confluences", {}).get("ote")]
        no_ote_trades = [t for t in trades if not t.get("confluences", {}).get("ote")]
        all_ote.extend(ote_trades)
        all_no_ote.extend(no_ote_trades)

        pip_size = 0.01 if "JPY" in pair else 0.0001
        m_ote = calculate_metrics(ote_trades, pip_size=pip_size)
        m_no = calculate_metrics(no_ote_trades, pip_size=pip_size)
        print(f"  {pair:<10} OTE: {m_ote.total_trades:>4}t Exp={m_ote.expectancy:>+.3f}R PnL={m_ote.total_pnl_pips:>+7.1f} | "
              f"NO-OTE: {m_no.total_trades:>4}t Exp={m_no.expectancy:>+.3f}R PnL={m_no.total_pnl_pips:>+7.1f}")

    m_ote_all = calculate_metrics(all_ote)
    m_no_all = calculate_metrics(all_no_ote)
    print(f"  {'-'*68}")
    print(f"  {'COMBINED':<10} OTE: {m_ote_all.total_trades:>4}t Exp={m_ote_all.expectancy:>+.3f}R PnL={m_ote_all.total_pnl_pips:>+7.1f} | "
          f"NO-OTE: {m_no_all.total_trades:>4}t Exp={m_no_all.expectancy:>+.3f}R PnL={m_no_all.total_pnl_pips:>+7.1f}")

    # =========================================================================
    # TEST 4: R:R comparison (2.0 vs 2.5 vs 3.0) at threshold 4
    # =========================================================================
    print_header("R:R COMPARISON AT THRESHOLD 4")
    for rr in [2.0, 2.5, 3.0]:
        all_trades_rr = []
        for pair, (ohlc, htf) in data.items():
            trades, _, _ = run_pair_backtest(pair, ohlc, htf, threshold=4, min_rr=rr)
            all_trades_rr.extend(trades)
        m = calculate_metrics(all_trades_rr)
        print(f"  R:R={rr:.1f} | {m.total_trades:>4} trades | WR={m.win_rate:>5.1%} | "
              f"Exp={m.expectancy:>+.3f}R | PnL={m.total_pnl_pips:>+8.1f} | PF={m.profit_factor:.2f}")

    # =========================================================================
    # TEST 5: Excluding USD_JPY (structural outlier)
    # =========================================================================
    print_header("THRESHOLD 4 — EXCLUDING USD_JPY")
    exclude_jpy_trades = []
    for pair, (ohlc, htf) in data.items():
        if pair == "USD_JPY":
            continue
        trades, metrics, _ = run_pair_backtest(pair, ohlc, htf, threshold=4)
        print_pair_row(pair, metrics)
        exclude_jpy_trades.extend(trades)

    m_no_jpy = calculate_metrics(exclude_jpy_trades)
    print(f"  {'-'*68}")
    print_pair_row("COMBINED", m_no_jpy)

    # =========================================================================
    # TEST 6: Walk-forward on best pairs with threshold 4
    # =========================================================================
    best_pairs = ["EUR_USD", "GBP_USD", "EUR_GBP"]
    print_header(f"WALK-FORWARD ANALYSIS (threshold=4, pairs={best_pairs})")

    for pair in best_pairs:
        if pair not in data:
            continue
        ohlc, htf = data[pair]
        config = BacktestConfig(
            pair=pair,
            swing_length=20,
            confluence_threshold=4,
            min_rr=2.0,
            sl_buffer_pips=10.0,
            spread_pips=1.5,
            slippage_pips=0.5,
            use_breakeven=False,
            require_htf_bias=True,
            pullback_window=20,
        )
        print(f"\n  --- {pair} ---")
        wf = walk_forward(ohlc, htf_ohlc=htf, config=config, train_months=6, test_months=2, step_months=1)
        if wf:
            positive = sum(1 for w in wf if w["pnl_pips"] > 0)
            total_wf_trades = sum(w["trades"] for w in wf)
            avg_exp = np.mean([w["expectancy"] for w in wf if w["trades"] > 0]) if wf else 0
            print(f"  Windows: {len(wf)} | Profitable: {positive}/{len(wf)} "
                  f"({positive/len(wf):.0%}) | Trades: {total_wf_trades} | Avg Exp: {avg_exp:+.3f}R")
            for w in wf:
                status = "+" if w["pnl_pips"] > 0 else "-"
                print(f"    [{status}] {w['test_start'].strftime('%Y-%m')} to {w['test_end'].strftime('%Y-%m')}: "
                      f"{w['trades']:>3}t | WR={w['win_rate']:>5.1%} | Exp={w['expectancy']:>+.3f}R | "
                      f"PnL={w['pnl_pips']:>+7.1f}")
        else:
            print("  No walk-forward windows (insufficient data)")

    # =========================================================================
    # TEST 7: Monte Carlo on best configuration
    # =========================================================================
    print_header("MONTE CARLO (threshold=4, best 3 pairs)")
    best_trades = []
    for pair in best_pairs:
        if pair not in data:
            continue
        ohlc, htf = data[pair]
        trades, _, _ = run_pair_backtest(pair, ohlc, htf, threshold=4)
        best_trades.extend(trades)

    if best_trades:
        mc = monte_carlo(best_trades, n_simulations=2000)
        print(format_monte_carlo_report(mc))

    # =========================================================================
    # TEST 8: Exit reason analysis at threshold 4
    # =========================================================================
    print_header("EXIT REASON ANALYSIS (threshold=4, all pairs)")
    all_t4_trades = []
    for pair, (ohlc, htf) in data.items():
        trades, _, _ = run_pair_backtest(pair, ohlc, htf, threshold=4)
        all_t4_trades.extend(trades)

    if all_t4_trades:
        df = pd.DataFrame(all_t4_trades)
        for reason in ["TP", "SL", "BE", "TIMEOUT"]:
            subset = df[df["exit_reason"] == reason]
            if len(subset) == 0:
                continue
            avg_pnl = subset["pnl_pips"].mean()
            print(f"  {reason:<8} {len(subset):>4} trades ({len(subset)/len(df):.1%}) | "
                  f"Avg PnL={avg_pnl:>+6.1f} pips | Total={subset['pnl_pips'].sum():>+8.1f} pips")

        # Score breakdown
        print(f"\n  Confluence score breakdown:")
        for score in sorted(df["confluence_score"].unique()):
            s = df[df["confluence_score"] == score]
            r_mult = s["pnl_pips"] / s["risk_pips"]
            print(f"    Score {score}: {len(s):>4} trades | WR={((s['pnl_pips']>0).mean()):>5.1%} | "
                  f"Exp={r_mult.mean():>+.3f}R | PnL={s['pnl_pips'].sum():>+8.1f}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY & RECOMMENDATIONS")
    print("""
  Key Findings:
  1. Threshold comparison: Which threshold maximizes risk-adjusted returns?
  2. OTE filter: Does requiring OTE improve edge significantly?
  3. Long-only: Do shorts drag down performance?
  4. Walk-forward: Does the edge persist across time windows?
  5. Monte Carlo: What are the confidence intervals?

  See results above for data-driven conclusions.
""")


if __name__ == "__main__":
    main()
