#!/usr/bin/env python3
"""Walk-forward validation for M5/M15 best configs and sw=5 on H1.

M5/M15: 83 days of data → use 2-week test windows with 4-week train
H1: 2.8 years → use standard 6-month test windows with 18-month train
Also test sw=5 on H1 since it was the winner on lower TFs.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ict_bot.backtest import BacktestConfig
from ict_bot.backtest.engine import simulate_trades
from ict_bot.backtest.metrics import calculate_metrics
from ict_bot.backtest.walk_forward import monte_carlo, format_monte_carlo_report
from ict_bot.data import load_pair
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals.detector import generate_signals

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PAIRS = ["EUR_USD", "GBP_USD"]


def resample_m5_to_m15(m5: pd.DataFrame) -> pd.DataFrame:
    return m5.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def run_window(ohlc, htf_ohlc, pair, config):
    """Generate signals and simulate trades for a single window."""
    rm = RiskManager(RiskConfig(
        max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0,
    ))

    signals = generate_signals(
        ohlc, htf_ohlc=htf_ohlc, pair=pair,
        swing_length=config.swing_length,
        confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
        skip_days=config.skip_days, use_displacement=config.use_displacement,
        pullback_window=config.pullback_window,
        require_htf_bias=config.require_htf_bias,
        fvg_lookback=config.fvg_lookback, compute_ob=False,
    )

    if not signals:
        return [], calculate_metrics([])

    trades = simulate_trades(signals, ohlc, config, risk_manager=rm)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)
    return trades, metrics


def walk_forward_days(ohlc, htf_ohlc, pair, config, train_days, test_days, step_days):
    """Walk-forward with day-based windows (for shorter datasets)."""
    results = []
    start = ohlc.index.min()
    end = ohlc.index.max()
    current = start
    window_num = 0

    while True:
        train_start = current
        train_end = train_start + pd.Timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_end > end:
            break

        test_ohlc = ohlc[(ohlc.index >= test_start) & (ohlc.index < test_end)]

        if len(test_ohlc) < 20:
            current += pd.Timedelta(days=step_days)
            continue

        trades, metrics = run_window(test_ohlc, htf_ohlc, pair, config)
        window_num += 1

        results.append({
            "window": window_num,
            "test_start": test_start,
            "test_end": test_end,
            "bars": len(test_ohlc),
            "trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "pnl_pips": metrics.total_pnl_pips,
            "profit_factor": metrics.profit_factor,
            "all_trades": trades,
        })

        current += pd.Timedelta(days=step_days)

    return results


def walk_forward_months(ohlc, htf_ohlc, pair, config, train_months, test_months, step_months):
    """Walk-forward with month-based windows (for longer datasets)."""
    results = []
    start = ohlc.index.min()
    end = ohlc.index.max()
    current = start
    window_num = 0

    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        test_ohlc = ohlc[(ohlc.index >= test_start) & (ohlc.index < test_end)]

        if len(test_ohlc) < 50:
            current += pd.DateOffset(months=step_months)
            continue

        trades, metrics = run_window(test_ohlc, htf_ohlc, pair, config)
        window_num += 1

        results.append({
            "window": window_num,
            "test_start": test_start,
            "test_end": test_end,
            "bars": len(test_ohlc),
            "trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "pnl_pips": metrics.total_pnl_pips,
            "profit_factor": metrics.profit_factor,
            "all_trades": trades,
        })

        current += pd.DateOffset(months=step_months)

    return results


def print_wf_results(pair, tf, config_name, results):
    """Print walk-forward results for a single pair/tf/config."""
    if not results:
        print(f"  {pair} {tf}: No valid windows")
        return

    positive = sum(1 for r in results if r["pnl_pips"] > 0)
    total_trades = sum(r["trades"] for r in results)
    total_pnl = sum(r["pnl_pips"] for r in results)
    avg_exp = np.mean([r["expectancy"] for r in results if r["trades"] > 0]) if any(r["trades"] > 0 for r in results) else 0

    print(f"\n  {pair} {tf} ({config_name})")
    print(f"  {'Window':<8} {'Period':<28} {'Bars':>6} {'Trades':>7} {'WR%':>7} {'Exp(R)':>8} {'PF':>6} {'PnL':>10}")
    print(f"  {'-' * 82}")

    for r in results:
        period = f"{r['test_start'].strftime('%Y-%m-%d')} to {r['test_end'].strftime('%Y-%m-%d')}"
        wr = f"{r['win_rate']:.1%}" if r['trades'] > 0 else "N/A"
        pf = f"{r['profit_factor']:.2f}" if r['trades'] > 0 else "N/A"
        print(f"  {r['window']:<8} {period:<28} {r['bars']:>6} {r['trades']:>7} "
              f"{wr:>7} {r['expectancy']:>+7.3f} {pf:>6} {r['pnl_pips']:>+9.1f}")

    print(f"  {'-' * 82}")
    print(f"  Windows: {len(results)} | Profitable: {positive}/{len(results)} "
          f"({positive/len(results):.0%}) | Trades: {total_trades} | "
          f"AvgExp: {avg_exp:+.3f}R | TotalPnL: {total_pnl:+.0f} pips")


def main():
    print("=" * 80)
    print("Walk-Forward Validation: M5, M15, and H1 (sw=5)")
    print("=" * 80)

    # Load data
    data = {}
    htf_data = {}
    for pair in PAIRS:
        print(f"\nLoading {pair}...")
        m5 = load_pair(pair, "M5")
        h1 = load_pair(pair, "H1")
        daily = load_pair(pair, "D")
        m15 = resample_m5_to_m15(m5)
        data[(pair, "M5")] = m5
        data[(pair, "M15")] = m15
        data[(pair, "H1")] = h1
        htf_data[pair] = daily
        print(f"  M5:  {len(m5)} bars ({m5.index[0].date()} to {m5.index[-1].date()})")
        print(f"  M15: {len(m15)} bars")
        print(f"  H1:  {len(h1)} bars ({h1.index[0].date()} to {h1.index[-1].date()})")

    # ── Configs to validate ──────────────────────────────────────
    configs = {
        "M5_best": BacktestConfig(
            swing_length=5, confluence_threshold=3, min_rr=2.0,
            sl_buffer_pips=5.0, skip_days=[], use_displacement=False,
            use_breakeven=False, use_partial_tp=False, compute_ob=False,
            fvg_lookback=16, pullback_window=80, max_sl_pips=50.0,
            spread_pips=0.5, require_htf_bias=True,
        ),
        "M15_best": BacktestConfig(
            swing_length=5, confluence_threshold=4, min_rr=2.0,
            sl_buffer_pips=5.0, skip_days=[], use_displacement=False,
            use_breakeven=False, use_partial_tp=False, compute_ob=False,
            fvg_lookback=16, pullback_window=40, max_sl_pips=100.0,
            spread_pips=1.0, require_htf_bias=True,
        ),
        "H1_sw10": BacktestConfig(
            swing_length=10, confluence_threshold=3, min_rr=2.0,
            sl_buffer_pips=10.0, skip_days=[0, 4], use_displacement=False,
            use_breakeven=False, use_partial_tp=False, compute_ob=False,
            fvg_lookback=16, pullback_window=40, max_sl_pips=100.0,
            spread_pips=1.0, require_htf_bias=True,
        ),
        "H1_sw5": BacktestConfig(
            swing_length=5, confluence_threshold=3, min_rr=2.0,
            sl_buffer_pips=5.0, skip_days=[], use_displacement=False,
            use_breakeven=False, use_partial_tp=False, compute_ob=False,
            fvg_lookback=16, pullback_window=40, max_sl_pips=100.0,
            spread_pips=1.0, require_htf_bias=True,
        ),
    }

    # ── Walk-Forward: M5 best (sw=5, thresh=3) ──────────────────
    # 83 days: train=28d, test=14d, step=7d → ~5 windows
    print("\n" + "=" * 80)
    print("WALK-FORWARD: M5 (sw=5, thresh=3, no skip)")
    print("Train=28d, Test=14d, Step=7d")
    print("=" * 80)

    all_m5_trades = []
    for pair in PAIRS:
        cfg = configs["M5_best"]
        cfg.pair = pair
        t0 = time.time()
        results = walk_forward_days(
            data[(pair, "M5")], htf_data[pair], pair, cfg,
            train_days=28, test_days=14, step_days=7,
        )
        elapsed = time.time() - t0
        print_wf_results(pair, "M5", "sw=5 thresh=3", results)
        print(f"  (computed in {elapsed:.0f}s)")
        for r in results:
            all_m5_trades.extend(r["all_trades"])

    # ── Walk-Forward: M15 best (sw=5, thresh=4) ─────────────────
    print("\n" + "=" * 80)
    print("WALK-FORWARD: M15 (sw=5, thresh=4, no skip)")
    print("Train=28d, Test=14d, Step=7d")
    print("=" * 80)

    all_m15_trades = []
    for pair in PAIRS:
        cfg = configs["M15_best"]
        cfg.pair = pair
        t0 = time.time()
        results = walk_forward_days(
            data[(pair, "M15")], htf_data[pair], pair, cfg,
            train_days=28, test_days=14, step_days=7,
        )
        elapsed = time.time() - t0
        print_wf_results(pair, "M15", "sw=5 thresh=4", results)
        print(f"  (computed in {elapsed:.0f}s)")
        for r in results:
            all_m15_trades.extend(r["all_trades"])

    # ── Walk-Forward: H1 sw=10 baseline (2.8 years) ─────────────
    print("\n" + "=" * 80)
    print("WALK-FORWARD: H1 baseline (sw=10, thresh=3, skip Mon/Fri)")
    print("Train=18m, Test=6m, Step=3m")
    print("=" * 80)

    all_h1_sw10_trades = []
    for pair in PAIRS:
        cfg = configs["H1_sw10"]
        cfg.pair = pair
        t0 = time.time()
        results = walk_forward_months(
            data[(pair, "H1")], htf_data[pair], pair, cfg,
            train_months=18, test_months=6, step_months=3,
        )
        elapsed = time.time() - t0
        print_wf_results(pair, "H1", "sw=10 thresh=3 skip", results)
        print(f"  (computed in {elapsed:.0f}s)")
        for r in results:
            all_h1_sw10_trades.extend(r["all_trades"])

    # ── Walk-Forward: H1 sw=5 (does the M5/M15 winner work on H1?) ──
    print("\n" + "=" * 80)
    print("WALK-FORWARD: H1 with sw=5 (testing if M5/M15 winner works on H1)")
    print("Train=18m, Test=6m, Step=3m")
    print("=" * 80)

    all_h1_sw5_trades = []
    for pair in PAIRS:
        cfg = configs["H1_sw5"]
        cfg.pair = pair
        t0 = time.time()
        results = walk_forward_months(
            data[(pair, "H1")], htf_data[pair], pair, cfg,
            train_months=18, test_months=6, step_months=3,
        )
        elapsed = time.time() - t0
        print_wf_results(pair, "H1", "sw=5 thresh=3 no skip", results)
        print(f"  (computed in {elapsed:.0f}s)")
        for r in results:
            all_h1_sw5_trades.extend(r["all_trades"])

    # ── Monte Carlo for best configs ─────────────────────────────
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATIONS (1000 runs)")
    print("=" * 80)

    for name, trades in [
        ("M5 (sw=5 thresh=3)", all_m5_trades),
        ("M15 (sw=5 thresh=4)", all_m15_trades),
        ("H1 baseline (sw=10)", all_h1_sw10_trades),
        ("H1 sw=5", all_h1_sw5_trades),
    ]:
        if trades:
            mc = monte_carlo(trades, n_simulations=1000)
            print(f"\n  {name}:")
            print(f"    Trades: {len(trades)}")
            print(f"    Median PnL: {mc['median_pnl']:+.0f} pips")
            print(f"    95% CI: [{mc['ci_low']:+.0f}, {mc['ci_high']:+.0f}] pips")
            print(f"    P(profitable): {mc['prob_profitable']:.1%}")
            print(f"    Median MaxDD: {mc['median_max_dd']:.0f} pips")
            print(f"    Worst-case DD (95%): {mc['worst_case_dd']:.0f} pips")
        else:
            print(f"\n  {name}: No trades")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nAll walk-forward validation complete.")
    print("Key question: Are M5/M15 profitable in ALL test windows?")
    print("If yes → genuine edge, safe to trade live.")
    print("If no → may be overfit to specific market conditions.")


if __name__ == "__main__":
    main()
