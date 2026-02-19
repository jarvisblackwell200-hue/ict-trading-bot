#!/usr/bin/env python3
"""Parallel M5/M15 backtest — spawns one worker per (tf, swing_length).

Each worker handles all 7 pairs for its combo, writes results to a temp file.
Main process collects and prints summary + walk-forward + Monte Carlo.

Usage:
    PYTHONPATH=src python scripts/backtest_parallel.py
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_bot.backtest.engine import BacktestConfig, simulate_trades, calculate_metrics
from ict_bot.backtest.walk_forward import monte_carlo
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals import generate_signals

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ALL_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]
THRESHOLDS = [3, 4]


def generate_and_backtest(args):
    """Worker: generate signals for one (tf, sw, pair) and backtest with all thresholds.

    Returns a list of result dicts.
    """
    tf, sw, pair = args
    path = DATA_DIR / f"{pair}_{tf}.parquet"
    if not path.exists():
        return []

    ohlc = pd.read_parquet(path)
    htf_path = DATA_DIR / f"{pair}_D.parquet"
    htf_ohlc = pd.read_parquet(htf_path) if htf_path.exists() else None
    days = (ohlc.index[-1] - ohlc.index[0]).days

    # Generate signals once with lowest threshold
    t0 = time.time()
    all_signals = generate_signals(
        ohlc=ohlc,
        htf_ohlc=htf_ohlc,
        pair=pair,
        swing_length=sw,
        confluence_threshold=min(THRESHOLDS),
        min_rr=2.0,
        sl_buffer_pips=10.0,
        skip_days=[],
        use_displacement=False,
        fvg_lookback=16,
        pullback_window=40,
        compute_ob=False,
    )
    elapsed = time.time() - t0
    all_signals = all_signals or []
    print(f"  [{tf} sw={sw}] {pair}: {len(all_signals)} signals ({elapsed:.0f}s)", flush=True)

    results = []

    # Backtest with each threshold
    for thresh in THRESHOLDS:
        signals = [s for s in all_signals if s.confluence_score >= thresh]
        if not signals:
            continue

        bc = BacktestConfig(use_breakeven=False, use_partial_tp=False, max_sl_pips=100.0)
        rm = RiskManager(RiskConfig(
            max_risk_per_trade=0.01, starting_balance=10_000,
            max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0,
        ))
        trades = simulate_trades(signals, ohlc, bc, rm)
        if not trades or len(trades) < 5:
            continue
        metrics = calculate_metrics(trades)

        results.append({
            "config": f"{tf} sw={sw} t={thresh}",
            "tf": tf, "sw": sw, "thresh": thresh,
            "pair": pair,
            "trades": len(trades),
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "profit_factor": metrics.profit_factor,
            "total_pips": metrics.total_pnl_pips,
            "sharpe": metrics.sharpe_ratio,
            "max_dd": metrics.max_drawdown_pct,
            "days": days,
            # Store R-values for Monte Carlo
            "r_values": [
                t.get("pnl_pips", 0) / t["risk_pips"]
                if t.get("risk_pips") and t["risk_pips"] > 0
                else t.get("pnl_r", 0)
                for t in trades
            ],
        })

    # Walk-forward for each threshold
    for thresh in THRESHOLDS:
        signals = [s for s in all_signals if s.confluence_score >= thresh]
        if not signals:
            continue

        # Walk-forward windows
        start = ohlc.index[0]
        end = ohlc.index[-1]
        days_avail = (end - start).days

        if days_avail >= 180:
            train, test, step = 90, 30, 15
        elif days_avail >= 120:
            train, test, step = 60, 30, 15
        else:
            train, test, step = 28, 14, 7

        wf_windows = []
        window_start = start
        while True:
            train_end = window_start + pd.Timedelta(days=train)
            test_end = train_end + pd.Timedelta(days=test)
            if test_end > end:
                break

            # Filter signals to test window
            test_signals = [s for s in signals if train_end <= s.timestamp < test_end]
            test_ohlc = ohlc[(ohlc.index >= train_end) & (ohlc.index < test_end)]

            if len(test_ohlc) >= 50 and test_signals:
                bc = BacktestConfig(use_breakeven=False, use_partial_tp=False, max_sl_pips=100.0)
                rm = RiskManager(RiskConfig(
                    max_risk_per_trade=0.01, starting_balance=10_000,
                    max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0,
                ))
                wf_trades = simulate_trades(test_signals, test_ohlc, bc, rm)
                if wf_trades and len(wf_trades) >= 3:
                    wf_m = calculate_metrics(wf_trades)
                    wf_windows.append({
                        "trades": len(wf_trades),
                        "expectancy": wf_m.expectancy,
                        "total_pnl": wf_m.total_pnl_pips,
                    })

            window_start += pd.Timedelta(days=step)

        if wf_windows:
            for r in results:
                if r["pair"] == pair and r["thresh"] == thresh and r["sw"] == sw:
                    r["wf_windows"] = len(wf_windows)
                    r["wf_profitable"] = sum(1 for w in wf_windows if w["total_pnl"] > 0)
                    r["wf_avg_trades"] = sum(w["trades"] for w in wf_windows) / len(wf_windows)
                    r["wf_avg_exp"] = sum(w["expectancy"] for w in wf_windows) / len(wf_windows)
                    r["wf_tot_pnl"] = sum(w["total_pnl"] for w in wf_windows)

    return results


def main():
    print("=" * 90)
    print("PARALLEL M5/M15 BACKTEST — IB Historical Data (7 months M5, 14 months M15)")
    print("=" * 90)
    print()

    # Check available data
    for tf in ["M5", "M15"]:
        for pair in ALL_PAIRS:
            path = DATA_DIR / f"{pair}_{tf}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                days = (df.index[-1] - df.index[0]).days
                print(f"  {pair} {tf}: {len(df):,} bars, {days} days")
    print()

    # Build work items: (tf, sw, pair) — one signal generation per item
    work_items = []
    for tf in ["M5", "M15"]:
        for sw in [5, 10]:
            for pair in ALL_PAIRS:
                path = DATA_DIR / f"{pair}_{tf}.parquet"
                if path.exists():
                    work_items.append((tf, sw, pair))

    n_workers = min(cpu_count(), 4)  # Don't overload the machine
    print(f"Launching {len(work_items)} tasks across {n_workers} workers...")
    print(f"(Each worker generates signals once, then backtests + walk-forwards all thresholds)")
    print()

    t_start = time.time()
    with Pool(n_workers) as pool:
        all_result_lists = pool.map(generate_and_backtest, work_items)

    total_time = time.time() - t_start
    print(f"\nTotal compute time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Flatten results
    all_results = []
    for result_list in all_result_lists:
        all_results.extend(result_list)

    if not all_results:
        print("No results!")
        return

    # ── Full-sample results ──────────────────────────────────────────
    print()
    print("=" * 90)
    print("FULL-SAMPLE BACKTEST RESULTS")
    print("=" * 90)
    print()
    print(f"{'Config':<18} {'Pair':<10} {'Trades':>7} {'WR':>6} {'Expect':>8} {'PF':>6} {'Pips':>8} {'Days':>5}")
    print("-" * 80)

    for r in sorted(all_results, key=lambda x: (x["tf"], x["sw"], x["thresh"], x["pair"])):
        print(
            f"  {r['config']:<16} {r['pair']:<10} {r['trades']:>7} "
            f"{r['win_rate']:>5.1f}% {r['expectancy']:>+7.3f}R "
            f"{r['profit_factor']:>5.2f} {r['total_pips']:>+8.0f} {r['days']:>5}"
        )

    # ── Summary by config ────────────────────────────────────────────
    print()
    print("=" * 90)
    print("SUMMARY BY CONFIG")
    print("=" * 90)
    print(f"{'Config':<18} {'Pairs':>5} {'Trades':>7} {'AvgWR':>6} {'AvgExp':>8} {'AvgPF':>6} {'TotPips':>9} {'Pairs+/-'}")
    print("-" * 80)

    df = pd.DataFrame(all_results)
    for cfg_name, group in df.groupby("config", sort=False):
        n_pos = (group["expectancy"] > 0).sum()
        n_neg = len(group) - n_pos
        print(
            f"  {cfg_name:<16} {len(group):>5} {group['trades'].sum():>7} "
            f"{group['win_rate'].mean():>5.1f}% {group['expectancy'].mean():>+7.3f}R "
            f"{group['profit_factor'].mean():>5.2f} {group['total_pips'].sum():>+9.0f} "
            f"{n_pos}+/{n_neg}-"
        )

    # ── Walk-forward results ─────────────────────────────────────────
    wf_results = [r for r in all_results if "wf_windows" in r]
    if wf_results:
        print()
        print("=" * 90)
        print("WALK-FORWARD VALIDATION")
        print("=" * 90)
        print()

        current_cfg = None
        for r in sorted(wf_results, key=lambda x: (x["tf"], x["sw"], x["thresh"], x["pair"])):
            cfg = r["config"]
            if cfg != current_cfg:
                if current_cfg is not None:
                    print()
                current_cfg = cfg
                print(f"--- {cfg} ---")
                print(f"  {'Pair':<10} {'Windows':>8} {'Win%':>6} {'AvgTrades':>10} {'AvgExp':>8} {'TotPnL':>8}")
                print("  " + "-" * 58)

            win_pct = r["wf_profitable"] / r["wf_windows"] * 100 if r["wf_windows"] else 0
            print(
                f"  {r['pair']:<10} {r['wf_windows']:>8} {win_pct:>5.0f}% "
                f"{r['wf_avg_trades']:>10.1f} {r['wf_avg_exp']:>+7.3f}R {r['wf_tot_pnl']:>+8.0f}"
            )

    # ── Monte Carlo ──────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("MONTE CARLO SIMULATION")
    print("=" * 90)
    print()

    for cfg_name, group in df.groupby("config", sort=False):
        all_r = []
        for _, row in group.iterrows():
            all_r.extend(row["r_values"])

        if len(all_r) >= 20:
            mc = monte_carlo([{"pnl_pips": r} for r in all_r], n_simulations=5000)
            print(f"  {cfg_name}: {len(all_r)} trades")
            print(f"    Median final R:  {mc['median_pnl']:+.1f}R")
            print(f"    95% CI:          [{mc['ci_low']:+.1f}R, {mc['ci_high']:+.1f}R]")
            print(f"    P(profitable):   {mc['prob_profitable']*100:.0f}%")
            print(f"    Median max DD:   {mc['median_max_dd']:.1f}R")
            print()

    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
