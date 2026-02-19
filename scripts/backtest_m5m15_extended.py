#!/usr/bin/env python3
"""Backtest M5 and M15 with extended IB data + walk-forward validation.

Pre-computes signals once per (pair, tf, swing_length), then post-filters
by confluence_threshold to avoid redundant expensive signal generation.

Usage:
    PYTHONPATH=src python scripts/backtest_m5m15_extended.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_bot.backtest.engine import BacktestConfig, simulate_trades, calculate_metrics
from ict_bot.backtest.walk_forward import monte_carlo
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals import generate_signals

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

ALL_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]

# Pre-compute with lowest threshold, then post-filter
SWING_LENGTHS = [5, 10]
THRESHOLDS = [3, 4]
TIMEFRAMES = ["M5", "M15"]

# Global signal cache: (pair, tf, swing_length) -> list of signals
_signal_cache: dict[tuple, list] = {}


def get_signals_cached(ohlc, htf_ohlc, pair, tf, swing_length, confluence_threshold):
    """Get signals, using cache if available. Generate with lowest threshold and filter."""
    cache_key = (pair, tf, swing_length, id(ohlc))

    if cache_key not in _signal_cache:
        # Generate with lowest threshold (2) so we can filter up later
        t0 = time.time()
        signals = generate_signals(
            ohlc=ohlc,
            htf_ohlc=htf_ohlc,
            pair=pair,
            swing_length=swing_length,
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
        _signal_cache[cache_key] = signals or []
        print(f"    [cache] Generated {len(_signal_cache[cache_key])} signals for "
              f"{pair} {tf} sw={swing_length} ({elapsed:.0f}s)", flush=True)

    # Post-filter by threshold
    return [s for s in _signal_cache[cache_key] if s.confluence_score >= confluence_threshold]


def run_backtest_from_signals(signals, ohlc):
    """Simulate trades from pre-filtered signals."""
    if not signals:
        return [], None

    bc = BacktestConfig(
        use_breakeven=False,
        use_partial_tp=False,
        max_sl_pips=100.0,
    )
    rm = RiskManager(RiskConfig(
        max_risk_per_trade=0.01,
        starting_balance=10_000,
        max_consecutive_losses=999,
        max_daily_loss=1.0,
        max_drawdown=1.0,
    ))
    trades = simulate_trades(signals, ohlc, bc, rm)
    if not trades:
        return [], None
    metrics = calculate_metrics(trades)
    return trades, metrics


def walk_forward_days(ohlc, htf_ohlc, pair, tf, swing_length, confluence_threshold,
                      train_days=60, test_days=30, step_days=15):
    """Walk-forward validation — generates signals on test slices (no cache)."""
    if len(ohlc) == 0:
        return []

    start = ohlc.index[0]
    end = ohlc.index[-1]
    results = []
    window_start = start

    while True:
        train_end = window_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > end:
            break

        test_ohlc = ohlc[(ohlc.index >= train_end) & (ohlc.index < test_end)]
        test_htf = htf_ohlc[htf_ohlc.index < test_end] if htf_ohlc is not None else None

        if len(test_ohlc) < 50:
            window_start += pd.Timedelta(days=step_days)
            continue

        # For walk-forward, we use the full signal cache filtered to test window
        signals = get_signals_cached(ohlc, htf_ohlc, pair, tf, swing_length, confluence_threshold)
        # Filter signals to test window only
        test_signals = [s for s in signals
                       if hasattr(s, 'timestamp') and train_end <= s.timestamp < test_end]

        if not test_signals:
            # Fallback: filter by entry_price timestamp using index
            test_signals = [s for s in signals
                          if hasattr(s, 'bar_index') and s.bar_index is not None]

        trades, metrics = run_backtest_from_signals(test_signals, test_ohlc)

        if metrics and len(trades) >= 3:
            results.append({
                "test_start": train_end,
                "test_end": test_end,
                "trades": len(trades),
                "win_rate": metrics.win_rate,
                "expectancy": metrics.expectancy,
                "total_pnl": metrics.total_pnl_pips,
                "profit_factor": metrics.profit_factor,
            })

        window_start += pd.Timedelta(days=step_days)

    return results


def main():
    print("=" * 80)
    print("M5/M15 EXTENDED BACKTEST — IB Historical Data")
    print("=" * 80)
    print()

    # Check available data
    available = {}
    for tf in TIMEFRAMES:
        for pair in ALL_PAIRS:
            path = DATA_DIR / f"{pair}_{tf}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                days = (df.index[-1] - df.index[0]).days
                available[(pair, tf)] = {"bars": len(df), "days": days}
                print(f"  {pair} {tf}: {len(df):,} bars, {days} days")
    print()

    # ── Phase 1: Pre-compute all signals ──────────────────────────────
    print("=" * 80)
    print("PHASE 1: PRE-COMPUTING SIGNALS (this is the slow part)")
    print("=" * 80)
    print()

    # Load all data
    data = {}
    for tf in TIMEFRAMES:
        for pair in ALL_PAIRS:
            path = DATA_DIR / f"{pair}_{tf}.parquet"
            if not path.exists():
                continue
            ohlc = pd.read_parquet(path)
            htf_path = DATA_DIR / f"{pair}_D.parquet"
            htf_ohlc = pd.read_parquet(htf_path) if htf_path.exists() else None
            data[(pair, tf)] = (ohlc, htf_ohlc)

    # Pre-compute signals for all (pair, tf, swing_length) combos
    total_combos = len(data) * len(SWING_LENGTHS)
    done = 0
    for (pair, tf), (ohlc, htf_ohlc) in data.items():
        for sw in SWING_LENGTHS:
            done += 1
            print(f"  [{done}/{total_combos}] {pair} {tf} sw={sw}...", flush=True)
            # This populates the cache
            get_signals_cached(ohlc, htf_ohlc, pair, tf, sw, min(THRESHOLDS))

    print(f"\nSignal cache populated: {len(_signal_cache)} entries\n")

    # ── Phase 2: Full-sample backtest (fast — just filters + simulates) ──
    print("=" * 80)
    print("PHASE 2: FULL-SAMPLE BACKTEST")
    print("=" * 80)
    print()
    print(f"{'Config':<22} {'Pair':<10} {'Trades':>7} {'WR':>6} {'Expect':>8} {'PF':>6} {'Pips':>8} {'Days':>5}")
    print("-" * 80)

    all_results = []
    configs = []
    for tf in TIMEFRAMES:
        for sw in SWING_LENGTHS:
            for thresh in THRESHOLDS:
                configs.append({
                    "name": f"{tf} sw={sw} t={thresh}",
                    "tf": tf,
                    "swing_length": sw,
                    "confluence_threshold": thresh,
                })

    for cfg in configs:
        tf = cfg["tf"]
        for pair in ALL_PAIRS:
            if (pair, tf) not in data:
                continue

            ohlc, htf_ohlc = data[(pair, tf)]
            signals = get_signals_cached(ohlc, htf_ohlc, pair, tf,
                                         cfg["swing_length"], cfg["confluence_threshold"])
            trades, metrics = run_backtest_from_signals(signals, ohlc)

            if metrics and len(trades) >= 5:
                row = {
                    "config": cfg["name"],
                    "pair": pair,
                    "trades": len(trades),
                    "win_rate": metrics.win_rate,
                    "expectancy": metrics.expectancy,
                    "profit_factor": metrics.profit_factor,
                    "total_pips": metrics.total_pnl_pips,
                    "sharpe": metrics.sharpe_ratio,
                    "max_dd": metrics.max_drawdown_pct,
                    "days": available.get((pair, tf), {}).get("days", 0),
                }
                all_results.append(row)
                print(
                    f"  {cfg['name']:<20} {pair:<10} {len(trades):>7} "
                    f"{metrics.win_rate:>5.1f}% {metrics.expectancy:>+7.3f}R "
                    f"{metrics.profit_factor:>5.2f} {metrics.total_pnl_pips:>+8.0f} "
                    f"{row['days']:>5}"
                )
            else:
                ntrades = len(trades) if trades else 0
                if ntrades > 0:
                    print(f"  {cfg['name']:<20} {pair:<10} {ntrades:>7} trades — insufficient (<5)")

    # ── Summary by config ────────────────────────────────────────────
    if all_results:
        print()
        print("=" * 80)
        print("SUMMARY BY CONFIG (averaged across pairs)")
        print("=" * 80)
        print(f"{'Config':<22} {'Pairs':>5} {'Trades':>7} {'AvgWR':>6} {'AvgExp':>8} {'AvgPF':>6} {'TotPips':>9} {'Pairs+/-'}")
        print("-" * 80)

        df_results = pd.DataFrame(all_results)
        for cfg_name, group in df_results.groupby("config", sort=False):
            n_pairs = len(group)
            n_positive = (group["expectancy"] > 0).sum()
            n_negative = n_pairs - n_positive
            print(
                f"  {cfg_name:<20} {n_pairs:>5} {group['trades'].sum():>7} "
                f"{group['win_rate'].mean():>5.1f}% {group['expectancy'].mean():>+7.3f}R "
                f"{group['profit_factor'].mean():>5.2f} {group['total_pips'].sum():>+9.0f} "
                f"{n_positive}+/{n_negative}-"
            )

    # ── Phase 3: Walk-forward validation ─────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 3: WALK-FORWARD VALIDATION")
    print("=" * 80)

    # Pick best configs from Phase 2 for walk-forward
    wf_configs = [
        ("M5",  5,  3),
        ("M5",  10, 3),
        ("M15", 5,  4),
        ("M15", 10, 3),
    ]

    for tf, sw, thresh in wf_configs:
        cfg_name = f"{tf} sw={sw} t={thresh}"
        print(f"\n--- {cfg_name} ---")
        print(f"{'Pair':<10} {'Windows':>8} {'Win%':>6} {'AvgTrades':>10} {'AvgExp':>8} {'TotPnL':>8}")
        print("-" * 60)

        for pair in ALL_PAIRS:
            if (pair, tf) not in data:
                continue

            ohlc, htf_ohlc = data[(pair, tf)]
            days_available = (ohlc.index[-1] - ohlc.index[0]).days

            if days_available >= 180:
                train, test, step = 90, 30, 15
            elif days_available >= 120:
                train, test, step = 60, 30, 15
            else:
                train, test, step = 28, 14, 7

            wf_results = walk_forward_days(
                ohlc, htf_ohlc, pair, tf, sw, thresh, train, test, step
            )

            if wf_results:
                n_windows = len(wf_results)
                n_profitable = sum(1 for r in wf_results if r["total_pnl"] > 0)
                avg_trades = sum(r["trades"] for r in wf_results) / n_windows
                avg_exp = sum(r["expectancy"] for r in wf_results) / n_windows
                tot_pnl = sum(r["total_pnl"] for r in wf_results)
                win_pct = n_profitable / n_windows * 100
                print(
                    f"  {pair:<10} {n_windows:>8} {win_pct:>5.0f}% "
                    f"{avg_trades:>10.1f} {avg_exp:>+7.3f}R {tot_pnl:>+8.0f}"
                )
            else:
                print(f"  {pair:<10} — insufficient data")

    # ── Phase 4: Monte Carlo ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("PHASE 4: MONTE CARLO SIMULATION")
    print("=" * 80)
    print()

    mc_configs = [
        ("M5",  5, 3),
        ("M5",  10, 3),
        ("M15", 5, 4),
        ("M15", 10, 3),
    ]

    for tf, sw, thresh in mc_configs:
        cfg_name = f"{tf} sw={sw} t={thresh}"
        all_trades = []

        for pair in ALL_PAIRS:
            if (pair, tf) not in data:
                continue
            ohlc, htf_ohlc = data[(pair, tf)]
            signals = get_signals_cached(ohlc, htf_ohlc, pair, tf, sw, thresh)
            trades, _ = run_backtest_from_signals(signals, ohlc)
            all_trades.extend(trades)

        if len(all_trades) >= 20:
            r_values = []
            for t in all_trades:
                if t.get("risk_pips") and t["risk_pips"] > 0:
                    r_values.append(t.get("pnl_pips", 0) / t["risk_pips"])
                else:
                    r_values.append(t.get("pnl_r", 0))

            mc = monte_carlo(r_values, n_simulations=5000, n_trades=len(r_values))
            print(f"  {cfg_name}: {len(all_trades)} trades")
            print(f"    Median final R: {mc['median_final_r']:+.1f}R")
            print(f"    5th percentile: {mc['percentile_5_final_r']:+.1f}R")
            print(f"    P(profitable):  {mc['prob_profitable']*100:.0f}%")
            print(f"    Median max DD:  {mc['median_max_drawdown_r']:.1f}R")
            print()

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
