#!/usr/bin/env python3
"""Backtest ICT strategy on M5 and M15 timeframes with parameter sweeps.

Optimized: pre-compute signals once per (pair, tf, swing_length) with min
threshold, then post-filter by confluence_threshold and skip_days. This
reduces expensive signal generation from 72 to 12 calls.
"""
from __future__ import annotations

import itertools
import logging
import time

import pandas as pd

from ict_bot.backtest import BacktestConfig
from ict_bot.backtest.engine import simulate_trades
from ict_bot.backtest.metrics import calculate_metrics
from ict_bot.data import load_pair
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals.detector import generate_signals

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PAIRS = ["EUR_USD", "GBP_USD"]

# Best H1 config as baseline reference
BEST_H1 = dict(
    swing_length=10,
    confluence_threshold=3,
    min_rr=2.0,
    skip_days=[0, 4],
    use_displacement=False,
    sl_buffer_pips=10.0,
)

# Parameter sweep dimensions
SWING_LENGTHS = [5, 10, 15]
CONFLUENCE_THRESHOLDS = [2, 3, 4]
SKIP_DAYS_OPTIONS = [None, [0, 4]]
FIXED_PARAMS = dict(min_rr=2.0, sl_buffer_pips=5.0, use_displacement=False)


def resample_m5_to_m15(m5: pd.DataFrame) -> pd.DataFrame:
    """Resample M5 OHLC to M15."""
    return m5.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def run_backtest_from_signals(signals, ohlc, pair, timeframe, params):
    """Run simulate_trades + metrics from pre-filtered signals."""
    config = BacktestConfig(
        pair=pair,
        swing_length=params["swing_length"],
        confluence_threshold=params["confluence_threshold"],
        min_rr=params["min_rr"],
        sl_buffer_pips=params["sl_buffer_pips"],
        skip_days=params.get("skip_days") or [],
        use_displacement=params["use_displacement"],
        use_breakeven=False,
        use_partial_tp=False,
        compute_ob=False,
        fvg_lookback=16,
        pullback_window=80 if timeframe == "M5" else 40,
        max_sl_pips=50.0 if timeframe == "M5" else 100.0,
        spread_pips=0.5 if timeframe == "M5" else 1.0,
        use_confluence_sizing=False,
        require_htf_bias=True,
    )

    rm = RiskManager(RiskConfig(
        max_consecutive_losses=999,
        max_daily_loss=1.0,
        max_drawdown=1.0,
    ))

    try:
        trades = simulate_trades(signals, ohlc, config, risk_manager=rm)
        metrics = calculate_metrics(trades, pip_size=config.pip_size)
    except Exception as e:
        logging.warning("Backtest failed for %s %s: %s", pair, timeframe, e)
        return None

    if metrics.total_trades < 5:
        return None

    return {
        "pair": pair,
        "timeframe": timeframe,
        "swing_length": params["swing_length"],
        "confluence_threshold": params["confluence_threshold"],
        "min_rr": params["min_rr"],
        "sl_buffer_pips": params["sl_buffer_pips"],
        "skip_days": str(params.get("skip_days") or "none"),
        "use_displacement": params["use_displacement"],
        "trades": metrics.total_trades,
        "win_rate": round(metrics.win_rate * 100, 1),
        "expectancy": round(metrics.expectancy, 3),
        "profit_factor": round(metrics.profit_factor, 2),
        "total_pips": round(metrics.total_pnl_pips, 0),
        "max_dd_pct": round(metrics.max_drawdown_pct * 100, 1),
        "avg_rr": round(metrics.avg_rr, 2),
        "sharpe": round(metrics.sharpe_ratio, 2),
    }


def run_h1_baseline(pair, h1, htf_ohlc, params):
    """Run H1 baseline using generate_signals directly."""
    config = BacktestConfig(
        pair=pair,
        swing_length=params["swing_length"],
        confluence_threshold=params["confluence_threshold"],
        min_rr=params["min_rr"],
        sl_buffer_pips=params["sl_buffer_pips"],
        skip_days=params.get("skip_days") or [],
        use_displacement=params["use_displacement"],
        use_breakeven=False,
        use_partial_tp=False,
        compute_ob=False,
        fvg_lookback=16,
        pullback_window=40,
        max_sl_pips=100.0,
        spread_pips=1.0,
        use_confluence_sizing=False,
        require_htf_bias=True,
    )

    rm = RiskManager(RiskConfig(
        max_consecutive_losses=999,
        max_daily_loss=1.0,
        max_drawdown=1.0,
    ))

    try:
        signals = generate_signals(
            h1, htf_ohlc=htf_ohlc, pair=pair,
            swing_length=config.swing_length,
            confluence_threshold=config.confluence_threshold,
            min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
            skip_days=config.skip_days, use_displacement=config.use_displacement,
            pullback_window=config.pullback_window,
            require_htf_bias=config.require_htf_bias,
            fvg_lookback=config.fvg_lookback, compute_ob=False,
        )
        if not signals:
            return None
        trades = simulate_trades(signals, h1, config, risk_manager=rm)
        metrics = calculate_metrics(trades, pip_size=config.pip_size)
    except Exception as e:
        return None

    if metrics.total_trades < 5:
        return None

    return {
        "pair": pair, "timeframe": "H1",
        "swing_length": params["swing_length"],
        "confluence_threshold": params["confluence_threshold"],
        "min_rr": params["min_rr"],
        "sl_buffer_pips": params["sl_buffer_pips"],
        "skip_days": str(params.get("skip_days") or "none"),
        "use_displacement": params["use_displacement"],
        "trades": metrics.total_trades,
        "win_rate": round(metrics.win_rate * 100, 1),
        "expectancy": round(metrics.expectancy, 3),
        "profit_factor": round(metrics.profit_factor, 2),
        "total_pips": round(metrics.total_pnl_pips, 0),
        "max_dd_pct": round(metrics.max_drawdown_pct * 100, 1),
        "avg_rr": round(metrics.avg_rr, 2),
        "sharpe": round(metrics.sharpe_ratio, 2),
    }


def main():
    print("=" * 80)
    print("ICT Strategy — Lower Timeframe Backtests (M5 & M15)")
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
        print(f"  H1:  {len(h1)} bars")
        print(f"  D:   {len(daily)} bars")

    # ── Phase 1: H1 baseline on same date range ──────────────────
    print("\n" + "=" * 80)
    print("PHASE 1: H1 baseline (same date range as M5 for fair comparison)")
    print("=" * 80)

    m5_start = data[(PAIRS[0], "M5")].index[0]
    results_h1 = []
    for pair in PAIRS:
        h1 = data[(pair, "H1")]
        h1_subset = h1[h1.index >= m5_start]
        params = {**BEST_H1}
        r = run_h1_baseline(pair, h1_subset, htf_data[pair], params)
        if r:
            results_h1.append(r)
            print(f"  {pair} H1: {r['trades']} trades, {r['win_rate']}% WR, "
                  f"+{r['expectancy']}R, PF={r['profit_factor']}, {r['total_pips']} pips")

    # ── Phase 2: Pre-compute signals ─────────────────────────────
    print("\n" + "=" * 80)
    print("PHASE 2: Pre-computing signals (expensive step)")
    print("=" * 80)

    # Generate signals ONCE per (pair, tf, swing_length) with lowest threshold
    # Then post-filter by confluence_threshold and skip_days
    min_thresh = min(CONFLUENCE_THRESHOLDS)
    signal_cache = {}  # (pair, tf, sw) -> list[Signal]

    n_combos = len(SWING_LENGTHS) * len(PAIRS) * 2
    print(f"Generating signals for {n_combos} unique (pair, tf, swing_length) combos...\n")

    for sw in SWING_LENGTHS:
        for pair in PAIRS:
            for tf in ["M5", "M15"]:
                ohlc = data[(pair, tf)]
                t0 = time.time()
                try:
                    signals = generate_signals(
                        ohlc, htf_ohlc=htf_data[pair], pair=pair,
                        swing_length=sw, confluence_threshold=min_thresh,
                        min_rr=FIXED_PARAMS["min_rr"],
                        sl_buffer_pips=FIXED_PARAMS["sl_buffer_pips"],
                        skip_days=[],  # no day filter — we'll post-filter
                        use_displacement=False,
                        pullback_window=80 if tf == "M5" else 40,
                        require_htf_bias=True,
                        fvg_lookback=16, compute_ob=False,
                    )
                except Exception as e:
                    signals = []
                    logging.warning("Signal gen failed for %s %s sw=%d: %s", pair, tf, sw, e)

                elapsed = time.time() - t0
                signal_cache[(pair, tf, sw)] = signals
                print(f"  {pair} {tf} sw={sw}: {len(signals)} signals in {elapsed:.1f}s")

    # ── Phase 3: Parameter sweep (fast — just filtering + trade sim) ──
    print("\n" + "=" * 80)
    print("PHASE 3: Parameter sweep (filtering + trade simulation)")
    print("=" * 80)

    n_sweeps = len(SWING_LENGTHS) * len(CONFLUENCE_THRESHOLDS) * len(SKIP_DAYS_OPTIONS)
    total = n_sweeps * len(PAIRS) * 2
    print(f"Running {n_sweeps} configs × {len(PAIRS)} pairs × 2 TFs = {total} backtests\n")

    results = []
    count = 0
    t0 = time.time()

    for sw in SWING_LENGTHS:
        for thresh in CONFLUENCE_THRESHOLDS:
            for skip in SKIP_DAYS_OPTIONS:
                params = {
                    "swing_length": sw,
                    "confluence_threshold": thresh,
                    "skip_days": skip,
                    **FIXED_PARAMS,
                }

                for pair in PAIRS:
                    for tf in ["M5", "M15"]:
                        count += 1
                        all_signals = signal_cache.get((pair, tf, sw), [])

                        # Post-filter: confluence threshold
                        filtered = [s for s in all_signals
                                    if s.confluence_score >= thresh]

                        # Post-filter: skip days
                        if skip:
                            filtered = [s for s in filtered
                                        if s.timestamp.weekday() not in skip]

                        if len(filtered) < 5:
                            continue

                        ohlc = data[(pair, tf)]
                        r = run_backtest_from_signals(
                            filtered, ohlc, pair, tf, params
                        )
                        if r:
                            results.append(r)

                        if count % 20 == 0:
                            elapsed = time.time() - t0
                            print(f"  [{count}/{total}] {elapsed:.0f}s elapsed... "
                                  f"({len(results)} results)")

    elapsed = time.time() - t0
    print(f"\nCompleted {count} backtests in {elapsed:.0f}s ({len(results)} valid results)")

    if not results:
        print("No results with >= 5 trades.")
        return

    df = pd.DataFrame(results)

    # ── Results by timeframe ─────────────────────────────────────
    for tf in ["M5", "M15"]:
        tf_df = df[df["timeframe"] == tf].copy()
        if tf_df.empty:
            print(f"\nNo results for {tf}")
            continue

        print(f"\n{'=' * 80}")
        print(f"RESULTS: {tf} — Top 15 by Expectancy (min 10 trades)")
        print(f"{'=' * 80}")

        tf_filtered = tf_df[tf_df["trades"] >= 10].sort_values("expectancy", ascending=False)
        if tf_filtered.empty:
            print(f"No configs with >= 10 trades on {tf}")
            print(f"\nAll {tf} results:")
            print(tf_df.sort_values("expectancy", ascending=False).to_string(index=False))
            continue

        top = tf_filtered.head(15)
        print(top.to_string(index=False))

        # Aggregate by swing_length
        print(f"\n--- {tf}: Average by swing_length ---")
        agg = tf_filtered.groupby("swing_length").agg(
            configs=("expectancy", "count"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_exp=("expectancy", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_pips=("total_pips", "mean"),
        ).round(3)
        print(agg.to_string())

        # Aggregate by confluence_threshold
        print(f"\n--- {tf}: Average by confluence_threshold ---")
        agg2 = tf_filtered.groupby("confluence_threshold").agg(
            configs=("expectancy", "count"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_exp=("expectancy", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_pips=("total_pips", "mean"),
        ).round(3)
        print(agg2.to_string())

        # Skip days impact
        print(f"\n--- {tf}: Average by skip_days ---")
        agg3 = tf_filtered.groupby("skip_days").agg(
            configs=("expectancy", "count"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_exp=("expectancy", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_pips=("total_pips", "mean"),
        ).round(3)
        print(agg3.to_string())

    # ── Best config per timeframe ────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BEST CONFIG PER TIMEFRAME (highest avg expectancy across both pairs)")
    print(f"{'=' * 80}")

    for tf in ["M5", "M15"]:
        tf_df = df[df["timeframe"] == tf].copy()
        if tf_df.empty:
            continue

        cfg_cols = ["swing_length", "confluence_threshold", "min_rr",
                    "sl_buffer_pips", "skip_days", "use_displacement"]
        grouped = tf_df.groupby(cfg_cols).agg(
            n_pairs=("pair", "nunique"),
            total_trades=("trades", "sum"),
            avg_exp=("expectancy", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_pf=("profit_factor", "mean"),
            total_pips=("total_pips", "sum"),
        ).reset_index()

        grouped = grouped[grouped["n_pairs"] == 2].sort_values("avg_exp", ascending=False)

        if grouped.empty:
            print(f"\n{tf}: No config with results on both pairs")
            continue

        best = grouped.iloc[0]
        print(f"\n{tf} BEST: sw={int(best['swing_length'])} thresh={int(best['confluence_threshold'])} "
              f"rr={best['min_rr']} sl_buf={best['sl_buffer_pips']} "
              f"skip={best['skip_days']} disp={best['use_displacement']}")
        print(f"  Trades: {int(best['total_trades'])}, AvgWR: {best['avg_wr']:.1f}%, "
              f"AvgExp: {best['avg_exp']:.3f}R, PF: {best['avg_pf']:.2f}, "
              f"Pips: {best['total_pips']:.0f}")

        # Per-pair breakdown
        mask = True
        for col in cfg_cols:
            mask = mask & (tf_df[col] == best[col])
        pair_results = tf_df[mask]
        for _, row in pair_results.iterrows():
            print(f"  {row['pair']}: {int(row['trades'])} trades, {row['win_rate']}% WR, "
                  f"+{row['expectancy']}R, PF={row['profit_factor']}, {row['total_pips']} pips")

    # ── Comparison vs H1 ──────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON: M5 vs M15 vs H1 baseline")
    print(f"{'=' * 80}")
    all_results = results_h1 + results
    if all_results:
        all_df = pd.DataFrame(all_results)
        comparison = all_df.groupby("timeframe").agg(
            configs=("expectancy", "count"),
            avg_trades=("trades", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_exp=("expectancy", "mean"),
            avg_pf=("profit_factor", "mean"),
            avg_pips=("total_pips", "mean"),
        ).round(3)
        print(comparison.to_string())


if __name__ == "__main__":
    main()
