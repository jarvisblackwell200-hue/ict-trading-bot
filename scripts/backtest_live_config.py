#!/usr/bin/env python3
"""Backtest the EXACT live trading configuration (M15 sw=5 t=4) across all 7 pairs.

Full-sample backtest + walk-forward validation.
Uses permissive risk manager (no circuit breakers) for clean signal evaluation.

Signal generation on 28K M15 bars takes ~27 min per pair via the smc library,
so walk-forward uses 7 non-overlapping windows (60d each) instead of sliding windows.
"""
from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from ict_bot.backtest import BacktestConfig
from ict_bot.backtest.engine import simulate_trades
from ict_bot.backtest.metrics import calculate_metrics
from ict_bot.data import load_pair
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals.detector import generate_signals

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]

# Variable spreads by session (matching real-world conditions)
SPREADS = {
    "EUR_USD": 1.5, "GBP_USD": 1.9, "USD_JPY": 1.6, "AUD_USD": 1.9,
    "USD_CAD": 2.0, "NZD_USD": 2.3, "EUR_GBP": 2.2,
}

# Exact live config: M15, sw=5, t=4
LIVE_CONFIG = dict(
    swing_length=5,
    confluence_threshold=4,
    min_rr=2.0,
    sl_buffer_pips=10.0,
    skip_days=[],
    use_displacement=False,
    use_breakeven=False,
    use_partial_tp=False,
    compute_ob=False,
    fvg_lookback=16,
    pullback_window=40,
    max_sl_pips=100.0,
    require_htf_bias=True,
    use_liquidity_targets=False,
    use_premium_discount=False,
    use_confluence_sizing=False,
    use_sweep_filter=False,
    use_ifvg=False,
    use_breaker_blocks=False,
    use_ce_entry=False,
)


def make_config(pair: str) -> BacktestConfig:
    return BacktestConfig(
        pair=pair,
        spread_pips=SPREADS.get(pair, 1.5),
        slippage_pips=0.5,
        **LIVE_CONFIG,
    )


def permissive_rm() -> RiskManager:
    return RiskManager(RiskConfig(
        max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0,
    ))


def generate_all_signals(ohlc, htf_ohlc, pair, config):
    """Generate signals on the full dataset once, then split by date for walk-forward."""
    return generate_signals(
        ohlc, htf_ohlc=htf_ohlc, pair=pair,
        swing_length=config.swing_length,
        confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
        skip_days=config.skip_days, use_displacement=config.use_displacement,
        pullback_window=config.pullback_window,
        require_htf_bias=config.require_htf_bias,
        fvg_lookback=config.fvg_lookback, compute_ob=False,
    )


def simulate_window(signals, ohlc_window, config):
    """Simulate trades for signals that fall within the ohlc_window date range."""
    rm = permissive_rm()
    window_start = ohlc_window.index.min()
    window_end = ohlc_window.index.max()
    window_signals = [s for s in signals if window_start <= s.timestamp <= window_end]
    if not window_signals:
        return [], calculate_metrics([])
    trades = simulate_trades(window_signals, ohlc_window, config, risk_manager=rm)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)
    return trades, metrics


def main():
    t_start = time.time()

    print("=" * 90, flush=True)
    print("BACKTEST: Live Config (M15, sw=5, t=4, no disp, no BE, no skip)", flush=True)
    print("Variable spreads per pair | Permissive risk manager", flush=True)
    print("=" * 90, flush=True)

    # ── Load data ──
    data = {}
    htf_data = {}
    for pair in PAIRS:
        m15 = load_pair(pair, "M15")
        daily = load_pair(pair, "D")
        data[pair] = m15
        htf_data[pair] = daily
        print(f"  {pair}: {len(m15)} M15 bars ({m15.index[0].date()} to {m15.index[-1].date()})", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # Generate signals ONCE per pair (expensive — ~27 min each)
    # Then split by date for full-sample and walk-forward
    # ═══════════════════════════════════════════════════════════════
    all_signals = {}
    for pair in PAIRS:
        config = make_config(pair)
        t0 = time.time()
        print(f"\n  Generating signals for {pair}...", end="", flush=True)
        sigs = generate_all_signals(data[pair], htf_data[pair], pair, config)
        elapsed = time.time() - t0
        all_signals[pair] = sigs
        print(f" {len(sigs)} signals in {elapsed:.0f}s", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Full-sample backtest
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("FULL-SAMPLE BACKTEST", flush=True)
    print("=" * 90, flush=True)

    header = f"\n{'Pair':<10} {'Trades':>7} {'WR%':>7} {'Expect':>8} {'PF':>6} {'Pips':>10} {'Spread':>7}"
    print(header, flush=True)
    print("-" * 60, flush=True)

    total_trades = 0
    total_pips = 0.0
    pair_summaries = []

    for pair in PAIRS:
        config = make_config(pair)
        rm = permissive_rm()
        trades = simulate_trades(all_signals[pair], data[pair], config, risk_manager=rm)
        metrics = calculate_metrics(trades, pip_size=config.pip_size)
        total_trades += metrics.total_trades
        total_pips += metrics.total_pnl_pips

        wr = f"{metrics.win_rate:.1%}" if metrics.total_trades > 0 else "N/A"
        pf = f"{metrics.profit_factor:.2f}" if metrics.total_trades > 0 else "N/A"
        print(f"{pair:<10} {metrics.total_trades:>7} {wr:>7} "
              f"{metrics.expectancy:>+7.3f}R {pf:>6} {metrics.total_pnl_pips:>+9.0f} "
              f"{SPREADS[pair]:>6.1f}p", flush=True)

        pair_summaries.append({
            "pair": pair, "trades": metrics.total_trades,
            "wr": metrics.win_rate, "exp": metrics.expectancy,
            "pf": metrics.profit_factor, "pips": metrics.total_pnl_pips,
        })

    print("-" * 60, flush=True)
    avg_wr = np.mean([s["wr"] for s in pair_summaries if s["trades"] > 0])
    avg_exp = np.mean([s["exp"] for s in pair_summaries if s["trades"] > 0])
    pairs_pos = sum(1 for s in pair_summaries if s["pips"] > 0)
    print(f"{'TOTAL':<10} {total_trades:>7} {avg_wr:>6.1%} {avg_exp:>+7.3f}R "
          f"{'':>6} {total_pips:>+9.0f}    {pairs_pos}/{len(PAIRS)} pairs +", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # PART 2: Walk-forward validation
    # Split the 423-day dataset into non-overlapping 60-day windows.
    # First 60 days = warm-up (needed for swing detection).
    # Remaining ~363 days = 6 windows of 60 days each.
    # Signals generated on FULL data (look-ahead in signal generation
    # is prevented by the detector only using past bars).
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("WALK-FORWARD VALIDATION (out-of-sample windows)", flush=True)
    print("6 non-overlapping 60-day test windows (first 60d = warm-up)", flush=True)
    print("=" * 90, flush=True)

    wf_summary = []

    for pair in PAIRS:
        config = make_config(pair)
        ohlc = data[pair]
        start = ohlc.index.min()
        end = ohlc.index.max()

        # Skip first 60 days as warm-up
        warmup_end = start + pd.Timedelta(days=60)
        remaining = ohlc[ohlc.index >= warmup_end]

        windows = []
        w_start = warmup_end
        wnum = 0
        while True:
            w_end = w_start + pd.Timedelta(days=60)
            if w_end > end:
                break
            window_ohlc = ohlc[(ohlc.index >= w_start) & (ohlc.index < w_end)]
            if len(window_ohlc) < 20:
                w_start = w_end
                continue

            trades, metrics = simulate_window(all_signals[pair], window_ohlc, config)
            wnum += 1
            windows.append({
                "window": wnum,
                "test_start": w_start,
                "test_end": w_end,
                "bars": len(window_ohlc),
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "expectancy": metrics.expectancy,
                "pnl_pips": metrics.total_pnl_pips,
                "profit_factor": metrics.profit_factor,
            })
            w_start = w_end

        if not windows:
            print(f"\n  {pair}: No valid windows", flush=True)
            continue

        positive = sum(1 for w in windows if w["pnl_pips"] > 0)
        total_wf_trades = sum(w["trades"] for w in windows)
        total_wf_pnl = sum(w["pnl_pips"] for w in windows)
        avg_wf_exp = np.mean([w["expectancy"] for w in windows if w["trades"] > 0]) if any(w["trades"] > 0 for w in windows) else 0

        print(f"\n  {pair}", flush=True)
        print(f"  {'Win':<5} {'Period':<28} {'Bars':>6} {'Trades':>7} {'WR%':>7} {'Exp(R)':>8} {'PF':>6} {'PnL':>10}", flush=True)
        print(f"  {'-' * 82}", flush=True)

        for w in windows:
            period = f"{w['test_start'].strftime('%Y-%m-%d')} to {w['test_end'].strftime('%Y-%m-%d')}"
            wr = f"{w['win_rate']:.1%}" if w["trades"] > 0 else "N/A"
            pf = f"{w['profit_factor']:.2f}" if w["trades"] > 0 else "N/A"
            print(f"  {w['window']:<5} {period:<28} {w['bars']:>6} {w['trades']:>7} "
                  f"{wr:>7} {w['expectancy']:>+7.3f} {pf:>6} {w['pnl_pips']:>+9.1f}", flush=True)

        print(f"  {'-' * 82}", flush=True)
        print(f"  Windows: {len(windows)} | Profitable: {positive}/{len(windows)} "
              f"({positive/len(windows):.0%}) | Trades: {total_wf_trades} | "
              f"AvgExp: {avg_wf_exp:+.3f}R | TotalPnL: {total_wf_pnl:+.0f}", flush=True)

        wf_summary.append({
            "pair": pair, "windows": len(windows), "positive": positive,
            "trades": total_wf_trades, "avg_exp": avg_wf_exp,
            "total_pnl": total_wf_pnl,
        })

    # ═══════════════════════════════════════════════════════════════
    # PART 3: Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 90, flush=True)
    print("WALK-FORWARD SUMMARY", flush=True)
    print("=" * 90, flush=True)

    print(f"\n{'Pair':<10} {'Windows':>8} {'Win%':>8} {'Trades':>8} {'AvgExp':>8} {'TotPnL':>10}", flush=True)
    print("-" * 55, flush=True)

    total_wins = 0
    total_windows = 0
    for s in wf_summary:
        win_pct = f"{s['positive']}/{s['windows']}" if s["windows"] > 0 else "N/A"
        print(f"{s['pair']:<10} {s['windows']:>8} {win_pct:>8} {s['trades']:>8} "
              f"{s['avg_exp']:>+7.3f} {s['total_pnl']:>+9.0f}", flush=True)
        total_wins += s["positive"]
        total_windows += s["windows"]

    print("-" * 55, flush=True)
    all_positive = sum(1 for s in wf_summary if s["total_pnl"] > 0)
    print(f"\nOverall: {total_wins}/{total_windows} windows profitable "
          f"({total_wins/total_windows:.0%})" if total_windows > 0 else "", flush=True)
    print(f"Pairs with positive total PnL: {all_positive}/{len(wf_summary)}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
