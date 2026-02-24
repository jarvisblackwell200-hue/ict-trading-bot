#!/usr/bin/env python3
"""True forward walk: in-sample / out-of-sample split.

Generates signals ONLY on truncated data (no look-ahead).
- In-sample: Dec 2024 → Jul 2025 (~7 months, for parameter validation)
- Out-of-sample: Jul 2025 → Feb 2026 (~7 months, pure holdout)

Signal generation is run TWICE per pair: once on IS data, once on OOS data
(truncated so the detector never sees future bars).
"""
from __future__ import annotations

import logging
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

SPREADS = {
    "EUR_USD": 1.5, "GBP_USD": 1.9, "USD_JPY": 1.6, "AUD_USD": 1.9,
    "USD_CAD": 2.0, "NZD_USD": 2.3, "EUR_GBP": 2.2,
}

# Exact live config
LIVE_CONFIG = dict(
    swing_length=5, confluence_threshold=4, min_rr=2.0, sl_buffer_pips=10.0,
    skip_days=[], use_displacement=False, use_breakeven=False, use_partial_tp=False,
    compute_ob=False, fvg_lookback=16, pullback_window=40, max_sl_pips=100.0,
    require_htf_bias=True, use_liquidity_targets=False, use_premium_discount=False,
    use_confluence_sizing=False, use_sweep_filter=False, use_ifvg=False,
    use_breaker_blocks=False, use_ce_entry=False,
)

SPLIT_DATE = pd.Timestamp("2025-07-15", tz="UTC")


def make_config(pair):
    return BacktestConfig(pair=pair, spread_pips=SPREADS.get(pair, 1.5), slippage_pips=0.5, **LIVE_CONFIG)


def permissive_rm():
    return RiskManager(RiskConfig(max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0))


def run_segment(ohlc, htf_ohlc, pair, config):
    """Generate signals on truncated data and simulate."""
    rm = permissive_rm()
    signals = generate_signals(
        ohlc, htf_ohlc=htf_ohlc, pair=pair,
        swing_length=config.swing_length, confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
        skip_days=config.skip_days, use_displacement=config.use_displacement,
        pullback_window=config.pullback_window, require_htf_bias=config.require_htf_bias,
        fvg_lookback=config.fvg_lookback, compute_ob=False,
    )
    if not signals:
        return [], calculate_metrics([])
    trades = simulate_trades(signals, ohlc, config, risk_manager=rm)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)
    return trades, metrics


def main():
    t_start = time.time()

    print("=" * 90, flush=True)
    print("TRUE FORWARD WALK: In-Sample vs Out-of-Sample", flush=True)
    print(f"Split date: {SPLIT_DATE.date()}", flush=True)
    print("Signals generated on TRUNCATED data only (no look-ahead)", flush=True)
    print("Config: M15, sw=5, t=4, no disp, no BE, no skip, variable spreads", flush=True)
    print("=" * 90, flush=True)

    # Load data
    data = {}
    htf_data = {}
    for pair in PAIRS:
        m15 = load_pair(pair, "M15")
        daily = load_pair(pair, "D")
        data[pair] = m15
        htf_data[pair] = daily
        is_bars = len(m15[m15.index < SPLIT_DATE])
        oos_bars = len(m15[m15.index >= SPLIT_DATE])
        print(f"  {pair}: {len(m15)} total | IS={is_bars} bars | OOS={oos_bars} bars", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # IN-SAMPLE (parameter validation period)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print(f"IN-SAMPLE: {data['EUR_USD'].index.min().date()} to {SPLIT_DATE.date()}", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n{'Pair':<10} {'Trades':>7} {'WR%':>7} {'Expect':>8} {'PF':>6} {'Pips':>10}", flush=True)
    print("-" * 50, flush=True)

    is_results = []
    for pair in PAIRS:
        config = make_config(pair)
        is_ohlc = data[pair][data[pair].index < SPLIT_DATE]
        is_htf = htf_data[pair][htf_data[pair].index < SPLIT_DATE]

        t0 = time.time()
        print(f"  {pair} generating...", end="", flush=True)
        trades, metrics = run_segment(is_ohlc, is_htf, pair, config)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s)", flush=True)

        wr = f"{metrics.win_rate:.1%}" if metrics.total_trades > 0 else "N/A"
        pf = f"{metrics.profit_factor:.2f}" if metrics.total_trades > 0 else "N/A"
        print(f"{pair:<10} {metrics.total_trades:>7} {wr:>7} "
              f"{metrics.expectancy:>+7.3f}R {pf:>6} {metrics.total_pnl_pips:>+9.0f}", flush=True)

        is_results.append({
            "pair": pair, "trades": metrics.total_trades, "wr": metrics.win_rate,
            "exp": metrics.expectancy, "pf": metrics.profit_factor, "pips": metrics.total_pnl_pips,
        })

    print("-" * 50, flush=True)
    is_total = sum(r["pips"] for r in is_results)
    is_trades = sum(r["trades"] for r in is_results)
    is_avg_exp = np.mean([r["exp"] for r in is_results if r["trades"] > 0])
    is_pos = sum(1 for r in is_results if r["pips"] > 0)
    print(f"{'TOTAL':<10} {is_trades:>7} {'':>7} {is_avg_exp:>+7.3f}R {'':>6} {is_total:>+9.0f}    {is_pos}/7 pairs +", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # OUT-OF-SAMPLE (pure holdout — never used for parameter tuning)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print(f"OUT-OF-SAMPLE: {SPLIT_DATE.date()} to {data['EUR_USD'].index.max().date()}", flush=True)
    print("*** This data was NEVER used for parameter selection ***", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n{'Pair':<10} {'Trades':>7} {'WR%':>7} {'Expect':>8} {'PF':>6} {'Pips':>10}", flush=True)
    print("-" * 50, flush=True)

    oos_results = []
    for pair in PAIRS:
        config = make_config(pair)
        oos_ohlc = data[pair][data[pair].index >= SPLIT_DATE]
        oos_htf = htf_data[pair][htf_data[pair].index >= SPLIT_DATE]

        t0 = time.time()
        print(f"  {pair} generating...", end="", flush=True)
        trades, metrics = run_segment(oos_ohlc, oos_htf, pair, config)
        elapsed = time.time() - t0
        print(f" done ({elapsed:.0f}s)", flush=True)

        wr = f"{metrics.win_rate:.1%}" if metrics.total_trades > 0 else "N/A"
        pf = f"{metrics.profit_factor:.2f}" if metrics.total_trades > 0 else "N/A"
        print(f"{pair:<10} {metrics.total_trades:>7} {wr:>7} "
              f"{metrics.expectancy:>+7.3f}R {pf:>6} {metrics.total_pnl_pips:>+9.0f}", flush=True)

        oos_results.append({
            "pair": pair, "trades": metrics.total_trades, "wr": metrics.win_rate,
            "exp": metrics.expectancy, "pf": metrics.profit_factor, "pips": metrics.total_pnl_pips,
        })

    print("-" * 50, flush=True)
    oos_total = sum(r["pips"] for r in oos_results)
    oos_trades = sum(r["trades"] for r in oos_results)
    oos_avg_exp = np.mean([r["exp"] for r in oos_results if r["trades"] > 0])
    oos_pos = sum(1 for r in oos_results if r["pips"] > 0)
    print(f"{'TOTAL':<10} {oos_trades:>7} {'':>7} {oos_avg_exp:>+7.3f}R {'':>6} {oos_total:>+9.0f}    {oos_pos}/7 pairs +", flush=True)

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"\n{'Pair':<10} {'IS Exp':>8} {'OOS Exp':>8} {'IS Pips':>10} {'OOS Pips':>10} {'OOS ok?':>8}", flush=True)
    print("-" * 58, flush=True)

    for is_r, oos_r in zip(is_results, oos_results):
        ok = "YES" if oos_r["pips"] > 0 else "NO"
        print(f"{is_r['pair']:<10} {is_r['exp']:>+7.3f} {oos_r['exp']:>+7.3f} "
              f"{is_r['pips']:>+9.0f} {oos_r['pips']:>+9.0f} {ok:>8}", flush=True)

    print("-" * 58, flush=True)
    print(f"{'TOTAL':<10} {is_avg_exp:>+7.3f} {oos_avg_exp:>+7.3f} "
          f"{is_total:>+9.0f} {oos_total:>+9.0f}", flush=True)

    degradation = ((oos_avg_exp - is_avg_exp) / is_avg_exp * 100) if is_avg_exp != 0 else 0
    print(f"\nExpectancy degradation IS→OOS: {degradation:+.1f}%", flush=True)
    print(f"OOS pairs profitable: {oos_pos}/7", flush=True)

    if oos_pos == 7 and oos_avg_exp > 0.1:
        print("\nVERDICT: Strategy is ROBUST — positive on all pairs out-of-sample", flush=True)
    elif oos_pos >= 5 and oos_avg_exp > 0:
        print("\nVERDICT: Strategy shows EDGE — majority of pairs profitable OOS", flush=True)
    elif oos_avg_exp > 0:
        print("\nVERDICT: Strategy is MARGINAL — positive overall but inconsistent", flush=True)
    else:
        print("\nVERDICT: Strategy may be OVERFIT — negative OOS expectancy", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nTotal runtime: {elapsed_total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
