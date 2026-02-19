#!/usr/bin/env python3
"""
ICT Concepts Backtest — test new concepts (sweep, IFVG, breaker blocks, CE entry)
against the proven sw=10 baseline.

Phases:
  1. BASELINE: sw=10, no disp, no BE (proven best config)
  2. NEW CONCEPTS: add each concept individually to baseline
  3. COMBINATIONS: promising concept combos
  4. WALK-FORWARD: validate best configs on rolling out-of-sample windows
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd

from src.ict_bot.backtest.engine import BacktestConfig, simulate_trades
from src.ict_bot.backtest.metrics import calculate_metrics
from src.ict_bot.data.loader import load_pair
from src.ict_bot.risk import RiskConfig, RiskManager
from src.ict_bot.signals.detector import generate_signals

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]


def permissive_rm():
    return RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10000.0,
        max_consecutive_losses=9999, max_daily_loss=1.0,
        max_drawdown=1.0, max_total_exposure=1.0,
    ))


def run_config(pair, h1, htf, config, label=""):
    signals = generate_signals(
        h1, htf_ohlc=htf, pair=config.pair,
        swing_length=config.swing_length,
        confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
        target_kill_zones=config.target_kill_zones,
        pullback_window=config.pullback_window,
        require_htf_bias=config.require_htf_bias,
        use_liquidity_targets=config.use_liquidity_targets,
        use_premium_discount=config.use_premium_discount,
        skip_days=config.skip_days,
        use_displacement=config.use_displacement,
        fvg_lookback=config.fvg_lookback,
        compute_ob=config.compute_ob,
        use_sweep_filter=config.use_sweep_filter,
        use_ifvg=config.use_ifvg,
        use_breaker_blocks=config.use_breaker_blocks,
        use_ce_entry=config.use_ce_entry,
    )

    if not signals:
        return {"label": label, "pair": pair, "signals": 0, "trades": 0,
                "wr": 0, "exp": 0, "pnl": 0, "avg_win": 0, "avg_loss": 0,
                "exits": {}, "pf": 0, "sharpe": 0}

    rm = permissive_rm()
    trades = simulate_trades(signals, h1, config, risk_manager=rm)
    m = calculate_metrics(trades, pip_size=config.pip_size)

    wins = [t["pnl_pips"] for t in trades if t["pnl_pips"] > 0]
    losses = [t["pnl_pips"] for t in trades if t["pnl_pips"] <= 0]
    exits = {}
    for t in trades:
        exits[t["exit_reason"]] = exits.get(t["exit_reason"], 0) + 1

    return {"label": label, "pair": pair,
            "signals": len(signals), "trades": m.total_trades,
            "wr": m.win_rate, "exp": m.expectancy, "pnl": m.total_pnl_pips,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "exits": exits, "pf": m.profit_factor, "sharpe": m.sharpe_ratio,
            "max_dd_pct": m.max_drawdown_pct}


def run_wf_config(pair, h1, htf, config, train_months=18, test_months=6, step_months=3):
    """Run walk-forward on a single pair. Returns per-window results."""
    start = h1.index.min()
    end = h1.index.max()
    current = start
    windows = []

    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        test_h1 = h1[(h1.index >= test_start) & (h1.index < test_end)]
        if len(test_h1) < 100:
            current += pd.DateOffset(months=step_months)
            continue

        # HTF data filtered to cover test period context
        test_htf = htf[htf.index < test_end] if htf is not None else None

        signals = generate_signals(
            test_h1, htf_ohlc=test_htf, pair=config.pair,
            swing_length=config.swing_length,
            confluence_threshold=config.confluence_threshold,
            min_rr=config.min_rr, sl_buffer_pips=config.sl_buffer_pips,
            target_kill_zones=config.target_kill_zones,
            pullback_window=config.pullback_window,
            require_htf_bias=config.require_htf_bias,
            use_liquidity_targets=config.use_liquidity_targets,
            use_premium_discount=config.use_premium_discount,
            skip_days=config.skip_days,
            use_displacement=config.use_displacement,
            fvg_lookback=config.fvg_lookback,
            compute_ob=config.compute_ob,
            use_sweep_filter=config.use_sweep_filter,
            use_ifvg=config.use_ifvg,
            use_breaker_blocks=config.use_breaker_blocks,
            use_ce_entry=config.use_ce_entry,
        )

        if signals:
            rm = permissive_rm()
            trades = simulate_trades(signals, test_h1, config, risk_manager=rm)
            m = calculate_metrics(trades, pip_size=config.pip_size)
            windows.append({
                "test_start": test_start,
                "test_end": test_end,
                "trades": m.total_trades,
                "wr": m.win_rate,
                "exp": m.expectancy,
                "pnl": m.total_pnl_pips,
                "pf": m.profit_factor,
            })
        else:
            windows.append({
                "test_start": test_start,
                "test_end": test_end,
                "trades": 0, "wr": 0, "exp": 0, "pnl": 0, "pf": 0,
            })

        current += pd.DateOffset(months=step_months)

    return windows


def print_aggregate(results_by_label, title):
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")
    print(f"  {'Config':<40} {'Trd':>5} {'AvgWR':>6} {'AvgExp':>8} {'AvgPF':>6} "
          f"{'TotalPnL':>10} {'Pairs+':>6} {'Pairs-':>6}")
    print(f"  {'-'*105}")

    for label, results in results_by_label.items():
        total_trades = sum(r["trades"] for r in results)
        active = [r for r in results if r["trades"] > 0]
        n = len(active)
        avg_wr = sum(r["wr"] for r in active) / n if n else 0
        avg_exp = sum(r["exp"] for r in active) / n if n else 0
        avg_pf = sum(r["pf"] for r in active) / n if n else 0
        total_pnl = sum(r["pnl"] for r in results)
        pairs_pos = sum(1 for r in results if r["pnl"] > 0)
        pairs_neg = sum(1 for r in results if r["pnl"] < 0)

        marker = ""
        if avg_exp > 0.1:
            marker = " ***"
        elif avg_exp > 0:
            marker = " *"

        print(f"  {label:<40} {total_trades:>5} {avg_wr:>5.1%} {avg_exp:>+7.3f}R "
              f"{avg_pf:>6.2f} {total_pnl:>+9.1f} {pairs_pos:>5}+ {pairs_neg:>5}-{marker}")


def print_per_pair(results_by_label, title):
    """Print detailed per-pair results for selected configs."""
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")

    for label, results in results_by_label.items():
        print(f"\n  {label}:")
        print(f"  {'Pair':<10} {'Trd':>5} {'WR':>6} {'Exp':>8} {'PF':>6} "
              f"{'PnL':>10} {'Sharpe':>7} {'MaxDD%':>7}")
        print(f"  {'-'*65}")
        for r in results:
            if r["trades"] > 0:
                dd_str = f"{r.get('max_dd_pct', 0):.1%}" if r.get("max_dd_pct") else "n/a"
                print(f"  {r['pair']:<10} {r['trades']:>5} {r['wr']:>5.1%} {r['exp']:>+7.3f}R "
                      f"{r['pf']:>6.2f} {r['pnl']:>+9.1f} {r['sharpe']:>7.2f} {dd_str:>7}")
            else:
                print(f"  {r['pair']:<10}     0     -        -      -          -       -       -")


def print_walk_forward(wf_results, label):
    """Print walk-forward results summary."""
    print(f"\n  {label}:")
    print(f"  {'Pair':<10} {'Windows':>8} {'WinPct':>7} {'AvgTrd':>7} {'AvgExp':>8} "
          f"{'AvgPnL':>9} {'TotPnL':>9}")
    print(f"  {'-'*70}")

    for pair, windows in wf_results.items():
        if not windows:
            print(f"  {pair:<10}        0       -       -        -         -         -")
            continue
        n = len(windows)
        win_pct = sum(1 for w in windows if w["pnl"] > 0) / n if n else 0
        avg_trd = sum(w["trades"] for w in windows) / n
        active = [w for w in windows if w["trades"] > 0]
        avg_exp = sum(w["exp"] for w in active) / len(active) if active else 0
        avg_pnl = sum(w["pnl"] for w in windows) / n
        tot_pnl = sum(w["pnl"] for w in windows)
        print(f"  {pair:<10} {n:>8} {win_pct:>6.0%} {avg_trd:>7.1f} {avg_exp:>+7.3f}R "
              f"{avg_pnl:>+8.1f} {tot_pnl:>+8.1f}")


# ── Config factories ─────────────────────────────────────────────

def baseline_config(pair):
    """Proven best: sw=10, no displacement, no BE."""
    return BacktestConfig(
        pair=pair, swing_length=10, confluence_threshold=3, min_rr=2.0,
        sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        risk_per_trade=0.01, starting_balance=10000.0,
        use_displacement=False, use_partial_tp=True,
        use_breakeven=False, be_threshold_r=1.5,
        use_confluence_sizing=False, skip_days=[0, 4],
        fvg_lookback=16, pullback_window=40,
        use_liquidity_targets=False, use_premium_discount=False,
        compute_ob=False,
    )


def baseline_with_skip(pair):
    """Baseline + skip Mon/Fri."""
    cfg = baseline_config(pair)
    cfg.skip_days = [0, 4]
    return cfg


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    data = {}
    for pair in PAIRS:
        try:
            h1 = load_pair(pair, "H1")
            htf = load_pair(pair, "D")
            data[pair] = (h1, htf)
            years = (h1.index[-1] - h1.index[0]).days / 365.25
            print(f"  {pair}: {len(h1)} H1 bars ({years:.1f} years)", flush=True)
        except FileNotFoundError:
            print(f"  {pair}: NO DATA", flush=True)

    pairs = list(data.keys())

    print(f"\n{'='*115}")
    print("  ICT CONCEPTS BACKTEST")
    print("  Testing: Liquidity Sweeps, IFVG, Breaker Blocks, CE Entry")
    print("  Base config: sw=10, no disp, no BE, skip Mon/Fri, thresh=3")
    print(f"{'='*115}", flush=True)

    # ── Phase 1: Baseline + individual concepts ────────────────────
    def run_phase(pairs, data, configs_dict, base_fn, title):
        t0 = time.time()
        results_by_label = {}
        for label, overrides in configs_dict.items():
            label_results = []
            for pair in pairs:
                h1, htf = data[pair]
                cfg = base_fn(pair)
                for k, v in overrides.items():
                    setattr(cfg, k, v)
                r = run_config(pair, h1, htf, cfg, label=label)
                label_results.append(r)
            results_by_label[label] = label_results
        elapsed = time.time() - t0
        print_aggregate(results_by_label, f"{title}  ({elapsed:.0f}s)")
        return results_by_label

    concepts = {
        "BASELINE (sw=10 no-disp no-BE)":   {},
        "+ sweep filter":                     {"use_sweep_filter": True},
        "+ IFVG zones":                       {"use_ifvg": True},
        "+ breaker blocks":                   {"use_breaker_blocks": True},
        "+ CE entry (50% midpoint)":          {"use_ce_entry": True},
        "+ sweep + IFVG":                     {"use_sweep_filter": True, "use_ifvg": True},
        "+ sweep + breaker":                  {"use_sweep_filter": True, "use_breaker_blocks": True},
        "+ IFVG + breaker":                   {"use_ifvg": True, "use_breaker_blocks": True},
        "+ sweep + CE":                       {"use_sweep_filter": True, "use_ce_entry": True},
        "+ IFVG + CE":                        {"use_ifvg": True, "use_ce_entry": True},
        "+ breaker + CE":                     {"use_breaker_blocks": True, "use_ce_entry": True},
        "+ all four":                         {"use_sweep_filter": True, "use_ifvg": True,
                                               "use_breaker_blocks": True, "use_ce_entry": True},
    }
    r1 = run_phase(pairs, data, concepts, baseline_config,
                    "PHASE 1: NEW ICT CONCEPTS — individual and combined")

    # Print per-pair detail for top configs
    # Pick configs with avg_exp > 0.1 for detail
    top_configs = {}
    for label, results in r1.items():
        active = [r for r in results if r["trades"] > 0]
        if active:
            avg_exp = sum(r["exp"] for r in active) / len(active)
            if avg_exp > 0.1:
                top_configs[label] = results
    if top_configs:
        print_per_pair(top_configs, "PER-PAIR DETAIL — Top configs (AvgExp > 0.1R)")

    # ── Phase 2: Threshold variations with best concepts ───────────
    threshold_tests = {
        "baseline thresh=1":                  {"confluence_threshold": 1},
        "baseline thresh=2":                  {"confluence_threshold": 2},
        "baseline thresh=3 (default)":        {},
        "baseline thresh=4":                  {"confluence_threshold": 4},
    }
    run_phase(pairs, data, threshold_tests, baseline_config,
              "PHASE 2: THRESHOLD SENSITIVITY")

    # ── Phase 3: Walk-forward validation on best configs ───────────
    print(f"\n{'='*115}")
    print("  PHASE 3: WALK-FORWARD VALIDATION")
    print("  Train: 18 months, Test: 6 months, Step: 3 months")
    print(f"{'='*115}", flush=True)

    wf_configs = {
        "BASELINE (sw=10 no-disp no-BE)":   {},
        "+ sweep filter":                     {"use_sweep_filter": True},
        "+ IFVG zones":                       {"use_ifvg": True},
        "+ breaker blocks":                   {"use_breaker_blocks": True},
        "+ CE entry":                         {"use_ce_entry": True},
    }

    t0 = time.time()
    for label, overrides in wf_configs.items():
        wf_results = {}
        for pair in pairs:
            h1, htf = data[pair]
            cfg = baseline_config(pair)
            for k, v in overrides.items():
                setattr(cfg, k, v)
            windows = run_wf_config(pair, h1, htf, cfg,
                                     train_months=18, test_months=6, step_months=3)
            wf_results[pair] = windows
        print_walk_forward(wf_results, label)

    elapsed = time.time() - t0
    print(f"\n  Walk-forward completed in {elapsed:.0f}s")

    print(f"\n{'='*115}")
    print("  DONE — Look for *** (strong expectancy) and walk-forward consistency")
    print(f"{'='*115}\n", flush=True)


if __name__ == "__main__":
    main()
