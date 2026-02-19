#!/usr/bin/env python3
"""
Filter Isolation Analysis — which ICT confluence factors actually contribute edge?

Three-phase analysis:
  Phase 1: ABLATION — start with full config (no OB for speed), remove one factor at a time
  Phase 2: ADDITION — start with minimal config (BOS only), add one factor at a time
  Phase 3: TRADE MANAGEMENT — test breakeven, partial TP, liquidity targets, sizing

Uses a permissive risk manager (no circuit breakers) so results aren't confounded.
OB computation is skipped (too slow) — tested as a single dedicated comparison.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
logging.basicConfig(level=logging.WARNING)

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
        use_sweep_filter=getattr(config, "use_sweep_filter", False),
        use_ifvg=getattr(config, "use_ifvg", False),
        use_breaker_blocks=getattr(config, "use_breaker_blocks", False),
        use_ce_entry=getattr(config, "use_ce_entry", False),
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
            "exits": exits, "pf": m.profit_factor, "sharpe": m.sharpe_ratio}


def print_aggregate(results_by_label, title):
    print(f"\n{'='*105}")
    print(f"  {title}")
    print(f"{'='*105}")
    print(f"  {'Config':<34} {'Trd':>5} {'AvgWR':>6} {'AvgExp':>8} {'AvgPF':>6} "
          f"{'TotalPnL':>10} {'Pairs+':>6} {'Pairs-':>6}")
    print(f"  {'-'*95}")

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

        print(f"  {label:<34} {total_trades:>5} {avg_wr:>5.1%} {avg_exp:>+7.3f}R "
              f"{avg_pf:>6.2f} {total_pnl:>+9.1f} {pairs_pos:>5}+ {pairs_neg:>5}-{marker}")


def run_phase(pairs, data, configs_dict, base_fn, title):
    """Run a set of configs and print aggregate results."""
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


# ── Config factories ─────────────────────────────────────────────

def full_config_no_ob(pair):
    """Full config but compute_ob=False for speed."""
    return BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=3, min_rr=2.0,
        sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        risk_per_trade=0.01, starting_balance=10000.0,
        use_displacement=True, use_partial_tp=True,
        use_breakeven=True, be_threshold_r=1.5,
        use_confluence_sizing=True, skip_days=[0, 4],
        fvg_lookback=16, pullback_window=40,
        use_liquidity_targets=True, use_premium_discount=True,
        compute_ob=False,
    )


def minimal_config(pair):
    return BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=1, min_rr=2.0,
        sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        risk_per_trade=0.01, starting_balance=10000.0,
        use_displacement=False, use_partial_tp=False,
        use_breakeven=False, use_confluence_sizing=False,
        skip_days=None, fvg_lookback=16, pullback_window=40,
        use_liquidity_targets=False, use_premium_discount=False,
        compute_ob=False,
    )


def mgmt_base_config(pair):
    """Displacement + threshold=2 as signal source for management tests."""
    return BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=2, min_rr=2.0,
        sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
        risk_per_trade=0.01, starting_balance=10000.0,
        use_displacement=True, use_partial_tp=False,
        use_breakeven=False, use_confluence_sizing=False,
        skip_days=None, fvg_lookback=16, pullback_window=40,
        use_liquidity_targets=False, use_premium_discount=False,
        compute_ob=False,
    )


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

    print(f"\n{'='*105}")
    print("  FILTER ISOLATION ANALYSIS")
    print("  Which ICT confluence factor actually contributes edge?")
    print("  Permissive risk manager (no circuit breakers). OB skipped (slow).")
    print(f"{'='*105}", flush=True)

    # ── Phase 1: Ablation from full config ───────────────────────
    ablations = {
        "FULL (no OB, baseline)":    {},
        "- displacement":            {"use_displacement": False},
        "- premium/discount":        {"use_premium_discount": False},
        "- liquidity targets":       {"use_liquidity_targets": False},
        "- skip Mon/Fri":            {"skip_days": None},
        "- partial TP":              {"use_partial_tp": False},
        "- breakeven":               {"use_breakeven": False},
        "- confluence sizing":       {"use_confluence_sizing": False},
        "pullback_window=20":        {"pullback_window": 20},
        "pullback_window=60":        {"pullback_window": 60},
        "threshold=2":               {"confluence_threshold": 2},
        "threshold=1":               {"confluence_threshold": 1},
        "threshold=4":               {"confluence_threshold": 4},
        "swing_length=10":           {"swing_length": 10},
        "swing_length=30":           {"swing_length": 30},
        "fvg_lookback=8":            {"fvg_lookback": 8},
        "fvg_lookback=32":           {"fvg_lookback": 32},
        "sl_buffer=5":               {"sl_buffer_pips": 5.0},
        "sl_buffer=15":              {"sl_buffer_pips": 15.0},
        "min_rr=1.5":                {"min_rr": 1.5},
        "min_rr=3.0":                {"min_rr": 3.0},
    }
    run_phase(pairs, data, ablations, full_config_no_ob,
              "PHASE 1: ABLATION — Remove/change one factor from full config")

    # ── Phase 2: Build up from minimal ───────────────────────────
    additions = {
        "MINIMAL (BOS+FVG, thresh=1)": {},
        "+ displacement":              {"use_displacement": True},
        "+ premium/discount":          {"use_premium_discount": True},
        "+ liquidity targets":         {"use_liquidity_targets": True},
        "+ skip Mon/Fri":              {"skip_days": [0, 4]},
        "+ breakeven (1.5R)":          {"use_breakeven": True, "be_threshold_r": 1.5},
        "+ partial TP (1R)":           {"use_partial_tp": True},
        "+ threshold=2":               {"confluence_threshold": 2},
        "+ threshold=3":               {"confluence_threshold": 3},
        "disp + thresh=2":             {"use_displacement": True, "confluence_threshold": 2},
        "disp + thresh=3":             {"use_displacement": True, "confluence_threshold": 3},
        "disp + BE 1.5R":             {"use_displacement": True, "use_breakeven": True, "be_threshold_r": 1.5},
        "disp + liq tgt":             {"use_displacement": True, "use_liquidity_targets": True},
        "disp + liq + BE":            {"use_displacement": True, "use_liquidity_targets": True,
                                        "use_breakeven": True, "be_threshold_r": 1.5},
        "disp + P/D + thresh=2":      {"use_displacement": True, "use_premium_discount": True,
                                        "confluence_threshold": 2},
        "disp + liq + P/D + t=2":     {"use_displacement": True, "use_liquidity_targets": True,
                                        "use_premium_discount": True, "confluence_threshold": 2},
        "disp + liq + BE + partial":  {"use_displacement": True, "use_liquidity_targets": True,
                                        "use_breakeven": True, "use_partial_tp": True},
    }
    run_phase(pairs, data, additions, minimal_config,
              "PHASE 2: ADDITION — Build up from minimal config")

    # ── Phase 3: Trade management ────────────────────────────────
    mgmt_variants = {
        "no BE, no partial":     {"use_breakeven": False, "use_partial_tp": False},
        "BE at 1.0R":            {"use_breakeven": True, "be_threshold_r": 1.0},
        "BE at 1.5R":            {"use_breakeven": True, "be_threshold_r": 1.5},
        "BE at 2.0R":            {"use_breakeven": True, "be_threshold_r": 2.0},
        "partial 1R, no BE":     {"use_partial_tp": True, "partial_tp_r": 1.0},
        "partial + BE 1.5R":     {"use_breakeven": True, "be_threshold_r": 1.5,
                                   "use_partial_tp": True, "partial_tp_r": 1.0},
        "liq targets, no BE":    {"use_liquidity_targets": True},
        "liq + BE 1.5R":         {"use_liquidity_targets": True,
                                   "use_breakeven": True, "be_threshold_r": 1.5},
        "liq + partial + BE":    {"use_liquidity_targets": True,
                                   "use_breakeven": True, "be_threshold_r": 1.5,
                                   "use_partial_tp": True},
        "fixed RR=1.5":          {"min_rr": 1.5},
        "fixed RR=2.0":          {"min_rr": 2.0},
        "fixed RR=3.0":          {"min_rr": 3.0},
        "fixed RR=4.0":          {"min_rr": 4.0},
    }
    run_phase(pairs, data, mgmt_variants, mgmt_base_config,
              "PHASE 3: TRADE MANAGEMENT — Same signals (disp+t=2), different exits")

    print(f"\n{'='*105}")
    print("  DONE — Look for *** (strong positive expectancy across pairs)")
    print(f"{'='*105}\n")


if __name__ == "__main__":
    main()
