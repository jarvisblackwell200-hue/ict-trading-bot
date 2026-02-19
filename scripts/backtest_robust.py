#!/usr/bin/env python3
"""Robust M15 backtest addressing simulation critiques.

Improvements over backtest_parallel.py:
1. Variable spread by session (Asian=2.0x, London=1.0x, NY overlap=0.8x, rollover=2.5x)
2. Max 3 concurrent positions across all pairs
3. Max 2 correlated same-direction USD pairs simultaneously
4. True out-of-sample walk-forward (signals generated on truncated data)
5. Portfolio-level equity curve, drawdown, and Monte Carlo
6. Fixed win rate display

Usage:
    PYTHONPATH=src python scripts/backtest_robust.py
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_bot.backtest.walk_forward import monte_carlo
from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals import generate_signals

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ALL_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]
USD_PAIRS = {"EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD"}

CONFIGS = [
    {"name": "M15 sw=5 t=3", "sw": 5, "thresh": 3},
    {"name": "M15 sw=5 t=4", "sw": 5, "thresh": 4},
    {"name": "M15 sw=10 t=3", "sw": 10, "thresh": 3},
]


# ── Variable Spread Model ─────────────────────────────────────────────

def get_spread_pips(hour_utc: int, pair: str) -> float:
    """Realistic IB CFD spread by session and pair."""
    if pair in ("EUR_USD", "USD_JPY"):
        base = 1.0
    elif pair in ("GBP_USD", "AUD_USD"):
        base = 1.2
    elif pair == "USD_CAD":
        base = 1.3
    elif pair == "NZD_USD":
        base = 1.5
    elif pair == "EUR_GBP":
        base = 1.4
    else:
        base = 1.5

    if 0 <= hour_utc < 7:        # Asian
        return base * 2.0
    elif 7 <= hour_utc < 8:      # London open
        return base * 1.5
    elif 8 <= hour_utc < 12:     # London
        return base * 1.0
    elif 12 <= hour_utc < 16:    # NY overlap
        return base * 0.8
    elif 16 <= hour_utc < 21:    # NY afternoon
        return base * 1.2
    else:                         # Rollover
        return base * 2.5


# ── Signal Generation Worker ──────────────────────────────────────────

def _gen_signals_worker(args):
    """Worker: generate signals for one (sw, pair) on M15 data."""
    sw, pair = args
    path = DATA_DIR / f"{pair}_M15.parquet"
    if not path.exists():
        return (pair, sw, [])

    ohlc = pd.read_parquet(path)
    htf_path = DATA_DIR / f"{pair}_D.parquet"
    htf_ohlc = pd.read_parquet(htf_path) if htf_path.exists() else None

    t0 = time.time()
    signals = generate_signals(
        ohlc=ohlc, htf_ohlc=htf_ohlc, pair=pair,
        swing_length=sw, confluence_threshold=3,
        min_rr=2.0, sl_buffer_pips=10.0, skip_days=[],
        use_displacement=False, fvg_lookback=16,
        pullback_window=40, compute_ob=False,
    )
    elapsed = time.time() - t0
    signals = signals or []
    print(f"  {pair} sw={sw}: {len(signals)} signals ({elapsed:.0f}s)", flush=True)
    return (pair, sw, signals)


# ── Simulate with Variable Spread ─────────────────────────────────────

def simulate_variable_spread(signals, ohlc, pair, max_sl_pips=100.0):
    """Simulate trades using time-of-day variable spread."""
    pip_size = 0.01 if "JPY" in pair else 0.0001
    slippage = 0.5 * pip_size

    rm = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
        max_consecutive_losses=999, max_daily_loss=1.0, max_drawdown=1.0,
    ))

    trades = []
    last_trade_date = None

    for signal in signals:
        signal_date = signal.timestamp.date()
        if last_trade_date is not None and signal_date != last_trade_date:
            rm.reset_daily()
        last_trade_date = signal_date

        try:
            entry_idx = ohlc.index.get_loc(signal.timestamp)
        except KeyError:
            continue

        entry_spread_pips = get_spread_pips(signal.timestamp.hour, pair)
        entry_spread = entry_spread_pips * pip_size

        if signal.direction == "long":
            actual_entry = signal.entry_price + entry_spread / 2 + slippage
        else:
            actual_entry = signal.entry_price - entry_spread / 2 - slippage

        actual_sl = signal.stop_loss
        actual_tp = signal.take_profit

        if signal.direction == "long":
            risk_pips = (actual_entry - actual_sl) / pip_size
        else:
            risk_pips = (actual_sl - actual_entry) / pip_size

        if risk_pips <= 0 or risk_pips > max_sl_pips:
            continue

        decision = rm.evaluate_signal(signal, pip_size)
        if not decision.approved:
            continue

        pip_value = decision.position_size
        risk_amount = decision.risk_amount
        rm.register_open_position(signal.pair, risk_amount)

        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + 200, len(ohlc))):
            candle = ohlc.iloc[j]
            high, low = candle["high"], candle["low"]
            bar_spread = get_spread_pips(ohlc.index[j].hour, pair) * pip_size

            if signal.direction == "long":
                if low <= actual_sl:
                    exit_price = actual_sl - slippage
                    exit_time = ohlc.index[j]
                    exit_reason = "SL"
                    break
                if high >= actual_tp:
                    exit_price = actual_tp - bar_spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break
            else:
                if high >= actual_sl:
                    exit_price = actual_sl + slippage
                    exit_time = ohlc.index[j]
                    exit_reason = "SL"
                    break
                if low <= actual_tp:
                    exit_price = actual_tp + bar_spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break

        if exit_price is None:
            ti = min(entry_idx + 200, len(ohlc) - 1)
            exit_price = ohlc.iloc[ti]["close"]
            exit_time = ohlc.index[ti]
            exit_reason = "TIMEOUT"

        if signal.direction == "long":
            pnl_pips = (exit_price - actual_entry) / pip_size
        else:
            pnl_pips = (actual_entry - exit_price) / pip_size

        pnl_amount = pnl_pips * pip_value
        rm.record_trade_result(pnl_amount, signal.pair)

        trades.append({
            "entry_time": signal.timestamp,
            "exit_time": exit_time,
            "pair": pair,
            "direction": signal.direction,
            "risk_pips": round(risk_pips, 1),
            "pnl_pips": round(pnl_pips, 1),
            "pnl_r": round(pnl_pips / risk_pips, 3) if risk_pips > 0 else 0,
            "exit_reason": exit_reason,
            "entry_spread_pips": round(entry_spread_pips, 1),
        })

    return trades


# ── Portfolio Position Filter ─────────────────────────────────────────

def portfolio_filter(trades_by_pair, max_positions=3, max_corr=2):
    """Reject trades exceeding concurrent position / correlation limits."""
    all_trades = []
    for trades in trades_by_pair.values():
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t["entry_time"])

    accepted = []
    rej_pos = rej_corr = 0
    open_pos = []  # (pair, direction, exit_time)
    max_conc = 0

    for t in all_trades:
        et = t["entry_time"]
        open_pos = [p for p in open_pos if p[2] > et]
        max_conc = max(max_conc, len(open_pos))

        if len(open_pos) >= max_positions:
            rej_pos += 1
            continue

        if t["pair"] in USD_PAIRS:
            same = sum(1 for p, d, _ in open_pos if p in USD_PAIRS and d == t["direction"])
            if same >= max_corr:
                rej_corr += 1
                continue

        accepted.append(t)
        open_pos.append((t["pair"], t["direction"], t["exit_time"]))

    return accepted, {
        "total": len(all_trades), "accepted": len(accepted),
        "rej_pos": rej_pos, "rej_corr": rej_corr,
        "max_concurrent": max_conc,
    }


# ── Walk-Forward Worker ───────────────────────────────────────────────

def _wf_worker(args):
    """True OOS walk-forward: generate signals on data truncated at test_end."""
    pair, sw, thresh, data_end_s, test_start_s, test_end_s = args

    path = DATA_DIR / f"{pair}_M15.parquet"
    if not path.exists():
        return None

    ohlc_full = pd.read_parquet(path)
    data_end = pd.Timestamp(data_end_s)
    test_start = pd.Timestamp(test_start_s)
    test_end = pd.Timestamp(test_end_s)

    ohlc = ohlc_full[ohlc_full.index < data_end]
    if len(ohlc) < 200:
        return None

    htf_path = DATA_DIR / f"{pair}_D.parquet"
    htf_ohlc = None
    if htf_path.exists():
        htf_full = pd.read_parquet(htf_path)
        htf_ohlc = htf_full[htf_full.index < data_end]

    t0 = time.time()
    signals = generate_signals(
        ohlc=ohlc, htf_ohlc=htf_ohlc, pair=pair,
        swing_length=sw, confluence_threshold=thresh,
        min_rr=2.0, sl_buffer_pips=10.0, skip_days=[],
        use_displacement=False, fvg_lookback=16,
        pullback_window=40, compute_ob=False,
    )
    elapsed = time.time() - t0
    signals = [s for s in (signals or []) if test_start <= s.timestamp < test_end]

    if not signals:
        return None

    test_ohlc = ohlc[(ohlc.index >= test_start) & (ohlc.index < test_end)]
    if len(test_ohlc) < 50:
        return None

    trades = simulate_variable_spread(signals, test_ohlc, pair)
    if len(trades) < 3:
        return None

    wins = sum(1 for t in trades if t["pnl_pips"] > 0)
    tot = sum(t["pnl_pips"] for t in trades)
    avg_r = sum(t["pnl_r"] for t in trades) / len(trades)

    print(f"    WF {pair} [{test_start.strftime('%Y-%m-%d')}→{test_end.strftime('%Y-%m-%d')}]: "
          f"{len(trades)}t {wins/len(trades)*100:.0f}%WR {avg_r:+.3f}R {tot:+.0f}p ({elapsed:.0f}s)",
          flush=True)

    return {
        "pair": pair, "test_start": test_start_s, "test_end": test_end_s,
        "trades": len(trades), "win_rate": wins / len(trades),
        "expectancy": avg_r, "total_pnl": tot,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("ROBUST M15 BACKTEST — Variable Spread + Position Limits + True OOS Walk-Forward")
    print("=" * 90)
    print()

    for pair in ALL_PAIRS:
        path = DATA_DIR / f"{pair}_M15.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            days = (df.index[-1] - df.index[0]).days
            print(f"  {pair} M15: {len(df):,} bars, {days} days")
    print()

    # ── Phase 1: Signal generation ────────────────────────────────────
    print("=" * 90)
    print("PHASE 1: SIGNAL GENERATION (parallel)")
    print("=" * 90)
    print()

    work = []
    for sw in [5, 10]:
        for pair in ALL_PAIRS:
            if (DATA_DIR / f"{pair}_M15.parquet").exists():
                work.append((sw, pair))

    n_workers = min(cpu_count(), 4)
    print(f"{len(work)} combos across {n_workers} workers...\n")

    t0 = time.time()
    with Pool(n_workers) as pool:
        gen_results = pool.map(_gen_signals_worker, work)
    print(f"\nSignal gen: {time.time()-t0:.0f}s\n")

    sig_cache = {}
    for pair, sw, sigs in gen_results:
        sig_cache[(pair, sw)] = sigs

    ohlc_data = {}
    for pair in ALL_PAIRS:
        p = DATA_DIR / f"{pair}_M15.parquet"
        if p.exists():
            ohlc_data[pair] = pd.read_parquet(p)

    # ── Phase 2: Full-sample with variable spread ─────────────────────
    print("=" * 90)
    print("PHASE 2: FULL-SAMPLE BACKTEST (variable spread, per-pair)")
    print("=" * 90)

    all_results = {}  # cfg_name -> {pair: trades}

    for cfg in CONFIGS:
        cn = cfg["name"]
        sw, thresh = cfg["sw"], cfg["thresh"]
        all_results[cn] = {}

        print(f"\n--- {cn} ---")
        print(f"  {'Pair':<10} {'Trades':>7} {'WR':>6} {'Expect':>8} {'PF':>6} "
              f"{'Pips':>8} {'AvgSpd':>7}")
        print("  " + "-" * 60)

        for pair in ALL_PAIRS:
            if pair not in ohlc_data or (pair, sw) not in sig_cache:
                continue

            sigs = [s for s in sig_cache[(pair, sw)] if s.confluence_score >= thresh]
            trades = simulate_variable_spread(sigs, ohlc_data[pair], pair)
            all_results[cn][pair] = trades

            if len(trades) >= 5:
                wins = sum(1 for t in trades if t["pnl_pips"] > 0)
                wr = wins / len(trades)
                tot_pnl = sum(t["pnl_pips"] for t in trades)
                avg_r = sum(t["pnl_r"] for t in trades) / len(trades)
                avg_spd = sum(t["entry_spread_pips"] for t in trades) / len(trades)
                gw = sum(t["pnl_pips"] for t in trades if t["pnl_pips"] > 0)
                gl = abs(sum(t["pnl_pips"] for t in trades if t["pnl_pips"] < 0))
                pf = gw / gl if gl > 0 else 99.99

                print(f"  {pair:<10} {len(trades):>7} {wr:>5.1%} {avg_r:>+7.3f}R "
                      f"{pf:>5.2f} {tot_pnl:>+8.0f} {avg_spd:>6.1f}p")

    # ── Phase 3: Portfolio replay ─────────────────────────────────────
    print()
    print("=" * 90)
    print("PHASE 3: PORTFOLIO SIMULATION (max 3 pos, max 2 correlated same-dir)")
    print("=" * 90)

    for cn, tmap in all_results.items():
        flat = [t for trades in tmap.values() for t in trades]
        if not flat:
            continue

        accepted, stats = portfolio_filter(tmap)
        if not accepted:
            continue

        uf_n = len(flat)
        uf_r = sum(t["pnl_r"] for t in flat) / uf_n
        uf_pnl = sum(t["pnl_pips"] for t in flat)
        uf_wr = sum(1 for t in flat if t["pnl_pips"] > 0) / uf_n

        pf_n = len(accepted)
        pf_r = sum(t["pnl_r"] for t in accepted) / pf_n
        pf_pnl = sum(t["pnl_pips"] for t in accepted)
        pf_wr = sum(1 for t in accepted if t["pnl_pips"] > 0) / pf_n

        eq = np.array([0] + [t["pnl_r"] for t in accepted]).cumsum()
        dd = (np.maximum.accumulate(eq) - eq).max()

        print(f"\n--- {cn} ---")
        print(f"  Unfiltered : {uf_n:>6} trades  {uf_wr:>5.1%} WR  {uf_r:>+.3f}R  {uf_pnl:>+.0f} pips")
        print(f"  Portfolio  : {pf_n:>6} trades  {pf_wr:>5.1%} WR  {pf_r:>+.3f}R  {pf_pnl:>+.0f} pips")
        print(f"  Rejected   : {stats['rej_pos']} pos-limit + {stats['rej_corr']} correlation "
              f"= {stats['rej_pos']+stats['rej_corr']} total")
        print(f"  Max concurrent: {stats['max_concurrent']}")
        print(f"  Portfolio max DD: {dd:.1f}R | Final: {eq[-1]:+.1f}R")

    # ── Phase 4: True OOS walk-forward ────────────────────────────────
    print()
    print("=" * 90)
    print("PHASE 4: TRUE OUT-OF-SAMPLE WALK-FORWARD")
    print("(Signals generated on truncated data — no look-ahead)")
    print("=" * 90)
    print()

    best_sw, best_thresh = 5, 3
    sample = ohlc_data[ALL_PAIRS[0]]
    d_start = sample.index[0]
    d_end = sample.index[-1]

    wf_work = []
    train_d, test_d, step_d = 90, 30, 45

    ws = d_start
    while True:
        te_start = ws + pd.Timedelta(days=train_d)
        te_end = te_start + pd.Timedelta(days=test_d)
        if te_end > d_end:
            break
        for pair in ALL_PAIRS:
            if pair in ohlc_data:
                wf_work.append((pair, best_sw, best_thresh,
                                str(te_end), str(te_start), str(te_end)))
        ws += pd.Timedelta(days=step_d)

    n_windows = len(wf_work) // len([p for p in ALL_PAIRS if p in ohlc_data])
    print(f"{len(wf_work)} tasks ({n_windows} windows × "
          f"{len([p for p in ALL_PAIRS if p in ohlc_data])} pairs)")
    print(f"Config: M15 sw={best_sw} t={best_thresh}")
    print()

    t_wf = time.time()
    with Pool(n_workers) as pool:
        wf_raw = pool.map(_wf_worker, wf_work)
    print(f"\nWalk-forward: {time.time()-t_wf:.0f}s ({(time.time()-t_wf)/60:.1f} min)")

    wf_results = [r for r in wf_raw if r is not None]

    if wf_results:
        by_window = defaultdict(list)
        for r in wf_results:
            by_window[(r["test_start"], r["test_end"])].append(r)

        print(f"\n{'Window':<28} {'Pairs':>5} {'Trades':>7} {'AvgWR':>7} "
              f"{'AvgExp':>8} {'TotPnL':>8}")
        print("-" * 72)

        n_prof = 0
        for (ts, te), wr in sorted(by_window.items()):
            np_ = len(wr)
            tt = sum(r["trades"] for r in wr)
            aw = sum(r["win_rate"] for r in wr) / np_
            ae = sum(r["expectancy"] for r in wr) / np_
            tp = sum(r["total_pnl"] for r in wr)
            if tp > 0:
                n_prof += 1
            s = pd.Timestamp(ts).strftime("%Y-%m-%d")
            e = pd.Timestamp(te).strftime("%Y-%m-%d")
            print(f"  {s} → {e}  {np_:>5} {tt:>7} {aw:>6.1%} {ae:>+7.3f}R {tp:>+8.0f}")

        tot_w = len(by_window)
        print(f"\nProfitable windows: {n_prof}/{tot_w} ({n_prof/tot_w*100:.0f}%)")

        by_pair = defaultdict(list)
        for r in wf_results:
            by_pair[r["pair"]].append(r)

        print(f"\n{'Pair':<10} {'Windows':>8} {'WinPct':>7} {'AvgExp':>8} {'TotPnL':>8}")
        print("-" * 50)
        for pair in ALL_PAIRS:
            if pair in by_pair:
                pr = by_pair[pair]
                n = len(pr)
                pw = sum(1 for r in pr if r["total_pnl"] > 0)
                ae = sum(r["expectancy"] for r in pr) / n
                tp = sum(r["total_pnl"] for r in pr)
                print(f"  {pair:<10} {n:>8} {pw/n*100:>6.0f}% {ae:>+7.3f}R {tp:>+8.0f}")

    # ── Phase 5: Monte Carlo (portfolio-level) ────────────────────────
    print()
    print("=" * 90)
    print("PHASE 5: MONTE CARLO (portfolio-filtered trades)")
    print("=" * 90)
    print()

    for cn, tmap in all_results.items():
        accepted, _ = portfolio_filter(tmap)
        if len(accepted) >= 20:
            mc = monte_carlo([{"pnl_pips": t["pnl_r"]} for t in accepted],
                             n_simulations=5000)
            print(f"  {cn} ({len(accepted)} portfolio trades):")
            print(f"    Median final R:  {mc['median_pnl']:+.1f}R")
            print(f"    95% CI:          [{mc['ci_low']:+.1f}R, {mc['ci_high']:+.1f}R]")
            print(f"    P(profitable):   {mc['prob_profitable']*100:.0f}%")
            print(f"    Median max DD:   {mc['median_max_dd']:.1f}R")
            print()

    # ── Comparison ────────────────────────────────────────────────────
    print("=" * 90)
    print("NAIVE vs ROBUST COMPARISON")
    print("=" * 90)
    print()
    print("Naive:  fixed 1.5 pip spread, unlimited concurrent positions")
    print("Robust: variable spread (0.8–3.75p), max 3 positions, max 2 corr same-dir")
    print()

    naive = {
        "M15 sw=5 t=3": (6414, 0.226, 35994),
        "M15 sw=5 t=4": (3660, 0.250, 21816),
        "M15 sw=10 t=3": (3632, 0.173, 21479),
    }

    print(f"  {'Config':<18} {'Method':<10} {'Trades':>7} {'AvgExp':>8} {'TotPips':>9}")
    print("  " + "-" * 55)

    for cn in all_results:
        if cn in naive:
            nt, ne, np_ = naive[cn]
            print(f"  {cn:<18} {'Naive':<10} {nt:>7} {ne:>+7.3f}R {np_:>+9.0f}")

        accepted, _ = portfolio_filter(all_results[cn])
        if accepted:
            ar = sum(t["pnl_r"] for t in accepted) / len(accepted)
            ap = sum(t["pnl_pips"] for t in accepted)
            print(f"  {cn:<18} {'Robust':<10} {len(accepted):>7} {ar:>+7.3f}R {ap:>+9.0f}")
        print()

    print("=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
