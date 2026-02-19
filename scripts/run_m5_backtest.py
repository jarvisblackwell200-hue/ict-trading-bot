#!/usr/bin/env python3
"""MTF backtest: H1 structure + M5 entry for EUR_USD & GBP_USD.

Tests multiple swing_length and risk configurations on M5 data.
Compares against H1-only baseline over the same 83-day period.
"""
from __future__ import annotations
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHONUNBUFFERED"] = "1"

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
from src.ict_bot.backtest.engine import BacktestConfig, simulate_trades, run_backtest
from src.ict_bot.backtest.metrics import calculate_metrics
from src.ict_bot.signals.detector import generate_signals_mtf
from src.ict_bot.data.loader import load_pair

PAIRS = ["EUR_USD", "GBP_USD"]

def print_row(label, trades, m, pip_size):
    if m.total_trades == 0:
        print(f"  {label:<36} {'no trades':>6}", flush=True)
        return
    be = sum(1 for t in trades if t["exit_reason"] == "BE")
    tp = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl = sum(1 for t in trades if t["exit_reason"] == "SL")
    longs = sum(1 for t in trades if t["direction"] == "long")
    shorts = sum(1 for t in trades if t["direction"] == "short")
    avg_sl = sum(t["risk_pips"] for t in trades) / len(trades)
    end_bal = trades[-1]["balance"]
    ret = (end_bal - 10000) / 10000
    print(f"  {label:<36} {m.total_trades:>4} {m.win_rate:>5.1%} {m.expectancy:>+7.3f}R "
          f"{m.total_pnl_pips:>+8.1f} ${end_bal:>7,.0f} {ret:>+6.1%} "
          f"{avg_sl:>5.1f} {m.max_drawdown_pips:>7.1f} {m.sharpe_ratio:>6.2f} "
          f"{be:>3}/{tp:>3}/{sl:<3} {longs:>2}L/{shorts:<2}S",
          flush=True)

# MTF configs to test
MTF_CONFIGS = [
    # M5 SL (tighter) — original approach
    ("M5entry M5sl sw=10 2%", dict(
        structure_swing_length=10, entry_swing_length=5,
        sl_buffer_pips=3.0, fvg_lookback=60, pullback_window=240,
        risk_per_trade=0.02, min_rr=2.0, use_structure_sl=False,
    )),
    # H1 SL (wider, more room) — hybrid approach
    ("M5entry H1sl sw=20 1%", dict(
        structure_swing_length=20, entry_swing_length=8,
        sl_buffer_pips=5.0, fvg_lookback=32, pullback_window=120,
        risk_per_trade=0.01, min_rr=2.0, use_structure_sl=True,
    )),
    ("M5entry H1sl sw=12 1%", dict(
        structure_swing_length=12, entry_swing_length=6,
        sl_buffer_pips=5.0, fvg_lookback=48, pullback_window=180,
        risk_per_trade=0.01, min_rr=2.0, use_structure_sl=True,
    )),
    ("M5entry H1sl sw=10 1%", dict(
        structure_swing_length=10, entry_swing_length=5,
        sl_buffer_pips=5.0, fvg_lookback=60, pullback_window=240,
        risk_per_trade=0.01, min_rr=2.0, use_structure_sl=True,
    )),
    ("M5entry H1sl sw=10 2%", dict(
        structure_swing_length=10, entry_swing_length=5,
        sl_buffer_pips=5.0, fvg_lookback=60, pullback_window=240,
        risk_per_trade=0.02, min_rr=2.0, use_structure_sl=True,
    )),
    ("M5entry H1sl sw=10 2% rr3", dict(
        structure_swing_length=10, entry_swing_length=5,
        sl_buffer_pips=5.0, fvg_lookback=60, pullback_window=240,
        risk_per_trade=0.02, min_rr=3.0, use_structure_sl=True,
    )),
]

for pair in PAIRS:
    print(f"\nLoading {pair}...", flush=True)
    h1 = load_pair(pair, "H1")
    htf = load_pair(pair, "D")
    m5 = load_pair(pair, "M5")
    print(f"  H1: {len(h1)} candles | M5: {len(m5)} candles ({m5.index[0].date()} → {m5.index[-1].date()})", flush=True)

    m5_start = m5.index[0]
    m5_end = m5.index[-1]

    # Filter H1 to include some lookback before M5 period
    h1_overlap = h1[(h1.index >= m5_start - pd.Timedelta(days=60)) & (h1.index <= m5_end)]
    htf_overlap = htf[htf.index <= m5_end]

    pip_size = 0.01 if "JPY" in pair else 0.0001

    print(f"\n{'='*130}", flush=True)
    print(f"  {pair} — H1+M5 MTF Comparison (83 days: {m5_start.date()} → {m5_end.date()})", flush=True)
    print(f"{'='*130}", flush=True)
    print(f"  {'Config':<36} {'Trd':>4} {'WR':>6} {'Expect':>8} "
          f"{'PnL':>9} {'End$':>8} {'Ret%':>7} "
          f"{'AvgSL':>5} {'MaxDD':>7} {'Shrpe':>6} "
          f"{'BE/TP/SL':>10} {'L/S':>7}", flush=True)
    print(f"  {'-'*125}", flush=True)

    # H1-only baselines over the same period
    h1_period = h1[(h1.index >= m5_start) & (h1.index <= m5_end)]
    for sw, risk in [(20, 0.01), (12, 0.01), (10, 0.02)]:
        config = BacktestConfig(
            pair=pair, swing_length=sw, confluence_threshold=3,
            min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
            use_displacement=True, use_partial_tp=True, skip_days=None,
            use_confluence_sizing=True, pullback_window=40, fvg_lookback=16,
            use_liquidity_targets=False, use_premium_discount=False, compute_ob=False,
            risk_per_trade=risk,
        )
        trades, m = run_backtest(h1_period, htf_ohlc=htf_overlap, config=config)
        print_row(f"H1-only sw={sw} {risk:.0%} (baseline)", trades, m, pip_size)

    print(f"  {'-'*125}", flush=True)

    # MTF H1+M5 configs
    for label, params in MTF_CONFIGS:
        struct_sw = params["structure_swing_length"]
        entry_sw = params["entry_swing_length"]
        sl_buf = params["sl_buffer_pips"]
        fvg_lb = params["fvg_lookback"]
        pb_win = params["pullback_window"]
        risk = params["risk_per_trade"]
        min_rr = params["min_rr"]

        config_mtf = BacktestConfig(
            pair=pair, swing_length=struct_sw, confluence_threshold=3,
            min_rr=min_rr, sl_buffer_pips=sl_buf, spread_pips=1.5, slippage_pips=0.5,
            use_partial_tp=True, use_confluence_sizing=True,
            use_liquidity_targets=False, use_premium_discount=False, compute_ob=False,
            risk_per_trade=risk,
        )

        use_struct_sl = params.get("use_structure_sl", False)
        signals = generate_signals_mtf(
            structure_ohlc=h1_overlap,
            entry_ohlc=m5,
            htf_ohlc=htf_overlap,
            pair=pair,
            structure_swing_length=struct_sw,
            entry_swing_length=entry_sw,
            confluence_threshold=3,
            min_rr=min_rr,
            sl_buffer_pips=sl_buf,
            use_liquidity_targets=False,
            use_premium_discount=False,
            use_displacement=True,
            fvg_lookback=fvg_lb,
            pullback_window=pb_win,
            compute_ob=False,
            use_structure_sl=use_struct_sl,
        )

        if signals:
            trades = simulate_trades(signals, m5, config_mtf)
            metrics = calculate_metrics(trades, pip_size=pip_size)
            print_row(label, trades, metrics, pip_size)
        else:
            print(f"  {label:<36} {'no signals':>6}", flush=True)

    # Sample trades for best M5 config
    print(f"\n  --- Sample trades: M5entry H1sl sw=10 2% ---", flush=True)
    best = MTF_CONFIGS[4]  # M5entry H1sl sw=10 2%
    bp = best[1]
    signals = generate_signals_mtf(
        structure_ohlc=h1_overlap, entry_ohlc=m5, htf_ohlc=htf_overlap,
        pair=pair, structure_swing_length=bp["structure_swing_length"],
        entry_swing_length=bp["entry_swing_length"], confluence_threshold=3,
        min_rr=bp["min_rr"], sl_buffer_pips=bp["sl_buffer_pips"],
        use_liquidity_targets=False, use_premium_discount=False,
        use_displacement=True, fvg_lookback=bp["fvg_lookback"],
        pullback_window=bp["pullback_window"], compute_ob=False,
        use_structure_sl=bp.get("use_structure_sl", False),
    )
    if signals:
        config_best = BacktestConfig(
            pair=pair, swing_length=bp["structure_swing_length"], confluence_threshold=3,
            min_rr=bp["min_rr"], sl_buffer_pips=bp["sl_buffer_pips"],
            spread_pips=1.5, slippage_pips=0.5,
            use_partial_tp=True, use_confluence_sizing=True,
            use_liquidity_targets=False, use_premium_discount=False, compute_ob=False,
            risk_per_trade=bp["risk_per_trade"],
        )
        trades = simulate_trades(signals, m5, config_best)
        print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'SL':>9} {'TP':>9} {'Exit':>9} "
              f"{'PnL':>8} {'Risk':>5} {'Out':>4} {'S':>2}", flush=True)
        print(f"  {'-'*100}", flush=True)
        for t in trades[:25]:
            print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} "
                  f"{t['entry_price']:>9.5f} {t['stop_loss']:>9.5f} "
                  f"{t['take_profit']:>9.5f} {t['exit_price']:>9.5f} "
                  f"{t['pnl_pips']:>+7.1f} {t['risk_pips']:>5.1f} "
                  f"{t['exit_reason']:>4} {t['confluence_score']:>2}", flush=True)

        avg_sl_m5 = sum(t["risk_pips"] for t in trades) / len(trades)
        print(f"\n  Average SL: M5={avg_sl_m5:.1f} pips | Trades: {len(trades)} in 83 days "
              f"({len(trades)/83*365:.0f}/year pace)", flush=True)
