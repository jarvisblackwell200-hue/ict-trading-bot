#!/usr/bin/env python3
"""Full comparison: old H1 vs improved H1 vs MTF (H1+M15) for EUR_USD & GBP_USD."""
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

pair = sys.argv[1] if len(sys.argv) > 1 else "EUR_USD"

# ── Load data ──────────────────────────────────────────────────
print(f"Loading {pair}...", flush=True)
h1 = load_pair(pair, "H1")
htf = load_pair(pair, "D")
print(f"  H1: {len(h1)} candles ({h1.index[0].date()} → {h1.index[-1].date()})", flush=True)

# Try loading M5 and resampling to M15
m15 = None
try:
    m5 = load_pair(pair, "M5")
    m15 = m5.resample("15min").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna()
    print(f"  M15 (from M5): {len(m15)} candles ({m15.index[0].date()} → {m15.index[-1].date()})", flush=True)
except FileNotFoundError:
    print(f"  M5 data not available for {pair} — skipping MTF", flush=True)

# ── Configs ────────────────────────────────────────────────────
def print_row(label, trades, m):
    be = sum(1 for t in trades if t["exit_reason"] == "BE")
    tp = sum(1 for t in trades if t["exit_reason"] == "TP")
    sl = sum(1 for t in trades if t["exit_reason"] == "SL")
    part = sum(1 for t in trades if t.get("partial_taken", False))
    longs = sum(1 for t in trades if t["direction"] == "long")
    shorts = sum(1 for t in trades if t["direction"] == "short")
    avg_rr_str = f"{m.avg_rr:>6.2f}" if m.total_trades > 0 else "   N/A"
    print(f"  {label:<26} {m.total_trades:>6} {m.win_rate:>5.1%} {avg_rr_str} "
          f"{m.expectancy:>+7.3f}R {m.profit_factor:>6.2f} {m.total_pnl_pips:>+9.1f} "
          f"{m.max_drawdown_pips:>8.1f} {m.sharpe_ratio:>7.2f} "
          f"{be:>3} {tp:>3} {sl:>3} {part:>4} {longs:>2}L/{shorts:<2}S",
          flush=True)

print(f"\n{'='*130}", flush=True)
print(f"  {pair} — Full Strategy Comparison", flush=True)
print(f"{'='*130}", flush=True)
print(f"{'Config':<28} {'Trd':>6} {'WR':>6} {'AvgRR':>6} {'Expect':>8} {'PF':>6} "
      f"{'PnL':>10} {'MaxDD':>8} {'Sharpe':>7} "
      f"{'BE':>3} {'TP':>3} {'SL':>3} {'Part':>4} {'L/S':>6}", flush=True)
print("-" * 130, flush=True)

# 1. Old defaults (H1, no new features)
config_old = BacktestConfig(
    pair=pair, swing_length=20, confluence_threshold=3,
    min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
    use_displacement=False, use_partial_tp=False, skip_days=None,
    use_confluence_sizing=False, fvg_lookback=8, pullback_window=20,
)
trades, m = run_backtest(h1, htf_ohlc=htf, config=config_old)
print_row("Old H1 (baseline)", trades, m)

# 2. Improved H1: wider pipeline + displacement + partials + sizing
config_improved = BacktestConfig(
    pair=pair, swing_length=20, confluence_threshold=3,
    min_rr=2.0, sl_buffer_pips=10.0, spread_pips=1.5, slippage_pips=0.5,
    use_displacement=True, use_partial_tp=True, skip_days=None,
    use_confluence_sizing=True, fvg_lookback=16, pullback_window=40,
)
trades_imp, m_imp = run_backtest(h1, htf_ohlc=htf, config=config_improved)
print_row("Improved H1 (wider+sized)", trades_imp, m_imp)

# 3. MTF: H1 structure + M15 entry (if M15 data available)
if m15 is not None:
    # Filter H1 and HTF to the M15 data range
    m15_start = m15.index[0]
    m15_end = m15.index[-1]
    h1_overlap = h1[(h1.index >= m15_start - pd.Timedelta(days=30)) & (h1.index <= m15_end)]
    htf_overlap = htf[htf.index <= m15_end]

    config_mtf = BacktestConfig(
        pair=pair, swing_length=20, confluence_threshold=3,
        min_rr=2.0, sl_buffer_pips=3.0, spread_pips=1.5, slippage_pips=0.5,
        use_partial_tp=True, use_confluence_sizing=True,
    )

    # Generate MTF signals
    mtf_signals = generate_signals_mtf(
        structure_ohlc=h1_overlap,
        entry_ohlc=m15,
        htf_ohlc=htf_overlap,
        pair=pair,
        structure_swing_length=20,
        entry_swing_length=10,
        confluence_threshold=3,
        min_rr=2.0,
        sl_buffer_pips=3.0,
        use_liquidity_targets=True,
        use_premium_discount=True,
        use_displacement=True,
        fvg_lookback=32,
    )

    if mtf_signals:
        mtf_trades = simulate_trades(mtf_signals, m15, config_mtf)
        mtf_metrics = calculate_metrics(mtf_trades, pip_size=config_mtf.pip_size)
        print_row(f"MTF H1+M15 ({len(m15)}bars)", mtf_trades, mtf_metrics)
    else:
        print(f"  {'MTF H1+M15':<26} {'no signals':>6}", flush=True)

    # Also run old H1 strategy on just the M15 period for fair comparison
    h1_period = h1[(h1.index >= m15_start) & (h1.index <= m15_end)]
    if len(h1_period) > 100:
        trades_period, m_period = run_backtest(h1_period, htf_ohlc=htf_overlap, config=config_old)
        print_row(f"Old H1 (same period)", trades_period, m_period)

# Show sample trades from improved config
print(f"\n--- Sample trades: Improved H1 (first 20) ---", flush=True)
if trades_imp:
    print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'SL':>9} {'TP':>9} {'Exit':>9} "
          f"{'PnL':>8} {'Out':>4} {'P':>1} {'S':>2} {'TP Method'}", flush=True)
    print(f"  {'-'*105}", flush=True)
    for t in trades_imp[:20]:
        print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} "
              f"{t['entry_price']:>9.5f} {t['stop_loss']:>9.5f} "
              f"{t['take_profit']:>9.5f} {t['exit_price']:>9.5f} "
              f"{t['pnl_pips']:>+7.1f} {t['exit_reason']:>4} "
              f"{'Y' if t.get('partial_taken') else 'N':>1} {t['confluence_score']:>2} "
              f"{t.get('tp_method', 'rr_based')}", flush=True)

# Show MTF sample trades
if m15 is not None and mtf_signals:
    print(f"\n--- Sample trades: MTF H1+M15 ---", flush=True)
    print(f"  {'Entry Time':<22} {'Dir':>5} {'Entry':>9} {'SL':>9} {'TP':>9} {'Exit':>9} "
          f"{'PnL':>8} {'RR':>5} {'Out':>4} {'S':>2} {'TP Method'}", flush=True)
    print(f"  {'-'*110}", flush=True)
    for t in mtf_trades[:20]:
        print(f"  {str(t['entry_time'])[:19]:<22} {t['direction']:>5} "
              f"{t['entry_price']:>9.5f} {t['stop_loss']:>9.5f} "
              f"{t['take_profit']:>9.5f} {t['exit_price']:>9.5f} "
              f"{t['pnl_pips']:>+7.1f} {t['risk_pips']:>5.1f} "
              f"{t['exit_reason']:>4} {t['confluence_score']:>2} "
              f"{t.get('tp_method', 'rr_based')}", flush=True)

    # Show SL comparison
    if trades_imp and mtf_trades:
        avg_sl_h1 = sum(t["risk_pips"] for t in trades_imp) / len(trades_imp)
        avg_sl_mtf = sum(t["risk_pips"] for t in mtf_trades) / len(mtf_trades)
        print(f"\n  Average SL: H1={avg_sl_h1:.1f} pips | M15={avg_sl_mtf:.1f} pips "
              f"({avg_sl_mtf/avg_sl_h1*100:.0f}% of H1)", flush=True)
