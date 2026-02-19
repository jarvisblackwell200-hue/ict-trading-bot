#!/usr/bin/env python3
"""
Adaptive Trend Continuation — backtest + head-to-head comparison with ICT strategy.

Runs both strategies across all 7 pairs, prints per-pair breakdown, and checks
parameter sensitivity (±20%) to verify robustness.

Usage:
  PYTHONPATH=src python scripts/run_adaptive_strategy.py
  PYTHONPATH=src python scripts/run_adaptive_strategy.py --pairs EUR_USD GBP_USD
  PYTHONPATH=src python scripts/run_adaptive_strategy.py --sensitivity
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ict_bot.backtest.engine import BacktestConfig, run_backtest, simulate_trades
from src.ict_bot.backtest.metrics import calculate_metrics
from src.ict_bot.data.loader import load_pair
from src.ict_bot.risk import RiskConfig, RiskManager
from src.ict_bot.signals.trend_adaptive import generate_adaptive_signals

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ALL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "EUR_GBP",
]


def run_ict_baseline(pair: str, h1, htf) -> tuple[list[dict], object]:
    """Run the current ICT strategy as baseline."""
    config = BacktestConfig(
        pair=pair,
        swing_length=20,
        confluence_threshold=3,
        min_rr=2.0,
        sl_buffer_pips=10.0,
        spread_pips=1.5,
        slippage_pips=0.5,
        risk_per_trade=0.01,
        use_displacement=True,
        use_partial_tp=True,
        use_confluence_sizing=True,
        fvg_lookback=16,
        pullback_window=40,
    )
    trades, metrics = run_backtest(h1, htf_ohlc=htf, config=config)
    return trades, metrics


def run_adaptive(pair: str, h1, htf) -> tuple[list[dict], object]:
    """Run the Adaptive Trend Continuation strategy."""
    # Generate signals
    signals = generate_adaptive_signals(
        h1,
        htf_ohlc=htf,
        pair=pair,
        swing_length=10,
        ema_period=50,
        atr_period=14,
        atr_sl_mult=2.0,
        max_sl_pips=100.0,
    )

    # Simulate with trailing stop
    config = BacktestConfig(
        pair=pair,
        spread_pips=1.5,
        slippage_pips=0.5,
        risk_per_trade=0.01,
        starting_balance=10000.0,
        max_sl_pips=100.0,
        use_breakeven=True,
        be_threshold_r=1.0,            # Move to BE after 1R
        use_partial_tp=False,          # No partial TP — trailing stop handles exits
        use_confluence_sizing=False,   # Fixed 1% risk
        use_trailing_stop=True,
        atr_trail_mult=2.5,
        trailing_activation_r=1.5,
        max_bars=500,
    )

    if not signals:
        return [], calculate_metrics([])

    trades = simulate_trades(signals, h1, config)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)
    return trades, metrics


def print_header():
    print(f"\n{'='*130}")
    print(f"  ADAPTIVE TREND CONTINUATION vs ICT BASELINE — Head-to-Head Comparison")
    print(f"{'='*130}")
    print(f"  {'Pair':<10} {'Strategy':<22} {'Trades':>6} {'WR':>6} {'AvgRR':>6} "
          f"{'Expect':>8} {'PF':>6} {'PnL':>10} {'MaxDD%':>7} {'Sharpe':>7} "
          f"{'Trd/Yr':>7} {'Exits'}")
    print(f"  {'-'*120}")


def print_row(pair: str, label: str, trades: list[dict], m, years: float):
    if m.total_trades == 0:
        print(f"  {pair:<10} {label:<22} {'no trades':>6}")
        return
    tpy = m.total_trades / years if years > 0 else 0
    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "?")
        reasons[r] = reasons.get(r, 0) + 1
    exit_str = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))

    print(f"  {pair:<10} {label:<22} {m.total_trades:>6} {m.win_rate:>5.1%} "
          f"{m.avg_rr:>6.2f} {m.expectancy:>+7.3f}R {m.profit_factor:>6.2f} "
          f"{m.total_pnl_pips:>+9.1f} {m.max_drawdown_pct:>6.1%} "
          f"{m.sharpe_ratio:>7.2f} {tpy:>6.1f}  {exit_str}")


def run_sensitivity(pair: str, h1, htf, years: float):
    """Test ±20% variation on each parameter."""
    base_params = {
        "swing_length": 10,
        "ema_period": 50,
        "atr_period": 14,
        "atr_sl_mult": 2.0,
    }

    print(f"\n  Parameter Sensitivity for {pair} (±20%):")
    print(f"  {'Param':<16} {'Value':>8} {'Trades':>6} {'WR':>6} {'Expect':>8} "
          f"{'PnL':>10} {'Sharpe':>7}")
    print(f"  {'-'*70}")

    for param_name, base_val in base_params.items():
        for mult in [0.8, 1.0, 1.2]:
            val = base_val * mult
            if param_name in ("swing_length", "ema_period", "atr_period"):
                val = max(2, int(round(val)))
            else:
                val = round(val, 2)

            kwargs = dict(base_params)
            kwargs[param_name] = val

            signals = generate_adaptive_signals(
                h1, htf_ohlc=htf, pair=pair,
                swing_length=int(kwargs["swing_length"]),
                ema_period=int(kwargs["ema_period"]),
                atr_period=int(kwargs["atr_period"]),
                atr_sl_mult=float(kwargs["atr_sl_mult"]),
            )

            config = BacktestConfig(
                pair=pair, spread_pips=1.5, slippage_pips=0.5,
                risk_per_trade=0.01, max_sl_pips=100.0,
                use_breakeven=True, be_threshold_r=1.0,
                use_partial_tp=False, use_confluence_sizing=False,
                use_trailing_stop=True, atr_trail_mult=2.5,
                trailing_activation_r=1.5, max_bars=500,
            )

            if signals:
                trades = simulate_trades(signals, h1, config)
                m = calculate_metrics(trades, pip_size=config.pip_size)
            else:
                m = calculate_metrics([])

            marker = " <-- base" if mult == 1.0 else ""
            status = "+" if m.expectancy > 0 else "-"
            print(f"  [{status}] {param_name:<14} {val:>8} {m.total_trades:>6} "
                  f"{m.win_rate:>5.1%} {m.expectancy:>+7.3f}R "
                  f"{m.total_pnl_pips:>+9.1f} {m.sharpe_ratio:>7.2f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Trend Continuation backtest")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Pairs to test (default: all 7)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run parameter sensitivity analysis")
    args = parser.parse_args()

    pairs = args.pairs or ALL_PAIRS

    print_header()

    # Aggregate results
    ict_total = {"trades": 0, "pnl": 0.0}
    adp_total = {"trades": 0, "pnl": 0.0}

    for pair in pairs:
        try:
            h1 = load_pair(pair, "H1")
            htf = load_pair(pair, "D")
        except FileNotFoundError:
            print(f"  {pair:<10} {'(no data)':>20}", flush=True)
            continue

        years = (h1.index[-1] - h1.index[0]).days / 365.25
        print(f"\n  --- {pair} ({len(h1)} bars, {years:.1f} years) ---", flush=True)

        # ICT baseline
        ict_trades, ict_m = run_ict_baseline(pair, h1, htf)
        print_row(pair, "ICT Baseline", ict_trades, ict_m, years)

        # Adaptive
        adp_trades, adp_m = run_adaptive(pair, h1, htf)
        print_row(pair, "Adaptive Trend", adp_trades, adp_m, years)

        ict_total["trades"] += ict_m.total_trades
        ict_total["pnl"] += ict_m.total_pnl_pips
        adp_total["trades"] += adp_m.total_trades
        adp_total["pnl"] += adp_m.total_pnl_pips

        # Parameter sensitivity (optional, for first pair or all)
        if args.sensitivity:
            run_sensitivity(pair, h1, htf, years)

    # --- Portfolio summary ---
    print(f"\n{'='*130}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"{'='*130}")
    print(f"  ICT Baseline:   {ict_total['trades']:>4} total trades | "
          f"{ict_total['pnl']:>+9.1f} total pips")
    print(f"  Adaptive Trend: {adp_total['trades']:>4} total trades | "
          f"{adp_total['pnl']:>+9.1f} total pips")

    if ict_total["trades"] > 0 and adp_total["trades"] > 0:
        ratio = adp_total["trades"] / ict_total["trades"]
        print(f"\n  Trade frequency: Adaptive generates {ratio:.1f}x more trades than ICT")

    print()


if __name__ == "__main__":
    main()
