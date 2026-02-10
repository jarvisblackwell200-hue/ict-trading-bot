"""
Walk-forward analysis and robustness testing.

Prevents overfitting by validating strategy across rolling time windows
and testing parameter sensitivity.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from .engine import BacktestConfig, run_backtest
from .metrics import BacktestMetrics, calculate_metrics

logger = logging.getLogger(__name__)


def walk_forward(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
    train_months: int = 6,
    test_months: int = 1,
    step_months: int = 1,
) -> list[dict]:
    """
    Walk-forward analysis.

    Rolls through data: train on N months, test on next M months, step forward.
    Returns metrics for each test window.
    """
    if config is None:
        config = BacktestConfig()

    results = []
    start = ohlc.index.min()
    end = ohlc.index.max()
    current = start

    window_num = 0
    while True:
        train_start = current
        train_end = train_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > end:
            break

        # Split data
        test_ohlc = ohlc[(ohlc.index >= test_start) & (ohlc.index < test_end)]

        if len(test_ohlc) < 50:
            current += pd.DateOffset(months=step_months)
            continue

        # Run backtest on test window only
        # (In a real walk-forward, you'd optimize on train, test on test.
        #  Since our parameters are fixed by ICT definitions, we just test.)
        trades, metrics = run_backtest(test_ohlc, htf_ohlc=htf_ohlc, config=config)

        window_num += 1
        result = {
            "window": window_num,
            "test_start": test_start,
            "test_end": test_end,
            "trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "pnl_pips": metrics.total_pnl_pips,
            "max_dd_pips": metrics.max_drawdown_pips,
            "profit_factor": metrics.profit_factor,
        }
        results.append(result)

        logger.info(
            f"Window {window_num}: {test_start.date()} to {test_end.date()} | "
            f"{metrics.total_trades} trades | WR={metrics.win_rate:.1%} | "
            f"Exp={metrics.expectancy:+.3f}R | PnL={metrics.total_pnl_pips:+.1f}"
        )

        current += pd.DateOffset(months=step_months)

    return results


def parameter_sensitivity(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    base_config: BacktestConfig | None = None,
) -> dict[str, list[dict]]:
    """
    Test parameter sensitivity by varying each key parameter +/- 20%.

    If results change drastically with small parameter changes,
    the strategy is likely overfitted.
    """
    if base_config is None:
        base_config = BacktestConfig()

    # Parameters to test and their variation ranges
    param_tests = {
        "swing_length": [10, 15, 20, 25, 30],
        "confluence_threshold": [2, 3, 4],
        "min_rr": [1.5, 2.0, 2.5, 3.0],
        "sl_buffer_pips": [5.0, 10.0, 15.0, 20.0],
    }

    results = {}

    for param_name, values in param_tests.items():
        param_results = []
        for val in values:
            cfg = deepcopy(base_config)
            setattr(cfg, param_name, val)

            trades, metrics = run_backtest(ohlc, htf_ohlc=htf_ohlc, config=cfg)

            param_results.append({
                "value": val,
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "expectancy": metrics.expectancy,
                "pnl_pips": metrics.total_pnl_pips,
                "profit_factor": metrics.profit_factor,
            })

            logger.info(
                f"  {param_name}={val}: {metrics.total_trades} trades | "
                f"WR={metrics.win_rate:.1%} | Exp={metrics.expectancy:+.3f}R"
            )

        results[param_name] = param_results

    return results


def monte_carlo(
    trades: list[dict],
    n_simulations: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """
    Monte Carlo simulation: randomize trade order to get confidence intervals.

    Tests whether the strategy's equity curve is robust or dependent
    on a lucky sequence of trades.
    """
    if not trades:
        return {"median_pnl": 0, "ci_low": 0, "ci_high": 0, "prob_profitable": 0}

    pnls = np.array([t["pnl_pips"] for t in trades])
    n_trades = len(pnls)

    final_pnls = []
    max_drawdowns = []

    rng = np.random.default_rng(42)

    for _ in range(n_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(pnls)
        cumulative = np.cumsum(shuffled)
        final_pnls.append(cumulative[-1])

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        dd = running_max - cumulative
        max_drawdowns.append(dd.max())

    final_pnls = np.array(final_pnls)
    max_drawdowns = np.array(max_drawdowns)

    ci_low_idx = int((1 - confidence) / 2 * n_simulations)
    ci_high_idx = int((1 + confidence) / 2 * n_simulations)
    sorted_pnls = np.sort(final_pnls)
    sorted_dd = np.sort(max_drawdowns)

    return {
        "median_pnl": float(np.median(final_pnls)),
        "mean_pnl": float(np.mean(final_pnls)),
        "ci_low": float(sorted_pnls[ci_low_idx]),
        "ci_high": float(sorted_pnls[ci_high_idx]),
        "prob_profitable": float((final_pnls > 0).mean()),
        "median_max_dd": float(np.median(max_drawdowns)),
        "worst_case_dd": float(sorted_dd[ci_high_idx]),
        "n_simulations": n_simulations,
    }


def format_walk_forward_report(results: list[dict]) -> str:
    """Format walk-forward results as a readable report."""
    if not results:
        return "No walk-forward results."

    lines = [
        f"\n{'='*70}",
        "  WALK-FORWARD ANALYSIS",
        f"{'='*70}",
        f"  {'Window':<8} {'Period':<25} {'Trades':>7} {'Win%':>7} {'Exp(R)':>8} {'PnL':>10}",
        f"  {'-'*65}",
    ]

    total_trades = 0
    positive_windows = 0

    for r in results:
        period = f"{r['test_start'].strftime('%Y-%m')} to {r['test_end'].strftime('%Y-%m')}"
        lines.append(
            f"  {r['window']:<8} {period:<25} {r['trades']:>7} "
            f"{r['win_rate']:>6.1%} {r['expectancy']:>+7.3f} {r['pnl_pips']:>+9.1f}"
        )
        total_trades += r["trades"]
        if r["pnl_pips"] > 0:
            positive_windows += 1

    lines.append(f"  {'-'*65}")
    lines.append(f"  Windows: {len(results)} | Profitable: {positive_windows}/{len(results)} "
                 f"({positive_windows/len(results):.0%}) | Total trades: {total_trades}")
    lines.append(f"{'='*70}")

    return "\n".join(lines)


def format_monte_carlo_report(mc: dict) -> str:
    """Format Monte Carlo results."""
    return (
        f"\n{'='*60}\n"
        f"  MONTE CARLO SIMULATION ({mc['n_simulations']} runs)\n"
        f"{'='*60}\n"
        f"  Median final P&L:     {mc['median_pnl']:+.1f} pips\n"
        f"  Mean final P&L:       {mc['mean_pnl']:+.1f} pips\n"
        f"  95% CI:               [{mc['ci_low']:+.1f}, {mc['ci_high']:+.1f}] pips\n"
        f"  P(profitable):        {mc['prob_profitable']:.1%}\n"
        f"  Median max drawdown:  {mc['median_max_dd']:.1f} pips\n"
        f"  Worst-case DD (95%):  {mc['worst_case_dd']:.1f} pips\n"
        f"{'='*60}"
    )
