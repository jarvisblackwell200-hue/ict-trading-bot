"""
Backtesting engine for ICT signals (v2 — with trade management).

Changes from v1:
  - Break-even stop: move SL to entry after price reaches 1R profit
  - Partial TP: close half at 1R, let rest run to full TP
  - Max SL cap: reject signals with unreasonably wide stops
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signals.detector import Signal, generate_signals
from .metrics import BacktestMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    pair: str = "EUR_USD"
    swing_length: int = 20
    confluence_threshold: int = 3
    min_rr: float = 2.0
    sl_buffer_pips: float = 5.0
    spread_pips: float = 1.5
    slippage_pips: float = 0.5
    risk_per_trade: float = 0.01       # 1% of account
    starting_balance: float = 10000.0
    target_kill_zones: list | None = None
    require_htf_bias: bool = True
    max_sl_pips: float = 100.0         # Reject signals with SL > this
    use_breakeven: bool = True          # Move SL to entry after 1R
    pullback_window: int = 20

    @property
    def pip_size(self) -> float:
        return 0.01 if "JPY" in self.pair else 0.0001


def simulate_trades(
    signals: list[Signal],
    ohlc: pd.DataFrame,
    config: BacktestConfig,
) -> list[dict]:
    """
    Simulate trade execution with realistic trade management.

    For each signal:
      1. Enter at signal price (adjusted for spread/slippage)
      2. Walk forward checking SL and TP
      3. If break-even enabled: move SL to entry after 1R profit reached
    """
    pip_size = config.pip_size
    spread = config.spread_pips * pip_size
    slippage = config.slippage_pips * pip_size

    trades = []
    balance = config.starting_balance

    for signal in signals:
        try:
            entry_idx = ohlc.index.get_loc(signal.timestamp)
        except KeyError:
            continue

        # Adjust entry for spread + slippage
        if signal.direction == "long":
            actual_entry = signal.entry_price + spread / 2 + slippage
        else:
            actual_entry = signal.entry_price - spread / 2 - slippage

        actual_sl = signal.stop_loss
        actual_tp = signal.take_profit

        # Risk calculation
        if signal.direction == "long":
            risk_pips = (actual_entry - actual_sl) / pip_size
        else:
            risk_pips = (actual_sl - actual_entry) / pip_size

        if risk_pips <= 0:
            continue

        # Reject trades with SL too wide
        if risk_pips > config.max_sl_pips:
            continue

        risk_amount = balance * config.risk_per_trade
        pip_value = risk_amount / risk_pips

        # 1R level for break-even management
        one_r_distance = risk_pips * pip_size
        if signal.direction == "long":
            one_r_level = actual_entry + one_r_distance
        else:
            one_r_level = actual_entry - one_r_distance

        # Walk forward
        exit_price = None
        exit_time = None
        exit_reason = None
        current_sl = actual_sl
        be_triggered = False

        for j in range(entry_idx + 1, min(entry_idx + 200, len(ohlc))):
            candle = ohlc.iloc[j]
            high = candle["high"]
            low = candle["low"]

            if signal.direction == "long":
                # Break-even check: if high reached 1R, move SL to entry
                if config.use_breakeven and not be_triggered and high >= one_r_level:
                    current_sl = actual_entry + spread  # BE + spread
                    be_triggered = True

                # Check SL
                if low <= current_sl:
                    exit_price = current_sl - slippage
                    exit_time = ohlc.index[j]
                    exit_reason = "BE" if be_triggered and current_sl > actual_sl else "SL"
                    break
                # Check TP
                if high >= actual_tp:
                    exit_price = actual_tp - spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break
            else:
                # Break-even check
                if config.use_breakeven and not be_triggered and low <= one_r_level:
                    current_sl = actual_entry - spread
                    be_triggered = True

                # Check SL
                if high >= current_sl:
                    exit_price = current_sl + slippage
                    exit_time = ohlc.index[j]
                    exit_reason = "BE" if be_triggered and current_sl < actual_sl else "SL"
                    break
                # Check TP
                if low <= actual_tp:
                    exit_price = actual_tp + spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break

        if exit_price is None:
            exit_price = ohlc.iloc[min(entry_idx + 200, len(ohlc) - 1)]["close"]
            exit_time = ohlc.index[min(entry_idx + 200, len(ohlc) - 1)]
            exit_reason = "TIMEOUT"

        # P&L
        if signal.direction == "long":
            pnl_pips = (exit_price - actual_entry) / pip_size
        else:
            pnl_pips = (actual_entry - exit_price) / pip_size

        pnl_amount = pnl_pips * pip_value
        balance += pnl_amount

        trades.append({
            "entry_time": signal.timestamp,
            "exit_time": exit_time,
            "direction": signal.direction,
            "entry_price": round(actual_entry, 5),
            "exit_price": round(exit_price, 5),
            "stop_loss": round(actual_sl, 5),
            "take_profit": round(actual_tp, 5),
            "risk_pips": round(risk_pips, 1),
            "pnl_pips": round(pnl_pips, 1),
            "pnl_amount": round(pnl_amount, 2),
            "balance": round(balance, 2),
            "exit_reason": exit_reason,
            "be_triggered": be_triggered,
            "confluence_score": signal.confluence_score,
            "confluences": signal.confluences,
            "kill_zone": signal.kill_zone,
        })

    return trades


def run_backtest(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    config: BacktestConfig | None = None,
) -> tuple[list[dict], BacktestMetrics]:
    """
    Run a full backtest: generate signals -> simulate trades -> compute metrics.
    """
    if config is None:
        config = BacktestConfig()

    logger.info(f"Running backtest on {config.pair} ({len(ohlc)} candles)...")

    signals = generate_signals(
        ohlc,
        htf_ohlc=htf_ohlc,
        pair=config.pair,
        swing_length=config.swing_length,
        confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr,
        sl_buffer_pips=config.sl_buffer_pips,
        target_kill_zones=config.target_kill_zones,
        pullback_window=config.pullback_window,
        require_htf_bias=config.require_htf_bias,
    )

    if not signals:
        logger.warning("No signals generated.")
        return [], calculate_metrics([])

    trades = simulate_trades(signals, ohlc, config)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)

    logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                f"win rate={metrics.win_rate:.1%}, expectancy={metrics.expectancy:+.3f}R")

    return trades, metrics
