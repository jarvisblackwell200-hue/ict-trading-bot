"""
Backtesting engine for ICT signals.

Feeds historical OHLC data through the signal detector and simulates
trade execution with realistic spread, slippage, and position sizing.
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
    sl_buffer_pips: float = 10.0
    spread_pips: float = 1.5
    slippage_pips: float = 0.5
    risk_per_trade: float = 0.01       # 1% of account
    starting_balance: float = 10000.0
    target_kill_zones: list | None = None

    @property
    def pip_size(self) -> float:
        return 0.01 if "JPY" in self.pair else 0.0001


def simulate_trades(
    signals: list[Signal],
    ohlc: pd.DataFrame,
    config: BacktestConfig,
) -> list[dict]:
    """
    Simulate trade execution for a list of signals against OHLC data.

    For each signal, walks forward through candles to check if SL or TP
    is hit first. Accounts for spread and slippage.
    """
    pip_size = config.pip_size
    spread = config.spread_pips * pip_size
    slippage = config.slippage_pips * pip_size

    trades = []
    balance = config.starting_balance

    for signal in signals:
        # Find the bar index for this signal's timestamp
        try:
            entry_idx = ohlc.index.get_loc(signal.timestamp)
        except KeyError:
            continue

        # Adjust entry for spread + slippage
        if signal.direction == "long":
            actual_entry = signal.entry_price + spread / 2 + slippage
            actual_sl = signal.stop_loss
            actual_tp = signal.take_profit
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

        risk_amount = balance * config.risk_per_trade
        pip_value = risk_amount / risk_pips

        # Walk forward to determine outcome
        exit_price = None
        exit_time = None
        exit_reason = None

        for j in range(entry_idx + 1, min(entry_idx + 500, len(ohlc))):
            candle = ohlc.iloc[j]
            high = candle["high"]
            low = candle["low"]

            if signal.direction == "long":
                # Check SL hit first (worst case)
                if low <= actual_sl:
                    exit_price = actual_sl - slippage  # slippage on exit too
                    exit_time = ohlc.index[j]
                    exit_reason = "SL"
                    break
                # Check TP hit
                if high >= actual_tp:
                    exit_price = actual_tp - spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break
            else:
                # Check SL hit first
                if high >= actual_sl:
                    exit_price = actual_sl + slippage
                    exit_time = ohlc.index[j]
                    exit_reason = "SL"
                    break
                # Check TP hit
                if low <= actual_tp:
                    exit_price = actual_tp + spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break

        if exit_price is None:
            # Trade expired without hitting SL or TP (timeout after 500 bars)
            exit_price = ohlc.iloc[min(entry_idx + 500, len(ohlc) - 1)]["close"]
            exit_time = ohlc.index[min(entry_idx + 500, len(ohlc) - 1)]
            exit_reason = "TIMEOUT"

        # Calculate P&L
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

    Args:
        ohlc: Entry timeframe OHLC data
        htf_ohlc: Higher timeframe data for bias
        config: Backtest configuration

    Returns:
        (list of trade dicts, BacktestMetrics)
    """
    if config is None:
        config = BacktestConfig()

    logger.info(f"Running backtest on {config.pair} ({len(ohlc)} candles)...")

    # Generate signals
    signals = generate_signals(
        ohlc,
        htf_ohlc=htf_ohlc,
        pair=config.pair,
        swing_length=config.swing_length,
        confluence_threshold=config.confluence_threshold,
        min_rr=config.min_rr,
        sl_buffer_pips=config.sl_buffer_pips,
        target_kill_zones=config.target_kill_zones,
    )

    if not signals:
        logger.warning("No signals generated.")
        return [], calculate_metrics([])

    # Simulate trades
    trades = simulate_trades(signals, ohlc, config)

    # Calculate metrics
    metrics = calculate_metrics(trades, pip_size=config.pip_size)

    logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                f"win rate={metrics.win_rate:.1%}, expectancy={metrics.expectancy:+.3f}R")

    return trades, metrics
