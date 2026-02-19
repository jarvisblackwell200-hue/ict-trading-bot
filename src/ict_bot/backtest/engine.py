"""
Backtesting engine for ICT signals (v2 — with trade management).

Changes from v1:
  - Break-even stop: move SL to entry after price reaches 1R profit
  - Partial TP: close half at 1R, let rest run to full TP
  - Max SL cap: reject signals with unreasonably wide stops
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..risk import RiskConfig, RiskManager
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
    use_breakeven: bool = True          # Move SL to entry after be_threshold_r
    be_threshold_r: float = 1.5          # R-multiple price must reach before BE triggers
    pullback_window: int = 20
    use_liquidity_targets: bool = True
    use_premium_discount: bool = True
    use_partial_tp: bool = True
    partial_tp_r: float = 1.0           # Take partial at this R-multiple
    partial_tp_pct: float = 0.5         # Close this fraction at partial TP
    skip_days: list[int] = field(default_factory=lambda: [0, 4])  # Mon/Fri
    use_displacement: bool = True
    fvg_lookback: int = 16              # FVG search window (bars before break)
    use_confluence_sizing: bool = True
    confluence_size_base: float = 0.75  # Multiplier at threshold score
    confluence_size_step: float = 0.25  # Increase per score point above threshold
    compute_ob: bool = True             # Compute order blocks (slow on large datasets)
    # New ICT concept flags
    use_sweep_filter: bool = False      # Require liquidity sweep before BOS
    use_ifvg: bool = False              # Use invalidated FVGs as entry zones
    use_breaker_blocks: bool = False    # Use breaker blocks as entry zones
    use_ce_entry: bool = False          # Use CE (50% midpoint) entry instead of 25% edge
    # Trailing stop fields (for adaptive strategy)
    use_trailing_stop: bool = False     # Enable ATR trailing stop
    atr_trail_mult: float = 2.5        # Trailing distance in ATR multiples
    trailing_activation_r: float = 1.5  # Activate trailing after this R-profit
    max_bars: int = 200                 # Timeout (default matches original; 500 for adaptive)

    @property
    def pip_size(self) -> float:
        return 0.01 if "JPY" in self.pair else 0.0001


def simulate_trades(
    signals: list[Signal],
    ohlc: pd.DataFrame,
    config: BacktestConfig,
    risk_manager: RiskManager | None = None,
) -> list[dict]:
    """
    Simulate trade execution with realistic trade management.

    For each signal:
      1. Enter at signal price (adjusted for spread/slippage)
      2. Walk forward checking SL and TP
      3. If break-even enabled: move SL to entry after 1R profit reached

    Parameters
    ----------
    risk_manager : RiskManager, optional
        When provided, gates entry via circuit breakers and exposure limits.
        When None, a default RiskManager is created from the BacktestConfig.
    """
    pip_size = config.pip_size
    spread = config.spread_pips * pip_size
    slippage = config.slippage_pips * pip_size

    if risk_manager is None:
        risk_manager = RiskManager(RiskConfig(
            max_risk_per_trade=config.risk_per_trade,
            starting_balance=config.starting_balance,
        ))

    trades = []
    last_trade_date = None

    for signal in signals:
        # Daily reset on date change
        signal_date = signal.timestamp.date()
        if last_trade_date is not None and signal_date != last_trade_date:
            risk_manager.reset_daily()
        last_trade_date = signal_date

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

        # Risk calculation via risk manager
        if signal.direction == "long":
            risk_pips = (actual_entry - actual_sl) / pip_size
        else:
            risk_pips = (actual_sl - actual_entry) / pip_size

        if risk_pips <= 0:
            continue

        # Reject trades with SL too wide
        if risk_pips > config.max_sl_pips:
            continue

        decision = risk_manager.evaluate_signal(signal, pip_size)
        if not decision.approved:
            logger.debug(f"Signal rejected: {decision.reason}")
            continue

        pip_value = decision.position_size
        risk_amount = decision.risk_amount

        # Dynamic sizing: scale by confluence score
        if config.use_confluence_sizing:
            size_mult = config.confluence_size_base + (
                signal.confluence_score - config.confluence_threshold
            ) * config.confluence_size_step
            size_mult = max(0.5, min(size_mult, 2.0))
            pip_value *= size_mult
            risk_amount *= size_mult

        risk_manager.register_open_position(signal.pair, risk_amount)

        # Break-even trigger level
        one_r_distance = risk_pips * pip_size
        be_distance = one_r_distance * config.be_threshold_r
        if signal.direction == "long":
            be_trigger_level = actual_entry + be_distance
            partial_level = actual_entry + one_r_distance * config.partial_tp_r
        else:
            be_trigger_level = actual_entry - be_distance
            partial_level = actual_entry - one_r_distance * config.partial_tp_r

        # Walk forward
        exit_price = None
        exit_time = None
        exit_reason = None
        current_sl = actual_sl
        be_triggered = False
        partial_taken = False

        # Trailing stop state
        atr_value = signal.meta.get("atr", 0) if config.use_trailing_stop else 0
        best_price = actual_entry

        for j in range(entry_idx + 1, min(entry_idx + config.max_bars, len(ohlc))):
            candle = ohlc.iloc[j]
            high = candle["high"]
            low = candle["low"]

            if signal.direction == "long":
                best_price = max(best_price, high)

                # Partial TP check: close half at 1R
                if config.use_partial_tp and not partial_taken and high >= partial_level:
                    partial_taken = True

                # Break-even check: move SL to entry after price reaches threshold
                if config.use_breakeven and not be_triggered and high >= be_trigger_level:
                    current_sl = actual_entry + spread  # BE + spread
                    be_triggered = True

                # Phase 3: ATR trailing stop (after trailing_activation_r profit)
                if config.use_trailing_stop and atr_value > 0:
                    profit_r = (best_price - actual_entry) / (risk_pips * pip_size)
                    if profit_r >= config.trailing_activation_r:
                        trail_sl = best_price - config.atr_trail_mult * atr_value
                        current_sl = max(current_sl, trail_sl)

                # Check SL
                if low <= current_sl:
                    exit_price = current_sl - slippage
                    exit_time = ohlc.index[j]
                    if config.use_trailing_stop and current_sl > actual_entry + spread:
                        exit_reason = "TRAIL"
                    elif be_triggered and current_sl > actual_sl:
                        exit_reason = "BE"
                    else:
                        exit_reason = "SL"
                    break
                # Check TP
                if high >= actual_tp:
                    exit_price = actual_tp - spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break
            else:
                best_price = min(best_price, low)

                # Partial TP check
                if config.use_partial_tp and not partial_taken and low <= partial_level:
                    partial_taken = True

                # Break-even check
                if config.use_breakeven and not be_triggered and low <= be_trigger_level:
                    current_sl = actual_entry - spread
                    be_triggered = True

                # Phase 3: ATR trailing stop (after trailing_activation_r profit)
                if config.use_trailing_stop and atr_value > 0:
                    profit_r = (actual_entry - best_price) / (risk_pips * pip_size)
                    if profit_r >= config.trailing_activation_r:
                        trail_sl = best_price + config.atr_trail_mult * atr_value
                        current_sl = min(current_sl, trail_sl)

                # Check SL
                if high >= current_sl:
                    exit_price = current_sl + slippage
                    exit_time = ohlc.index[j]
                    if config.use_trailing_stop and current_sl < actual_entry - spread:
                        exit_reason = "TRAIL"
                    elif be_triggered and current_sl < actual_sl:
                        exit_reason = "BE"
                    else:
                        exit_reason = "SL"
                    break
                # Check TP
                if low <= actual_tp:
                    exit_price = actual_tp + spread / 2
                    exit_time = ohlc.index[j]
                    exit_reason = "TP"
                    break

        if exit_price is None:
            timeout_idx = min(entry_idx + config.max_bars, len(ohlc) - 1)
            exit_price = ohlc.iloc[timeout_idx]["close"]
            exit_time = ohlc.index[timeout_idx]
            exit_reason = "TIMEOUT"

        # P&L — composite if partial was taken
        if signal.direction == "long":
            remaining_pnl_pips = (exit_price - actual_entry) / pip_size
        else:
            remaining_pnl_pips = (actual_entry - exit_price) / pip_size

        if partial_taken and config.use_partial_tp:
            partial_pnl_pips = config.partial_tp_r * risk_pips
            pct = config.partial_tp_pct
            pnl_pips = pct * partial_pnl_pips + (1 - pct) * remaining_pnl_pips
        else:
            pnl_pips = remaining_pnl_pips

        pnl_amount = pnl_pips * pip_value
        risk_manager.record_trade_result(pnl_amount, signal.pair)

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
            "balance": round(risk_manager.balance, 2),
            "exit_reason": exit_reason,
            "be_triggered": be_triggered,
            "partial_taken": partial_taken,
            "confluence_score": signal.confluence_score,
            "confluences": signal.confluences,
            "kill_zone": signal.kill_zone,
            "tp_method": signal.meta.get("tp_method", "rr_based"),
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
        logger.warning("No signals generated.")
        return [], calculate_metrics([])

    trades = simulate_trades(signals, ohlc, config)
    metrics = calculate_metrics(trades, pip_size=config.pip_size)

    logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                f"win rate={metrics.win_rate:.1%}, expectancy={metrics.expectancy:+.3f}R")

    return trades, metrics
