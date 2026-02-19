"""LiveTradingSession — main orchestrator tying broker, signals, risk, positions."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone

import pandas as pd

from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals import generate_signals

from .broker import IBKRBroker
from .config import LiveConfig, pip_size_for
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

# Midnight ET in UTC (ET = UTC-5, so midnight ET = 05:00 UTC)
DAILY_RESET_UTC = time(5, 0)


class LiveTradingSession:
    """Orchestrates live trading: connects, subscribes, detects signals, manages positions."""

    def __init__(self, config: LiveConfig) -> None:
        self.config = config
        self.broker = IBKRBroker(config)
        self.risk_manager = RiskManager(
            RiskConfig(
                max_risk_per_trade=config.risk_per_trade,
                starting_balance=config.starting_balance,
            )
        )
        self.position_manager = PositionManager(self.broker, config)
        self._running = False
        self._last_daily_reset: datetime | None = None
        self._htf_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._started_at: datetime | None = None
        self._last_trade_time: datetime | None = None  # rate limiter

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect, reconcile, subscribe to bars, run event loop."""
        logger.info("Starting live trading session...")
        logger.info(
            "Config: %s pairs=%s port=%d risk=%.1f%% sw=%d thresh=%d",
            self.config.timeframe, self.config.pairs, self.config.ib_port,
            self.config.risk_per_trade * 100,
            self.config.swing_length, self.config.confluence_threshold,
        )

        await self.broker.connect()

        # Cancel any stale orders from previous runs
        await self.broker.cancel_all_orders()

        # Sync balance from IB
        balance = await self.broker.get_account_balance()
        self.risk_manager._balance = balance
        self.risk_manager._peak_balance = max(self.risk_manager._peak_balance, balance)
        logger.info("Account balance: $%.2f", balance)

        # Reconcile positions from last run
        await self.position_manager.reconcile_on_startup()
        for pair, pos in self.position_manager.positions.items():
            self.risk_manager.register_open_position(
                pair, pos.risk_pips * pip_size_for(pair) * pos.units
            )

        # Subscribe to bars for each pair
        for pair in self.config.pairs:
            await self.broker.subscribe_bars(pair, self._on_bar_update)

        self._running = True
        self._started_at = datetime.now(timezone.utc)
        logger.info("Live session running on %s — listening for signals...", self.config.timeframe)
        await self._run_loop()

    async def _run_loop(self) -> None:
        """Main loop: heartbeat every N seconds."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval)
            try:
                await self._heartbeat()
            except Exception as exc:
                logger.error("Heartbeat error: %s", exc, exc_info=True)

    async def stop(self) -> None:
        """Graceful shutdown — does NOT close open positions (SL/TP remain on IB)."""
        logger.info("Stopping live session...")
        self._running = False
        self.position_manager.save_state(self.config.state_file)
        await self.broker.disconnect()
        logger.info("Live session stopped. Open positions retain SL/TP on IB servers.")

    # ── Bar Update Callback ────────────────────────────────────────

    async def _on_bar_update(self, pair: str, bars, has_new_bar: bool) -> None:
        """Called when bar data updates. Run signal detection on new bar."""
        if not has_new_bar:
            return

        logger.debug("New %s bar for %s (total bars: %d)", self.config.timeframe, pair, len(bars))

        try:
            h1_df = self.broker._bars_to_dataframe(bars)
            if len(h1_df) < 100:
                logger.warning("Not enough H1 bars for %s (%d), skipping", pair, len(h1_df))
                return

            # Get HTF (daily) bars for bias alignment
            htf_df = await self._get_htf_bars(pair)

            signals = generate_signals(
                ohlc=h1_df,
                htf_ohlc=htf_df,
                pair=pair,
                swing_length=self.config.swing_length,
                confluence_threshold=self.config.confluence_threshold,
                min_rr=self.config.min_rr,
                sl_buffer_pips=self.config.sl_buffer_pips,
                skip_days=self.config.skip_days,
                use_displacement=self.config.use_displacement,
                fvg_lookback=self.config.fvg_lookback,
                pullback_window=self.config.pullback_window,
                compute_ob=self.config.compute_ob,
            )

            if signals:
                # Only act on signals from bars that completed after bot started
                latest = signals[-1]
                if self._started_at and latest.timestamp < self._started_at:
                    logger.debug("Skipping historical signal for %s (pre-startup)", pair)
                    return
                logger.info("Generated %d signal(s) for %s, processing latest", len(signals), pair)
                await self._process_signal(latest)

        except Exception as exc:
            logger.error("Error processing bars for %s: %s", pair, exc, exc_info=True)

    async def _get_htf_bars(self, pair: str) -> pd.DataFrame:
        """Get daily bars, cached for 1 hour."""
        now = datetime.now(timezone.utc)
        if pair in self._htf_cache:
            cached_time, cached_df = self._htf_cache[pair]
            if (now - cached_time).total_seconds() < 3600:
                return cached_df

        htf_df = await self.broker.request_historical_bars(pair, "180 D", "1 day")
        self._htf_cache[pair] = (now, htf_df)
        return htf_df

    # ── Signal Processing ──────────────────────────────────────────

    async def _process_signal(self, signal) -> None:
        """Risk check → validation → position sizing → place bracket order.

        Safety gates (in order):
        1. Duplicate pair check (internal state)
        2. Max positions (internal state)
        3. Rate limit (max 1 new trade per bar interval)
        4. SL/TP direction validation
        5. SL distance check
        6. Verify against actual IB positions (not just internal state)
        7. Refresh balance from IB before sizing
        8. Risk manager gate
        9. Position size + max notional cap
        """
        pair = signal.pair

        # Gate 1: Skip if already in a trade for this pair
        if pair in self.position_manager.positions:
            logger.debug("Already in position for %s, skipping signal", pair)
            return

        # Gate 2: Check max positions
        if len(self.position_manager.positions) >= self.config.max_positions:
            logger.info("Max positions (%d) reached, skipping signal", self.config.max_positions)
            return

        # Gate 3: Rate limit — max 1 new trade per bar interval
        now = datetime.now(timezone.utc)
        if self._last_trade_time is not None:
            bar_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}
            min_interval = bar_seconds.get(self.config.timeframe, 900)
            elapsed = (now - self._last_trade_time).total_seconds()
            if elapsed < min_interval:
                logger.info(
                    "Rate limit: last trade %.0fs ago (min %ds), skipping %s",
                    elapsed, min_interval, pair,
                )
                return

        # Gate 4: Validate SL/TP direction
        if signal.direction == "long":
            if signal.stop_loss >= signal.entry_price:
                logger.warning("REJECTED %s: long SL (%.5f) >= entry (%.5f)", pair, signal.stop_loss, signal.entry_price)
                return
            if signal.take_profit <= signal.entry_price:
                logger.warning("REJECTED %s: long TP (%.5f) <= entry (%.5f)", pair, signal.take_profit, signal.entry_price)
                return
        elif signal.direction == "short":
            if signal.stop_loss <= signal.entry_price:
                logger.warning("REJECTED %s: short SL (%.5f) <= entry (%.5f)", pair, signal.stop_loss, signal.entry_price)
                return
            if signal.take_profit >= signal.entry_price:
                logger.warning("REJECTED %s: short TP (%.5f) >= entry (%.5f)", pair, signal.take_profit, signal.entry_price)
                return

        # Gate 5: Check SL distance
        pip_size = pip_size_for(pair)
        sl_pips = abs(signal.entry_price - signal.stop_loss) / pip_size
        if sl_pips > self.config.max_sl_pips:
            logger.info("SL too wide for %s (%.1f pips > %.1f max)", pair, sl_pips, self.config.max_sl_pips)
            return
        if sl_pips < 1:
            logger.warning("REJECTED %s: SL too tight (%.1f pips)", pair, sl_pips)
            return

        # Gate 6: Verify against actual IB positions (prevent duplicates from multiple instances)
        if not self.config.dry_run:
            try:
                ib_positions = await self.broker.get_open_positions()
                if pair in ib_positions and ib_positions[pair] != 0:
                    logger.warning(
                        "REJECTED %s: IB already has position (%.0f units) — state mismatch",
                        pair, ib_positions[pair],
                    )
                    return
            except Exception as exc:
                logger.warning("Could not verify IB positions for %s: %s", pair, exc)

        # Gate 7: Refresh balance from IB before sizing
        try:
            balance = await self.broker.get_account_balance()
            self.risk_manager._balance = balance
        except Exception as exc:
            logger.warning("Could not refresh balance: %s — using cached", exc)

        # Gate 8: Risk manager gate
        decision = self.risk_manager.evaluate_signal(signal, pip_size)
        if not decision.approved:
            logger.info("Signal rejected for %s: %s", pair, decision.reason)
            return

        # Gate 9: Calculate units + max notional cap
        units = self._calculate_units(decision.position_size, pip_size, pair)
        if units < 1:
            logger.info("Position size too small for %s (%.2f units)", pair, units)
            return

        # Notional value cap: max 10x account balance per position
        notional = units * signal.entry_price
        max_notional = self.risk_manager._balance * 10
        if notional > max_notional:
            capped_units = round(max_notional / signal.entry_price)
            logger.warning(
                "Notional cap: %s %.0f units ($%.0f) exceeds 10x balance ($%.0f) — "
                "capping to %.0f units",
                pair, units, notional, max_notional, capped_units,
            )
            units = capped_units
            if units < 1:
                return

        logger.info(
            "Signal approved for %s: %s entry=%.5f SL=%.5f TP=%.5f units=%.0f "
            "confluence=%d risk=$%.2f notional=$%.0f",
            pair, signal.direction, signal.entry_price, signal.stop_loss,
            signal.take_profit, units, signal.confluence_score, decision.risk_amount,
            units * signal.entry_price,
        )

        # Open position (market + SL + TP)
        pos = await self.position_manager.open_position(signal, units)
        self.risk_manager.register_open_position(pair, decision.risk_amount)
        self._last_trade_time = datetime.now(timezone.utc)

    def _calculate_units(self, pip_value: float, pip_size: float, pair: str) -> float:
        """Convert pip_value ($/pip) to forex units.

        For most pairs: 1 standard lot (100,000 units) = $10/pip.
        pip_value / 10 * 100,000 = pip_value * 10,000
        For JPY pairs: 1 lot = $10/pip (same calculation).
        """
        # pip_value is dollars per pip
        # For standard forex: units = pip_value / pip_size_in_dollars_per_unit
        # Simplified: units = pip_value * (1 / pip_size) for most pairs
        # But this depends on the quote currency. For USD-quoted pairs:
        #   1 unit move of pip_size = pip_size USD → pip_value_per_unit = pip_size
        #   units = pip_value / pip_size
        # This is a simplification; IB handles the exact conversion.
        units = pip_value / pip_size
        return round(units)

    # ── Heartbeat ──────────────────────────────────────────────────

    async def _heartbeat(self) -> None:
        """Periodic: reconnect, manage positions, daily reset."""
        # Reconnect if needed
        if not self.broker.is_connected():
            logger.info("Reconnecting to IB Gateway...")
            await self.broker.connect()

        # Manage positions (OCO emulation, BE)
        closed = await self.position_manager.check_and_manage()
        for trade_record in closed:
            pnl = trade_record.get("pnl_amount", 0.0)
            pair = trade_record.get("pair", "")
            self.risk_manager.record_trade_result(pnl, pair)
            logger.info(
                "Trade closed: %s %s PnL=%.2f (%.1f pips) reason=%s",
                pair, trade_record.get("direction"),
                pnl, trade_record.get("pnl_pips", 0),
                trade_record.get("exit_reason"),
            )

        # Daily reset at midnight ET (05:00 UTC)
        await self._check_daily_reset()

    async def _check_daily_reset(self) -> None:
        """Reset risk manager daily at midnight ET."""
        now = datetime.now(timezone.utc)
        if self._last_daily_reset is not None:
            # Already reset today?
            if self._last_daily_reset.date() == now.date():
                return

        if now.time() >= DAILY_RESET_UTC:
            if self._last_daily_reset is None or self._last_daily_reset.date() < now.date():
                self.risk_manager.reset_daily()
                self._last_daily_reset = now
                # Refresh balance
                balance = await self.broker.get_account_balance()
                self.risk_manager._balance = balance
                logger.info(
                    "Daily reset — balance: $%.2f, positions: %d",
                    balance, len(self.position_manager.positions),
                )
                # Clear HTF cache to force fresh daily bars
                self._htf_cache.clear()
