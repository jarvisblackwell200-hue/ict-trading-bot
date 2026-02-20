"""LiveTradingSession — main orchestrator tying broker, signals, risk, positions."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from ict_bot.risk import RiskConfig, RiskManager
from ict_bot.signals import generate_signals

from .broker import IBKRBroker
from .config import LiveConfig, pip_size_for
from .news_filter import NewsFilter
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

ET_TZ = ZoneInfo("America/New_York")


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
        self.news_filter = NewsFilter(
            blackout_minutes=config.news_blackout_minutes,
            close_before_news=config.news_close_before_events,
        ) if config.news_filter_enabled else None
        self._running = False
        self._last_daily_reset: datetime | None = None
        self._htf_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._started_at: datetime | None = None
        self._last_trade_time: dict[str, datetime] = {}  # per-pair rate limiter (#14)
        self._quote_currency_rates: dict[str, float] = {}  # pair -> quote/USD rate (#1)

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

        # Cancel stale orders FIRST, then reconcile will re-place SL/TP (#3)
        await self.broker.cancel_all_orders()

        # Sync balance from IB
        balance = await self.broker.get_account_balance()
        self.risk_manager._balance = balance
        self.risk_manager._peak_balance = max(self.risk_manager._peak_balance, balance)
        logger.info("Account balance: $%.2f", balance)

        # Reconcile positions from last run (re-places SL/TP orders)
        await self.position_manager.reconcile_on_startup()
        for pair, pos in self.position_manager.positions.items():
            self.risk_manager.register_open_position(
                pair, pos.risk_pips * pip_size_for(pair) * pos.units
            )

        # Subscribe to bars for each pair
        for pair in self.config.pairs:
            await self.broker.subscribe_bars(pair, self._on_bar_update)

        # Fetch quote currency rates for position sizing
        await self._refresh_quote_rates()

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

    async def _refresh_quote_rates(self) -> None:
        """Fetch current exchange rates for position sizing of non-USD-quoted pairs.

        For USD-base pairs (USD/JPY, USD/CAD): stores the pair's own rate (e.g. 150.0).
        For cross pairs (EUR/GBP): stores the quote currency vs USD rate (GBP/USD).
        """
        for pair in self.config.pairs:
            try:
                if pair.endswith("_USD"):
                    continue  # no conversion needed
                elif pair.startswith("USD_"):
                    # USD/JPY, USD/CAD — the pair's own mid-price IS the rate
                    bars = self.broker.get_live_bars(pair)
                    if bars is not None and not bars.empty:
                        rate = bars["close"].iloc[-1]
                        self._quote_currency_rates[pair] = rate
                else:
                    # Cross pair (EUR/GBP) — need quote_currency/USD rate
                    # EUR/GBP quote is GBP, so we need GBP/USD
                    quote_ccy = pair.split("_")[1]  # "GBP"
                    usd_pair = f"{quote_ccy}_USD"  # "GBP_USD"
                    bars = self.broker.get_live_bars(usd_pair)
                    if bars is not None and not bars.empty:
                        rate = bars["close"].iloc[-1]
                        self._quote_currency_rates[pair] = rate
            except Exception as exc:
                logger.debug("Could not get quote rate for %s: %s", pair, exc)

    # ── Signal Processing ──────────────────────────────────────────

    async def _process_signal(self, signal) -> None:
        """Risk check → validation → position sizing → place bracket order.

        Safety gates (in order):
        1. Duplicate pair check (internal state)
        2. Max positions (internal state)
        3. Rate limit (max 1 new trade per bar interval)
        4. News blackout check
        5. SL/TP direction validation
        6. SL distance check
        7. Verify against actual IB positions (not just internal state)
        8. Refresh balance from IB before sizing
        9. Risk manager gate
        10. Position size + max notional cap
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

        # Gate 3: Per-pair rate limit — max 1 new trade per bar interval per pair
        now = datetime.now(timezone.utc)
        last_trade_for_pair = self._last_trade_time.get(pair)
        if last_trade_for_pair is not None:
            bar_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400}
            min_interval = bar_seconds.get(self.config.timeframe, 900)
            elapsed = (now - last_trade_for_pair).total_seconds()
            if elapsed < min_interval:
                logger.info(
                    "Rate limit: %s last trade %.0fs ago (min %ds), skipping",
                    pair, elapsed, min_interval,
                )
                return

        # Gate 4: News blackout check
        if self.news_filter is not None and self.news_filter.is_pair_blocked(pair):
            logger.info("News blackout: %s blocked by high-impact event", pair)
            return

        # Gate 5: Validate SL/TP direction
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

        # Gate 6: Check SL distance
        pip_size = pip_size_for(pair)
        sl_pips = abs(signal.entry_price - signal.stop_loss) / pip_size
        if sl_pips > self.config.max_sl_pips:
            logger.info("SL too wide for %s (%.1f pips > %.1f max)", pair, sl_pips, self.config.max_sl_pips)
            return
        if sl_pips < 1:
            logger.warning("REJECTED %s: SL too tight (%.1f pips)", pair, sl_pips)
            return

        # Gate 7: Verify against actual IB positions (prevent duplicates from multiple instances)
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

        # Gate 8: Refresh balance from IB before sizing
        try:
            balance = await self.broker.get_account_balance()
            self.risk_manager._balance = balance
        except Exception as exc:
            logger.warning("Could not refresh balance: %s — using cached", exc)

        # Gate 9: Risk manager gate
        decision = self.risk_manager.evaluate_signal(signal, pip_size)
        if not decision.approved:
            logger.info("Signal rejected for %s: %s", pair, decision.reason)
            return

        # Gate 10: Calculate units + max notional cap
        units = self._calculate_units(decision.position_size, pip_size, pair)
        if units < 1:
            logger.info("Position size too small for %s (%.2f units)", pair, units)
            return

        # Notional value cap: max 10x account balance per position (normalized to USD)
        # For USD-quoted pairs: notional_usd = units * price
        # For USD-base pairs (USD/JPY): each unit IS 1 USD, so notional_usd = units
        # For crosses: notional_usd = units * base_ccy/USD_rate
        if pair.startswith("USD_"):
            notional_usd = units  # each unit is 1 USD
        else:
            notional_usd = units * signal.entry_price
            if not pair.endswith("_USD"):
                # Cross pair — entry_price is in quote ccy, need base/USD
                base_ccy = pair.split("_")[0]
                base_pair = f"{base_ccy}_USD"
                base_rate = self._quote_currency_rates.get(base_pair)
                if base_rate:
                    notional_usd = units * base_rate
        max_notional = self.risk_manager._balance * 10
        if notional_usd > max_notional:
            if pair.startswith("USD_"):
                capped_units = round(max_notional)
            else:
                capped_units = round(max_notional / signal.entry_price)
            logger.warning(
                "Notional cap: %s %.0f units ($%.0f) exceeds 10x balance ($%.0f) — "
                "capping to %.0f units",
                pair, units, notional_usd, max_notional, capped_units,
            )
            units = capped_units
            if units < 1:
                return

        logger.info(
            "Signal approved for %s: %s entry=%.5f SL=%.5f TP=%.5f units=%.0f "
            "confluence=%d risk=$%.2f",
            pair, signal.direction, signal.entry_price, signal.stop_loss,
            signal.take_profit, units, signal.confluence_score, decision.risk_amount,
        )

        # Open position (market + SL + TP)
        pos = await self.position_manager.open_position(signal, units)
        self.risk_manager.register_open_position(pair, decision.risk_amount)
        self._last_trade_time[pair] = datetime.now(timezone.utc)

    def _calculate_units(self, pip_value: float, pip_size: float, pair: str) -> float:
        """Convert pip_value ($/pip) to forex units.

        pip_value is the dollar amount we want to risk per pip.
        The formula depends on which currency is the quote:

        - USD-quoted (EUR/USD, GBP/USD, AUD/USD, NZD/USD):
          1 unit moves pip_size USD per pip → units = pip_value / pip_size

        - USD-base (USD/JPY, USD/CAD):
          1 unit moves pip_size in quote currency per pip.
          In USD: pip_size / quote_rate → units = pip_value * quote_rate / pip_size

        - Cross (EUR/GBP):
          1 unit moves pip_size GBP per pip.
          In USD: pip_size * GBPUSD → units = pip_value / (pip_size * GBPUSD)

        We use cached quote rates; if unavailable, fall back to the simple formula.
        """
        quote_rate = self._quote_currency_rates.get(pair)

        if pair.endswith("_USD"):
            # Quote currency is USD — simple case
            units = pip_value / pip_size
        elif pair.startswith("USD_"):
            # USD is base, quote is JPY/CAD — need quote currency rate
            if quote_rate is not None and quote_rate > 0:
                # quote_rate = e.g. 150.0 for USD/JPY, 1.36 for USD/CAD
                units = pip_value * quote_rate / pip_size
            else:
                # Fallback: estimate from pair name
                logger.warning("No quote rate for %s, using simple formula", pair)
                units = pip_value / pip_size
        else:
            # Cross pair (e.g. EUR/GBP) — quote is GBP, need GBP/USD rate
            if quote_rate is not None and quote_rate > 0:
                # quote_rate = GBP/USD rate (e.g. 1.27)
                units = pip_value / (pip_size * quote_rate)
            else:
                logger.warning("No quote rate for %s, using simple formula", pair)
                units = pip_value / pip_size

        return round(units)

    # ── Heartbeat ──────────────────────────────────────────────────

    async def _heartbeat(self) -> None:
        """Periodic: reconnect, manage positions, daily reset."""
        # Reconnect if needed (re-subscribe bars after reconnect)
        if not self.broker.is_connected():
            logger.info("Reconnecting to IB Gateway...")
            await self.broker.connect()
            # Re-subscribe to bars — subscriptions are lost on disconnect (#6)
            for pair in self.config.pairs:
                try:
                    await self.broker.subscribe_bars(pair, self._on_bar_update)
                except Exception as exc:
                    logger.error("Failed to re-subscribe bars for %s: %s", pair, exc)

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

        # Close positions before major news (if enabled)
        if self.news_filter is not None:
            for pair, event in self.news_filter.get_pairs_to_close_before_news(
                list(self.position_manager.positions.keys())
            ):
                record = await self.position_manager.close_position(pair, "pre_news_close")
                if record:
                    self.risk_manager.record_trade_result(record.get("pnl_amount", 0), pair)
                    logger.info(
                        "Pre-news close: %s closed before '%s' (%s)",
                        pair, event.title, event.country,
                    )

        # Daily reset at midnight ET (DST-aware)
        await self._check_daily_reset()

        # Refresh quote currency rates periodically
        await self._refresh_quote_rates()

    async def _check_daily_reset(self) -> None:
        """Reset risk manager daily at midnight ET (DST-aware)."""
        now_et = datetime.now(ET_TZ)
        today_et = now_et.date()

        if self._last_daily_reset is not None:
            last_et = self._last_daily_reset.astimezone(ET_TZ).date()
            if last_et == today_et:
                return

        # Reset once per ET day (after midnight ET)
        if self._last_daily_reset is None or self._last_daily_reset.astimezone(ET_TZ).date() < today_et:
            self.risk_manager.reset_daily()
            self._last_daily_reset = datetime.now(timezone.utc)
            # Refresh balance
            balance = await self.broker.get_account_balance()
            self.risk_manager._balance = balance
            logger.info(
                "Daily reset — balance: $%.2f, positions: %d (ET date: %s)",
                balance, len(self.position_manager.positions), today_et,
            )
            # Clear HTF cache to force fresh daily bars
            self._htf_cache.clear()
