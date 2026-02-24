"""PositionManager — track live positions, OCA order groups, break-even moves."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ib_insync import Trade

from .broker import IBKRBroker
from .config import LiveConfig, pip_size_for

logger = logging.getLogger(__name__)


@dataclass
class LivePosition:
    """A single live position with SL/TP orders."""

    pair: str
    direction: str               # "long" or "short"
    entry_price: float
    units: float
    stop_loss: float
    take_profit: float
    entry_time: str              # ISO format for JSON serialization
    risk_pips: float
    confluence_score: int
    be_triggered: bool = False
    partial_taken: bool = False

    # IB order references (not serialized)
    sl_order: Trade | None = field(default=None, repr=False)
    tp_order: Trade | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict (excludes IB order objects)."""
        return {
            "pair": self.pair,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "units": self.units,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_time": self.entry_time,
            "risk_pips": self.risk_pips,
            "confluence_score": self.confluence_score,
            "be_triggered": self.be_triggered,
            "partial_taken": self.partial_taken,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LivePosition:
        return cls(
            pair=d["pair"],
            direction=d["direction"],
            entry_price=d["entry_price"],
            units=d["units"],
            stop_loss=d["stop_loss"],
            take_profit=d["take_profit"],
            entry_time=d["entry_time"],
            risk_pips=d["risk_pips"],
            confluence_score=d["confluence_score"],
            be_triggered=d.get("be_triggered", False),
            partial_taken=d.get("partial_taken", False),
        )


class PositionManager:
    """Tracks live positions, emulates OCO, manages BE moves."""

    def __init__(self, broker: IBKRBroker, config: LiveConfig) -> None:
        self.broker = broker
        self.config = config
        self.positions: dict[str, LivePosition] = {}  # pair -> position

    # ── Open / Close ───────────────────────────────────────────────

    async def open_position(self, signal, units: float) -> LivePosition:
        """Open a new position: market entry + OCA-grouped SL/TP.

        Waits for market fill before placing SL/TP (#7).
        Uses actual fill price to adjust SL/TP levels (#8).
        Uses OCA group so IB cancels the counterpart automatically (#2).

        Parameters
        ----------
        signal : Signal
            Must have .pair, .direction, .entry_price, .stop_loss,
            .take_profit, .confluence_score attributes.
        units : float
            Position size in base currency units.
        """
        pair = signal.pair
        direction = signal.direction
        pip_size = pip_size_for(pair)

        actual_entry = signal.entry_price
        sl_price = signal.stop_loss
        tp_price = signal.take_profit
        sl_trade = None
        tp_trade = None

        if not self.config.dry_run:
            # Place market entry and wait for fill (#7)
            try:
                entry_trade = await self.broker.place_market_order(pair, direction, units)
            except Exception as exc:
                logger.error("Failed to place entry order for %s: %s", pair, exc)
                return None
            try:
                for _ in range(30):  # wait up to 15s for fill
                    await asyncio.sleep(0.5)
                    if entry_trade.orderStatus.status == "Filled":
                        break
                if entry_trade.orderStatus.status != "Filled":
                    logger.error(
                        "Market order not filled for %s (status=%s) — aborting",
                        pair, entry_trade.orderStatus.status,
                    )
                    try:
                        await self.broker.cancel_order(entry_trade)
                    except Exception:
                        pass
                    return None
            except Exception as exc:
                logger.error("Error waiting for fill on %s: %s", pair, exc)
                return None

            # Use actual fill price and adjust SL/TP (#8)
            actual_entry = entry_trade.orderStatus.avgFillPrice
            slippage = actual_entry - signal.entry_price
            sl_price = round(signal.stop_loss + slippage, 6)
            tp_price = round(signal.take_profit + slippage, 6)

            # Place SL/TP with OCA group so IB cancels the other on fill (#2)
            sl_dir = "short" if direction == "long" else "long"
            oca_group = f"ict_{pair}_{int(datetime.now(timezone.utc).timestamp())}"
            try:
                sl_trade = await self.broker.place_stop_order(
                    pair, sl_dir, units, sl_price, oca_group=oca_group,
                )
                tp_trade = await self.broker.place_limit_order(
                    pair, sl_dir, units, tp_price, oca_group=oca_group,
                )
            except Exception as exc:
                logger.error(
                    "Failed to place SL/TP for %s: %s — CLOSING POSITION",
                    pair, exc,
                )
                close_dir = "short" if direction == "long" else "long"
                try:
                    await self.broker.place_market_order(pair, close_dir, units)
                except Exception as exc2:
                    logger.critical(
                        "FAILED to emergency-close %s after SL/TP failure: %s — "
                        "MANUAL INTERVENTION REQUIRED", pair, exc2,
                    )
                return None

            # Verify SL/TP orders were accepted by IB
            accepted_statuses = ("PreSubmitted", "Submitted", "Filled")
            for _ in range(10):  # wait up to 5s
                await asyncio.sleep(0.5)
                sl_ok = sl_trade and sl_trade.orderStatus.status in accepted_statuses
                tp_ok = tp_trade and tp_trade.orderStatus.status in accepted_statuses
                if sl_ok and tp_ok:
                    break

            sl_ok = sl_trade and sl_trade.orderStatus.status in accepted_statuses
            tp_ok = tp_trade and tp_trade.orderStatus.status in accepted_statuses

            if not sl_ok or not tp_ok:
                logger.error(
                    "SL/TP ORDER REJECTED for %s (SL=%s, TP=%s) — "
                    "CLOSING POSITION IMMEDIATELY to avoid unprotected exposure",
                    pair,
                    sl_trade.orderStatus.status if sl_trade else "None",
                    tp_trade.orderStatus.status if tp_trade else "None",
                )
                # Cancel whichever order did go through
                for t in (sl_trade, tp_trade):
                    if t and t.orderStatus.status in accepted_statuses:
                        try:
                            await self.broker.cancel_order(t)
                        except Exception:
                            pass
                # Close the position immediately
                close_dir = "short" if direction == "long" else "long"
                try:
                    await self.broker.place_market_order(pair, close_dir, units)
                    logger.info("Emergency close of %s after SL/TP rejection", pair)
                except Exception as exc:
                    logger.critical(
                        "FAILED to emergency-close %s after SL/TP rejection: %s — "
                        "MANUAL INTERVENTION REQUIRED",
                        pair, exc,
                    )
                return None

        prefix = "[DRY-RUN] " if self.config.dry_run else ""
        risk_pips = abs(actual_entry - sl_price) / pip_size

        pos = LivePosition(
            pair=pair,
            direction=direction,
            entry_price=actual_entry,
            units=units,
            stop_loss=sl_price,
            take_profit=tp_price,
            entry_time=datetime.now(timezone.utc).isoformat(),
            risk_pips=risk_pips,
            confluence_score=signal.confluence_score,
            sl_order=sl_trade,
            tp_order=tp_trade,
        )

        self.positions[pair] = pos
        self.save_state(self.config.state_file)
        logger.info(
            "%sOpened %s %s: entry=%.5f (signal=%.5f) SL=%.5f TP=%.5f units=%.0f",
            prefix, direction, pair, actual_entry, signal.entry_price,
            sl_price, tp_price, units,
        )
        return pos

    async def close_position(self, pair: str, reason: str) -> dict:
        """Close a position by market order and cancel SL/TP orders.

        Returns a trade record dict for the risk manager.
        """
        pos = self.positions.get(pair)
        if pos is None:
            logger.warning("No position to close for %s", pair)
            return {}

        if not self.config.dry_run:
            # Cancel outstanding SL/TP
            if pos.sl_order is not None:
                try:
                    await self.broker.cancel_order(pos.sl_order)
                except Exception as exc:
                    logger.warning("Failed to cancel SL for %s: %s", pair, exc)

            if pos.tp_order is not None:
                try:
                    await self.broker.cancel_order(pos.tp_order)
                except Exception as exc:
                    logger.warning("Failed to cancel TP for %s: %s", pair, exc)

            # Close via market order (opposite direction)
            close_dir = "short" if pos.direction == "long" else "long"
            try:
                await self.broker.place_market_order(pair, close_dir, pos.units)
            except Exception as exc:
                logger.critical(
                    "FAILED to close %s: %s — MANUAL INTERVENTION REQUIRED",
                    pair, exc,
                )
                return {}

        record = self._make_trade_record(pos, reason)
        del self.positions[pair]
        self.save_state(self.config.state_file)
        logger.info("Closed %s (%s): %s", pair, reason, record)
        return record

    # ── Heartbeat: OCO Emulation & BE ──────────────────────────────

    async def check_and_manage(self) -> list[dict]:
        """Check all positions for SL/TP fills, BE moves. Returns closed trade records."""
        closed: list[dict] = []
        pairs_to_close: list[tuple[str, str]] = []

        for pair, pos in list(self.positions.items()):
            if self.config.dry_run:
                # Dry-run: check SL/TP against live prices
                hit = self._check_sl_tp_price(pos)
                if hit:
                    pairs_to_close.append((pair, hit))
                    continue
            else:
                # Live: check IB order fill status
                # With OCA groups, IB should auto-cancel the counterpart.
                # But verify and handle double fills as safety net (#2).
                sl_filled = pos.sl_order is not None and self._is_filled(pos.sl_order)
                tp_filled = pos.tp_order is not None and self._is_filled(pos.tp_order)

                if sl_filled and tp_filled:
                    # Both filled (OCA race) — we're now accidentally reversed.
                    # Long: SL sold X + TP sold X = short X. Need to BUY X (pos.direction).
                    # Short: SL bought X + TP bought X = long X. Need to SELL X (pos.direction).
                    logger.warning(
                        "DOUBLE FILL for %s: both SL and TP filled! Closing extra position.",
                        pair,
                    )
                    try:
                        await self.broker.place_market_order(pair, pos.direction, pos.units)
                    except Exception as exc:
                        logger.error("Failed to close double-fill for %s: %s", pair, exc)
                    pairs_to_close.append((pair, "stop_loss"))
                    continue

                if sl_filled:
                    logger.info("SL filled for %s", pair)
                    if pos.tp_order is not None:
                        try:
                            await self.broker.cancel_order(pos.tp_order)
                        except Exception:
                            pass
                    pairs_to_close.append((pair, "stop_loss"))
                    continue

                if tp_filled:
                    logger.info("TP filled for %s", pair)
                    if pos.sl_order is not None:
                        try:
                            await self.broker.cancel_order(pos.sl_order)
                        except Exception:
                            pass
                    pairs_to_close.append((pair, "take_profit"))
                    continue

                # Watchdog: detect missing or IB-cancelled/inactive SL/TP orders
                _DEAD = ("Cancelled", "ApiCancelled", "Inactive")
                sl_dead = pos.sl_order is None or pos.sl_order.orderStatus.status in _DEAD
                tp_dead = pos.tp_order is None or pos.tp_order.orderStatus.status in _DEAD
                if sl_dead or tp_dead:
                    logger.warning(
                        "WATCHDOG: SL/TP died for %s (SL=%s, TP=%s) — re-placing",
                        pair,
                        pos.sl_order.orderStatus.status if pos.sl_order else "N/A",
                        pos.tp_order.orderStatus.status if pos.tp_order else "N/A",
                    )
                    ok = await self._replace_sl_tp(pos)
                    if not ok:
                        pairs_to_close.append((pair, "sl_tp_failed"))
                        continue

            # Break-even move (if enabled)
            if self.config.use_breakeven and not pos.be_triggered:
                await self._check_breakeven(pos)

        # Process closures
        prefix = "[DRY-RUN] " if self.config.dry_run else ""
        for pair, reason in pairs_to_close:
            pos = self.positions[pair]

            # sl_tp_failed: position is still open on IB — must emergency close
            if reason == "sl_tp_failed" and not self.config.dry_run:
                close_dir = "short" if pos.direction == "long" else "long"
                try:
                    await self.broker.place_market_order(pair, close_dir, abs(pos.units))
                    logger.info("Emergency close %s — SL/TP replacement failed", pair)
                except Exception as exc:
                    logger.critical(
                        "FAILED to emergency-close %s after SL/TP failure: %s — "
                        "MANUAL INTERVENTION REQUIRED", pair, exc,
                    )

            record = self._make_trade_record(pos, reason)
            closed.append(record)
            del self.positions[pair]
            logger.info("%s%s hit for %s", prefix, reason.upper(), pair)

        if pairs_to_close:
            self.save_state(self.config.state_file)

        return closed

    async def _replace_sl_tp(self, pos: LivePosition) -> bool:
        """Cancel existing and re-place SL/TP orders. Returns True on success."""
        pair = pos.pair
        _DEAD = ("Cancelled", "ApiCancelled", "Inactive")

        # Cancel any existing orders and VERIFY they are actually dead
        for t in (pos.sl_order, pos.tp_order):
            if t is None:
                continue
            status = t.orderStatus.status
            if status in _DEAD:
                continue  # already dead
            try:
                await self.broker.cancel_order(t)
            except Exception:
                pass
            # Wait for cancellation confirmation
            for _ in range(10):
                await asyncio.sleep(0.3)
                if t.orderStatus.status in _DEAD:
                    break
            if t.orderStatus.status not in _DEAD:
                logger.warning(
                    "Order %s for %s still %s after cancel — using global cancel",
                    t.order.orderId, pair, t.orderStatus.status,
                )
                await self.broker.cancel_all_orders()
                await asyncio.sleep(0.5)  # let IB process
                # Global cancel killed all orders on IB but Python Trade
                # objects retain stale status (see CLAUDE.md #3). Clear refs
                # instead of checking the now-meaningless Python status.
                pos.sl_order = None
                pos.tp_order = None
                break  # both orders killed — proceed to re-place

        sl_dir = "short" if pos.direction == "long" else "long"
        oca_group = f"ict_{pair}_{int(datetime.now(timezone.utc).timestamp())}"
        try:
            pos.sl_order = await self.broker.place_stop_order(
                pair, sl_dir, abs(pos.units), pos.stop_loss, oca_group=oca_group,
            )
            pos.tp_order = await self.broker.place_limit_order(
                pair, sl_dir, abs(pos.units), pos.take_profit, oca_group=oca_group,
            )
        except Exception as exc:
            logger.error("Failed to re-place SL/TP for %s: %s", pair, exc)
            return False

        accepted_statuses = ("PreSubmitted", "Submitted", "Filled")
        for _ in range(10):
            await asyncio.sleep(0.5)
            sl_ok = pos.sl_order and pos.sl_order.orderStatus.status in accepted_statuses
            tp_ok = pos.tp_order and pos.tp_order.orderStatus.status in accepted_statuses
            if sl_ok and tp_ok:
                break

        sl_ok = pos.sl_order and pos.sl_order.orderStatus.status in accepted_statuses
        tp_ok = pos.tp_order and pos.tp_order.orderStatus.status in accepted_statuses

        if not sl_ok or not tp_ok:
            logger.error(
                "SL/TP re-placement REJECTED for %s (SL=%s, TP=%s)",
                pair,
                pos.sl_order.orderStatus.status if pos.sl_order else "None",
                pos.tp_order.orderStatus.status if pos.tp_order else "None",
            )
            return False

        logger.info(
            "Re-placed SL/TP for %s: SL=%.5f TP=%.5f (OCA=%s)",
            pair, pos.stop_loss, pos.take_profit, oca_group,
        )
        self.save_state(self.config.state_file)
        return True

    def _check_sl_tp_price(self, pos: LivePosition) -> str | None:
        """Dry-run: check if live price has hit SL or TP. Returns exit reason or None."""
        bars_df = self.broker.get_live_bars(pos.pair)
        if bars_df is None or bars_df.empty:
            return None

        high = bars_df["high"].iloc[-1]
        low = bars_df["low"].iloc[-1]

        if pos.direction == "long":
            if low <= pos.stop_loss:
                return "stop_loss"
            if high >= pos.take_profit:
                return "take_profit"
        else:
            if high >= pos.stop_loss:
                return "stop_loss"
            if low <= pos.take_profit:
                return "take_profit"
        return None

    async def _check_breakeven(self, pos: LivePosition) -> None:
        """Move SL to entry if price has reached BE threshold."""
        bars_df = self.broker.get_live_bars(pos.pair)
        if bars_df is None or bars_df.empty:
            return

        current_price = bars_df["close"].iloc[-1]
        pip_size = pip_size_for(pos.pair)

        if pos.direction == "long":
            profit_pips = (current_price - pos.entry_price) / pip_size
        else:
            profit_pips = (pos.entry_price - current_price) / pip_size

        threshold_pips = pos.risk_pips * self.config.be_threshold_r

        if profit_pips >= threshold_pips and pos.sl_order is not None:
            new_sl = pos.entry_price
            try:
                await self.broker.modify_order(pos.sl_order, new_sl, pair=pos.pair)
                pos.be_triggered = True
                pos.stop_loss = new_sl
                self.save_state(self.config.state_file)
                logger.info("BE triggered for %s: SL moved to %.5f", pos.pair, new_sl)
            except Exception as exc:
                logger.error("Failed to move SL to BE for %s: %s", pos.pair, exc)

    # ── Startup Reconciliation ─────────────────────────────────────

    async def reconcile_on_startup(self) -> None:
        """Load saved state, reconcile with IB, and re-place SL/TP orders (#3).

        Checks for existing valid orders before placing new ones to prevent
        duplicates when multiple bot instances run concurrently.
        """
        self.load_state(self.config.state_file)

        if self.config.dry_run:
            if self.positions:
                logger.info(
                    "[DRY-RUN] Restored %d simulated positions: %s",
                    len(self.positions), list(self.positions.keys()),
                )
            else:
                logger.info("[DRY-RUN] No positions to restore")
            self.save_state(self.config.state_file)
            return

        ib_positions = await self.broker.get_open_positions()

        # Remove positions from state that no longer exist in IB
        for pair in list(self.positions.keys()):
            if pair not in ib_positions:
                logger.warning(
                    "Position %s in state file but not in IB — removing from state", pair
                )
                del self.positions[pair]

        # Warn about IB positions not in our state
        for pair, units in ib_positions.items():
            if pair not in self.positions and pair in [p for p in self.config.pairs]:
                logger.warning(
                    "IB has position in %s (%.0f units) not in our state — "
                    "managing manually or restart needed",
                    pair, units,
                )

        # Check existing orders on IB BEFORE placing new ones
        existing_orders = self.broker.get_open_orders_by_pair()

        # Re-place SL/TP orders for reconciled positions (#3)
        accepted_statuses = ("PreSubmitted", "Submitted", "Filled")
        pairs_to_emergency_close = []
        for pair, pos in self.positions.items():
            # Check if valid SL+TP already exist on IB for this pair
            pair_orders = existing_orders.get(pair, [])
            has_stop = any(
                t.order.orderType == "STP"
                and t.orderStatus.status in accepted_statuses
                for t in pair_orders
            )
            has_limit = any(
                t.order.orderType == "LMT"
                and t.orderStatus.status in accepted_statuses
                for t in pair_orders
            )
            if has_stop and has_limit:
                # Adopt existing orders instead of placing duplicates
                for t in pair_orders:
                    if t.order.orderType == "STP" and t.orderStatus.status in accepted_statuses:
                        pos.sl_order = t
                    elif t.order.orderType == "LMT" and t.orderStatus.status in accepted_statuses:
                        pos.tp_order = t
                logger.info(
                    "Adopted existing SL/TP orders for %s (SL orderId=%s, TP orderId=%s)",
                    pair,
                    pos.sl_order.order.orderId if pos.sl_order else "?",
                    pos.tp_order.order.orderId if pos.tp_order else "?",
                )
                continue

            # No valid orders exist — cancel any partial/stale ones and place fresh
            for t in pair_orders:
                try:
                    await self.broker.cancel_order(t)
                except Exception:
                    pass

            sl_dir = "short" if pos.direction == "long" else "long"
            oca_group = f"ict_{pair}_{int(datetime.now(timezone.utc).timestamp())}"
            try:
                pos.sl_order = await self.broker.place_stop_order(
                    pair, sl_dir, abs(pos.units), pos.stop_loss, oca_group=oca_group,
                )
                pos.tp_order = await self.broker.place_limit_order(
                    pair, sl_dir, abs(pos.units), pos.take_profit, oca_group=oca_group,
                )
                # Verify acceptance
                for _ in range(10):
                    await asyncio.sleep(0.5)
                    sl_ok = pos.sl_order and pos.sl_order.orderStatus.status in accepted_statuses
                    tp_ok = pos.tp_order and pos.tp_order.orderStatus.status in accepted_statuses
                    if sl_ok and tp_ok:
                        break
                sl_ok = pos.sl_order and pos.sl_order.orderStatus.status in accepted_statuses
                tp_ok = pos.tp_order and pos.tp_order.orderStatus.status in accepted_statuses
                if not sl_ok or not tp_ok:
                    logger.error(
                        "SL/TP re-placement REJECTED for %s (SL=%s, TP=%s) — will emergency close",
                        pair,
                        pos.sl_order.orderStatus.status if pos.sl_order else "None",
                        pos.tp_order.orderStatus.status if pos.tp_order else "None",
                    )
                    pairs_to_emergency_close.append(pair)
                else:
                    logger.info(
                        "Re-placed SL/TP for %s: SL=%.5f TP=%.5f (OCA=%s)",
                        pair, pos.stop_loss, pos.take_profit, oca_group,
                    )
            except Exception as exc:
                logger.error("Failed to re-place SL/TP for %s: %s — will emergency close", pair, exc)
                pairs_to_emergency_close.append(pair)

        # Emergency close any positions where SL/TP could not be placed
        for pair in pairs_to_emergency_close:
            pos = self.positions.get(pair)
            if pos is None:
                continue
            # Cancel any partial orders
            for t in (pos.sl_order, pos.tp_order):
                if t and t.orderStatus.status in accepted_statuses:
                    try:
                        await self.broker.cancel_order(t)
                    except Exception:
                        pass
            # Close position
            close_dir = "short" if pos.direction == "long" else "long"
            try:
                await self.broker.place_market_order(pair, close_dir, abs(pos.units))
                logger.info("Emergency close of %s on startup — SL/TP could not be placed", pair)
            except Exception as exc:
                logger.critical(
                    "FAILED to emergency-close %s on startup: %s — MANUAL INTERVENTION REQUIRED",
                    pair, exc,
                )
            del self.positions[pair]

        if self.positions:
            logger.info(
                "Reconciled %d positions from state: %s",
                len(self.positions),
                list(self.positions.keys()),
            )
        else:
            logger.info("No positions to reconcile")

        # Post-placement audit: detect and remove any remaining duplicates
        await self.audit_and_dedup_orders()

        self.save_state(self.config.state_file)

    # ── Order Deduplication ─────────────────────────────────────────

    async def audit_and_dedup_orders(self) -> None:
        """Scan IB for duplicate SL/TP orders per pair and cancel extras.

        Keeps the orders we're tracking (pos.sl_order / pos.tp_order) and
        cancels any other stop/limit orders for the same pair. This handles
        duplicates left by crashed instances or multiple clientIds.
        """
        orders_by_pair = self.broker.get_open_orders_by_pair()
        accepted = ("PreSubmitted", "Submitted")

        for pair, trades in orders_by_pair.items():
            stops = [t for t in trades if t.order.orderType == "STP" and t.orderStatus.status in accepted]
            limits = [t for t in trades if t.order.orderType == "LMT" and t.orderStatus.status in accepted]

            pos = self.positions.get(pair)
            tracked_sl_id = pos.sl_order.order.orderId if pos and pos.sl_order else None
            tracked_tp_id = pos.tp_order.order.orderId if pos and pos.tp_order else None

            # Cancel duplicate stops (keep the one we're tracking)
            if len(stops) > 1:
                logger.warning(
                    "DEDUP: %d stop orders for %s (expected 1) — cancelling extras",
                    len(stops), pair,
                )
                for t in stops:
                    if t.order.orderId != tracked_sl_id:
                        try:
                            await self.broker.cancel_order(t)
                            logger.info("DEDUP: cancelled extra stop %s for %s", t.order.orderId, pair)
                        except Exception as exc:
                            logger.warning("DEDUP: failed to cancel stop %s: %s", t.order.orderId, exc)

            # Cancel duplicate limits (keep the one we're tracking)
            if len(limits) > 1:
                logger.warning(
                    "DEDUP: %d limit orders for %s (expected 1) — cancelling extras",
                    len(limits), pair,
                )
                for t in limits:
                    if t.order.orderId != tracked_tp_id:
                        try:
                            await self.broker.cancel_order(t)
                            logger.info("DEDUP: cancelled extra limit %s for %s", t.order.orderId, pair)
                        except Exception as exc:
                            logger.warning("DEDUP: failed to cancel limit %s: %s", t.order.orderId, exc)

            # Warn about orphan orders (pair has orders but no tracked position)
            if not pos and (stops or limits):
                logger.warning(
                    "DEDUP: %d orphan orders for %s (no tracked position) — cancelling all",
                    len(stops) + len(limits), pair,
                )
                for t in stops + limits:
                    try:
                        await self.broker.cancel_order(t)
                    except Exception:
                        pass

    # ── State Persistence ──────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Save positions to JSON file (atomic write via temp file + rename)."""
        state = {pair: pos.to_dict() for pair, pos in self.positions.items()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file in same dir, then rename (#11)
        fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load_state(self, path: str) -> None:
        """Load positions from JSON file (without IB order references)."""
        p = Path(path)
        if not p.exists():
            logger.info("No state file at %s", path)
            return
        try:
            data = json.loads(p.read_text())
            for pair, d in data.items():
                self.positions[pair] = LivePosition.from_dict(d)
            logger.info("Loaded %d positions from %s", len(self.positions), path)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load state from %s: %s", path, exc)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_filled(trade: Trade) -> bool:
        """Check if an IB trade/order has been filled."""
        return trade.orderStatus.status == "Filled"

    @staticmethod
    def _make_trade_record(pos: LivePosition, exit_reason: str) -> dict:
        """Create a trade result record for the risk manager.

        P&L is converted to USD regardless of quote currency (#4):
        - USD-quoted pairs (EUR/USD): pnl_usd = pnl_pips * units * pip_size
        - USD-base pairs (USD/JPY): pnl is in quote ccy; approximate using entry rate
        - Cross pairs (EUR/GBP): pnl is in GBP; approximate as risk_pips * risk_per_trade
        """
        pip_size = pip_size_for(pos.pair)

        if exit_reason == "take_profit":
            exit_price = pos.take_profit
        elif exit_reason == "stop_loss":
            exit_price = pos.stop_loss
        else:
            exit_price = pos.entry_price  # manual/unknown

        if pos.direction == "long":
            pnl_pips = (exit_price - pos.entry_price) / pip_size
        else:
            pnl_pips = (pos.entry_price - exit_price) / pip_size

        # Convert P&L to USD (#4)
        if pos.pair.endswith("_USD"):
            # Quote is USD — pip value in USD = units * pip_size
            pnl_amount = pnl_pips * pos.units * pip_size
        elif pos.pair.startswith("USD_"):
            # Quote is JPY/CAD — pip value in quote ccy, divide by rate
            # pnl_quote = pnl_pips * units * pip_size (in JPY/CAD)
            # pnl_usd = pnl_quote / rate
            rate = exit_price if exit_price > 0 else pos.entry_price
            pnl_amount = (pnl_pips * pos.units * pip_size) / rate
        else:
            # Cross pair — approximate using R-multiple and risk_pips
            # This is approximate since we don't have the cross rate here
            pnl_amount = pnl_pips * pos.units * pip_size

        return {
            "pair": pos.pair,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl_pips": pnl_pips,
            "pnl_amount": pnl_amount,
            "risk_pips": pos.risk_pips,
            "rr_achieved": pnl_pips / pos.risk_pips if pos.risk_pips > 0 else 0.0,
            "exit_reason": exit_reason,
            "entry_time": pos.entry_time,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "confluence_score": pos.confluence_score,
            "be_triggered": pos.be_triggered,
        }
