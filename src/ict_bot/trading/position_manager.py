"""PositionManager — track live positions, OCO emulation, break-even moves."""
from __future__ import annotations

import json
import logging
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
        """Open a new position: market entry + SL stop + TP limit.

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
        risk_pips = abs(signal.entry_price - signal.stop_loss) / pip_size

        sl_trade = None
        tp_trade = None

        if not self.config.dry_run:
            # Place market entry
            await self.broker.place_market_order(pair, direction, units)

            # SL: opposite direction stop order
            sl_dir = "short" if direction == "long" else "long"
            sl_trade = await self.broker.place_stop_order(
                pair, sl_dir, units, signal.stop_loss
            )

            # TP: opposite direction limit order
            tp_trade = await self.broker.place_limit_order(
                pair, sl_dir, units, signal.take_profit
            )

        prefix = "[DRY-RUN] " if self.config.dry_run else ""

        pos = LivePosition(
            pair=pair,
            direction=direction,
            entry_price=signal.entry_price,
            units=units,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=datetime.now(timezone.utc).isoformat(),
            risk_pips=risk_pips,
            confluence_score=signal.confluence_score,
            sl_order=sl_trade,
            tp_order=tp_trade,
        )

        self.positions[pair] = pos
        self.save_state(self.config.state_file)
        logger.info(
            "%sOpened %s %s: entry=%.5f SL=%.5f TP=%.5f units=%.0f",
            prefix, direction, pair, signal.entry_price, signal.stop_loss,
            signal.take_profit, units,
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
            await self.broker.place_market_order(pair, close_dir, pos.units)

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
                # Live: check IB order fill status (OCO emulation)
                if pos.sl_order is not None and self._is_filled(pos.sl_order):
                    logger.info("SL filled for %s", pair)
                    if pos.tp_order is not None:
                        try:
                            await self.broker.cancel_order(pos.tp_order)
                        except Exception:
                            pass
                    pairs_to_close.append((pair, "stop_loss"))
                    continue

                if pos.tp_order is not None and self._is_filled(pos.tp_order):
                    logger.info("TP filled for %s", pair)
                    if pos.sl_order is not None:
                        try:
                            await self.broker.cancel_order(pos.sl_order)
                        except Exception:
                            pass
                    pairs_to_close.append((pair, "take_profit"))
                    continue

            # Break-even move (if enabled)
            if self.config.use_breakeven and not pos.be_triggered:
                await self._check_breakeven(pos)

        # Process closures
        prefix = "[DRY-RUN] " if self.config.dry_run else ""
        for pair, reason in pairs_to_close:
            record = self._make_trade_record(self.positions[pair], reason)
            closed.append(record)
            del self.positions[pair]
            logger.info("%s%s hit for %s", prefix, reason.upper(), pair)

        if pairs_to_close:
            self.save_state(self.config.state_file)

        return closed

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
            await self.broker.modify_order(pos.sl_order, new_sl)
            pos.be_triggered = True
            pos.stop_loss = new_sl
            self.save_state(self.config.state_file)
            logger.info("BE triggered for %s: SL moved to %.5f", pos.pair, new_sl)

    # ── Startup Reconciliation ─────────────────────────────────────

    async def reconcile_on_startup(self) -> None:
        """Load saved state and reconcile with actual IB positions."""
        self.load_state(self.config.state_file)

        if self.config.dry_run:
            # In dry-run, keep all saved positions (they're simulated)
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

        if self.positions:
            logger.info(
                "Reconciled %d positions from state: %s",
                len(self.positions),
                list(self.positions.keys()),
            )
        else:
            logger.info("No positions to reconcile")

        self.save_state(self.config.state_file)

    # ── State Persistence ──────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Save positions to JSON file."""
        state = {pair: pos.to_dict() for pair, pos in self.positions.items()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2))

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
        """Create a trade result record for the risk manager."""
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

        # Approximate P&L in account currency
        pnl_amount = pnl_pips * (pos.units * pip_size)

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
