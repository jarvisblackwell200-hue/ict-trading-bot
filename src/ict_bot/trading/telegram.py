"""Telegram trade notifications via Bot API."""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramNotifier:
    """Sends trade notifications to a Telegram chat.

    Auto-discovers the chat_id from recent messages on first use.
    Set TELEGRAM_BOT_TOKEN env var or pass token to LiveConfig.
    """

    def __init__(self, token: str, chat_id: int = 0) -> None:
        self._token = token
        self._api = TELEGRAM_API.format(token=token)
        self._chat_id: int | None = chat_id if chat_id else None
        self._enabled = bool(token)
        if not self._enabled:
            logger.info("Telegram notifications disabled (no token)")
        elif self._chat_id:
            logger.info("Telegram notifications enabled (chat_id=%s)", self._chat_id)

    async def _discover_chat_id(self) -> int | None:
        """Auto-discover chat_id from recent bot updates."""
        if self._chat_id:
            return self._chat_id

        def _fetch():
            url = f"{self._api}/getUpdates?limit=10&offset=-10"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        try:
            data = await asyncio.to_thread(_fetch)
            if data.get("ok") and data.get("result"):
                for update in reversed(data["result"]):
                    # Try message.chat.id first
                    msg = update.get("message")
                    if msg and msg.get("chat", {}).get("id"):
                        self._chat_id = msg["chat"]["id"]
                        logger.info("Telegram chat_id discovered: %s", self._chat_id)
                        return self._chat_id
                    # Try my_chat_member (when user starts bot)
                    mcm = update.get("my_chat_member")
                    if mcm and mcm.get("chat", {}).get("id"):
                        self._chat_id = mcm["chat"]["id"]
                        logger.info("Telegram chat_id discovered: %s", self._chat_id)
                        return self._chat_id
        except Exception as exc:
            logger.warning("Telegram getUpdates failed: %s", exc)
        return None

    async def send(self, text: str) -> bool:
        """Send a message to the Telegram chat. Returns True on success."""
        if not self._enabled:
            return False

        chat_id = await self._discover_chat_id()
        if not chat_id:
            logger.warning("No Telegram chat_id — send /start to your bot first")
            return False

        def _post():
            payload = json.dumps({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }).encode()
            req = urllib.request.Request(
                f"{self._api}/sendMessage",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200

        try:
            return await asyncio.to_thread(_post)
        except Exception as exc:
            logger.warning("Telegram send failed: %s", exc)
            return False


# ── Message Formatting ─────────────────────────────────────────────────


def _fmt_price(price: float, pair: str) -> str:
    """Format price with appropriate decimal places."""
    if "JPY" in pair:
        return f"{price:.3f}"
    return f"{price:.5f}"


def _positions_block(positions: dict, max_positions: int) -> str:
    """Format the open positions summary block."""
    count = len(positions)
    header = f"\n\U0001F4CA <b>Open Positions ({count}/{max_positions})</b>"
    if count == 0:
        return header + "\nNo open positions"

    lines = [header]
    for pair, pos in positions.items():
        arrow = "\u2191" if pos.direction == "long" else "\u2193"
        lines.append(
            f"  {arrow} {pair} {pos.direction.upper()} @ {_fmt_price(pos.entry_price, pair)}"
            f"  SL {_fmt_price(pos.stop_loss, pair)} / TP {_fmt_price(pos.take_profit, pair)}"
        )
    return "\n".join(lines)


def format_trade_opened(pos, all_positions: dict, max_positions: int) -> str:
    """Format a trade-opened notification message."""
    arrow = "\u2191" if pos.direction == "long" else "\u2193"
    lines = [
        f"\U0001F7E2 <b>OPENED {pos.pair} {pos.direction.upper()}</b> {arrow}",
        f"Entry: {_fmt_price(pos.entry_price, pos.pair)}",
        f"SL: {_fmt_price(pos.stop_loss, pos.pair)}  |  TP: {_fmt_price(pos.take_profit, pos.pair)}",
        f"Risk: {pos.risk_pips:.1f} pips  |  Units: {pos.units:,.0f}",
        f"Confluence: {pos.confluence_score}",
        _positions_block(all_positions, max_positions),
    ]
    return "\n".join(lines)


def format_trade_closed(record: dict, all_positions: dict, max_positions: int) -> str:
    """Format a trade-closed notification message."""
    pair = record["pair"]
    direction = record["direction"]
    pnl_pips = record.get("pnl_pips", 0)
    rr = record.get("rr_achieved", 0)
    reason = record.get("exit_reason", "unknown")

    reason_label = {
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "pre_news_close": "Pre-News Close",
        "sl_tp_failed": "SL/TP Failed",
        "manual": "Manual",
    }.get(reason, reason.replace("_", " ").title())

    icon = "\U0001F534" if pnl_pips < 0 else "\U0001F7E2"
    pnl_sign = "+" if pnl_pips >= 0 else ""
    rr_sign = "+" if rr >= 0 else ""

    lines = [
        f"{icon} <b>CLOSED {pair} {direction.upper()} \u2014 {reason_label}</b>",
        f"Entry: {_fmt_price(record['entry_price'], pair)} \u2192 Exit: {_fmt_price(record['exit_price'], pair)}",
        f"P&L: {pnl_sign}{pnl_pips:.1f} pips ({rr_sign}{rr:.1f}R)",
        _positions_block(all_positions, max_positions),
    ]
    return "\n".join(lines)
