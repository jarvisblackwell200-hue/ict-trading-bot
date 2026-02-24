#!/usr/bin/env python3
"""
Independent position watchdog — monitors IB for unprotected or duplicate orders.

Connects to IB Gateway with its own clientId (separate from bot and dashboard).
Checks every 30 seconds that:
  1. Every position has a matching STP (stop loss) order
  2. Every position has a matching LMT (take profit) order
  3. Order units match position units
  4. No duplicate SL/TP orders exist for the same pair

Alerts via Telegram if any issue is detected. Keeps alerting every cycle until resolved.

Usage:
    TELEGRAM_BOT_TOKEN="..." TELEGRAM_CHAT_ID="..." python scripts/watchdog.py [--port 4002] [--interval 30]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ib_insync import IB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("watchdog")

CLIENT_ID = 98
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


# ── Telegram ──────────────────────────────────────────────────────────


def send_telegram(token: str, chat_id: int, text: str) -> bool:
    """Send a Telegram message. Returns True on success."""
    if not token or not chat_id:
        return False
    try:
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            TELEGRAM_API.format(token=token),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


# ── Data structures ───────────────────────────────────────────────────


@dataclass
class PositionInfo:
    pair: str
    units: float        # positive = long, negative = short
    direction: str      # "long" or "short"


@dataclass
class OrderInfo:
    pair: str
    order_type: str     # "STP" or "LMT"
    action: str         # "BUY" or "SELL"
    units: float
    price: float
    order_id: int


def symbol_to_pair(contract) -> str:
    """Convert IB contract to internal pair name (e.g. EUR_USD)."""
    return f"{contract.symbol}_{contract.currency}"


# ── Core check ────────────────────────────────────────────────────────


def check_positions(ib: IB) -> list[str]:
    """
    Check all IB positions for protection issues.
    Returns a list of alert strings (empty = all clear).
    """
    alerts: list[str] = []

    # Get positions (skip zero-size)
    positions: list[PositionInfo] = []
    for p in ib.positions():
        if p.position == 0:
            continue
        pair = symbol_to_pair(p.contract)
        direction = "long" if p.position > 0 else "short"
        positions.append(PositionInfo(pair=pair, units=p.position, direction=direction))

    if not positions:
        return alerts

    # Get ALL open orders across all client IDs
    all_orders: list[OrderInfo] = []
    try:
        trades = ib.reqAllOpenOrders()
        ib.sleep(1)  # wait for IB to deliver order data
    except Exception:
        trades = []

    # Also include own client's trades as fallback
    seen_ids = {t.order.orderId for t in trades if t.order}
    for t in ib.openTrades():
        if t.order and t.order.orderId not in seen_ids:
            trades.append(t)

    for trade in trades:
        order = trade.order
        status = trade.orderStatus.status
        # Only count live orders
        if status not in ("PreSubmitted", "Submitted"):
            continue
        pair = symbol_to_pair(trade.contract)
        price = order.auxPrice if order.orderType == "STP" else order.lmtPrice
        all_orders.append(OrderInfo(
            pair=pair,
            order_type=order.orderType,
            action=order.action,
            units=order.totalQuantity,
            price=price,
            order_id=order.orderId,
        ))

    # Check each position
    for pos in positions:
        abs_units = abs(pos.units)
        # Expected SL/TP direction: opposite of position
        expected_action = "SELL" if pos.direction == "long" else "BUY"

        # Find matching orders for this pair
        pair_orders = [o for o in all_orders if o.pair == pos.pair and o.action == expected_action]
        stops = [o for o in pair_orders if o.order_type == "STP"]
        limits = [o for o in pair_orders if o.order_type == "LMT"]

        # Check 1: Missing stop loss
        if len(stops) == 0:
            alerts.append(
                f"\u26a0\ufe0f <b>NO STOP LOSS</b>\n"
                f"{pos.pair} {pos.direction.upper()} {abs_units:,.0f} units"
            )

        # Check 2: Missing take profit
        if len(limits) == 0:
            alerts.append(
                f"\u26a0\ufe0f <b>NO TAKE PROFIT</b>\n"
                f"{pos.pair} {pos.direction.upper()} {abs_units:,.0f} units"
            )

        # Check 3: Duplicate stop losses
        if len(stops) > 1:
            prices = ", ".join(f"{o.price:.5f} (id={o.order_id})" for o in stops)
            total_units = sum(o.units for o in stops)
            alerts.append(
                f"\U0001f534 <b>DUPLICATE STOP LOSS</b>\n"
                f"{pos.pair} — {len(stops)} SL orders totaling {total_units:,.0f} units\n"
                f"Prices: {prices}\n"
                f"Position is only {abs_units:,.0f} units — excess will create reverse position"
            )

        # Check 4: Duplicate take profits
        if len(limits) > 1:
            prices = ", ".join(f"{o.price:.5f} (id={o.order_id})" for o in limits)
            total_units = sum(o.units for o in limits)
            alerts.append(
                f"\U0001f534 <b>DUPLICATE TAKE PROFIT</b>\n"
                f"{pos.pair} — {len(limits)} TP orders totaling {total_units:,.0f} units\n"
                f"Prices: {prices}\n"
                f"Position is only {abs_units:,.0f} units — excess will create reverse position"
            )

        # Check 5: Unit mismatch on stop loss
        if len(stops) == 1 and stops[0].units != abs_units:
            alerts.append(
                f"\u26a0\ufe0f <b>SL SIZE MISMATCH</b>\n"
                f"{pos.pair} position={abs_units:,.0f} but SL order={stops[0].units:,.0f} units"
            )

        # Check 6: Unit mismatch on take profit
        if len(limits) == 1 and limits[0].units != abs_units:
            alerts.append(
                f"\u26a0\ufe0f <b>TP SIZE MISMATCH</b>\n"
                f"{pos.pair} position={abs_units:,.0f} but TP order={limits[0].units:,.0f} units"
            )

    return alerts


# ── Main loop ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="IB Position Watchdog")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

    if not token or not chat_id:
        logger.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars required")
        sys.exit(1)

    ib = IB()
    logger.info("Watchdog starting (port=%d, interval=%ds, clientId=%d)", args.port, args.interval, CLIENT_ID)

    # Send startup message
    send_telegram(token, chat_id,
        "\u2705 <b>Watchdog started</b>\n"
        f"Checking IB positions every {args.interval}s\n"
        f"Port {args.port} | clientId {CLIENT_ID}"
    )

    consecutive_errors = 0

    while True:
        try:
            # Connect if needed
            if not ib.isConnected():
                try:
                    ib.connect(args.host, args.port, clientId=CLIENT_ID, readonly=True)
                    logger.info("Connected to IB Gateway (clientId=%d)", CLIENT_ID)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    logger.warning("Connect failed: %s (attempt %d)", exc, consecutive_errors)
                    if consecutive_errors >= 5 and consecutive_errors % 5 == 0:
                        send_telegram(token, chat_id,
                            f"\U0001f534 <b>WATCHDOG DISCONNECTED</b>\n"
                            f"Cannot reach IB Gateway at {args.host}:{args.port}\n"
                            f"Failed {consecutive_errors} times — positions may be unmonitored"
                        )
                    time.sleep(args.interval)
                    continue

            # Run check
            ib.sleep(0.1)  # process pending events
            alerts = check_positions(ib)

            if alerts:
                now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                header = f"\U0001f6a8 <b>WATCHDOG ALERT</b> — {now}\n\n"
                body = "\n\n".join(alerts)
                footer = "\n\n\U0001f6a8 <b>Immediate action required</b>"
                send_telegram(token, chat_id, header + body + footer)
                logger.warning("ALERT: %d issue(s) detected", len(alerts))
                for a in alerts:
                    logger.warning("  %s", a.replace("<b>", "").replace("</b>", "").replace("\n", " | "))
            else:
                logger.info("OK — all positions protected")

        except Exception as exc:
            logger.error("Watchdog error: %s", exc, exc_info=True)
            try:
                ib.disconnect()
            except Exception:
                pass

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
