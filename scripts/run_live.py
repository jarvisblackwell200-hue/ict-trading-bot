#!/usr/bin/env python3
"""Run the ICT trading bot live via Interactive Brokers.

Usage:
    PYTHONPATH=src python scripts/run_live.py --port 4002 --pairs EUR_USD GBP_USD
    PYTHONPATH=src python scripts/run_live.py --port 4002  # all 7 pairs
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys

import ib_insync.util as ib_util

ib_util.patchAsyncio()  # must be called before any asyncio usage
import asyncio  # noqa: E402

from ict_bot.trading import LiveConfig, LiveTradingSession

DEFAULT_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD",
    "USD_CAD", "NZD_USD", "EUR_GBP",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="ICT Trading Bot — Live via IBKR")
    parser.add_argument(
        "--pairs", nargs="+", default=DEFAULT_PAIRS,
        help="Pairs to trade (default: all 7)",
    )
    parser.add_argument(
        "--port", type=int, default=4002,
        help="IB Gateway port (4002=paper, 4001=live)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway host")
    parser.add_argument("--client-id", type=int, default=1, help="IB client ID")
    parser.add_argument("--risk", type=float, default=0.01, help="Risk per trade (0.01=1%%)")
    parser.add_argument("--balance", type=float, default=10_000.0, help="Starting balance")
    parser.add_argument("--max-positions", type=int, default=3, help="Max simultaneous positions")
    parser.add_argument("--timeframe", default="M15", help="Timeframe: M5, M15, H1 (default: M15)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders using real prices")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    # Safety warning for live port
    if args.port == 4001:
        print("\n*** WARNING: Port 4001 is LIVE TRADING — real money at risk! ***")
        confirm = input("Type 'LIVE' to confirm: ")
        if confirm != "LIVE":
            print("Aborted.")
            sys.exit(1)

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = LiveConfig(
        pairs=args.pairs,
        ib_host=args.host,
        ib_port=args.port,
        ib_client_id=args.client_id,
        risk_per_trade=args.risk,
        starting_balance=args.balance,
        max_positions=args.max_positions,
        timeframe=args.timeframe,
        dry_run=args.dry_run,
    )
    session = LiveTradingSession(config)

    # Graceful shutdown on SIGINT/SIGTERM
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logging.getLogger(__name__).info("Shutdown signal received")
        asyncio.ensure_future(session.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(session.start())
    except KeyboardInterrupt:
        loop.run_until_complete(session.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
