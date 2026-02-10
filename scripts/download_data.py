#!/usr/bin/env python3
"""
Download historical forex data for backtesting.

Usage:
  # Quick start with yfinance (no API key needed):
  python scripts/download_data.py --source yfinance --pair EUR_USD --timeframe H1
  python scripts/download_data.py --source yfinance --pair EUR_USD --timeframe D

  # Comprehensive data from OANDA (requires free practice account):
  python scripts/download_data.py --source oanda --pair EUR_USD --timeframe M5 --start 2022-01-01

  # Import from CSV (HistData.com):
  python scripts/download_data.py --source csv --file data/raw/EURUSD_M1.csv --pair EUR_USD --resample M1

  # Download all standard pairs and timeframes:
  python scripts/download_data.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ict_bot.data.downloader import (
    download_oanda,
    download_yfinance,
    import_csv,
)
from src.ict_bot.data.loader import list_available_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STANDARD_PAIRS = ["EUR_USD", "GBP_USD"]
STANDARD_TIMEFRAMES = ["D", "H4", "H1", "M5"]


def download_all_yfinance():
    """Download all standard pairs and timeframes from yfinance."""
    for pair in STANDARD_PAIRS:
        for tf in STANDARD_TIMEFRAMES:
            try:
                download_yfinance(pair, tf)
            except Exception as e:
                logger.warning(f"Failed to download {pair} {tf}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download historical forex data")
    parser.add_argument("--source", choices=["yfinance", "oanda", "csv"], default="yfinance")
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--file", help="CSV file path (for csv source)")
    parser.add_argument("--resample", help="Resample from this timeframe (e.g., M1)")
    parser.add_argument("--all", action="store_true", help="Download all standard pairs/timeframes")
    parser.add_argument("--list", action="store_true", help="List available data")
    args = parser.parse_args()

    if args.list:
        data = list_available_data()
        if not data:
            print("No data files found. Run a download first.")
            return
        print(f"\n{'Pair':<12} {'TF':<6} {'Candles':>10} {'Start':<22} {'End':<22}")
        print("-" * 75)
        for d in data:
            print(f"{d['pair']:<12} {d['timeframe']:<6} {d['candles']:>10} {d['start']:<22} {d['end']:<22}")
        return

    if args.all:
        download_all_yfinance()
        return

    if args.source == "yfinance":
        download_yfinance(args.pair, args.timeframe, args.start, args.end)
    elif args.source == "oanda":
        download_oanda(args.pair, args.timeframe, args.start, args.end)
    elif args.source == "csv":
        if not args.file:
            parser.error("--file required for csv source")
        import_csv(args.file, args.pair, args.timeframe, resample_from=args.resample)


if __name__ == "__main__":
    main()
