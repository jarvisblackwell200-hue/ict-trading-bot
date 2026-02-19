#!/usr/bin/env python3
"""Download M5 and M15 historical data from IB Gateway for all 7 pairs.

IB provides ~6 months of 5-min data and ~1-2 years of 15-min data.
Downloads in 30-day chunks to stay within IB's per-request limits.

Usage:
    PYTHONPATH=src python scripts/download_ib_data.py --port 4002 --client-id 50
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ib_insync.util as ib_util

ib_util.patchAsyncio()

import pandas as pd
from ib_insync import IB, Forex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD", "EUR_GBP"]

PAIR_TO_IB = {
    "EUR_USD": "EURUSD", "GBP_USD": "GBPUSD", "USD_JPY": "USDJPY",
    "AUD_USD": "AUDUSD", "USD_CAD": "USDCAD", "NZD_USD": "NZDUSD",
    "EUR_GBP": "EURGBP",
}

# IB bar size settings and how many months to go back
TIMEFRAMES = {
    "M5":  {"bar_size": "5 mins",  "months_back": 6,  "chunk_days": 30},
    "M15": {"bar_size": "15 mins", "months_back": 12, "chunk_days": 60},
}


def bars_to_dataframe(bars) -> pd.DataFrame:
    """Convert IB BarDataList to standard OHLC DataFrame."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    records = []
    for bar in bars:
        records.append({
            "timestamp": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume if bar.volume > 0 else 0,
        })
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


async def download_pair_tf(ib: IB, pair: str, tf_name: str, tf_config: dict) -> pd.DataFrame:
    """Download historical data for one pair/timeframe in chunks."""
    ib_symbol = PAIR_TO_IB[pair]
    contract = Forex(pair=ib_symbol, exchange="IDEALPRO")

    bar_size = tf_config["bar_size"]
    months_back = tf_config["months_back"]
    chunk_days = tf_config["chunk_days"]

    now = datetime.now(timezone.utc)
    end_dt = now
    start_limit = now - timedelta(days=months_back * 30)

    all_dfs = []
    chunk_num = 0

    while end_dt > start_limit:
        chunk_num += 1
        end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")
        duration = f"{chunk_days} D"

        logger.info(
            "  %s %s chunk %d: end=%s duration=%s",
            pair, tf_name, chunk_num, end_str, duration,
        )

        try:
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="MIDPOINT",
                useRTH=False,
                formatDate=2,
            )
        except Exception as exc:
            logger.warning("  Request failed for %s %s chunk %d: %s", pair, tf_name, chunk_num, exc)
            # Try smaller chunk
            if chunk_days > 7:
                chunk_days = chunk_days // 2
                logger.info("  Reducing chunk size to %d days", chunk_days)
                continue
            else:
                logger.error("  Giving up on %s %s", pair, tf_name)
                break

        if not bars:
            logger.info("  No more data for %s %s (got empty response)", pair, tf_name)
            break

        df = bars_to_dataframe(bars)
        if len(df) == 0:
            break

        all_dfs.append(df)
        logger.info("  Got %d bars (%s to %s)", len(df), df.index[0], df.index[-1])

        # Move end_dt back
        end_dt = df.index[0] - timedelta(minutes=1)

        # IB pacing: wait between requests to avoid throttling
        await asyncio.sleep(2)

    if not all_dfs:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    combined = pd.concat(all_dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    logger.info(
        "  %s %s TOTAL: %d bars from %s to %s",
        pair, tf_name, len(combined), combined.index[0], combined.index[-1],
    )
    return combined


async def main_async(host: str, port: int, client_id: int) -> None:
    ib = IB()
    await ib.connectAsync(host=host, port=port, clientId=client_id, readonly=True)
    logger.info("Connected to IB Gateway at %s:%d (clientId=%d)", host, port, client_id)

    for tf_name, tf_config in TIMEFRAMES.items():
        logger.info("=" * 60)
        logger.info("Downloading %s data for all pairs", tf_name)
        logger.info("=" * 60)

        for pair in PAIRS:
            logger.info("Downloading %s %s...", pair, tf_name)
            df = await download_pair_tf(ib, pair, tf_name, tf_config)

            if len(df) == 0:
                logger.warning("No data for %s %s — skipping", pair, tf_name)
                continue

            # Save
            out_path = DATA_DIR / f"{pair}_{tf_name}.parquet"
            df.to_parquet(out_path, engine="pyarrow")
            logger.info("Saved %s (%d bars, %.1f KB)", out_path.name, len(df), out_path.stat().st_size / 1024)

            # Pacing between pairs
            await asyncio.sleep(1)

    ib.disconnect()
    logger.info("Done! Disconnected from IB Gateway.")

    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    for tf_name in TIMEFRAMES:
        for pair in PAIRS:
            path = DATA_DIR / f"{pair}_{tf_name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                days = (df.index[-1] - df.index[0]).days
                print(f"  {pair} {tf_name}: {len(df):,} bars, {days} days ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"  {pair} {tf_name}: MISSING")


def main():
    parser = argparse.ArgumentParser(description="Download M5/M15 data from IB Gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--client-id", type=int, default=50, help="Use different ID from live bot")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async(args.host, args.port, args.client_id))


if __name__ == "__main__":
    main()
