"""
Multi-source OHLC data downloader.

Sources:
  - yfinance: Quick start, no API key. Daily/1H go back years; 5M limited to ~60 days.
  - OANDA v20: Comprehensive 5M data going back years. Requires free practice account.
  - CSV import: For HistData.com or Dukascopy manual downloads.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_parquet(df: pd.DataFrame, pair: str, timeframe: str) -> Path:
    _ensure_data_dir()
    path = DATA_DIR / f"{pair}_{timeframe}.parquet"
    df.to_parquet(path, engine="pyarrow")
    logger.info(f"Saved {len(df)} candles to {path}")
    return path


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names: open, high, low, close, volume."""
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    df = df.rename(columns=col_map)
    for required in ["open", "high", "low", "close"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")
    if "volume" not in df.columns:
        df["volume"] = 0
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# yfinance provider
# ---------------------------------------------------------------------------

YFINANCE_PAIR_MAP = {
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "AUD_USD": "AUDUSD=X",
    "USD_CAD": "USDCAD=X",
    "NZD_USD": "NZDUSD=X",
    "USD_CHF": "USDCHF=X",
    "EUR_GBP": "EURGBP=X",
    "EUR_JPY": "EURJPY=X",
    "GBP_JPY": "GBPJPY=X",
}

YFINANCE_TF_MAP = {
    "M5": "5m",
    "M15": "15m",
    "H1": "1h",
    "H4": "4h",   # yfinance doesn't support 4h natively, we resample from 1h
    "D": "1d",
    "W": "1wk",
}

# yfinance max periods for intraday
YFINANCE_MAX_PERIOD = {
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "max",
    "1wk": "max",
}


def download_yfinance(
    pair: str = "EUR_USD",
    timeframe: str = "H1",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Download forex data via yfinance.

    Limitations:
      - 5M/15M: last 60 days only
      - 1H: last 730 days
      - Daily: full history
    """
    import yfinance as yf

    ticker = YFINANCE_PAIR_MAP.get(pair)
    if not ticker:
        raise ValueError(f"Unknown pair {pair}. Available: {list(YFINANCE_PAIR_MAP.keys())}")

    yf_tf = YFINANCE_TF_MAP.get(timeframe)
    need_resample = False
    if timeframe == "H4":
        yf_tf = "1h"
        need_resample = True

    if yf_tf is None:
        raise ValueError(f"Unknown timeframe {timeframe}. Available: {list(YFINANCE_TF_MAP.keys())}")

    logger.info(f"Downloading {pair} {timeframe} from yfinance...")

    if start and end:
        df = yf.download(ticker, start=start, end=end, interval=yf_tf, progress=False)
    else:
        period = YFINANCE_MAX_PERIOD[yf_tf]
        df = yf.download(ticker, period=period, interval=yf_tf, progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for {pair} {timeframe}")

    # yfinance returns MultiIndex columns for single ticker, flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = _normalize_columns(df)

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"

    # Resample to 4H if needed
    if need_resample:
        df = df.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    # Remove weekends (forex is closed)
    df = df[df.index.dayofweek < 5]

    path = _save_parquet(df, pair, timeframe)
    logger.info(f"Downloaded {len(df)} candles for {pair} {timeframe} -> {path}")
    return df


# ---------------------------------------------------------------------------
# OANDA v20 provider
# ---------------------------------------------------------------------------

OANDA_GRANULARITIES = {
    "M1": "M1",
    "M5": "M5",
    "M15": "M15",
    "H1": "H1",
    "H4": "H4",
    "D": "D",
    "W": "W",
}

# OANDA returns max 5000 candles per request
OANDA_MAX_CANDLES = 5000


def download_oanda(
    pair: str = "EUR_USD",
    timeframe: str = "M5",
    start: str = "2022-01-01",
    end: str | None = None,
    api_key: str | None = None,
    account_id: str | None = None,
    environment: str = "practice",
) -> pd.DataFrame:
    """
    Download forex data from OANDA v20 REST API.

    Requires a free OANDA practice account.
    Set OANDA_API_KEY and OANDA_ACCOUNT_ID env vars or pass directly.
    """
    import os

    import requests

    api_key = api_key or os.environ.get("OANDA_API_KEY")
    account_id = account_id or os.environ.get("OANDA_ACCOUNT_ID")

    if not api_key:
        raise ValueError(
            "OANDA API key required. Set OANDA_API_KEY env var or pass api_key parameter. "
            "Get a free practice account at https://www.oanda.com/"
        )

    granularity = OANDA_GRANULARITIES.get(timeframe)
    if not granularity:
        raise ValueError(f"Unknown timeframe {timeframe}. Available: {list(OANDA_GRANULARITIES.keys())}")

    base_url = (
        "https://api-fxpractice.oanda.com" if environment == "practice"
        else "https://api-fxtrade.oanda.com"
    )
    instrument = pair  # OANDA uses EUR_USD format

    end_dt = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)

    logger.info(f"Downloading {pair} {timeframe} from OANDA ({start} to {end_dt.date()})...")

    all_candles = []
    current_from = start_dt

    while current_from < end_dt:
        params = {
            "granularity": granularity,
            "from": current_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "count": OANDA_MAX_CANDLES,
            "price": "M",  # midpoint
        }

        resp = requests.get(
            f"{base_url}/v3/instruments/{instrument}/candles",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        candles = data.get("candles", [])
        if not candles:
            break

        for c in candles:
            if c["complete"]:
                mid = c["mid"]
                all_candles.append({
                    "timestamp": pd.Timestamp(c["time"]),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c["volume"]),
                })

        # Move to after the last candle
        last_time = pd.Timestamp(candles[-1]["time"])
        if last_time <= current_from:
            break
        current_from = last_time.to_pydatetime()

        logger.info(f"  Fetched {len(all_candles)} candles so far (up to {current_from.date()})...")

    if not all_candles:
        raise RuntimeError(f"No data returned for {pair} {timeframe}")

    df = pd.DataFrame(all_candles).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df.index.name = "timestamp"

    path = _save_parquet(df, pair, timeframe)
    logger.info(f"Downloaded {len(df)} candles for {pair} {timeframe} -> {path}")
    return df


# ---------------------------------------------------------------------------
# CSV import (for HistData.com / Dukascopy manual downloads)
# ---------------------------------------------------------------------------

def import_csv(
    csv_path: str,
    pair: str = "EUR_USD",
    timeframe: str = "M5",
    datetime_col: str | int = 0,
    datetime_format: str | None = None,
    sep: str = ";",
    resample_from: str | None = None,
) -> pd.DataFrame:
    """
    Import OHLC data from CSV file.

    Works with HistData.com format (semicolon-separated, no header) and
    standard CSV exports. Set resample_from="1min" to resample 1M data to 5M.
    """
    logger.info(f"Importing {csv_path}...")

    df = pd.read_csv(csv_path, sep=sep, header=None if sep == ";" else "infer")

    # HistData.com format: DateTime;Open;High;Low;Close;Volume
    if df.columns.dtype == "int64":  # no header
        if len(df.columns) == 6:
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        elif len(df.columns) == 5:
            df.columns = ["timestamp", "open", "high", "low", "close"]
            df["volume"] = 0

    # Parse datetime
    ts_col = df.columns[datetime_col] if isinstance(datetime_col, int) else datetime_col
    df["timestamp"] = pd.to_datetime(df[ts_col], format=datetime_format)
    df = df.set_index("timestamp").sort_index()
    df = _normalize_columns(df)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Resample if needed (e.g., 1M -> 5M)
    if resample_from:
        resample_map = {"M1": "5min", "1min": "5min", "M5": "5min"}
        target = resample_map.get(resample_from, "5min")
        df = df.resample(target).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    path = _save_parquet(df, pair, timeframe)
    logger.info(f"Imported {len(df)} candles for {pair} {timeframe} -> {path}")
    return df
