"""Load stored OHLC data from Parquet files."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def load_pair(pair: str, timeframe: str) -> pd.DataFrame:
    """Load OHLC data for a pair/timeframe from Parquet storage."""
    path = DATA_DIR / f"{pair}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No data found at {path}. "
            f"Run download first: download_yfinance('{pair}', '{timeframe}') "
            f"or download_oanda('{pair}', '{timeframe}')"
        )
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"Loaded {len(df)} candles from {path}")
    return df


def list_available_data() -> list[dict]:
    """List all available data files with metadata."""
    if not DATA_DIR.exists():
        return []
    results = []
    for path in sorted(DATA_DIR.glob("*.parquet")):
        name = path.stem
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            pair_parts = name.split("_")
            # Handle EUR_USD_M5 -> pair=EUR_USD, tf=M5
            tf = pair_parts[-1]
            pair = "_".join(pair_parts[:-1])
        else:
            pair, tf = name, "unknown"
        df = pd.read_parquet(path, engine="pyarrow")
        results.append({
            "pair": pair,
            "timeframe": tf,
            "candles": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "path": str(path),
        })
    return results
