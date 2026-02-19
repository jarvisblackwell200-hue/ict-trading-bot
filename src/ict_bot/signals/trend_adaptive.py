"""
Adaptive Trend Continuation signal generator.

Simpler than the full ICT model — uses BOS + EMA trend filter + ATR-based
stops/sizing. Designed for higher trade frequency with trailing stop exits.

Parameters (5 total):
  - swing_length: 10 (BOS detection lookback)
  - ema_period: 50 (trend direction filter)
  - atr_period: 14 (volatility measurement)
  - atr_sl_mult: 2.0 (initial stop distance)
  - max_sl_pips: 100.0 (safety cap)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .detector import Signal, detect_primitives

logger = logging.getLogger(__name__)


def _compute_atr(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def generate_adaptive_signals(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    pair: str = "EUR_USD",
    swing_length: int = 10,
    ema_period: int = 50,
    atr_period: int = 14,
    atr_sl_mult: float = 2.0,
    max_sl_pips: float = 100.0,
) -> list[Signal]:
    """
    Generate Adaptive Trend Continuation signals.

    Entry rules:
      1. BOS detected (price breaks swing high/low)
      2. EMA filter — longs: close > EMA; shorts: close < EMA
      3. Pullback filter — price within 3x ATR of EMA (not overextended)
      4. Enter at next bar's open after conditions met

    Parameters
    ----------
    ohlc : DataFrame
        OHLC data with DatetimeIndex, columns: open, high, low, close, volume.
    htf_ohlc : DataFrame, optional
        Higher timeframe OHLC for bias alignment (bonus scoring, not required).
    pair : str
        Currency pair name.
    swing_length : int
        Lookback for swing high/low detection (BOS).
    ema_period : int
        EMA period for trend direction filter.
    atr_period : int
        ATR period for volatility measurement.
    atr_sl_mult : float
        Stop loss distance in ATR multiples.
    max_sl_pips : float
        Maximum allowed SL in pips (safety cap).

    Returns
    -------
    list[Signal]
        Signals with ATR and EMA values in meta dict.
    """
    pip_size = 0.01 if "JPY" in pair else 0.0001

    # Detect BOS using existing SMC infrastructure (skip OB for speed)
    primitives = detect_primitives(
        ohlc, swing_length=swing_length, compute_ob=False,
    )
    bos_choch = primitives["bos_choch"]
    df = primitives["ohlc"]

    # Compute EMA and ATR
    ema = df["close"].ewm(span=ema_period, adjust=False).mean()
    atr = _compute_atr(df, period=atr_period)

    # HTF bias (optional — used as bonus metadata, not a gate)
    htf_bias = 0
    if htf_ohlc is not None:
        htf_prims = detect_primitives(
            htf_ohlc, swing_length=swing_length, compute_ob=False,
        )
        htf_bos = htf_prims["bos_choch"]
        # Last BOS direction on HTF
        htf_bos_vals = htf_bos["BOS"].dropna()
        htf_bos_vals = htf_bos_vals[htf_bos_vals != 0]
        if len(htf_bos_vals) > 0:
            htf_bias = int(htf_bos_vals.iloc[-1])

    signals: list[Signal] = []
    min_bar = max(swing_length, ema_period, atr_period) + 1

    for i in range(min_bar, len(df)):
        bos_val = bos_choch.iloc[i]["BOS"]

        # Only trigger on BOS (not CHoCH — we want trend continuation)
        if np.isnan(bos_val) or bos_val == 0:
            continue

        direction = int(bos_val)  # 1 = bullish, -1 = bearish
        close_price = df.iloc[i]["close"]
        current_ema = ema.iloc[i]
        current_atr = atr.iloc[i]

        if np.isnan(current_atr) or current_atr <= 0:
            continue

        # --- Filter 1: EMA trend alignment ---
        if direction == 1 and close_price <= current_ema:
            continue
        if direction == -1 and close_price >= current_ema:
            continue

        # --- Filter 2: Not overextended (within 3x ATR of EMA) ---
        distance_to_ema = abs(close_price - current_ema)
        if distance_to_ema > 3.0 * current_atr:
            continue

        # --- Entry: next bar's open ---
        if i + 1 >= len(df):
            continue
        entry_price = df.iloc[i + 1]["open"]

        # --- Stop loss: ATR-based ---
        sl_distance = atr_sl_mult * current_atr
        sl_pips = sl_distance / pip_size
        if sl_pips > max_sl_pips or sl_pips <= 0:
            continue

        if direction == 1:
            stop_loss = entry_price - sl_distance
        else:
            stop_loss = entry_price + sl_distance

        # --- Take profit: 10x ATR placeholder (trailing stop handles real exit) ---
        tp_distance = 10.0 * current_atr
        if direction == 1:
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # HTF alignment check (metadata, not a gate)
        htf_aligned = (htf_bias == direction) if htf_bias != 0 else False

        # Confluence is minimal — just BOS + EMA alignment
        confluences = {
            "structure_break": "BOS",
            "ema_aligned": True,
        }
        if htf_aligned:
            confluences["htf_aligned"] = True

        confluence_score = 2 + (1 if htf_aligned else 0)

        signal = Signal(
            timestamp=df.index[i + 1],  # Entry at next bar
            pair=pair,
            direction="long" if direction == 1 else "short",
            entry_price=round(entry_price, 5),
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            rr_ratio=round(rr_ratio, 2),
            confluence_score=confluence_score,
            confluences=confluences,
            kill_zone=None,
            meta={
                "bar_index": i + 1,
                "break_idx": i,
                "atr": current_atr,
                "ema": current_ema,
                "distance_to_ema_atr": round(distance_to_ema / current_atr, 2),
                "htf_aligned": htf_aligned,
                "tp_method": "trailing",
            },
        )
        signals.append(signal)

    logger.info(
        f"Adaptive: Generated {len(signals)} signals from {len(df)} candles "
        f"(swing={swing_length}, ema={ema_period}, atr={atr_period})"
    )
    return signals
