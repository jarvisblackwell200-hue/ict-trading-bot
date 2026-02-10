"""
ICT Signal Detection Engine.

Wraps the smartmoneyconcepts library to detect ICT primitives
and combines them into tradeable signals with confluence scoring.

Pure function: DataFrame in -> list of Signal dicts out.
No broker dependency — works with any OHLC data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from smartmoneyconcepts import smc

from .kill_zones import get_kill_zone, mark_kill_zones

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A detected ICT trade signal."""
    timestamp: pd.Timestamp
    pair: str
    direction: str           # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confluence_score: int    # 0-5
    confluences: dict        # which factors are present
    kill_zone: Optional[str]
    meta: dict = field(default_factory=dict)


def detect_primitives(
    ohlc: pd.DataFrame,
    swing_length: int = 50,
) -> dict[str, pd.DataFrame]:
    """
    Run all ICT primitive detections on an OHLC DataFrame.

    Returns a dict of detection results keyed by primitive name.
    """
    # Ensure columns are lowercase
    df = ohlc.copy()
    df.columns = [c.lower() for c in df.columns]

    # 1. Swing highs and lows (foundation for everything else)
    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)

    # 2. Break of Structure / Change of Character
    bos_choch = smc.bos_choch(df, swing_hl, close_break=True)

    # 3. Fair Value Gaps
    fvg = smc.fvg(df, join_consecutive=False)

    # 4. Order Blocks
    ob = smc.ob(df, swing_hl, close_mitigation=False)

    # 5. Liquidity levels
    liquidity = smc.liquidity(df, swing_hl, range_percent=0.01)

    # 6. Retracements (for OTE zone detection)
    retracements = smc.retracements(df, swing_hl)

    # 7. Kill zones
    kill_zones = mark_kill_zones(df)

    return {
        "swing_hl": swing_hl,
        "bos_choch": bos_choch,
        "fvg": fvg,
        "ob": ob,
        "liquidity": liquidity,
        "retracements": retracements,
        "kill_zones": kill_zones,
        "ohlc": df,
    }


def score_confluence(
    idx: int,
    direction: int,  # 1 = bullish, -1 = bearish
    primitives: dict[str, pd.DataFrame],
    lookback: int = 10,
) -> tuple[int, dict]:
    """
    Score confluence at a given bar index.

    Checks how many ICT factors align for the given direction.
    Returns (score 0-5, dict of which factors are present).
    """
    fvg = primitives["fvg"]
    ob = primitives["ob"]
    bos_choch = primitives["bos_choch"]
    retracements = primitives["retracements"]
    kill_zones = primitives["kill_zones"]

    score = 0
    factors = {}

    # 1. FVG present in lookback window, matching direction
    fvg_window = fvg.iloc[max(0, idx - lookback):idx + 1]
    has_fvg = (fvg_window["FVG"] == direction).any()
    if has_fvg:
        score += 1
        factors["fvg"] = True

    # 2. Order Block present in lookback window, matching direction
    ob_window = ob.iloc[max(0, idx - lookback):idx + 1]
    has_ob = (ob_window["OB"] == direction).any()
    if has_ob:
        score += 1
        factors["ob"] = True

    # 3. Within a kill zone
    kz = kill_zones.iloc[idx]
    if kz is not None and not (isinstance(kz, float) and np.isnan(kz)):
        score += 1
        factors["kill_zone"] = kz

    # 4. MSS/BOS confirmation in lookback window
    bos_window = bos_choch.iloc[max(0, idx - lookback):idx + 1]
    has_bos = (bos_window["BOS"] == direction).any()
    has_choch = (bos_window["CHOCH"] == direction).any()
    if has_bos or has_choch:
        score += 1
        factors["structure_break"] = "BOS" if has_bos else "CHoCH"

    # 5. In OTE zone (retracement between 62% and 79%)
    ret = retracements.iloc[idx]
    if "CurrentRetracement%" in ret.index:
        current_ret = ret["CurrentRetracement%"]
        if not np.isnan(current_ret):
            in_ote = 0.62 <= current_ret <= 0.79
            if in_ote:
                score += 1
                factors["ote"] = round(current_ret, 3)

    return score, factors


def _find_nearest_level(primitives: dict, idx: int, direction: int, level_type: str) -> float:
    """Find nearest OB or FVG level for SL/TP placement."""
    if level_type == "ob":
        data = primitives["ob"]
        col_dir = "OB"
    else:
        data = primitives["fvg"]
        col_dir = "FVG"

    window = data.iloc[max(0, idx - 20):idx + 1]
    matching = window[window[col_dir] == direction]

    if matching.empty:
        return np.nan

    last = matching.iloc[-1]
    if direction == 1:  # bullish — SL below bottom
        return last["Bottom"]
    else:  # bearish — SL above top
        return last["Top"]


def generate_signals(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    pair: str = "EUR_USD",
    swing_length: int = 50,
    confluence_threshold: int = 3,
    min_rr: float = 2.0,
    sl_buffer_pips: float = 10.0,
    target_kill_zones: list[str] | None = None,
) -> list[Signal]:
    """
    Generate ICT trade signals from OHLC data.

    This is the main entry point. It:
    1. Detects all ICT primitives
    2. Looks for MSS/CHoCH confirmations
    3. Scores confluence at each confirmation
    4. Emits signals that meet the threshold

    Args:
        ohlc: OHLC DataFrame (5M or 1H candles) with UTC DatetimeIndex
        htf_ohlc: Higher timeframe OHLC for bias (Daily/4H). If provided,
                   adds HTF bias to confluence scoring.
        pair: Currency pair name
        swing_length: Lookback for swing detection
        confluence_threshold: Minimum score to emit signal (0-5)
        min_rr: Minimum risk:reward ratio
        sl_buffer_pips: Buffer pips beyond structural SL level
        target_kill_zones: Only generate signals in these kill zones (None = all)

    Returns:
        List of Signal objects, sorted by timestamp
    """
    pip_size = 0.0001 if "JPY" not in pair else 0.01
    sl_buffer = sl_buffer_pips * pip_size

    # Detect primitives on entry timeframe
    primitives = detect_primitives(ohlc, swing_length=swing_length)

    # Detect HTF bias if provided
    htf_bias = 0  # 0 = neutral, 1 = bullish, -1 = bearish
    if htf_ohlc is not None:
        htf_primitives = detect_primitives(htf_ohlc, swing_length=swing_length)
        htf_bos = htf_primitives["bos_choch"]
        # Use most recent HTF structure break as bias
        recent_bos = htf_bos[htf_bos["BOS"] != 0]["BOS"].dropna()
        if len(recent_bos) > 0:
            htf_bias = int(recent_bos.iloc[-1])

    bos_choch = primitives["bos_choch"]
    df = primitives["ohlc"]
    signals = []

    # Scan for structure breaks as entry triggers
    for i in range(swing_length, len(df)):
        bos_val = bos_choch.iloc[i]["BOS"]
        choch_val = bos_choch.iloc[i]["CHOCH"]

        # Determine direction from structure break
        direction = 0
        trigger_type = None
        if bos_val != 0 and not np.isnan(bos_val):
            direction = int(bos_val)
            trigger_type = "BOS"
        elif choch_val != 0 and not np.isnan(choch_val):
            direction = int(choch_val)
            trigger_type = "CHoCH"

        if direction == 0:
            continue

        # Check kill zone filter
        kz = get_kill_zone(df.index[i])
        if target_kill_zones and kz not in target_kill_zones:
            continue

        # Score confluence
        score, factors = score_confluence(i, direction, primitives)

        # Add HTF bias to score if it aligns
        if htf_bias == direction:
            score += 1
            factors["htf_bias"] = "bullish" if direction == 1 else "bearish"

        # Cap at 5 for display consistency, but allow > 5 internally
        factors["trigger"] = trigger_type

        if score < confluence_threshold:
            continue

        # Calculate entry, SL, TP
        entry_price = df.iloc[i]["close"]

        # SL: beyond the nearest OB or structural level + buffer
        ob_level = _find_nearest_level(primitives, i, direction, "ob")
        if np.isnan(ob_level):
            # Fallback: use recent swing
            swing_hl = primitives["swing_hl"]
            recent_swings = swing_hl.iloc[max(0, i - 30):i + 1]
            if direction == 1:
                lows = recent_swings[recent_swings["HighLow"] == -1]["Level"].dropna()
                ob_level = lows.iloc[-1] if len(lows) > 0 else entry_price - 30 * pip_size
            else:
                highs = recent_swings[recent_swings["HighLow"] == 1]["Level"].dropna()
                ob_level = highs.iloc[-1] if len(highs) > 0 else entry_price + 30 * pip_size

        if direction == 1:
            stop_loss = ob_level - sl_buffer
            sl_distance = entry_price - stop_loss
        else:
            stop_loss = ob_level + sl_buffer
            sl_distance = stop_loss - entry_price

        if sl_distance <= 0:
            continue

        # TP: minimum R:R
        tp_distance = sl_distance * min_rr
        take_profit = entry_price + tp_distance if direction == 1 else entry_price - tp_distance

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        if rr_ratio < min_rr:
            continue

        signal = Signal(
            timestamp=df.index[i],
            pair=pair,
            direction="long" if direction == 1 else "short",
            entry_price=round(entry_price, 5),
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            rr_ratio=round(rr_ratio, 2),
            confluence_score=min(score, 5),
            confluences=factors,
            kill_zone=kz,
            meta={
                "bar_index": i,
                "htf_bias": "bullish" if htf_bias == 1 else "bearish" if htf_bias == -1 else "neutral",
            },
        )
        signals.append(signal)

    logger.info(f"Generated {len(signals)} signals from {len(df)} candles (threshold={confluence_threshold})")
    return signals
