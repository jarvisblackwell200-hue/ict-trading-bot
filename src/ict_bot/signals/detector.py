"""
ICT Signal Detection Engine (v2 — pullback entry model).

Changes from v1:
  1. Two-phase entry: detect structure break, then WAIT for price to
     pull back into an FVG or OB zone before entering. This is core ICT
     methodology — never chase the displacement candle.
  2. HTF bias is a hard filter, not a bonus point. ICT never trades
     against higher-timeframe structure.
  3. SL placed at the swing that caused the structure break (structural SL),
     not a fixed buffer from an arbitrary OB level.

These are methodology corrections, not parameter optimizations.
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
    df = ohlc.copy()
    df.columns = [c.lower() for c in df.columns]

    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    bos_choch = smc.bos_choch(df, swing_hl, close_break=True)
    fvg = smc.fvg(df, join_consecutive=False)
    ob = smc.ob(df, swing_hl, close_mitigation=False)
    liquidity = smc.liquidity(df, swing_hl, range_percent=0.01)
    retracements = smc.retracements(df, swing_hl)
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


def _get_htf_bias_at(htf_bos_choch: pd.DataFrame, htf_index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    """
    Get the HTF directional bias that was active at a specific timestamp.

    Looks at the most recent HTF BOS before this timestamp.
    Returns 1 (bullish), -1 (bearish), or 0 (neutral).
    """
    # Find most recent HTF candle before this timestamp
    mask = htf_index <= timestamp
    if not mask.any():
        return 0

    # Get all BOS values up to this point
    relevant = htf_bos_choch.loc[mask]
    bos_signals = relevant[relevant["BOS"] != 0]["BOS"].dropna()

    if len(bos_signals) == 0:
        return 0

    return int(bos_signals.iloc[-1])


def _find_structural_sl(
    primitives: dict,
    break_idx: int,
    direction: int,
    pip_size: float,
    buffer_pips: float = 5.0,
) -> float:
    """
    Find the structural stop loss — the swing point that would invalidate
    the trade thesis if broken.

    For a bullish trade: SL below the most recent swing low before the break.
    For a bearish trade: SL above the most recent swing high before the break.
    """
    swing_hl = primitives["swing_hl"]
    ohlc = primitives["ohlc"]
    buffer = buffer_pips * pip_size

    # Look back from the structure break for the relevant swing
    window = swing_hl.iloc[max(0, break_idx - 50):break_idx + 1]

    if direction == 1:  # bullish — find the swing low the break came from
        lows = window[window["HighLow"] == -1]["Level"].dropna()
        if len(lows) > 0:
            return lows.iloc[-1] - buffer
    else:  # bearish — find the swing high
        highs = window[window["HighLow"] == 1]["Level"].dropna()
        if len(highs) > 0:
            return highs.iloc[-1] + buffer

    # Absolute fallback: use recent candle extreme
    recent = ohlc.iloc[max(0, break_idx - 20):break_idx + 1]
    if direction == 1:
        return recent["low"].min() - buffer
    else:
        return recent["high"].max() + buffer


def _find_entry_zone(
    primitives: dict,
    break_idx: int,
    direction: int,
) -> list[dict]:
    """
    Find FVG and OB zones created by or near the structure break that
    price should pull back into for entry.

    Returns list of zones: [{type, top, bottom, idx}]
    """
    fvg = primitives["fvg"]
    ob = primitives["ob"]
    zones = []

    # Look for FVGs near the break (created by the displacement)
    fvg_window = fvg.iloc[max(0, break_idx - 5):break_idx + 1]
    matching_fvg = fvg_window[fvg_window["FVG"] == direction]
    for idx_label, row in matching_fvg.iterrows():
        if not np.isnan(row.get("Top", np.nan)) and not np.isnan(row.get("Bottom", np.nan)):
            zones.append({
                "type": "FVG",
                "top": row["Top"],
                "bottom": row["Bottom"],
            })

    # Look for OBs near the break
    ob_window = ob.iloc[max(0, break_idx - 10):break_idx + 1]
    matching_ob = ob_window[ob_window["OB"] == direction]
    for idx_label, row in matching_ob.iterrows():
        if not np.isnan(row.get("Top", np.nan)) and not np.isnan(row.get("Bottom", np.nan)):
            zones.append({
                "type": "OB",
                "top": row["Top"],
                "bottom": row["Bottom"],
            })

    return zones


def generate_signals(
    ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    pair: str = "EUR_USD",
    swing_length: int = 50,
    confluence_threshold: int = 3,
    min_rr: float = 2.0,
    sl_buffer_pips: float = 5.0,
    target_kill_zones: list[str] | None = None,
    pullback_window: int = 20,
    require_htf_bias: bool = True,
) -> list[Signal]:
    """
    Generate ICT trade signals using a two-phase pullback entry model.

    Phase 1: Detect structure breaks (BOS/CHoCH) during kill zones
    Phase 2: Wait for price to pull back into an FVG or OB zone, then enter

    This prevents chasing displacement candles and gives much better
    entry prices with tighter, structural stop losses.
    """
    pip_size = 0.0001 if "JPY" not in pair else 0.01

    primitives = detect_primitives(ohlc, swing_length=swing_length)

    # Prepare HTF bias lookup
    htf_primitives = None
    htf_bos_choch = None
    htf_index = None
    if htf_ohlc is not None:
        htf_primitives = detect_primitives(htf_ohlc, swing_length=swing_length)
        htf_bos_choch = htf_primitives["bos_choch"]
        htf_index = htf_ohlc.index

    bos_choch = primitives["bos_choch"]
    df = primitives["ohlc"]
    kill_zones = primitives["kill_zones"]
    signals = []

    # Track pending setups (structure breaks waiting for pullback)
    pending_setups = []

    for i in range(swing_length, len(df)):
        current_price = df.iloc[i]["close"]
        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]
        current_time = df.index[i]
        current_kz = kill_zones.iloc[i] if not (isinstance(kill_zones.iloc[i], float) and np.isnan(kill_zones.iloc[i])) else None

        # --- Phase 1: Detect new structure breaks ---
        bos_val = bos_choch.iloc[i]["BOS"]
        choch_val = bos_choch.iloc[i]["CHOCH"]

        direction = 0
        trigger_type = None
        if not np.isnan(bos_val) and bos_val != 0:
            direction = int(bos_val)
            trigger_type = "BOS"
        elif not np.isnan(choch_val) and choch_val != 0:
            direction = int(choch_val)
            trigger_type = "CHoCH"

        if direction != 0:
            # Check HTF bias alignment (hard filter)
            if require_htf_bias and htf_bos_choch is not None:
                htf_bias = _get_htf_bias_at(htf_bos_choch, htf_index, current_time)
                if htf_bias != direction:
                    continue  # Skip — trading against HTF structure

            # Find entry zones (FVGs/OBs from the displacement)
            entry_zones = _find_entry_zone(primitives, i, direction)
            if not entry_zones:
                continue  # No pullback zone — can't define entry

            # Find structural SL
            structural_sl = _find_structural_sl(
                primitives, i, direction, pip_size, buffer_pips=sl_buffer_pips
            )

            # Register pending setup
            pending_setups.append({
                "break_idx": i,
                "break_time": current_time,
                "direction": direction,
                "trigger": trigger_type,
                "entry_zones": entry_zones,
                "structural_sl": structural_sl,
                "break_price": current_price,
                "bars_waiting": 0,
            })

        # --- Phase 2: Check if price pulls back into any pending zone ---
        still_pending = []
        for setup in pending_setups:
            setup["bars_waiting"] += 1

            # Expire setups that are too old
            if setup["bars_waiting"] > pullback_window:
                continue

            d = setup["direction"]
            sl = setup["structural_sl"]

            # Check if SL was hit while waiting (setup invalidated)
            if d == 1 and current_low <= sl:
                continue  # Invalidated
            if d == -1 and current_high >= sl:
                continue  # Invalidated

            # Check if price pulled back into any entry zone
            entered = False
            entry_zone_used = None

            for zone in setup["entry_zones"]:
                zone_top = zone["top"]
                zone_bottom = zone["bottom"]

                # For bullish: price must dip DOWN into the zone
                if d == 1 and current_low <= zone_top and current_close_or_low_in_zone(current_low, current_price, zone_bottom, zone_top, d):
                    entered = True
                    entry_zone_used = zone
                    break

                # For bearish: price must push UP into the zone
                if d == -1 and current_high >= zone_bottom and current_close_or_high_in_zone(current_high, current_price, zone_bottom, zone_top, d):
                    entered = True
                    entry_zone_used = zone
                    break

            if not entered:
                still_pending.append(setup)
                continue

            # --- Entry triggered! ---
            # Entry at the zone midpoint (conservative; in live trading you'd use limit order)
            zone_mid = (entry_zone_used["top"] + entry_zone_used["bottom"]) / 2
            if d == 1:
                entry_price = zone_mid  # Buy in the zone
                sl_distance = entry_price - sl
            else:
                entry_price = zone_mid  # Sell in the zone
                sl_distance = sl - entry_price

            if sl_distance <= 0 or sl_distance > 300 * pip_size:
                # SL too tight or too wide — skip
                continue

            # TP at min R:R
            tp_distance = sl_distance * min_rr
            if d == 1:
                take_profit = entry_price + tp_distance
            else:
                take_profit = entry_price - tp_distance

            rr_ratio = tp_distance / sl_distance

            # Score confluence at the entry bar
            score, factors = _score_entry_confluence(
                i, d, primitives, setup, entry_zone_used, current_kz,
            )

            if score < confluence_threshold:
                still_pending.append(setup)
                continue

            factors["trigger"] = setup["trigger"]
            factors["entry_zone"] = entry_zone_used["type"]
            factors["bars_to_entry"] = setup["bars_waiting"]

            signal = Signal(
                timestamp=current_time,
                pair=pair,
                direction="long" if d == 1 else "short",
                entry_price=round(entry_price, 5),
                stop_loss=round(sl, 5),
                take_profit=round(take_profit, 5),
                rr_ratio=round(rr_ratio, 2),
                confluence_score=min(score, 5),
                confluences=factors,
                kill_zone=current_kz,
                meta={
                    "bar_index": i,
                    "break_idx": setup["break_idx"],
                    "structural_sl": round(sl, 5),
                },
            )
            signals.append(signal)
            # Setup consumed — don't add back to pending

        pending_setups = still_pending

    logger.info(f"Generated {len(signals)} signals from {len(df)} candles (threshold={confluence_threshold})")
    return signals


def current_close_or_low_in_zone(low: float, close: float, zone_bottom: float, zone_top: float, direction: int) -> bool:
    """Check if price action interacted with the zone."""
    if direction == 1:
        # Bullish: candle low dipped into the zone (between bottom and top)
        return low <= zone_top and low >= zone_bottom * 0.999
    else:
        return False


def current_close_or_high_in_zone(high: float, close: float, zone_bottom: float, zone_top: float, direction: int) -> bool:
    """Check if price action interacted with the zone."""
    if direction == -1:
        # Bearish: candle high pushed into the zone
        return high >= zone_bottom and high <= zone_top * 1.001
    else:
        return False


def _score_entry_confluence(
    idx: int,
    direction: int,
    primitives: dict,
    setup: dict,
    entry_zone: dict,
    current_kz: str | None,
) -> tuple[int, dict]:
    """
    Score confluence at the pullback entry point.

    Factors:
      1. Entry in FVG or OB zone (already guaranteed — 1 point)
      2. Within a kill zone
      3. Structure break confirmation (BOS or CHoCH triggered the setup)
      4. Price in OTE zone (0.62–0.79 retracement)
      5. Multiple zone overlap (FVG + OB both present)
    """
    score = 0
    factors = {}

    # 1. Entry zone (always true since we entered a zone)
    score += 1
    factors["entry_zone_type"] = entry_zone["type"]

    # 2. Kill zone
    if current_kz is not None:
        score += 1
        factors["kill_zone"] = current_kz

    # 3. Structure break (always true — setup was triggered by BOS/CHoCH)
    score += 1
    factors["structure_break"] = setup["trigger"]

    # 4. OTE zone — check if entry price is within 62-79% retracement
    retracements = primitives["retracements"]
    ret = retracements.iloc[idx]
    if "CurrentRetracement%" in ret.index:
        current_ret = ret["CurrentRetracement%"]
        if not np.isnan(current_ret) and 0.62 <= current_ret <= 0.79:
            score += 1
            factors["ote"] = round(current_ret, 3)

    # 5. Multiple zones overlapping (FVG + OB at this level)
    if len(setup["entry_zones"]) >= 2:
        zone_types = set(z["type"] for z in setup["entry_zones"])
        if len(zone_types) >= 2:
            score += 1
            factors["zone_overlap"] = True

    return score, factors
