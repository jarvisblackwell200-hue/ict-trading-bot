"""
ICT Signal Detection Engine (v3 — fixed confluence scoring + entry precision).

Fixes from v2:
  1. Custom OTE calculation (library's retracements function returns 0s)
  2. Entry at zone edge instead of midpoint (better R:R)
  3. Functional confluence scoring (v2 always scored 3 because OTE and OB were broken)
  4. FVG stacking detection (multiple untested FVGs at the same level = stronger zone)
  5. Exclude Asian session (data shows -0.37R, consistent with ICT guidance)
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
    confluence_score: int    # 0-6
    confluences: dict        # which factors are present
    kill_zone: Optional[str]
    meta: dict = field(default_factory=dict)


def detect_primitives(
    ohlc: pd.DataFrame,
    swing_length: int = 50,
    compute_liquidity_targets: bool = False,
) -> dict[str, pd.DataFrame]:
    df = ohlc.copy()
    df.columns = [c.lower() for c in df.columns]

    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    bos_choch = smc.bos_choch(df, swing_hl, close_break=True)
    fvg = smc.fvg(df, join_consecutive=False)
    ob = smc.ob(df, swing_hl, close_mitigation=False)
    liquidity = smc.liquidity(df, swing_hl, range_percent=0.01)
    kill_zones = mark_kill_zones(df)

    pdh_pdl = None
    pwh_pwl = None
    if compute_liquidity_targets:
        pdh_pdl = smc.previous_high_low(df, time_frame="1D")
        pwh_pwl = smc.previous_high_low(df, time_frame="1W")

    return {
        "swing_hl": swing_hl,
        "bos_choch": bos_choch,
        "fvg": fvg,
        "ob": ob,
        "liquidity": liquidity,
        "kill_zones": kill_zones,
        "ohlc": df,
        "pdh_pdl": pdh_pdl,
        "pwh_pwl": pwh_pwl,
    }


def _get_htf_bias_at(htf_bos_choch: pd.DataFrame, htf_index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    mask = htf_index <= timestamp
    if not mask.any():
        return 0
    relevant = htf_bos_choch.loc[mask]
    bos_signals = relevant[relevant["BOS"] != 0]["BOS"].dropna()
    if len(bos_signals) == 0:
        return 0
    return int(bos_signals.iloc[-1])


def _find_structural_sl(primitives: dict, break_idx: int, direction: int, pip_size: float, buffer_pips: float = 5.0) -> float:
    swing_hl = primitives["swing_hl"]
    ohlc = primitives["ohlc"]
    buffer = buffer_pips * pip_size

    window = swing_hl.iloc[max(0, break_idx - 50):break_idx + 1]
    if direction == 1:
        lows = window[window["HighLow"] == -1]["Level"].dropna()
        if len(lows) > 0:
            return lows.iloc[-1] - buffer
    else:
        highs = window[window["HighLow"] == 1]["Level"].dropna()
        if len(highs) > 0:
            return highs.iloc[-1] + buffer

    recent = ohlc.iloc[max(0, break_idx - 20):break_idx + 1]
    if direction == 1:
        return recent["low"].min() - buffer
    else:
        return recent["high"].max() + buffer


def _compute_ote_zone(primitives: dict, break_idx: int, direction: int) -> tuple[float, float, float]:
    """
    Compute the OTE (Optimal Trade Entry) zone for a structure break.

    Finds the swing high and swing low that define the move, then
    calculates the 0.62 and 0.79 Fibonacci retracement levels.

    Returns (ote_top, ote_bottom, fib_705) or (nan, nan, nan) if can't compute.
    """
    swing_hl = primitives["swing_hl"]
    ohlc = primitives["ohlc"]

    window = swing_hl.iloc[max(0, break_idx - 50):break_idx + 1]

    if direction == 1:  # bullish — swing from low to high, OTE is retracement down
        lows = window[window["HighLow"] == -1]["Level"].dropna()
        highs = window[window["HighLow"] == 1]["Level"].dropna()
        if len(lows) == 0 or len(highs) == 0:
            return np.nan, np.nan, np.nan
        swing_low = lows.iloc[-1]
        # The high should be at or near the break point
        swing_high = ohlc.iloc[break_idx]["high"]
        move = swing_high - swing_low
        if move <= 0:
            return np.nan, np.nan, np.nan
        # OTE zone: 62-79% retracement from the high
        ote_top = swing_high - 0.62 * move     # shallower
        ote_bottom = swing_high - 0.79 * move  # deeper
        fib_705 = swing_high - 0.705 * move
        return ote_top, ote_bottom, fib_705
    else:  # bearish — swing from high to low, OTE is retracement up
        highs = window[window["HighLow"] == 1]["Level"].dropna()
        lows = window[window["HighLow"] == -1]["Level"].dropna()
        if len(highs) == 0 or len(lows) == 0:
            return np.nan, np.nan, np.nan
        swing_high = highs.iloc[-1]
        swing_low = ohlc.iloc[break_idx]["low"]
        move = swing_high - swing_low
        if move <= 0:
            return np.nan, np.nan, np.nan
        ote_top = swing_low + 0.79 * move     # deeper retracement
        ote_bottom = swing_low + 0.62 * move  # shallower
        fib_705 = swing_low + 0.705 * move
        return ote_top, ote_bottom, fib_705


def _find_entry_zones(primitives: dict, break_idx: int, direction: int) -> list[dict]:
    """Find FVG and OB zones near the structure break for pullback entry."""
    fvg = primitives["fvg"]
    ob = primitives["ob"]
    zones = []

    # FVGs within 8 bars of break
    fvg_window = fvg.iloc[max(0, break_idx - 8):break_idx + 1]
    matching_fvg = fvg_window[fvg_window["FVG"] == direction]
    for _, row in matching_fvg.iterrows():
        top = row.get("Top", np.nan)
        bottom = row.get("Bottom", np.nan)
        if not np.isnan(top) and not np.isnan(bottom):
            zones.append({"type": "FVG", "top": top, "bottom": bottom})

    # OBs within 30 bars (they're rarer so wider window)
    ob_window = ob.iloc[max(0, break_idx - 30):break_idx + 1]
    matching_ob = ob_window[ob_window["OB"] == direction]
    for _, row in matching_ob.iterrows():
        top = row.get("Top", np.nan)
        bottom = row.get("Bottom", np.nan)
        if not np.isnan(top) and not np.isnan(bottom):
            zones.append({"type": "OB", "top": top, "bottom": bottom})

    return zones


def _price_in_zone(low: float, high: float, zone_bottom: float, zone_top: float, direction: int) -> bool:
    """Check if candle interacted with a zone."""
    if direction == 1:
        # Bullish: candle low dipped into zone
        return low <= zone_top and low >= zone_bottom * 0.998
    else:
        # Bearish: candle high pushed into zone
        return high >= zone_bottom and high <= zone_top * 1.002


def _count_fvg_stack(primitives: dict, idx: int, direction: int, zone_top: float, zone_bottom: float) -> int:
    """Count how many FVGs overlap with this zone (FVG stacking = stronger level)."""
    fvg = primitives["fvg"]
    window = fvg.iloc[max(0, idx - 30):idx + 1]
    matching = window[window["FVG"] == direction]
    count = 0
    for _, row in matching.iterrows():
        t = row.get("Top", np.nan)
        b = row.get("Bottom", np.nan)
        if np.isnan(t) or np.isnan(b):
            continue
        # Check overlap
        if max(b, zone_bottom) < min(t, zone_top):
            count += 1
    return count


def _find_liquidity_target(
    primitives: dict,
    idx: int,
    direction: int,
    entry_price: float,
    sl_distance: float,
    min_rr: float,
    pdh_pdl: pd.DataFrame | None = None,
    pwh_pwl: pd.DataFrame | None = None,
) -> tuple[float | None, str | None]:
    """
    Find the nearest valid liquidity target for take-profit placement.

    Gathers candidates from 4 sources (equal highs/lows, PDH/PDL, PWH/PWL,
    swing highs/lows), filters by direction and min R:R, returns the nearest.
    """
    candidates: list[tuple[float, str]] = []

    # 1. Equal highs/lows from liquidity pools
    liq = primitives["liquidity"]
    for j in range(len(liq)):
        liq_val = liq.iloc[j].get("Liquidity", np.nan)
        if np.isnan(liq_val):
            continue
        level = liq.iloc[j]["Level"]
        end = int(liq.iloc[j]["End"])
        swept = liq.iloc[j]["Swept"]
        # Pool must have formed by now and not be swept
        if end > idx:
            continue
        if swept != 0:
            continue
        if direction == 1 and liq_val == 1 and level > entry_price:
            candidates.append((level, "equal_highs"))
        elif direction == -1 and liq_val == -1 and level < entry_price:
            candidates.append((level, "equal_lows"))

    # 2. Previous Day High/Low
    if pdh_pdl is not None and idx < len(pdh_pdl):
        row = pdh_pdl.iloc[idx]
        if direction == 1:
            pdh = row.get("PreviousHigh", np.nan)
            broken = row.get("BrokenHigh", 0)
            if not np.isnan(pdh) and broken == 0 and pdh > entry_price:
                candidates.append((float(pdh), "PDH"))
        else:
            pdl = row.get("PreviousLow", np.nan)
            broken = row.get("BrokenLow", 0)
            if not np.isnan(pdl) and broken == 0 and pdl < entry_price:
                candidates.append((float(pdl), "PDL"))

    # 3. Previous Week High/Low
    if pwh_pwl is not None and idx < len(pwh_pwl):
        row = pwh_pwl.iloc[idx]
        if direction == 1:
            pwh = row.get("PreviousHigh", np.nan)
            broken = row.get("BrokenHigh", 0)
            if not np.isnan(pwh) and broken == 0 and pwh > entry_price:
                candidates.append((float(pwh), "PWH"))
        else:
            pwl = row.get("PreviousLow", np.nan)
            broken = row.get("BrokenLow", 0)
            if not np.isnan(pwl) and broken == 0 and pwl < entry_price:
                candidates.append((float(pwl), "PWL"))

    # 4. Swing highs/lows as fallback targets
    swing_hl = primitives["swing_hl"]
    start = max(0, idx - 100)
    window = swing_hl.iloc[start:idx + 1]
    if direction == 1:
        swings = window[window["HighLow"] == 1]["Level"].dropna()
        for level in swings:
            if level > entry_price:
                candidates.append((level, "swing_high"))
    else:
        swings = window[window["HighLow"] == -1]["Level"].dropna()
        for level in swings:
            if level < entry_price:
                candidates.append((level, "swing_low"))

    # Filter: each candidate must meet min R:R
    valid = [
        (price, label)
        for price, label in candidates
        if sl_distance > 0 and abs(price - entry_price) / sl_distance >= min_rr
    ]

    if not valid:
        return None, None

    # Sort by distance from entry (ascending) and return nearest
    valid.sort(key=lambda x: abs(x[0] - entry_price))
    return valid[0]


def _get_dealing_range(primitives: dict, idx: int, lookback: int = 100) -> tuple[float, float, float]:
    """
    Compute the dealing range from recent swing structure.

    Returns (range_high, range_low, equilibrium) where equilibrium is the
    midpoint. Returns (nan, nan, nan) if insufficient swing data.
    """
    swing_hl = primitives["swing_hl"]
    start = max(0, idx - lookback)
    window = swing_hl.iloc[start:idx + 1]

    highs = window[window["HighLow"] == 1]["Level"].dropna()
    lows = window[window["HighLow"] == -1]["Level"].dropna()

    if len(highs) == 0 or len(lows) == 0:
        return np.nan, np.nan, np.nan

    range_high = highs.max()
    range_low = lows.min()
    equilibrium = (range_high + range_low) / 2.0
    return range_high, range_low, equilibrium


def _is_premium_discount_valid(entry_price: float, direction: int, equilibrium: float) -> bool:
    """
    Check if entry is in the correct premium/discount zone.

    Longs should enter in discount (below equilibrium).
    Shorts should enter in premium (above equilibrium).
    """
    if direction == 1:
        return entry_price < equilibrium
    else:
        return entry_price > equilibrium


def _score_entry(
    idx: int,
    direction: int,
    primitives: dict,
    setup: dict,
    entry_zone: dict,
    current_kz: str | None,
    entry_price: float,
    use_premium_discount: bool = False,
) -> tuple[int, dict]:
    """
    Score confluence at the pullback entry point.

    Factors (max 7):
      1. Entry in FVG or OB zone (+1, always true)
      2. Within a kill zone (+1)
      3. Structure break confirmation (+1, always true)
      4. Entry price in OTE zone 0.62-0.79 retracement (+1)
      5. FVG stacking — multiple FVGs at this level (+1)
      6. FVG + OB overlap at the entry level (+1)
      7. Premium/Discount zone (+1)
    """
    score = 0
    factors = {}

    # 1. Entry zone
    score += 1
    factors["entry_zone_type"] = entry_zone["type"]

    # 2. Kill zone
    if current_kz is not None:
        score += 1
        factors["kill_zone"] = current_kz

    # 3. Structure break
    score += 1
    factors["structure_break"] = setup["trigger"]

    # 4. OTE zone
    ote_top = setup.get("ote_top")
    ote_bottom = setup.get("ote_bottom")
    if ote_top is not None and ote_bottom is not None and not np.isnan(ote_top):
        if direction == 1:
            in_ote = ote_bottom <= entry_price <= ote_top
        else:
            in_ote = ote_bottom <= entry_price <= ote_top
        if in_ote:
            score += 1
            factors["ote"] = True

    # 5. FVG stacking
    stack_count = _count_fvg_stack(primitives, idx, direction, entry_zone["top"], entry_zone["bottom"])
    if stack_count >= 2:
        score += 1
        factors["fvg_stack"] = stack_count

    # 6. FVG + OB overlap
    zone_types = set(z["type"] for z in setup["entry_zones"])
    if len(zone_types) >= 2:
        score += 1
        factors["zone_overlap"] = True

    # 7. Premium/Discount zone
    if use_premium_discount:
        range_high, range_low, equilibrium = _get_dealing_range(primitives, idx)
        if not np.isnan(equilibrium):
            if _is_premium_discount_valid(entry_price, direction, equilibrium):
                score += 1
                factors["premium_discount"] = "discount" if direction == 1 else "premium"

    return score, factors


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
    use_liquidity_targets: bool = True,
    use_premium_discount: bool = True,
) -> list[Signal]:
    """
    Generate ICT trade signals using a two-phase pullback entry model (v3).

    Phase 1: Detect structure breaks (BOS/CHoCH)
    Phase 2: Wait for price to pull back into an FVG/OB zone, enter at zone edge
    """
    pip_size = 0.0001 if "JPY" not in pair else 0.01

    primitives = detect_primitives(ohlc, swing_length=swing_length,
                                    compute_liquidity_targets=use_liquidity_targets)

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
    pending_setups = []

    for i in range(swing_length, len(df)):
        current_high = df.iloc[i]["high"]
        current_low = df.iloc[i]["low"]
        current_time = df.index[i]
        kz_val = kill_zones.iloc[i]
        current_kz = kz_val if not (isinstance(kz_val, float) and np.isnan(kz_val)) else None

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
            # HTF bias hard filter
            if require_htf_bias and htf_bos_choch is not None:
                htf_bias = _get_htf_bias_at(htf_bos_choch, htf_index, current_time)
                if htf_bias != direction:
                    continue

            # Find entry zones
            entry_zones = _find_entry_zones(primitives, i, direction)
            if not entry_zones:
                continue

            # Find structural SL
            structural_sl = _find_structural_sl(primitives, i, direction, pip_size, buffer_pips=sl_buffer_pips)

            # Compute OTE zone
            ote_top, ote_bottom, fib_705 = _compute_ote_zone(primitives, i, direction)

            pending_setups.append({
                "break_idx": i,
                "break_time": current_time,
                "direction": direction,
                "trigger": trigger_type,
                "entry_zones": entry_zones,
                "structural_sl": structural_sl,
                "ote_top": ote_top,
                "ote_bottom": ote_bottom,
                "fib_705": fib_705,
                "bars_waiting": 0,
            })

        # --- Phase 2: Check pullback entries ---
        still_pending = []
        for setup in pending_setups:
            setup["bars_waiting"] += 1

            if setup["bars_waiting"] > pullback_window:
                continue

            d = setup["direction"]
            sl = setup["structural_sl"]

            # Invalidated if SL hit while waiting
            if d == 1 and current_low <= sl:
                continue
            if d == -1 and current_high >= sl:
                continue

            # Check zone interaction
            entered = False
            entry_zone_used = None

            for zone in setup["entry_zones"]:
                if _price_in_zone(current_low, current_high, zone["bottom"], zone["top"], d):
                    entered = True
                    entry_zone_used = zone
                    break

            if not entered:
                still_pending.append(setup)
                continue

            # --- Entry at zone edge (not midpoint) ---
            if d == 1:
                # Buy near the bottom of the zone (better entry)
                entry_price = entry_zone_used["bottom"] + (entry_zone_used["top"] - entry_zone_used["bottom"]) * 0.25
                sl_distance = entry_price - sl
            else:
                # Sell near the top of the zone
                entry_price = entry_zone_used["top"] - (entry_zone_used["top"] - entry_zone_used["bottom"]) * 0.25
                sl_distance = sl - entry_price

            if sl_distance <= 0 or sl_distance > 300 * pip_size:
                continue

            # Default R:R TP
            tp_distance = sl_distance * min_rr
            take_profit = entry_price + tp_distance if d == 1 else entry_price - tp_distance
            liq_target_type = None

            # Override with liquidity target if valid
            if use_liquidity_targets:
                liq_price, liq_type = _find_liquidity_target(
                    primitives, i, d, entry_price, sl_distance, min_rr,
                    pdh_pdl=primitives.get("pdh_pdl"),
                    pwh_pwl=primitives.get("pwh_pwl"),
                )
                if liq_price is not None:
                    take_profit = liq_price
                    tp_distance = abs(liq_price - entry_price)
                    liq_target_type = liq_type

            rr_ratio = tp_distance / sl_distance

            # Kill zone filter at entry time (not at break time)
            if target_kill_zones and current_kz not in target_kill_zones:
                still_pending.append(setup)
                continue

            # Score confluence
            score, factors = _score_entry(
                i, d, primitives, setup, entry_zone_used, current_kz, entry_price,
                use_premium_discount=use_premium_discount,
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
                confluence_score=score,
                confluences=factors,
                kill_zone=current_kz,
                meta={
                    "bar_index": i,
                    "break_idx": setup["break_idx"],
                    "tp_method": liq_target_type or "rr_based",
                    "liq_target_type": liq_target_type,
                },
            )
            signals.append(signal)
            # Setup consumed

        pending_setups = still_pending

    logger.info(f"Generated {len(signals)} signals from {len(df)} candles (threshold={confluence_threshold})")
    return signals
