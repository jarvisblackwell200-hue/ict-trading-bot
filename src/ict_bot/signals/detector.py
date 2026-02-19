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


_primitives_cache: dict[tuple, dict] = {}


def detect_primitives(
    ohlc: pd.DataFrame,
    swing_length: int = 50,
    compute_liquidity_targets: bool = False,
    compute_ob: bool = True,
) -> dict[str, pd.DataFrame]:
    cache_key = (id(ohlc), len(ohlc), swing_length, compute_liquidity_targets, compute_ob)
    if cache_key in _primitives_cache:
        return _primitives_cache[cache_key]

    df = ohlc.copy()
    df.columns = [c.lower() for c in df.columns]

    swing_hl = smc.swing_highs_lows(df, swing_length=swing_length)
    bos_choch = smc.bos_choch(df, swing_hl, close_break=True)
    fvg = smc.fvg(df, join_consecutive=False)
    if compute_ob:
        ob = smc.ob(df, swing_hl, close_mitigation=False)
    else:
        ob = pd.DataFrame(0, index=df.index, columns=["OB", "Top", "Bottom", "OBVolume", "Percentage"])
    liquidity = smc.liquidity(df, swing_hl, range_percent=0.01)
    kill_zones = mark_kill_zones(df)

    pdh_pdl = None
    pwh_pwl = None
    if compute_liquidity_targets:
        pdh_pdl = smc.previous_high_low(df, time_frame="1D")
        pwh_pwl = smc.previous_high_low(df, time_frame="1W")

    result = {
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
    _primitives_cache[cache_key] = result
    return result


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


def _detect_liquidity_sweep(
    primitives: dict,
    break_idx: int,
    direction: int,
    lookback: int = 20,
) -> bool:
    """
    Check if there was a liquidity sweep before the structure break.

    A sweep occurs when price wicks beyond a prior swing high/low but closes
    back inside — indicating institutional stop hunting before the real move.

    For bullish BOS: look for a sweep of swing lows (false breakdown) before break up.
    For bearish BOS: look for a sweep of swing highs (false breakout) before break down.
    """
    swing_hl = primitives["swing_hl"]
    ohlc = primitives["ohlc"]

    start = max(0, break_idx - lookback)
    window = swing_hl.iloc[start:break_idx]

    if direction == 1:  # Bullish — look for prior sweep of lows
        swing_lows = window[window["HighLow"] == -1]["Level"].dropna()
        if len(swing_lows) == 0:
            return False
        for sl_level in swing_lows:
            for k in range(max(start, int(swing_lows.index[0]) if isinstance(swing_lows.index[0], (int, np.integer)) else start), break_idx):
                if k >= len(ohlc):
                    break
                candle = ohlc.iloc[k]
                # Wick below swing low but close above it = sweep
                if candle["low"] < sl_level and candle["close"] > sl_level:
                    return True
    else:  # Bearish — look for prior sweep of highs
        swing_highs = window[window["HighLow"] == 1]["Level"].dropna()
        if len(swing_highs) == 0:
            return False
        for sh_level in swing_highs:
            for k in range(start, break_idx):
                if k >= len(ohlc):
                    break
                candle = ohlc.iloc[k]
                # Wick above swing high but close below it = sweep
                if candle["high"] > sh_level and candle["close"] < sh_level:
                    return True

    return False


def _find_ifvg_zones(
    primitives: dict,
    break_idx: int,
    direction: int,
    lookback: int = 50,
) -> list[dict]:
    """
    Find Inverse FVGs — FVGs in the opposite direction that were subsequently
    invalidated (price closed through them), which now act as zones in the
    current direction.

    For a long setup: find bearish FVGs that price later closed above → bullish IFVG.
    For a short setup: find bullish FVGs that price later closed below → bearish IFVG.
    """
    fvg = primitives["fvg"]
    ohlc = primitives["ohlc"]
    zones = []

    start = max(0, break_idx - lookback)
    window = fvg.iloc[start:break_idx]

    # Look for opposite-direction FVGs that were invalidated
    opposite = -direction
    matching = window[window["FVG"] == opposite]

    for idx_label, row in matching.iterrows():
        top = row.get("Top", np.nan)
        bottom = row.get("Bottom", np.nan)
        if np.isnan(top) or np.isnan(bottom):
            continue

        # Find the positional index for this FVG
        try:
            fvg_pos = ohlc.index.get_loc(idx_label)
        except KeyError:
            continue

        # Check if this FVG was invalidated between its formation and the break
        invalidated = False
        for k in range(fvg_pos + 1, break_idx):
            candle = ohlc.iloc[k]
            if opposite == -1:  # Was a bearish FVG → invalidated if close > top
                if candle["close"] > top:
                    invalidated = True
                    break
            else:  # Was a bullish FVG → invalidated if close < bottom
                if candle["close"] < bottom:
                    invalidated = True
                    break

        if invalidated:
            zones.append({"type": "IFVG", "top": top, "bottom": bottom})

    return zones


def _find_breaker_blocks(
    primitives: dict,
    break_idx: int,
    direction: int,
    lookback: int = 50,
) -> list[dict]:
    """
    Find breaker blocks — structural zones (swing highs/lows that held as S/R)
    that were subsequently broken through and now flip role.

    A breaker block forms when a prior swing high/low that was respected as S/R
    gets decisively broken. The zone around that level becomes an entry zone
    in the new direction.

    For longs: find prior resistance levels (swing highs) that were broken → now support.
    For shorts: find prior support levels (swing lows) that were broken → now resistance.
    """
    swing_hl = primitives["swing_hl"]
    ohlc = primitives["ohlc"]
    zones = []

    start = max(0, break_idx - lookback)
    window = swing_hl.iloc[start:break_idx]

    if direction == 1:  # Long — find broken resistance (swing highs that became support)
        swing_highs = window[window["HighLow"] == 1]
        for idx_label, row in swing_highs.iterrows():
            level = row.get("Level", np.nan)
            if np.isnan(level):
                continue
            try:
                sh_pos = ohlc.index.get_loc(idx_label)
            except KeyError:
                continue

            # Check: was this high first respected (price failed to close above),
            # then later broken (price closed above)?
            respected = False
            broken = False
            for k in range(sh_pos + 1, break_idx):
                candle = ohlc.iloc[k]
                if not respected and candle["high"] >= level * 0.999 and candle["close"] < level:
                    respected = True
                if respected and candle["close"] > level:
                    broken = True
                    break

            if respected and broken:
                # Create a zone around the level (±0.3× recent ATR or small fixed buffer)
                buffer = (ohlc.iloc[max(0, sh_pos-5):sh_pos+1]["high"] -
                          ohlc.iloc[max(0, sh_pos-5):sh_pos+1]["low"]).mean() * 0.5
                zones.append({
                    "type": "BREAKER",
                    "top": level + buffer * 0.3,
                    "bottom": level - buffer * 0.7,
                })

    else:  # Short — find broken support (swing lows that became resistance)
        swing_lows = window[window["HighLow"] == -1]
        for idx_label, row in swing_lows.iterrows():
            level = row.get("Level", np.nan)
            if np.isnan(level):
                continue
            try:
                sl_pos = ohlc.index.get_loc(idx_label)
            except KeyError:
                continue

            respected = False
            broken = False
            for k in range(sl_pos + 1, break_idx):
                candle = ohlc.iloc[k]
                if not respected and candle["low"] <= level * 1.001 and candle["close"] > level:
                    respected = True
                if respected and candle["close"] < level:
                    broken = True
                    break

            if respected and broken:
                buffer = (ohlc.iloc[max(0, sl_pos-5):sl_pos+1]["high"] -
                          ohlc.iloc[max(0, sl_pos-5):sl_pos+1]["low"]).mean() * 0.5
                zones.append({
                    "type": "BREAKER",
                    "top": level + buffer * 0.7,
                    "bottom": level - buffer * 0.3,
                })

    return zones


def _find_entry_zones(
    primitives: dict,
    break_idx: int,
    direction: int,
    fvg_lookback: int = 8,
    use_ifvg: bool = False,
    use_breaker_blocks: bool = False,
) -> list[dict]:
    """Find FVG, OB, IFVG, and Breaker Block zones near the structure break for pullback entry."""
    fvg = primitives["fvg"]
    ob = primitives["ob"]
    zones = []

    # FVGs within lookback bars of break
    fvg_window = fvg.iloc[max(0, break_idx - fvg_lookback):break_idx + 1]
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

    # IFVG zones (invalidated FVGs that flipped role)
    if use_ifvg:
        ifvg_zones = _find_ifvg_zones(primitives, break_idx, direction)
        zones.extend(ifvg_zones)

    # Breaker blocks (structural zones that flipped role)
    if use_breaker_blocks:
        breaker_zones = _find_breaker_blocks(primitives, break_idx, direction)
        zones.extend(breaker_zones)

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


def _validate_displacement(
    primitives: dict,
    break_idx: int,
    direction: int,
    lookback: int = 5,
    min_body_ratio: float = 0.6,
    min_body_candles: int = 2,
) -> bool:
    """
    Validate that a structure break was preceded by displacement.

    Displacement = consecutive large-bodied candles showing institutional intent.
    Checks the N candles before the break for:
      1. Body-to-range ratio >= min_body_ratio on at least min_body_candles
      2. Majority of candles close in the break direction
      3. Total displacement move >= 1.5x average candle range
    """
    ohlc = primitives["ohlc"]
    start = max(0, break_idx - lookback)
    window = ohlc.iloc[start:break_idx]

    if len(window) < 2:
        return False

    # Average candle range for context
    context = ohlc.iloc[max(0, break_idx - 20):break_idx]
    avg_range = (context["high"] - context["low"]).mean()
    if avg_range <= 0:
        return True  # can't validate, pass through

    body_count = 0
    directional_count = 0
    total_displacement = 0.0

    for _, candle in window.iterrows():
        body = abs(candle["close"] - candle["open"])
        full_range = candle["high"] - candle["low"]

        if full_range <= 0:
            continue

        if body / full_range >= min_body_ratio:
            body_count += 1

        if direction == 1 and candle["close"] > candle["open"]:
            directional_count += 1
            total_displacement += body
        elif direction == -1 and candle["close"] < candle["open"]:
            directional_count += 1
            total_displacement += body

    return (body_count >= min_body_candles and
            directional_count >= len(window) / 2 and
            total_displacement >= avg_range * 1.5)


def _find_liquidity_target(
    primitives: dict,
    idx: int,
    direction: int,
    entry_price: float,
    sl_distance: float,
    min_rr: float,
    pdh_pdl: pd.DataFrame | None = None,
    pwh_pwl: pd.DataFrame | None = None,
    min_target_rr: float = 1.0,
) -> tuple[float | None, str | None]:
    """
    Find the nearest valid liquidity target for take-profit placement.

    Gathers candidates from 4 sources (equal highs/lows, PDH/PDL, PWH/PWL,
    swing highs/lows), filters by direction and min_target_rr floor, returns
    the nearest. Uses min_target_rr (default 1.0R) instead of min_rr so that
    closer, higher-probability liquidity pools can be targeted.
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
        if sl_distance > 0 and abs(price - entry_price) / sl_distance >= min_target_rr
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

    Factors (max 8):
      1. Entry in FVG or OB zone (+1, always true)
      2. Within a kill zone (+1)
      3. Structure break confirmation (+1, always true)
      4. Entry price in OTE zone 0.62-0.79 retracement (+1)
      5. FVG stacking — multiple FVGs at this level (+1)
      6. FVG + OB overlap at the entry level (+1)
      7. Premium/Discount zone (+1)
      8. HTF bias alignment (+1)
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

    # 8. HTF bias alignment (numbered 8 but checked early for visibility)
    if setup.get("htf_aligned", False):
        score += 1
        factors["htf_aligned"] = True

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
    skip_days: list[int] | None = None,
    use_displacement: bool = True,
    fvg_lookback: int = 8,
    compute_ob: bool = True,
    use_sweep_filter: bool = False,
    use_ifvg: bool = False,
    use_breaker_blocks: bool = False,
    use_ce_entry: bool = False,
) -> list[Signal]:
    """
    Generate ICT trade signals using a two-phase pullback entry model (v3).

    Phase 1: Detect structure breaks (BOS/CHoCH)
    Phase 2: Wait for price to pull back into an FVG/OB zone, enter at zone edge
    """
    pip_size = 0.0001 if "JPY" not in pair else 0.01

    primitives = detect_primitives(ohlc, swing_length=swing_length,
                                    compute_liquidity_targets=use_liquidity_targets,
                                    compute_ob=compute_ob)

    htf_bos_choch = None
    htf_index = None
    if htf_ohlc is not None:
        htf_primitives = detect_primitives(htf_ohlc, swing_length=swing_length,
                                            compute_ob=compute_ob)
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
            # Validate displacement before the break (reject weak/wick-driven breaks)
            if use_displacement and not _validate_displacement(primitives, i, direction):
                continue

            # Liquidity sweep filter: require sweep of prior S/R before BOS
            if use_sweep_filter and not _detect_liquidity_sweep(primitives, i, direction):
                continue

            # HTF bias: check alignment (used as confluence score, not hard gate)
            htf_aligned = False
            if htf_bos_choch is not None:
                htf_bias = _get_htf_bias_at(htf_bos_choch, htf_index, current_time)
                htf_aligned = (htf_bias == direction)

            # Find entry zones
            entry_zones = _find_entry_zones(
                primitives, i, direction, fvg_lookback=fvg_lookback,
                use_ifvg=use_ifvg, use_breaker_blocks=use_breaker_blocks,
            )
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
                "htf_aligned": htf_aligned,
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

            # Day of week filter — keep pending but don't enter on skip days
            if skip_days is not None and current_time.dayofweek in skip_days:
                still_pending.append(setup)
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

            # --- Entry pricing ---
            # CE (Consequent Encroachment): enter at 50% midpoint of zone
            # Default: enter at 25% from zone edge (closer to favorable side)
            entry_pct = 0.5 if use_ce_entry else 0.25
            if d == 1:
                # Buy near the bottom of the zone (better entry)
                entry_price = entry_zone_used["bottom"] + (entry_zone_used["top"] - entry_zone_used["bottom"]) * entry_pct
                sl_distance = entry_price - sl
            else:
                # Sell near the top of the zone
                entry_price = entry_zone_used["top"] - (entry_zone_used["top"] - entry_zone_used["bottom"]) * entry_pct
                sl_distance = sl - entry_price

            if sl_distance <= 0 or sl_distance > 300 * pip_size:
                continue

            # Default R:R TP
            tp_distance = sl_distance * min_rr
            take_profit = entry_price + tp_distance if d == 1 else entry_price - tp_distance
            liq_target_type = None

            # Override with nearest liquidity target (min 1R floor)
            if use_liquidity_targets:
                liq_price, liq_type = _find_liquidity_target(
                    primitives, i, d, entry_price, sl_distance, min_rr,
                    pdh_pdl=primitives.get("pdh_pdl"),
                    pwh_pwl=primitives.get("pwh_pwl"),
                    min_target_rr=1.0,
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


def generate_signals_mtf(
    structure_ohlc: pd.DataFrame,
    entry_ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame | None = None,
    pair: str = "EUR_USD",
    structure_swing_length: int = 20,
    entry_swing_length: int = 10,
    confluence_threshold: int = 3,
    min_rr: float = 2.0,
    sl_buffer_pips: float = 3.0,
    pullback_window: int = 80,
    use_liquidity_targets: bool = True,
    use_premium_discount: bool = True,
    use_displacement: bool = True,
    skip_days: list[int] | None = None,
    fvg_lookback: int = 32,
    compute_ob: bool = True,
    use_structure_sl: bool = False,
) -> list[Signal]:
    """
    Multi-timeframe signal generation: structure TF for BOS, entry TF for zones/SL.

    Uses higher TF (e.g. H1) for structure break detection and lower TF (e.g. M15/M5)
    for FVG/OB entry zones. SL can come from either the entry TF (tighter) or
    the structure TF (wider, more room to breathe).

    Parameters
    ----------
    structure_ohlc : H1 (or H4) OHLC for BOS/CHoCH detection
    entry_ohlc : M15 (or M5) OHLC for entry zones
    htf_ohlc : Daily OHLC for HTF bias
    use_structure_sl : if True, use structure TF (H1) for SL instead of entry TF (M5)
    """
    pip_size = 0.0001 if "JPY" not in pair else 0.01

    # Detect structure on higher TF
    h_prims = detect_primitives(structure_ohlc, swing_length=structure_swing_length,
                                 compute_liquidity_targets=use_liquidity_targets,
                                 compute_ob=compute_ob)
    h_bos = h_prims["bos_choch"]

    # Detect entry zones on lower TF
    e_prims = detect_primitives(entry_ohlc, swing_length=entry_swing_length,
                                 compute_ob=compute_ob)
    e_df = e_prims["ohlc"]
    e_kz = e_prims["kill_zones"]

    # HTF bias
    htf_bos_choch = None
    htf_index = None
    if htf_ohlc is not None:
        htf_prims = detect_primitives(htf_ohlc, swing_length=structure_swing_length,
                                       compute_ob=compute_ob)
        htf_bos_choch = htf_prims["bos_choch"]
        htf_index = htf_ohlc.index

    # Collect structure break events from higher TF
    bos_events = []
    for i in range(structure_swing_length, len(structure_ohlc)):
        bos_val = h_bos.iloc[i]["BOS"]
        choch_val = h_bos.iloc[i]["CHOCH"]
        d = 0
        trigger = None
        if not np.isnan(bos_val) and bos_val != 0:
            d = int(bos_val)
            trigger = "BOS"
        elif not np.isnan(choch_val) and choch_val != 0:
            d = int(choch_val)
            trigger = "CHoCH"
        if d == 0:
            continue

        if use_displacement and not _validate_displacement(h_prims, i, d):
            continue

        # HTF bias
        htf_aligned = False
        if htf_bos_choch is not None:
            htf_bias = _get_htf_bias_at(htf_bos_choch, htf_index, structure_ohlc.index[i])
            htf_aligned = (htf_bias == d)

        # OTE from structure TF
        ote_top, ote_bottom, fib_705 = _compute_ote_zone(h_prims, i, d)

        bos_events.append({
            "time": structure_ohlc.index[i],
            "direction": d,
            "trigger": trigger,
            "h_idx": i,
            "htf_aligned": htf_aligned,
            "ote_top": ote_top,
            "ote_bottom": ote_bottom,
        })

    logger.info(f"MTF: {len(bos_events)} structure breaks from {len(structure_ohlc)} bars")

    signals = []
    for event in bos_events:
        d = event["direction"]
        bos_time = event["time"]

        # Map to entry TF index
        e_idx = e_df.index.searchsorted(bos_time)
        if e_idx >= len(e_df):
            continue

        # Find entry zones on entry TF near the break
        entry_zones = _find_entry_zones(e_prims, e_idx, d, fvg_lookback=fvg_lookback)
        if not entry_zones:
            continue

        # SL: from structure TF (wider, safer) or entry TF (tighter)
        if use_structure_sl:
            sl = _find_structural_sl(h_prims, event["h_idx"], d, pip_size, buffer_pips=sl_buffer_pips)
        else:
            sl = _find_structural_sl(e_prims, e_idx, d, pip_size, buffer_pips=sl_buffer_pips)

        # Walk entry TF bars for pullback
        for j in range(e_idx + 1, min(e_idx + pullback_window, len(e_df))):
            e_candle = e_df.iloc[j]
            current_time = e_df.index[j]
            e_high = e_candle["high"]
            e_low = e_candle["low"]

            # SL invalidation
            if d == 1 and e_low <= sl:
                break
            if d == -1 and e_high >= sl:
                break

            # Day filter
            if skip_days is not None and current_time.dayofweek in skip_days:
                continue

            # Check zone interaction
            entered = False
            entry_zone_used = None
            for zone in entry_zones:
                if _price_in_zone(e_low, e_high, zone["bottom"], zone["top"], d):
                    entered = True
                    entry_zone_used = zone
                    break

            if not entered:
                continue

            # Entry at zone edge
            if d == 1:
                entry_price = entry_zone_used["bottom"] + (entry_zone_used["top"] - entry_zone_used["bottom"]) * 0.25
                sl_distance = entry_price - sl
            else:
                entry_price = entry_zone_used["top"] - (entry_zone_used["top"] - entry_zone_used["bottom"]) * 0.25
                sl_distance = sl - entry_price

            if sl_distance <= 0 or sl_distance > 300 * pip_size:
                continue

            # TP from liquidity targets (use structure TF primitives)
            tp_distance = sl_distance * min_rr
            take_profit = entry_price + tp_distance if d == 1 else entry_price - tp_distance
            liq_target_type = None

            if use_liquidity_targets:
                liq_price, liq_type = _find_liquidity_target(
                    h_prims, event["h_idx"], d, entry_price, sl_distance, min_rr,
                    pdh_pdl=h_prims.get("pdh_pdl"),
                    pwh_pwl=h_prims.get("pwh_pwl"),
                    min_target_rr=1.0,
                )
                if liq_price is not None:
                    take_profit = liq_price
                    tp_distance = abs(liq_price - entry_price)
                    liq_target_type = liq_type

            rr_ratio = tp_distance / sl_distance

            # Kill zone at entry time
            kz_val = e_kz.iloc[j]
            current_kz = kz_val if not (isinstance(kz_val, float) and np.isnan(kz_val)) else None

            # Score confluence
            setup = {
                "trigger": event["trigger"],
                "entry_zones": entry_zones,
                "ote_top": event["ote_top"],
                "ote_bottom": event["ote_bottom"],
                "break_idx": event["h_idx"],
                "htf_aligned": event["htf_aligned"],
            }
            score, factors = _score_entry(
                j, d, e_prims, setup, entry_zone_used, current_kz, entry_price,
                use_premium_discount=use_premium_discount,
            )

            if score < confluence_threshold:
                continue

            factors["trigger"] = event["trigger"]
            factors["entry_zone"] = entry_zone_used["type"]
            factors["entry_tf"] = "LTF"

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
                    "bar_index": j,
                    "break_idx": event["h_idx"],
                    "tp_method": liq_target_type or "rr_based",
                    "liq_target_type": liq_target_type,
                    "entry_tf": "LTF",
                },
            )
            signals.append(signal)
            break  # Setup consumed, move to next BOS event

    logger.info(f"MTF: Generated {len(signals)} signals (threshold={confluence_threshold})")
    return signals
