"""
ICT Kill Zone time filters.

All times are Eastern Time (ET) as per ICT methodology.
Kill zones define when institutional activity is highest.
"""
from __future__ import annotations

import pandas as pd


# Kill zone definitions in ET (Eastern Time)
KILL_ZONES = {
    "asian": {"start": "19:00", "end": "22:00"},       # 7:00-10:00 PM ET
    "london": {"start": "02:00", "end": "05:00"},       # 2:00-5:00 AM ET
    "new_york": {"start": "07:00", "end": "10:00"},     # 7:00-10:00 AM ET
    "london_close": {"start": "10:00", "end": "12:00"}, # 10:00 AM-12:00 PM ET
}


def get_kill_zone(timestamp: pd.Timestamp) -> str | None:
    """
    Return the active kill zone for a given timestamp, or None if outside all zones.

    Expects UTC timestamp — converts to ET internally.
    """
    # Convert UTC to ET
    et_time = timestamp.tz_convert("US/Eastern").time()

    for zone_name, zone in KILL_ZONES.items():
        start = pd.Timestamp(zone["start"]).time()
        end = pd.Timestamp(zone["end"]).time()

        if start <= end:
            if start <= et_time < end:
                return zone_name
        else:
            # Handles overnight zones (asian wraps midnight)
            if et_time >= start or et_time < end:
                return zone_name

    return None


def mark_kill_zones(df: pd.DataFrame) -> pd.Series:
    """
    Add kill zone labels to a DataFrame with a UTC DatetimeIndex.

    Returns a Series with kill zone names or None.
    """
    return pd.Series(
        [get_kill_zone(ts) for ts in df.index],
        index=df.index,
        name="kill_zone",
    )


def in_kill_zone(df: pd.DataFrame, zones: list[str] | None = None) -> pd.Series:
    """
    Return boolean mask for candles within specified kill zones.

    If zones is None, returns True for any active kill zone.
    """
    kz = mark_kill_zones(df)
    if zones is None:
        return kz.notna()
    return kz.isin(zones)
