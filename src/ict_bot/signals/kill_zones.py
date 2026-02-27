"""
ICT Kill Zone time filters.

All times are Eastern Time (ET) as per ICT methodology.
Kill zones define when institutional activity is highest.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd

ET_TZ = ZoneInfo("America/New_York")


# Kill zone definitions in ET (Eastern Time)
KILL_ZONES = {
    "asian": {"start": "19:00", "end": "02:00"},       # 7:00 PM - 2:00 AM ET (overnight)
    "london": {"start": "02:00", "end": "05:00"},       # 2:00-5:00 AM ET
    "new_york": {"start": "07:00", "end": "10:00"},     # 7:00-10:00 AM ET
    "london_close": {"start": "10:00", "end": "12:00"}, # 10:00 AM-12:00 PM ET
}


def is_asian_session(utc_time: datetime | None = None) -> bool:
    """Check if the given UTC time falls within the Asian kill zone (19:00-02:00 ET).

    Uses wall-clock time as defense-in-depth against mislabeled signals.
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    et_time = utc_time.astimezone(ET_TZ).time()
    # Asian session: 19:00 ET to 02:00 ET (overnight, wraps midnight)
    return et_time >= time(19, 0) or et_time < time(2, 0)


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
