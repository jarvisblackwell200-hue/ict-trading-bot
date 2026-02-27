"""Economic calendar news filter — blocks trading around high-impact events.

Data source: ForexFactory calendar via faireconomy.media JSON API.
Fail-open design: network errors never block trading.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Which pairs are affected by each currency's news
CURRENCY_TO_PAIRS: dict[str, list[str]] = {
    "USD": ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "NZD_USD"],
    "EUR": ["EUR_USD", "EUR_GBP"],
    "GBP": ["GBP_USD", "EUR_GBP"],
    "JPY": ["USD_JPY"],
    "AUD": ["AUD_USD"],
    "CAD": ["USD_CAD"],
    "NZD": ["NZD_USD"],
}


@dataclass
class NewsEvent:
    """A single economic calendar event."""

    title: str
    country: str
    date: datetime  # UTC
    impact: str  # High, Medium, Low, Holiday
    forecast: Optional[str] = None
    previous: Optional[str] = None
    affected_pairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON API."""
        return {
            "title": self.title,
            "country": self.country,
            "date": self.date.isoformat(),
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
            "affected_pairs": self.affected_pairs,
        }


class NewsFilter:
    """Thread-safe economic calendar filter that blocks trading around high-impact events."""

    def __init__(
        self,
        blackout_minutes: int = 30,
        close_before_news: bool = False,
    ) -> None:
        self._blackout_minutes = blackout_minutes
        self._close_before_news = close_before_news
        self._events: list[NewsEvent] = []
        self._last_fetch: datetime | None = None
        self._last_attempt: datetime | None = None
        self._lock = threading.Lock()

    def refresh_if_needed(self) -> None:
        """Fetch calendar if cache is stale (>1 hour). Fail-open on errors."""
        with self._lock:
            now = datetime.now(timezone.utc)
            if self._last_fetch is not None and (now - self._last_fetch).total_seconds() < 3600:
                return
            # Back off 5 min after failed attempts to avoid 429 rate limits
            if self._last_attempt is not None and (now - self._last_attempt).total_seconds() < 300:
                return

        events = self._fetch_calendar()
        with self._lock:
            self._last_attempt = datetime.now(timezone.utc)
        if events is not None:
            with self._lock:
                self._events = events
                self._last_fetch = datetime.now(timezone.utc)
            logger.info("News calendar refreshed: %d events loaded", len(events))

    def is_pair_blocked(self, pair: str) -> bool:
        """True if pair is within ±blackout_minutes of a high-impact event."""
        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=self._blackout_minutes)

        with self._lock:
            for event in self._events:
                if event.impact != "High":
                    continue
                if pair not in event.affected_pairs:
                    continue
                if abs(event.date - now) <= window:
                    return True
        return False

    def get_blocked_pairs(self) -> list[str]:
        """Return all currently blocked pairs."""
        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=self._blackout_minutes)
        blocked: set[str] = set()

        with self._lock:
            for event in self._events:
                if event.impact != "High":
                    continue
                if abs(event.date - now) <= window:
                    blocked.update(event.affected_pairs)
        return sorted(blocked)

    def get_upcoming_events(self, hours_ahead: int = 48) -> list[NewsEvent]:
        """Return upcoming events within hours_ahead, sorted by date."""
        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        with self._lock:
            upcoming = [
                e for e in self._events
                if e.date >= now - timedelta(hours=1) and e.date <= cutoff
            ]
        return sorted(upcoming, key=lambda e: e.date)

    def get_pairs_to_close_before_news(
        self, open_pairs: list[str]
    ) -> list[tuple[str, NewsEvent]]:
        """If close_before_news enabled, return pairs that should be closed.

        Triggers exactly 30 minutes before high-impact events (#36).
        The caller is responsible for filtering by profitability.
        """
        if not self._close_before_news:
            return []

        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        # Close 30 min before event (with 1-min tolerance for heartbeat jitter)
        close_outer = timedelta(minutes=31)
        close_inner = timedelta(minutes=28)
        result: list[tuple[str, NewsEvent]] = []

        with self._lock:
            for event in self._events:
                if event.impact != "High":
                    continue
                time_to_event = event.date - now
                # Close if event is 28-31 minutes away
                if close_inner <= time_to_event <= close_outer:
                    for pair in open_pairs:
                        if pair in event.affected_pairs:
                            result.append((pair, event))
        return result

    def get_status_dict(self) -> dict:
        """Status for /api/status endpoint."""
        self.refresh_if_needed()
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=self._blackout_minutes)
        cutoff = now + timedelta(hours=48)

        blocked: set[str] = set()
        upcoming: list[NewsEvent] = []

        with self._lock:
            for event in self._events:
                if event.date >= now - timedelta(hours=1) and event.date <= cutoff:
                    upcoming.append(event)
                if event.impact == "High" and abs(event.date - now) <= window:
                    blocked.update(event.affected_pairs)

        upcoming.sort(key=lambda e: e.date)
        return {
            "news_blocked_pairs": sorted(blocked),
            "news_events": [e.to_dict() for e in upcoming],
        }

    def _fetch_calendar(self) -> list[NewsEvent] | None:
        """HTTP fetch of ForexFactory calendar. Returns None on error (fail-open)."""
        try:
            import requests

            resp = requests.get(CALENDAR_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._parse_events(data)
        except Exception as exc:
            logger.warning("Failed to fetch news calendar: %s", exc)
            return None

    def _parse_events(self, data: list[dict]) -> list[NewsEvent]:
        """Parse ForexFactory JSON into NewsEvent list with UTC dates."""
        events: list[NewsEvent] = []
        for item in data:
            try:
                title = item.get("title", "")
                country = item.get("country", "")
                impact = item.get("impact", "")
                forecast = item.get("forecast")
                previous = item.get("previous")

                date_str = item.get("date", "")
                if not date_str:
                    continue

                # Parse ISO 8601 date string and convert to UTC
                dt = datetime.fromisoformat(date_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)

                affected = CURRENCY_TO_PAIRS.get(country, [])

                events.append(NewsEvent(
                    title=title,
                    country=country,
                    date=dt,
                    impact=impact,
                    forecast=forecast,
                    previous=previous,
                    affected_pairs=list(affected),
                ))
            except Exception as exc:
                logger.debug("Skipping unparseable calendar event: %s", exc)
                continue

        return events
