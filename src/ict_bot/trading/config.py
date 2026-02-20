"""Live trading configuration and pair mapping."""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Pair name mapping: internal EUR_USD ↔ IBKR EURUSD ─────────────────
PAIR_TO_IB: dict[str, str] = {
    "EUR_USD": "EURUSD",
    "GBP_USD": "GBPUSD",
    "USD_JPY": "USDJPY",
    "AUD_USD": "AUDUSD",
    "USD_CAD": "USDCAD",
    "NZD_USD": "NZDUSD",
    "EUR_GBP": "EURGBP",
}
IB_TO_PAIR: dict[str, str] = {v: k for k, v in PAIR_TO_IB.items()}

# CFD symbol mapping: internal pair -> (symbol, currency)
# CFDs use the base currency as symbol and quote currency as currency
PAIR_TO_CFD: dict[str, tuple[str, str]] = {
    "EUR_USD": ("EUR", "USD"),
    "GBP_USD": ("GBP", "USD"),
    "USD_JPY": ("USD", "JPY"),
    "AUD_USD": ("AUD", "USD"),
    "USD_CAD": ("USD", "CAD"),
    "NZD_USD": ("NZD", "USD"),
    "EUR_GBP": ("EUR", "GBP"),
}

# Pip sizes per pair
PIP_SIZES: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "USD_JPY": 0.01,
    "AUD_USD": 0.0001,
    "USD_CAD": 0.0001,
    "NZD_USD": 0.0001,
    "EUR_GBP": 0.0001,
}


def pip_size_for(pair: str) -> float:
    """Return pip size for a pair, defaulting to 0.0001."""
    return PIP_SIZES.get(pair, 0.0001)


@dataclass
class LiveConfig:
    """Configuration for live trading via Interactive Brokers."""

    pairs: list[str] = field(default_factory=lambda: list(PAIR_TO_IB.keys()))

    # IB Gateway connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002              # 4002=paper, 4001=live
    ib_client_id: int = 1

    # Risk
    risk_per_trade: float = 0.01     # 1% per trade
    starting_balance: float = 5_000.0  # USD fallback (paper acct ~50K SEK ≈ $5K)

    # Timeframe — determines bar subscription size
    timeframe: str = "M15"           # "M15", "H1", "M5", etc.

    # Signal config (M15 sw=5 t=4 — best risk-adjusted return)
    swing_length: int = 5
    confluence_threshold: int = 4
    min_rr: float = 2.0
    sl_buffer_pips: float = 10.0
    skip_days: list[int] = field(default_factory=list)  # empty = trade all days
    fvg_lookback: int = 16
    pullback_window: int = 40
    compute_ob: bool = False
    use_displacement: bool = False

    # Trade management
    use_breakeven: bool = False
    be_threshold_r: float = 1.5
    use_partial_tp: bool = False
    max_sl_pips: float = 100.0

    # Safety
    max_positions: int = 3
    heartbeat_interval: int = 60     # seconds
    state_file: str = "data/live_state.json"
    dry_run: bool = False            # simulate orders, use real prices

    # News filter
    news_filter_enabled: bool = True
    news_blackout_minutes: int = 30          # ±30 min around high-impact events
    news_close_before_events: bool = False   # close positions before major events

    @property
    def pip_size(self) -> float:
        """Default pip size (use pip_size_for() for per-pair)."""
        return 0.0001

    @property
    def bar_size(self) -> str:
        """IB bar size string for the configured timeframe."""
        mapping = {
            "M1": "1 min", "M5": "5 mins", "M15": "15 mins",
            "M30": "30 mins", "H1": "1 hour", "H4": "4 hours",
            "D": "1 day",
        }
        return mapping.get(self.timeframe, "15 mins")

    @property
    def bar_duration(self) -> str:
        """IB duration string for initial historical bar request."""
        mapping = {
            "M1": "5 D", "M5": "10 D", "M15": "30 D",
            "M30": "30 D", "H1": "30 D", "H4": "60 D",
            "D": "365 D",
        }
        return mapping.get(self.timeframe, "30 D")
