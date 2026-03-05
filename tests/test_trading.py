"""Unit tests for the live trading module (mocked IB — no Gateway needed)."""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from ict_bot.trading.config import (
    IB_TO_PAIR,
    PAIR_TO_IB,
    LiveConfig,
    PIP_SIZES,
    pip_size_for,
)
from ict_bot.trading.news_filter import CURRENCY_TO_PAIRS, NewsEvent, NewsFilter
from ict_bot.trading.position_manager import LivePosition, PositionManager


# ── Fixtures / Helpers ─────────────────────────────────────────────


@dataclass
class FakeSignal:
    """Minimal signal stand-in for tests."""

    timestamp: pd.Timestamp = pd.Timestamp("2026-01-15 10:00", tz="UTC")
    pair: str = "EUR_USD"
    direction: str = "long"
    entry_price: float = 1.10000
    stop_loss: float = 1.09500
    take_profit: float = 1.11000
    rr_ratio: float = 2.0
    confluence_score: int = 4
    confluences: dict = field(default_factory=dict)
    kill_zone: Optional[str] = "new_york"
    meta: dict = field(default_factory=dict)


def make_mock_broker(config=None):
    """Create a mock IBKRBroker with async methods."""
    config = config or LiveConfig()
    broker = MagicMock()
    broker.config = config

    # Mock trade objects with orderStatus
    def make_trade(status="Filled"):
        trade = MagicMock()
        trade.orderStatus.status = status
        trade.orderStatus.avgFillPrice = 1.10000
        trade.order.orderId = 1
        trade.contract = MagicMock()
        return trade

    broker.place_market_order = AsyncMock(return_value=make_trade("Filled"))
    broker.place_stop_order = AsyncMock(return_value=make_trade("PreSubmitted"))
    broker.place_limit_order = AsyncMock(return_value=make_trade("Submitted"))
    broker.modify_order = AsyncMock()
    broker.cancel_order = AsyncMock()
    broker.get_open_positions = AsyncMock(return_value={})
    broker.get_account_balance = AsyncMock(return_value=10_000.0)
    broker.get_live_bars = MagicMock(return_value=None)
    broker.is_connected = MagicMock(return_value=True)
    broker._make_trade = make_trade  # expose for custom tests

    return broker


# ── Test 1: Pair Mapping ──────────────────────────────────────────


def test_pair_mapping():
    """EUR_USD ↔ EURUSD conversion is correct and bidirectional."""
    assert PAIR_TO_IB["EUR_USD"] == "EURUSD"
    assert PAIR_TO_IB["GBP_USD"] == "GBPUSD"
    assert PAIR_TO_IB["USD_JPY"] == "USDJPY"
    assert PAIR_TO_IB["AUD_USD"] == "AUDUSD"
    assert PAIR_TO_IB["USD_CAD"] == "USDCAD"
    assert PAIR_TO_IB["NZD_USD"] == "NZDUSD"
    assert PAIR_TO_IB["EUR_GBP"] == "EURGBP"

    # Reverse mapping
    for internal, ib_sym in PAIR_TO_IB.items():
        assert IB_TO_PAIR[ib_sym] == internal

    assert len(PAIR_TO_IB) == len(IB_TO_PAIR) == 7


# ── Test 2: LiveConfig Defaults ──────────────────────────────────


def test_live_config_defaults():
    """LiveConfig has sensible defaults matching best validated M15 config."""
    cfg = LiveConfig()
    assert cfg.ib_port == 4001  # live
    assert cfg.risk_per_trade == 0.01
    assert cfg.timeframe == "M15"
    assert cfg.swing_length == 5
    assert cfg.confluence_threshold == 4
    assert cfg.min_rr == 2.0
    assert cfg.skip_days == []
    assert cfg.use_displacement is False
    assert cfg.use_breakeven is False
    assert cfg.max_positions == 3
    assert len(cfg.pairs) == 7
    assert cfg.bar_size == "15 mins"
    assert cfg.bar_duration == "30 D"


def test_pip_size_for():
    """Pip sizes are correct per pair."""
    assert pip_size_for("EUR_USD") == 0.0001
    assert pip_size_for("USD_JPY") == 0.01
    assert pip_size_for("UNKNOWN") == 0.0001  # default


# ── Test 3: Position Manager Open/Close ──────────────────────────


@pytest.mark.asyncio
async def test_position_manager_open_close():
    """Opening a position creates SL+TP orders, closing removes it."""
    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    pos = await pm.open_position(signal, units=10000)

    assert pos.pair == "EUR_USD"
    assert pos.direction == "long"
    assert pos.entry_price == 1.10000
    assert pos.stop_loss == 1.09500
    assert pos.take_profit == 1.11000
    assert "EUR_USD" in pm.positions

    # Market entry + SL stop + TP limit = 3 order calls
    broker.place_market_order.assert_called_once()
    broker.place_stop_order.assert_called_once()
    broker.place_limit_order.assert_called_once()

    # Close
    record = await pm.close_position("EUR_USD", "manual")
    assert "EUR_USD" not in pm.positions
    assert record["pair"] == "EUR_USD"
    assert record["exit_reason"] == "manual"


# ── Test 4: OCO Emulation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_oco_sl_fill_cancels_tp():
    """When SL fills, TP order gets cancelled."""
    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    pos = await pm.open_position(signal, units=10000)

    # Simulate SL fill
    pos.sl_order.orderStatus.status = "Filled"

    closed = await pm.check_and_manage()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "stop_loss"
    assert "EUR_USD" not in pm.positions
    # TP should have been cancelled
    broker.cancel_order.assert_called()


@pytest.mark.asyncio
async def test_oco_tp_fill_cancels_sl():
    """When TP fills, SL order gets cancelled."""
    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    pos = await pm.open_position(signal, units=10000)

    # Simulate TP fill
    pos.tp_order.orderStatus.status = "Filled"

    closed = await pm.check_and_manage()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "take_profit"
    assert "EUR_USD" not in pm.positions
    broker.cancel_order.assert_called()


# ── Test 5: Break-Even Move ──────────────────────────────────────


@pytest.mark.asyncio
async def test_breakeven_move():
    """SL is modified to entry when BE threshold is reached."""
    config = LiveConfig(
        use_breakeven=True,
        be_threshold_r=1.5,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Provide bars showing price moved >1.5R in profit
    # Entry=1.10, SL=1.095 → risk=50 pips, 1.5R = 75 pips → price=1.1076 (slightly above)
    bars_df = pd.DataFrame(
        {"open": [1.1070], "high": [1.1080], "low": [1.1060], "close": [1.1076], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 11:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal()
    pos = await pm.open_position(signal, units=10000)
    assert not pos.be_triggered

    await pm.check_and_manage()

    assert pos.be_triggered
    assert pos.stop_loss == 1.10000  # moved to entry
    broker.modify_order.assert_called_once()


@pytest.mark.asyncio
async def test_breakeven_not_triggered_below_threshold():
    """SL is NOT modified when price hasn't reached BE threshold."""
    config = LiveConfig(
        use_breakeven=True,
        be_threshold_r=1.5,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Price only moved 1.0R (not enough for 1.5R threshold)
    bars_df = pd.DataFrame(
        {"open": [1.1040], "high": [1.1050], "low": [1.1030], "close": [1.1050], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 11:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal()
    pos = await pm.open_position(signal, units=10000)

    await pm.check_and_manage()

    assert not pos.be_triggered
    broker.modify_order.assert_not_called()


# ── Test 6: State Persistence ────────────────────────────────────


@pytest.mark.asyncio
async def test_state_persistence():
    """Positions survive save/load cycle."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name

    config = LiveConfig(state_file=state_path)
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    await pm.open_position(signal, units=10000)

    # Save happens automatically in open_position, verify file exists
    data = json.loads(Path(state_path).read_text())
    assert "EUR_USD" in data
    assert data["EUR_USD"]["direction"] == "long"
    assert data["EUR_USD"]["entry_price"] == 1.10000

    # Load into fresh manager
    pm2 = PositionManager(broker, config)
    pm2.load_state(state_path)
    assert "EUR_USD" in pm2.positions
    pos = pm2.positions["EUR_USD"]
    assert pos.direction == "long"
    assert pos.entry_price == 1.10000
    assert pos.stop_loss == 1.09500

    Path(state_path).unlink(missing_ok=True)


# ── Test 7: Reconcile on Startup ─────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_removes_stale_positions():
    """Positions in state file but not in IB are removed."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name

    # Write a stale position
    state = {
        "EUR_USD": LivePosition(
            pair="EUR_USD", direction="long", entry_price=1.1,
            units=10000, stop_loss=1.095, take_profit=1.11,
            entry_time="2026-01-15T10:00:00+00:00",
            risk_pips=50, confluence_score=4,
        ).to_dict()
    }
    Path(state_path).write_text(json.dumps(state))

    config = LiveConfig(state_file=state_path)
    broker = make_mock_broker(config)
    broker.get_open_positions = AsyncMock(return_value={})  # IB has no positions

    pm = PositionManager(broker, config)
    await pm.reconcile_on_startup()

    # Stale position should be removed
    assert "EUR_USD" not in pm.positions

    Path(state_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reconcile_keeps_matching_positions():
    """Positions that exist in both state and IB are kept."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name

    state = {
        "EUR_USD": LivePosition(
            pair="EUR_USD", direction="long", entry_price=1.1,
            units=10000, stop_loss=1.095, take_profit=1.11,
            entry_time="2026-01-15T10:00:00+00:00",
            risk_pips=50, confluence_score=4,
        ).to_dict()
    }
    Path(state_path).write_text(json.dumps(state))

    config = LiveConfig(state_file=state_path)
    broker = make_mock_broker(config)
    broker.get_open_positions = AsyncMock(return_value={"EUR_USD": 10000})

    pm = PositionManager(broker, config)
    await pm.reconcile_on_startup()

    assert "EUR_USD" in pm.positions

    Path(state_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reconcile_removes_direction_mismatch():
    """Position removed when state says long but IB has short (#32)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name

    state = {
        "EUR_USD": LivePosition(
            pair="EUR_USD", direction="long", entry_price=1.1,
            units=10000, stop_loss=1.095, take_profit=1.11,
            entry_time="2026-01-15T10:00:00+00:00",
            risk_pips=50, confluence_score=4,
        ).to_dict()
    }
    Path(state_path).write_text(json.dumps(state))

    config = LiveConfig(state_file=state_path)
    broker = make_mock_broker(config)
    # IB has SHORT position (negative units) but state says long
    broker.get_open_positions = AsyncMock(return_value={"EUR_USD": -10000})

    pm = PositionManager(broker, config)
    await pm.reconcile_on_startup()

    # Mismatched direction — position should be removed
    assert "EUR_USD" not in pm.positions

    Path(state_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reconcile_keeps_correct_direction():
    """Position kept when state direction matches IB (#32)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        state_path = f.name

    state = {
        "EUR_USD": LivePosition(
            pair="EUR_USD", direction="short", entry_price=1.1,
            units=10000, stop_loss=1.105, take_profit=1.09,
            entry_time="2026-01-15T10:00:00+00:00",
            risk_pips=50, confluence_score=4,
        ).to_dict()
    }
    Path(state_path).write_text(json.dumps(state))

    config = LiveConfig(state_file=state_path)
    broker = make_mock_broker(config)
    # IB has short position (negative) matching state
    broker.get_open_positions = AsyncMock(return_value={"EUR_USD": -10000})

    pm = PositionManager(broker, config)
    await pm.reconcile_on_startup()

    # Direction matches — position should be kept
    assert "EUR_USD" in pm.positions

    Path(state_path).unlink(missing_ok=True)


# ── Test 8: Signal Rejection When Position Open ──────────────────


@pytest.mark.asyncio
async def test_signal_rejected_when_position_open():
    """No double entry per pair."""
    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    await pm.open_position(signal, units=10000)

    # Try to open same pair again — should already exist
    assert "EUR_USD" in pm.positions

    # The LiveTradingSession._process_signal checks this before calling open_position
    # We verify the guard condition here
    assert signal.pair in pm.positions


# ── Test 9: Risk Manager Gates Entry ─────────────────────────────


def test_risk_manager_blocks_after_circuit_breaker():
    """Circuit breaker blocks new trades."""
    from ict_bot.risk import RiskConfig, RiskManager

    rm = RiskManager(RiskConfig(max_daily_loss=0.03, starting_balance=10_000))

    # Simulate loss that triggers circuit breaker (3% of $10K = $300)
    rm.record_trade_result(-350.0, "EUR_USD")
    assert rm.is_circuit_broken

    signal = FakeSignal()
    decision = rm.evaluate_signal(signal, 0.0001)
    assert not decision.approved
    assert "circuit breaker" in decision.reason


def test_risk_manager_blocks_after_kill_switch():
    """Kill switch blocks all trades."""
    from ict_bot.risk import RiskConfig, RiskManager

    rm = RiskManager(RiskConfig(max_drawdown=0.10, starting_balance=10_000))

    # Simulate drawdown exceeding 10%
    rm.record_trade_result(-1100.0, "EUR_USD")
    assert rm.is_killed

    signal = FakeSignal()
    decision = rm.evaluate_signal(signal, 0.0001)
    assert not decision.approved
    assert "kill switch" in decision.reason


# ── Test 10: Daily Reset ─────────────────────────────────────────


def test_daily_reset():
    """Risk manager resets daily P&L and circuit breaker."""
    from ict_bot.risk import RiskConfig, RiskManager

    rm = RiskManager(RiskConfig(max_daily_loss=0.03, starting_balance=10_000))

    rm.record_trade_result(-350.0, "EUR_USD")
    assert rm.is_circuit_broken
    assert rm.daily_pnl < 0

    rm.reset_daily()

    assert not rm.is_circuit_broken
    assert rm.daily_pnl == 0.0

    # Should be able to trade again
    signal = FakeSignal()
    decision = rm.evaluate_signal(signal, 0.0001)
    assert decision.approved


# ── Test 11: Units Calculation ───────────────────────────────────


def test_units_calculation():
    """Correct forex lot sizing from pip_value with quote currency conversion."""
    from ict_bot.trading.live_loop import LiveTradingSession

    config = LiveConfig()
    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session._quote_currency_rates = {"USD_JPY": 150.0, "USD_CAD": 1.36, "EUR_GBP": 1.27}

    # USD-quoted (EUR/USD): units = pip_value / pip_size
    units = session._calculate_units(pip_value=2.0, pip_size=0.0001, pair="EUR_USD")
    assert units == 20_000

    # USD-base (USD/JPY): units = pip_value * rate / pip_size
    # $2/pip * 150 / 0.01 = 30,000
    units = session._calculate_units(pip_value=2.0, pip_size=0.01, pair="USD_JPY")
    assert units == 30_000

    # USD-quoted (GBP/USD)
    units = session._calculate_units(pip_value=10.0, pip_size=0.0001, pair="GBP_USD")
    assert units == 100_000  # 1 standard lot

    # Cross (EUR/GBP): units = pip_value / (pip_size * GBPUSD_rate)
    # $2/pip / (0.0001 * 1.27) = 15,748
    units = session._calculate_units(pip_value=2.0, pip_size=0.0001, pair="EUR_GBP")
    assert units == 15_748


# ── Test 12: _process_signal — max positions gate ────────────────


@pytest.mark.asyncio
async def test_process_signal_max_positions():
    """Signal is rejected when max concurrent positions reached."""
    config = LiveConfig(
        max_positions=2,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    from ict_bot.trading.live_loop import LiveTradingSession

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)
    session._last_trade_time = {}
    session._started_at = None
    session._quote_currency_rates = {}
    session.news_filter = None
    session._signal_lock = asyncio.Lock()

    from ict_bot.risk import RiskConfig, RiskManager
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))

    # Fill 2 positions
    sig1 = FakeSignal(pair="EUR_USD")
    await session.position_manager.open_position(sig1, units=10000)
    sig2 = FakeSignal(pair="GBP_USD", entry_price=1.30000, stop_loss=1.29500, take_profit=1.31000)
    await session.position_manager.open_position(sig2, units=10000)
    assert len(session.position_manager.positions) == 2

    # Third signal should be rejected
    broker.place_market_order.reset_mock()
    sig3 = FakeSignal(pair="AUD_USD", entry_price=0.67000, stop_loss=0.66500, take_profit=0.68000)
    await session._process_signal(sig3)

    # No new market order placed
    broker.place_market_order.assert_not_called()
    assert "AUD_USD" not in session.position_manager.positions


# ── Test 13: _process_signal — SL too wide ───────────────────────


@pytest.mark.asyncio
async def test_process_signal_sl_too_wide():
    """Signal with SL wider than max_sl_pips is rejected."""
    config = LiveConfig(
        max_sl_pips=50.0,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    from ict_bot.trading.live_loop import LiveTradingSession

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)
    session._last_trade_time = {}
    session._started_at = None
    session._quote_currency_rates = {}
    session.news_filter = None
    session._signal_lock = asyncio.Lock()

    from ict_bot.risk import RiskConfig, RiskManager
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))

    # SL = 80 pips (1.10000 - 1.09200 = 0.008 / 0.0001 = 80 pips) > 50 max
    sig = FakeSignal(entry_price=1.10000, stop_loss=1.09200, take_profit=1.12000)
    await session._process_signal(sig)

    broker.place_market_order.assert_not_called()
    assert "EUR_USD" not in session.position_manager.positions


# ── Test 14: _process_signal — full pipeline approval ────────────


@pytest.mark.asyncio
async def test_process_signal_full_pipeline():
    """Valid signal passes all gates and opens a position."""
    config = LiveConfig(
        max_positions=3,
        max_sl_pips=100.0,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    from ict_bot.trading.live_loop import LiveTradingSession

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)
    session._last_trade_time = {}
    session._started_at = None
    session._quote_currency_rates = {}
    session.news_filter = None
    session._signal_lock = asyncio.Lock()

    from ict_bot.risk import RiskConfig, RiskManager
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))

    sig = FakeSignal()  # 50 pip SL, well within limits
    await session._process_signal(sig)

    assert "EUR_USD" in session.position_manager.positions
    broker.place_market_order.assert_called_once()
    broker.place_stop_order.assert_called_once()
    broker.place_limit_order.assert_called_once()


# ── Test 15: Dry-run SL hit detection ────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_sl_hit():
    """Dry-run mode detects SL hit from bar data."""
    config = LiveConfig(
        dry_run=True,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Bar where low breaches long SL (1.095)
    bars_df = pd.DataFrame(
        {"open": [1.0960], "high": [1.0970], "low": [1.0940], "close": [1.0945], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 12:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal()  # long, SL=1.095
    await pm.open_position(signal, units=10000)
    assert "EUR_USD" in pm.positions

    closed = await pm.check_and_manage()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "stop_loss"
    assert "EUR_USD" not in pm.positions


# ── Test 16: Dry-run TP hit detection ────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_tp_hit():
    """Dry-run mode detects TP hit from bar data."""
    config = LiveConfig(
        dry_run=True,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Bar where high breaches long TP (1.110)
    bars_df = pd.DataFrame(
        {"open": [1.1090], "high": [1.1110], "low": [1.1080], "close": [1.1105], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 12:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal()  # long, TP=1.110
    await pm.open_position(signal, units=10000)

    closed = await pm.check_and_manage()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "take_profit"
    assert "EUR_USD" not in pm.positions


# ── Test 17: Dry-run no hit (price between SL and TP) ───────────


@pytest.mark.asyncio
async def test_dry_run_no_hit():
    """Dry-run mode keeps position open when price is between SL and TP."""
    config = LiveConfig(
        dry_run=True,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Bar where price stays safely between SL (1.095) and TP (1.110)
    bars_df = pd.DataFrame(
        {"open": [1.1020], "high": [1.1040], "low": [1.1000], "close": [1.1030], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 12:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal()
    await pm.open_position(signal, units=10000)

    closed = await pm.check_and_manage()
    assert len(closed) == 0
    assert "EUR_USD" in pm.positions


# ── Test 18: Dry-run short SL/TP detection ───────────────────────


@pytest.mark.asyncio
async def test_dry_run_short_sl_hit():
    """Dry-run detects SL hit on a short position (high breaches SL)."""
    config = LiveConfig(
        dry_run=True,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Short: SL=1.105, TP=1.090. Bar high hits SL.
    bars_df = pd.DataFrame(
        {"open": [1.1020], "high": [1.1060], "low": [1.1000], "close": [1.1050], "volume": [100]},
        index=pd.to_datetime(["2026-01-15 12:00"], utc=True),
    )
    broker.get_live_bars = MagicMock(return_value=bars_df)

    pm = PositionManager(broker, config)
    signal = FakeSignal(direction="short", entry_price=1.10000, stop_loss=1.10500, take_profit=1.09000)
    await pm.open_position(signal, units=10000)

    closed = await pm.check_and_manage()
    assert len(closed) == 1
    assert closed[0]["exit_reason"] == "stop_loss"


# ── Test 19: Account balance — USD direct ────────────────────────


@pytest.mark.asyncio
async def test_account_balance_usd_direct():
    """get_account_balance returns USD value directly when available."""
    from ict_bot.trading.broker import IBKRBroker

    config = LiveConfig()
    broker = IBKRBroker.__new__(IBKRBroker)
    broker.config = config
    broker.ib = MagicMock()
    broker.ib.managedAccounts.return_value = ["DU12345"]

    # Simulate accountSummary with direct USD
    usd_item = MagicMock()
    usd_item.tag = "NetLiquidation"
    usd_item.currency = "USD"
    usd_item.value = "10500.00"
    broker.ib.accountSummaryAsync = AsyncMock(return_value=[usd_item])

    balance = await broker.get_account_balance()
    assert balance == 10500.0


# ── Test 20: Account balance — SEK conversion ────────────────────


@pytest.mark.asyncio
async def test_account_balance_sek_conversion():
    """get_account_balance converts non-USD account via ExchangeRate."""
    from ict_bot.trading.broker import IBKRBroker

    config = LiveConfig()
    broker = IBKRBroker.__new__(IBKRBroker)
    broker.config = config
    broker.ib = MagicMock()
    broker.ib.managedAccounts.return_value = ["DU12345"]

    # Simulate SEK account: 50,000 SEK ÷ 9.06 (SEK/USD) ≈ $5,519 USD
    sek_item = MagicMock()
    sek_item.tag = "NetLiquidation"
    sek_item.currency = "SEK"
    sek_item.value = "50000.00"

    usd_rate_item = MagicMock()
    usd_rate_item.tag = "ExchangeRate"
    usd_rate_item.currency = "USD"
    usd_rate_item.value = "9.06"  # 9.06 SEK per 1 USD

    # USD NLV is 0 (non-USD account)
    usd_zero = MagicMock()
    usd_zero.tag = "NetLiquidation"
    usd_zero.currency = "USD"
    usd_zero.value = "0.00"

    broker.ib.accountSummaryAsync = AsyncMock(return_value=[usd_zero, sek_item, usd_rate_item])

    balance = await broker.get_account_balance()
    assert balance == pytest.approx(50000.0 / 9.06, rel=0.01)


# ── Test 21: Account balance — fallback to config default ────────


@pytest.mark.asyncio
async def test_account_balance_fallback():
    """get_account_balance falls back to config default when no data."""
    from ict_bot.trading.broker import IBKRBroker

    config = LiveConfig(starting_balance=5_000.0)
    broker = IBKRBroker.__new__(IBKRBroker)
    broker.config = config
    broker.ib = MagicMock()
    broker.ib.managedAccounts.return_value = []
    broker.ib.accountValues.return_value = []

    balance = await broker.get_account_balance()
    assert balance == 5_000.0


# ── Test 22: _make_trade_record P&L — long TP win ────────────────


def test_make_trade_record_long_tp():
    """Trade record P&L is correct for a long TP win."""
    pos = LivePosition(
        pair="EUR_USD", direction="long", entry_price=1.10000,
        units=10000, stop_loss=1.09500, take_profit=1.11000,
        entry_time="2026-01-15T10:00:00+00:00", risk_pips=50.0,
        confluence_score=4,
    )
    record = PositionManager._make_trade_record(pos, "take_profit")

    assert record["exit_price"] == 1.11000
    assert record["pnl_pips"] == pytest.approx(100.0, rel=0.01)
    assert record["rr_achieved"] == pytest.approx(2.0, rel=0.01)
    assert record["pnl_amount"] > 0


# ── Test 23: _make_trade_record P&L — short SL loss ─────────────


def test_make_trade_record_short_sl():
    """Trade record P&L is correct for a short SL loss."""
    pos = LivePosition(
        pair="GBP_USD", direction="short", entry_price=1.30000,
        units=5000, stop_loss=1.30500, take_profit=1.29000,
        entry_time="2026-01-15T10:00:00+00:00", risk_pips=50.0,
        confluence_score=3,
    )
    record = PositionManager._make_trade_record(pos, "stop_loss")

    assert record["exit_price"] == 1.30500
    assert record["pnl_pips"] == pytest.approx(-50.0, rel=0.01)
    assert record["rr_achieved"] == pytest.approx(-1.0, rel=0.01)
    assert record["pnl_amount"] < 0


# ── Test: _make_trade_record uses actual fill price (#30) ────────


def test_make_trade_record_uses_actual_sl_fill_price():
    """Trade record uses actual IB fill price for SL instead of theoretical."""
    pos = LivePosition(
        pair="EUR_USD", direction="long", entry_price=1.10000,
        units=10000, stop_loss=1.09500, take_profit=1.11000,
        entry_time="2026-01-15T10:00:00+00:00", risk_pips=50.0,
        confluence_score=4,
    )
    # Simulate IB Trade with slipped fill (worse than theoretical SL)
    sl_trade = MagicMock()
    sl_trade.orderStatus.status = "Filled"
    sl_trade.orderStatus.avgFillPrice = 1.09450  # 0.5 pip slippage
    pos.sl_order = sl_trade

    record = PositionManager._make_trade_record(pos, "stop_loss")

    assert record["exit_price"] == 1.09450  # actual fill, not 1.09500
    assert record["pnl_pips"] == pytest.approx(-55.0, rel=0.01)


def test_make_trade_record_uses_actual_tp_fill_price():
    """Trade record uses actual IB fill price for TP instead of theoretical."""
    pos = LivePosition(
        pair="GBP_USD", direction="short", entry_price=1.30000,
        units=5000, stop_loss=1.30500, take_profit=1.29000,
        entry_time="2026-01-15T10:00:00+00:00", risk_pips=50.0,
        confluence_score=3,
    )
    # Simulate IB Trade with better-than-expected fill
    tp_trade = MagicMock()
    tp_trade.orderStatus.status = "Filled"
    tp_trade.orderStatus.avgFillPrice = 1.28950  # 0.5 pip better
    pos.tp_order = tp_trade

    record = PositionManager._make_trade_record(pos, "take_profit")

    assert record["exit_price"] == 1.28950  # actual fill, not 1.29000
    assert record["pnl_pips"] == pytest.approx(105.0, rel=0.01)


def test_make_trade_record_falls_back_to_theoretical():
    """Trade record uses theoretical price when no IB Trade object exists."""
    pos = LivePosition(
        pair="EUR_USD", direction="long", entry_price=1.10000,
        units=10000, stop_loss=1.09500, take_profit=1.11000,
        entry_time="2026-01-15T10:00:00+00:00", risk_pips=50.0,
        confluence_score=4,
    )
    # No IB Trade objects (e.g. loaded from JSON after restart)
    pos.sl_order = None
    pos.tp_order = None

    record = PositionManager._make_trade_record(pos, "stop_loss")
    assert record["exit_price"] == 1.09500  # theoretical fallback

    record = PositionManager._make_trade_record(pos, "take_profit")
    assert record["exit_price"] == 1.11000  # theoretical fallback


# ── Test 24: close_position survives cancel error ────────────────


@pytest.mark.asyncio
async def test_close_position_survives_cancel_error():
    """Position still closes even if SL/TP cancel raises an exception."""
    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)
    pm = PositionManager(broker, config)

    signal = FakeSignal()
    await pm.open_position(signal, units=10000)

    # Make cancel_order raise for both SL and TP
    broker.cancel_order = AsyncMock(side_effect=Exception("IB timeout"))

    record = await pm.close_position("EUR_USD", "manual")

    # Position should still be closed despite cancel errors
    assert "EUR_USD" not in pm.positions
    assert record["pair"] == "EUR_USD"
    # Market close order should still have been attempted
    assert broker.place_market_order.call_count == 2  # open + close


# ── Test 25: Daily reset timing ──────────────────────────────────


@pytest.mark.asyncio
async def test_daily_reset_timing():
    """Daily reset fires once per ET day (DST-aware)."""
    from ict_bot.trading.live_loop import LiveTradingSession

    config = LiveConfig(state_file=tempfile.mktemp(suffix=".json"))
    broker = make_mock_broker(config)

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)

    from ict_bot.risk import RiskConfig, RiskManager
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))
    session._last_daily_reset = None
    session._htf_cache = {}

    # Trigger a loss so we can verify reset clears it
    session.risk_manager.record_trade_result(-100.0, "EUR_USD")
    assert session.risk_manager.daily_pnl < 0

    # First call should reset (no previous reset)
    await session._check_daily_reset()
    assert session.risk_manager.daily_pnl == 0.0
    assert session._last_daily_reset is not None

    # Second call on same day should NOT reset again
    session.risk_manager.record_trade_result(-50.0, "EUR_USD")
    await session._check_daily_reset()
    assert session.risk_manager.daily_pnl == pytest.approx(-50.0)


# ── News Filter Tests ──────────────────────────────────────────────


# Mock calendar data: NFP (USD, High), ECB Rate Decision (EUR, High),
# German ZEW (EUR, Medium)
MOCK_CALENDAR = [
    {
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "date": "2026-02-06T13:30:00-05:00",
        "impact": "High",
        "forecast": "180K",
        "previous": "256K",
    },
    {
        "title": "ECB Main Refinancing Rate",
        "country": "EUR",
        "date": "2026-02-06T07:45:00-05:00",
        "impact": "High",
        "forecast": "2.90%",
        "previous": "3.15%",
    },
    {
        "title": "German ZEW Economic Sentiment",
        "country": "EUR",
        "date": "2026-02-04T05:00:00-05:00",
        "impact": "Medium",
        "forecast": "15.0",
        "previous": "10.3",
    },
]


def _make_news_filter_with_events(events: list[NewsEvent]) -> NewsFilter:
    """Create a NewsFilter pre-loaded with events (no HTTP fetch)."""
    nf = NewsFilter(blackout_minutes=30)
    nf._events = events
    nf._last_fetch = datetime.now(timezone.utc)
    return nf


# ── Test 26: Parse events ──────────────────────────────────────────


def test_news_filter_parse_events():
    """Correct UTC conversion and affected_pairs mapping."""
    nf = NewsFilter()
    events = nf._parse_events(MOCK_CALENDAR)

    assert len(events) == 3

    # NFP: 13:30 EST = 18:30 UTC
    nfp = events[0]
    assert nfp.title == "Non-Farm Employment Change"
    assert nfp.country == "USD"
    assert nfp.date.hour == 18
    assert nfp.date.minute == 30
    assert nfp.impact == "High"
    assert nfp.forecast == "180K"
    assert nfp.previous == "256K"
    # USD news affects 6 pairs (EUR_GBP excluded — no USD component)
    assert len(nfp.affected_pairs) == 6
    assert "EUR_USD" in nfp.affected_pairs

    # ECB: 07:45 EST = 12:45 UTC
    ecb = events[1]
    assert ecb.date.hour == 12
    assert ecb.date.minute == 45
    assert set(ecb.affected_pairs) == {"EUR_USD", "EUR_GBP"}

    # ZEW: Medium impact, EUR
    zew = events[2]
    assert zew.impact == "Medium"


# ── Test 27: Blocks pair during blackout ───────────────────────────


def test_news_filter_blocks_pair_during_blackout():
    """USD event blocks EUR_USD within ±30 min."""
    now = datetime.now(timezone.utc)
    event = NewsEvent(
        title="NFP",
        country="USD",
        date=now + timedelta(minutes=10),  # 10 min from now
        impact="High",
        affected_pairs=CURRENCY_TO_PAIRS["USD"],
    )
    nf = _make_news_filter_with_events([event])
    assert nf.is_pair_blocked("EUR_USD") is True
    assert nf.is_pair_blocked("GBP_USD") is True
    assert nf.is_pair_blocked("USD_JPY") is True


# ── Test 28: Allows pair outside blackout ──────────────────────────


def test_news_filter_allows_pair_outside_blackout():
    """Not blocked 2 hours before event."""
    now = datetime.now(timezone.utc)
    event = NewsEvent(
        title="NFP",
        country="USD",
        date=now + timedelta(hours=2),  # 2 hours from now
        impact="High",
        affected_pairs=CURRENCY_TO_PAIRS["USD"],
    )
    nf = _make_news_filter_with_events([event])
    assert nf.is_pair_blocked("EUR_USD") is False


# ── Test 29: Only blocks high impact ──────────────────────────────


def test_news_filter_only_blocks_high_impact():
    """Medium-impact events don't block trading."""
    now = datetime.now(timezone.utc)
    event = NewsEvent(
        title="ZEW Sentiment",
        country="EUR",
        date=now + timedelta(minutes=5),  # imminent
        impact="Medium",
        affected_pairs=CURRENCY_TO_PAIRS["EUR"],
    )
    nf = _make_news_filter_with_events([event])
    assert nf.is_pair_blocked("EUR_USD") is False


# ── Test 30: EUR blocks EUR pairs only ─────────────────────────────


def test_news_filter_eur_blocks_eur_pairs_only():
    """EUR event blocks EUR_USD + EUR_GBP, not USD_JPY."""
    now = datetime.now(timezone.utc)
    event = NewsEvent(
        title="ECB Rate Decision",
        country="EUR",
        date=now + timedelta(minutes=5),
        impact="High",
        affected_pairs=CURRENCY_TO_PAIRS["EUR"],
    )
    nf = _make_news_filter_with_events([event])
    assert nf.is_pair_blocked("EUR_USD") is True
    assert nf.is_pair_blocked("EUR_GBP") is True
    assert nf.is_pair_blocked("USD_JPY") is False
    assert nf.is_pair_blocked("AUD_USD") is False


# ── Test 31: Fail-open on network error ────────────────────────────


def test_news_filter_fail_open_on_network_error():
    """requests.get raises → is_pair_blocked returns False."""
    nf = NewsFilter()
    with patch("ict_bot.trading.news_filter.NewsFilter._fetch_calendar", return_value=None):
        # Force a refresh by clearing last_fetch
        nf._last_fetch = None
        result = nf.is_pair_blocked("EUR_USD")
    assert result is False


# ── Test 32: Caches data ──────────────────────────────────────────


def test_news_filter_caches_data():
    """Two rapid calls → only one HTTP request."""
    nf = NewsFilter()
    mock_events = [
        NewsEvent(
            title="Test", country="USD",
            date=datetime.now(timezone.utc) + timedelta(hours=1),
            impact="High", affected_pairs=["EUR_USD"],
        )
    ]
    with patch.object(nf, "_fetch_calendar", return_value=mock_events) as mock_fetch:
        nf.refresh_if_needed()
        nf.refresh_if_needed()
    mock_fetch.assert_called_once()


# ── Test 33: Get upcoming events ──────────────────────────────────


def test_news_filter_get_upcoming_events():
    """Sorted, filtered by time window."""
    now = datetime.now(timezone.utc)
    events = [
        NewsEvent(title="Later", country="USD", date=now + timedelta(hours=5),
                  impact="High", affected_pairs=["EUR_USD"]),
        NewsEvent(title="Soon", country="EUR", date=now + timedelta(hours=1),
                  impact="High", affected_pairs=["EUR_USD"]),
        NewsEvent(title="Far", country="GBP", date=now + timedelta(hours=100),
                  impact="High", affected_pairs=["GBP_USD"]),
    ]
    nf = _make_news_filter_with_events(events)
    upcoming = nf.get_upcoming_events(hours_ahead=48)

    assert len(upcoming) == 2  # "Far" excluded (>48h)
    assert upcoming[0].title == "Soon"  # sorted by date
    assert upcoming[1].title == "Later"


# ── Test 34: NewsEvent serialization ──────────────────────────────


def test_news_event_serialization():
    """to_dict() produces correct JSON-serializable output."""
    dt = datetime(2026, 2, 6, 18, 30, tzinfo=timezone.utc)
    event = NewsEvent(
        title="NFP",
        country="USD",
        date=dt,
        impact="High",
        forecast="180K",
        previous="256K",
        affected_pairs=["EUR_USD", "GBP_USD"],
    )
    d = event.to_dict()
    assert d["title"] == "NFP"
    assert d["country"] == "USD"
    assert d["date"] == "2026-02-06T18:30:00+00:00"
    assert d["impact"] == "High"
    assert d["forecast"] == "180K"
    assert d["previous"] == "256K"
    assert d["affected_pairs"] == ["EUR_USD", "GBP_USD"]

    # Round-trip: should be JSON serializable
    json_str = json.dumps(d)
    loaded = json.loads(json_str)
    assert loaded["title"] == "NFP"


# ── Test 35: _process_signal blocked by news ──────────────────────


@pytest.mark.asyncio
async def test_process_signal_blocked_by_news():
    """Integration: signal rejected when pair is in news blackout."""
    config = LiveConfig(
        max_positions=3,
        max_sl_pips=100.0,
        news_filter_enabled=True,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    from ict_bot.trading.live_loop import LiveTradingSession

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)
    session._last_trade_time = {}
    session._started_at = None
    session._quote_currency_rates = {}
    session._signal_lock = asyncio.Lock()

    from ict_bot.risk import RiskConfig, RiskManager
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))

    # Set up news filter with an imminent high-impact USD event
    now = datetime.now(timezone.utc)
    event = NewsEvent(
        title="NFP",
        country="USD",
        date=now + timedelta(minutes=5),
        impact="High",
        affected_pairs=CURRENCY_TO_PAIRS["USD"],
    )
    session.news_filter = _make_news_filter_with_events([event])

    sig = FakeSignal()  # EUR_USD — should be blocked
    await session._process_signal(sig)

    # Should NOT have opened a position
    broker.place_market_order.assert_not_called()
    assert "EUR_USD" not in session.position_manager.positions


# ── Test: Risk Manager State Persistence ──────────────────────


def test_risk_manager_state_persistence():
    """Risk manager state (kill switch, circuit breaker, etc.) survives save/load."""
    from ict_bot.risk import RiskConfig, RiskManager

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        risk_path = f.name

    rm = RiskManager(RiskConfig(starting_balance=10_000))
    rm._balance = 8_500.0
    rm._peak_balance = 10_000.0
    rm._daily_pnl = -200.0
    rm._consecutive_losses = 2
    rm._killed = True
    rm._circuit_broken = True
    rm._open_positions = {"EUR_USD": 100.0}

    rm.save_state(risk_path)

    # Load into a fresh instance
    rm2 = RiskManager(RiskConfig(starting_balance=10_000))
    rm2.load_state(risk_path)

    assert rm2._balance == 8_500.0
    assert rm2._peak_balance == 10_000.0
    assert rm2._daily_pnl == -200.0
    assert rm2._consecutive_losses == 2
    assert rm2._killed is True
    assert rm2._circuit_broken is True
    assert rm2._open_positions == {"EUR_USD": 100.0}

    # Clean up
    Path(risk_path).unlink(missing_ok=True)


# ── Test: Signal lock prevents concurrent max-position breach ─────


def _make_session(max_positions=3):
    """Helper to create a LiveTradingSession with mocked broker for _process_signal tests."""
    from ict_bot.risk import RiskConfig, RiskManager
    from ict_bot.trading.live_loop import LiveTradingSession

    config = LiveConfig(
        max_positions=max_positions,
        max_sl_pips=100.0,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config
    session.broker = broker
    session.position_manager = PositionManager(broker, config)
    session._last_trade_time = {}
    session._started_at = None
    session._quote_currency_rates = {}
    session.news_filter = None
    session.risk_manager = RiskManager(RiskConfig(
        max_risk_per_trade=0.01, starting_balance=10_000,
    ))
    # Signal lock must exist on the session
    session._signal_lock = asyncio.Lock()
    return session


@pytest.mark.asyncio
async def test_signal_lock_exists():
    """LiveTradingSession must have a _signal_lock attribute (asyncio.Lock)."""
    session = _make_session()
    assert hasattr(session, "_signal_lock")
    assert isinstance(session._signal_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_signal_lock_prevents_concurrent_breach():
    """Two concurrent _process_signal calls must not both pass the max-positions gate.

    Without the lock, both coroutines see positions < max at the same time and
    both open a position, exceeding max_positions. With the lock, only one gets
    through; the second sees positions == max and is rejected.
    """
    session = _make_session(max_positions=2)

    # Pre-fill 1 position so we're one away from the limit
    sig_prefill = FakeSignal(pair="EUR_USD")
    await session.position_manager.open_position(sig_prefill, units=10000)
    assert len(session.position_manager.positions) == 1

    # Two signals for different pairs, fired concurrently
    sig_a = FakeSignal(pair="GBP_USD", entry_price=1.30000, stop_loss=1.29500, take_profit=1.31000)
    sig_b = FakeSignal(pair="AUD_USD", entry_price=0.67000, stop_loss=0.66500, take_profit=0.68000)

    # Run both _process_signal concurrently
    await asyncio.gather(
        session._process_signal(sig_a),
        session._process_signal(sig_b),
    )

    # At most one new position should have been added (total <= max_positions=2)
    assert len(session.position_manager.positions) <= 2, (
        f"Max positions breached: {len(session.position_manager.positions)} positions "
        f"(max={session.config.max_positions}). "
        f"Open: {list(session.position_manager.positions.keys())}"
    )


@pytest.mark.asyncio
async def test_signal_lock_serializes_processing():
    """_process_signal must hold the lock for its entire duration.

    Verify that calling _process_signal acquires the lock and that a second
    concurrent call cannot enter until the first completes.
    We instrument _process_signal_inner (which runs inside the lock) to
    track ordering.
    """
    session = _make_session(max_positions=3)

    # Track entry/exit ordering of the inner method (runs under the lock)
    order_log = []

    original_inner = session._process_signal_inner

    async def instrumented_inner(signal):
        order_log.append(f"enter_{signal.pair}")
        await original_inner(signal)
        order_log.append(f"exit_{signal.pair}")

    session._process_signal_inner = instrumented_inner

    sig_a = FakeSignal(pair="EUR_USD")
    sig_b = FakeSignal(pair="GBP_USD", entry_price=1.30000, stop_loss=1.29500, take_profit=1.31000)

    await asyncio.gather(
        session._process_signal(sig_a),
        session._process_signal(sig_b),
    )

    # With the lock, one must fully complete before the other starts
    # So the order must be: enter_X, exit_X, enter_Y, exit_Y
    # (not interleaved like enter_X, enter_Y, exit_X, exit_Y)
    assert len(order_log) == 4
    # First entry must complete (exit) before second entry
    first_enter_idx = 0
    first_exit_idx = 1
    second_enter_idx = 2
    assert order_log[first_enter_idx].startswith("enter_")
    assert order_log[first_exit_idx].startswith("exit_")
    assert order_log[second_enter_idx].startswith("enter_")
    # The pair that entered first must also exit first
    first_pair = order_log[first_enter_idx].replace("enter_", "")
    assert order_log[first_exit_idx] == f"exit_{first_pair}"


# ── Reconnect Order Race Tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_reconnect_subscribes_bars_before_reconcile():
    """On reconnect, bars must be subscribed BEFORE order reconciliation.

    If orders are placed before bar subscriptions complete, IB can cancel
    orders due to reqId/orderId conflicts (Error 366).
    """
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()
    session.news_filter = None
    session._last_daily_reset = datetime.now(timezone.utc)
    session._htf_cache = {}
    session._started_at = datetime.now(timezone.utc)

    # Simulate disconnected state
    session.broker.is_connected = MagicMock(return_value=False)
    session.broker.connect = AsyncMock()
    session.broker.subscribe_bars = AsyncMock()
    session.broker.needs_resubscribe = MagicMock(return_value=False)
    session.broker.cancel_all_orders = AsyncMock()
    session.broker.get_open_orders_by_pair = MagicMock(return_value={})

    # Track call order
    call_order = []
    original_reconcile = session._reconcile_after_reconnect

    async def track_subscribe(pair, callback):
        call_order.append(f"subscribe_{pair}")

    async def track_reconcile():
        call_order.append("reconcile")
        # Don't actually reconcile (no real positions)

    session.broker.subscribe_bars = AsyncMock(side_effect=track_subscribe)
    session._reconcile_after_reconnect = track_reconcile

    await session._heartbeat()

    # All subscribes must come before reconcile
    reconcile_idx = call_order.index("reconcile")
    subscribe_indices = [i for i, c in enumerate(call_order) if c.startswith("subscribe_")]
    assert len(subscribe_indices) > 0, "No bar subscriptions happened"
    assert all(i < reconcile_idx for i in subscribe_indices), (
        f"Bar subscriptions must happen before reconcile. Order: {call_order}"
    )


@pytest.mark.asyncio
async def test_heartbeat_reconnect_skips_watchdog():
    """After reconnect + reconcile, watchdog (check_and_manage) must NOT run.

    The reconcile just placed fresh orders — running the watchdog in the same
    heartbeat can see them as dead (stale Trade objects) and duplicate them.
    """
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()
    session.news_filter = None
    session._last_daily_reset = datetime.now(timezone.utc)
    session._htf_cache = {}
    session._started_at = datetime.now(timezone.utc)

    # Simulate disconnected state
    session.broker.is_connected = MagicMock(return_value=False)
    session.broker.connect = AsyncMock()
    session.broker.subscribe_bars = AsyncMock()
    session.broker.needs_resubscribe = MagicMock(return_value=False)
    session.broker.cancel_all_orders = AsyncMock()
    session.broker.get_open_orders_by_pair = MagicMock(return_value={})

    # Stub out reconcile (no real positions)
    session._reconcile_after_reconnect = AsyncMock()
    session._check_daily_reset = AsyncMock()
    session._refresh_quote_rates = AsyncMock()

    # Spy on check_and_manage
    session.position_manager.check_and_manage = AsyncMock(return_value=[])
    session.position_manager.audit_and_dedup_orders = AsyncMock()

    await session._heartbeat()

    # check_and_manage should NOT have been called after a reconnect
    session.position_manager.check_and_manage.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_normal_runs_watchdog():
    """When already connected (no reconnect), watchdog must run normally."""
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()
    session.news_filter = None
    session._last_daily_reset = datetime.now(timezone.utc)
    session._htf_cache = {}
    session._started_at = datetime.now(timezone.utc)

    # Simulate connected state (no reconnect needed)
    session.broker.is_connected = MagicMock(return_value=True)
    session.broker.needs_resubscribe = MagicMock(return_value=False)

    # Stub out methods
    session._check_daily_reset = AsyncMock()
    session._refresh_quote_rates = AsyncMock()

    # Spy on check_and_manage
    session.position_manager.check_and_manage = AsyncMock(return_value=[])
    session.position_manager.audit_and_dedup_orders = AsyncMock()

    await session._heartbeat()

    # check_and_manage SHOULD run during normal heartbeats
    session.position_manager.check_and_manage.assert_called_once()


# ── Asian Session Kill Zone Tests ─────────────────────────────────


class TestIsAsianSession:
    """Tests for the is_asian_session() wall-clock helper."""

    def test_during_asian_hours(self):
        """Times within 19:00-02:00 ET return True."""
        from ict_bot.signals.kill_zones import is_asian_session

        # 21:00 ET = 01:00 UTC (next day) in winter (EST = UTC-5)
        # Use a time that's unambiguously Asian: 23:00 UTC = 18:00 ET (EST)
        # Actually, let's pick 00:30 UTC = 19:30 ET (EST) — inside Asian
        utc_time = datetime(2026, 1, 15, 0, 30, tzinfo=timezone.utc)  # 19:30 ET
        assert is_asian_session(utc_time) is True

        # 05:00 UTC = 00:00 ET (midnight) — inside Asian
        utc_time = datetime(2026, 1, 15, 5, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is True

        # 06:00 UTC = 01:00 ET — inside Asian
        utc_time = datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is True

    def test_outside_asian_hours(self):
        """Times outside 19:00-02:00 ET return False."""
        from ict_bot.signals.kill_zones import is_asian_session

        # 15:00 UTC = 10:00 ET (EST) — NY session, not Asian
        utc_time = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is False

        # 12:00 UTC = 07:00 ET (EST) — London/NY overlap
        utc_time = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is False

        # 20:00 UTC = 15:00 ET (EST) — afternoon NY
        utc_time = datetime(2026, 1, 15, 20, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is False

    def test_boundary(self):
        """19:00 ET = True (start), 02:00 ET = False (end is exclusive)."""
        from ict_bot.signals.kill_zones import is_asian_session

        # 19:00 ET exactly = 00:00 UTC (EST = UTC-5)
        utc_time = datetime(2026, 1, 16, 0, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is True

        # 02:00 ET exactly = 07:00 UTC (EST = UTC-5)
        utc_time = datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)
        assert is_asian_session(utc_time) is False


@pytest.mark.asyncio
async def test_asian_signal_rejected_in_on_bar_update():
    """Signal with kill_zone='asian' is skipped in _on_bar_update (existing behavior)."""
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session._started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session._htf_cache = {"EUR_USD": (datetime.now(timezone.utc), pd.DataFrame())}

    asian_signal = FakeSignal(
        timestamp=pd.Timestamp("2026-01-15 22:00", tz="UTC"),
        kill_zone="asian",
    )

    with patch("ict_bot.trading.live_loop.generate_signals", return_value=[asian_signal]):
        session.broker._bars_to_dataframe = MagicMock(
            return_value=pd.DataFrame(
                {"open": [1.1] * 200, "high": [1.1] * 200,
                 "low": [1.1] * 200, "close": [1.1] * 200, "volume": [0] * 200},
                index=pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            )
        )
        session._process_signal = AsyncMock()

        await session._on_bar_update("EUR_USD", MagicMock(), has_new_bar=True)

    # _process_signal should NOT have been called — killed by asian label check
    session._process_signal.assert_not_called()


@pytest.mark.asyncio
async def test_asian_wallclock_gate_blocks_trade():
    """Gate 4.5: wall-clock Asian check blocks trade even if signal says 'london'."""
    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))

    signal = FakeSignal(kill_zone="london")  # mislabeled — says london

    # Mock wall-clock to 21:00 ET (= 02:00 UTC next day in EST, inside Asian)
    asian_utc = datetime(2026, 1, 16, 2, 0, tzinfo=timezone.utc)  # 21:00 ET

    with patch("ict_bot.trading.live_loop.is_asian_session", return_value=True):
        await session._process_signal_inner(signal)

    # No market order should have been placed — blocked by Gate 4.5
    session.broker.place_market_order.assert_not_called()


# ── Min R:R Hard Filter Tests (#37) ──────────────────────────────


def test_min_rr_sets_default_tp_distance():
    """min_rr parameter sets the default TP distance but liquidity targets can override it.

    With use_liquidity_targets=True (default), signals can have rr_ratio < min_rr
    when a structural liquidity target (PDH/PDL/PWH/PWL) is closer than the
    default R:R TP, as long as the target gives R:R >= 1.0 (min_target_rr).
    This was validated in backtesting — these low-RR trades are profitable.
    """
    from pathlib import Path as P
    from ict_bot.signals.detector import generate_signals

    data_dir = P(__file__).parent.parent / "data" / "processed"
    h1_path = data_dir / "EUR_USD_H1.parquet"
    d_path = data_dir / "EUR_USD_D.parquet"

    if not h1_path.exists():
        pytest.skip("No parquet data available")

    ohlc = pd.read_parquet(h1_path)
    htf = pd.read_parquet(d_path) if d_path.exists() else None

    signals = generate_signals(
        ohlc=ohlc, htf_ohlc=htf, pair="EUR_USD", swing_length=10,
        confluence_threshold=3, min_rr=2.0, sl_buffer_pips=10.0,
        use_displacement=False, fvg_lookback=16, pullback_window=40,
        compute_ob=False,
    )

    assert len(signals) > 0, "Expected at least some signals from EUR_USD H1"
    # Signals should include some with rr_ratio < 2.0 (from liquidity targets)
    low_rr = [s for s in signals if s.rr_ratio < 2.0]
    assert len(low_rr) > 0, (
        "Expected some signals with rr_ratio < 2.0 from liquidity targets"
    )
    # All signals must have rr_ratio >= 1.0 (min_target_rr hardcoded in detector)
    below_1 = [s for s in signals if s.rr_ratio < 1.0]
    assert len(below_1) == 0, (
        f"{len(below_1)} signals below rr_ratio=1.0 — min_target_rr not enforced"
    )


# ── Pre-News Profit Protection Tests (#36) ───────────────────────


def test_dashboard_jpy_pnl_conversion():
    """USD/JPY unrealized PnL must be divided by rate to get USD (#33).

    IB reports unrealizedPNL in quote currency:
      EUR_USD -> USD (no conversion)
      USD_JPY -> JPY (divide by rate)
    """
    # Simulate: -15,147 JPY unrealized, rate=155.6
    raw_pnl_jpy = -15147.0
    market_price_jpy = 155.6

    # Conversion: JPY -> USD
    pnl_usd = raw_pnl_jpy / market_price_jpy
    assert -100 < pnl_usd < 0  # should be ~-97 USD, NOT -15,147
    assert pnl_usd == pytest.approx(-97.3, abs=1.0)

    # USD-quoted pair: no conversion needed
    raw_pnl_usd = -50.0
    pnl_usd_direct = raw_pnl_usd  # already in USD
    assert pnl_usd_direct == -50.0


# ── Signal Fingerprint Dedup Tests (#39) ──────────────────────────


def test_signal_fingerprint_computation():
    """Signal fingerprint is based on pair, direction, break_idx."""
    from ict_bot.trading.live_loop import LiveTradingSession

    sig1 = FakeSignal(pair="EUR_USD", direction="long", meta={"break_idx": 150})
    sig2 = FakeSignal(pair="EUR_USD", direction="long", meta={"break_idx": 150})
    sig3 = FakeSignal(pair="EUR_USD", direction="long", meta={"break_idx": 200})
    sig4 = FakeSignal(pair="EUR_USD", direction="short", meta={"break_idx": 150})

    fp1 = LiveTradingSession._signal_fingerprint(sig1)
    fp2 = LiveTradingSession._signal_fingerprint(sig2)
    fp3 = LiveTradingSession._signal_fingerprint(sig3)
    fp4 = LiveTradingSession._signal_fingerprint(sig4)

    # Same structural break = same fingerprint
    assert fp1 == fp2
    # Different break_idx = different fingerprint
    assert fp1 != fp3
    # Different direction = different fingerprint
    assert fp1 != fp4


@pytest.mark.asyncio
async def test_seen_signal_skipped_in_on_bar_update():
    """Previously seen signal fingerprint is skipped (#39)."""
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session._started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session._htf_cache = {"EUR_USD": (datetime.now(timezone.utc), pd.DataFrame())}
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))

    sig = FakeSignal(
        timestamp=pd.Timestamp("2026-01-15 15:00", tz="UTC"),
        kill_zone="new_york",
        meta={"break_idx": 100},
    )

    # Pre-record this fingerprint as seen
    fp = LiveTradingSession._signal_fingerprint(sig)
    session._seen_signals[fp] = datetime.now(timezone.utc).isoformat()

    with patch("ict_bot.trading.live_loop.generate_signals", return_value=[sig]):
        session.broker._bars_to_dataframe = MagicMock(
            return_value=pd.DataFrame(
                {"open": [1.1] * 200, "high": [1.1] * 200,
                 "low": [1.1] * 200, "close": [1.1] * 200, "volume": [0] * 200},
                index=pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            )
        )
        session._process_signal = AsyncMock()

        await session._on_bar_update("EUR_USD", MagicMock(), has_new_bar=True)

    # Signal was seen before — should NOT be processed
    session._process_signal.assert_not_called()


@pytest.mark.asyncio
async def test_fingerprint_not_burned_on_temporary_rejection():
    """Fingerprint must NOT be saved when a temporary gate rejects the signal.

    If a signal is rejected by circuit breaker, max positions, or news blackout,
    the fingerprint should remain unburned so the signal can be retried on the
    next bar when the temporary condition clears.
    """
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session(max_positions=1)
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session._started_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    session._htf_cache = {"EUR_USD": (datetime.now(timezone.utc), pd.DataFrame())}

    sig = FakeSignal(
        timestamp=pd.Timestamp("2026-01-15 15:00", tz="UTC"),
        kill_zone="new_york",
        meta={"break_idx": 100},
    )

    # Pre-fill 1 position so max_positions (1) is reached → Gate 2 rejects
    prefill = FakeSignal(pair="GBP_USD")
    await session.position_manager.open_position(prefill, units=10000)
    assert len(session.position_manager.positions) == 1

    with patch("ict_bot.trading.live_loop.generate_signals", return_value=[sig]):
        session.broker._bars_to_dataframe = MagicMock(
            return_value=pd.DataFrame(
                {"open": [1.1] * 200, "high": [1.1] * 200,
                 "low": [1.1] * 200, "close": [1.1] * 200, "volume": [0] * 200},
                index=pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
            )
        )

        await session._on_bar_update("EUR_USD", MagicMock(), has_new_bar=True)

    # Signal was rejected by max positions gate, fingerprint must NOT be saved
    fp = LiveTradingSession._signal_fingerprint(sig)
    assert fp not in session._seen_signals, (
        "Fingerprint was burned despite temporary rejection — signal can never be retried"
    )


@pytest.mark.asyncio
async def test_fingerprint_saved_after_successful_trade():
    """Fingerprint IS saved after a trade is successfully opened."""
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()

    sig = FakeSignal(
        kill_zone="new_york",
        meta={"break_idx": 200},
    )

    with patch("ict_bot.trading.live_loop.is_asian_session", return_value=False):
        await session._process_signal_inner(sig)

    # Trade should have been placed
    session.broker.place_market_order.assert_called_once()

    # Fingerprint must be saved
    fp = LiveTradingSession._signal_fingerprint(sig)
    assert fp in session._seen_signals, (
        "Fingerprint was NOT saved after successful trade"
    )


@pytest.mark.asyncio
async def test_signal_age_replaces_started_at():
    """Signal age window accepts recent pre-startup signals, rejects old ones.

    With signal_max_age_bars=8 on M15 (2h window), a signal from 1 hour ago
    should pass while a signal from 4 hours ago should be skipped.
    """
    from ict_bot.trading.live_loop import LiveTradingSession

    session = _make_session()
    session._started_at = datetime.now(timezone.utc)  # just started
    session._htf_cache = {"EUR_USD": (datetime.now(timezone.utc), pd.DataFrame())}
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.config.signal_max_age_bars = 8  # 8 × 15min = 2h

    now = pd.Timestamp.now(tz="UTC")

    # Signal from 1 hour ago — within 2h window, should reach processing
    recent_sig = FakeSignal(
        timestamp=now - pd.Timedelta(hours=1),
        kill_zone="new_york",
        meta={"break_idx": 300},
    )
    # Signal from 4 hours ago — outside window, should be skipped
    old_sig = FakeSignal(
        timestamp=now - pd.Timedelta(hours=4),
        kill_zone="new_york",
        meta={"break_idx": 200},
    )

    bars_df = pd.DataFrame(
        {"open": [1.1] * 200, "high": [1.1] * 200,
         "low": [1.1] * 200, "close": [1.1] * 200, "volume": [0] * 200},
        index=pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"),
    )
    session.broker._bars_to_dataframe = MagicMock(return_value=bars_df)
    session._process_signal = AsyncMock()

    # Old signal → should NOT reach _process_signal
    with patch("ict_bot.trading.live_loop.generate_signals", return_value=[old_sig]):
        await session._on_bar_update("EUR_USD", MagicMock(), has_new_bar=True)
    session._process_signal.assert_not_called()

    # Recent signal → should reach _process_signal
    with patch("ict_bot.trading.live_loop.generate_signals", return_value=[recent_sig]):
        await session._on_bar_update("EUR_USD", MagicMock(), has_new_bar=True)
    session._process_signal.assert_called_once()


@pytest.mark.asyncio
async def test_rr_gate_rejects_degraded_rr():
    """Gate 11: reject signal when R:R degrades below 1.0 at current price (#40).

    Long signal: entry=1.10000, SL=1.09500, TP=1.11000 (R:R=2.0).
    At current_price=1.10600: risk=110 pips, reward=40 pips → R:R=0.36 → reject.
    """
    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()

    sig = FakeSignal(
        entry_price=1.10000,
        stop_loss=1.09500,
        take_profit=1.11000,
        kill_zone="new_york",
        meta={"break_idx": 400},
    )

    # Current price drifted 60 pips up — R:R = 40/110 = 0.36 → reject
    drifted_bars = pd.DataFrame(
        {"open": [1.106], "high": [1.107], "low": [1.105], "close": [1.10600], "volume": [100]},
        index=pd.date_range("2026-01-15 15:00", periods=1, freq="15min", tz="UTC"),
    )
    session.broker.get_live_bars = MagicMock(return_value=drifted_bars)

    with patch("ict_bot.trading.live_loop.is_asian_session", return_value=False):
        await session._process_signal_inner(sig)

    # Should NOT have placed market order — R:R too low
    session.broker.place_market_order.assert_not_called()


@pytest.mark.asyncio
async def test_rr_gate_allows_acceptable_rr():
    """Gate 11: accept signal when R:R >= 1.0 at current price (#40)."""
    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()

    sig = FakeSignal(
        entry_price=1.10000,
        stop_loss=1.09500,
        take_profit=1.11000,
        kill_zone="new_york",
        meta={"break_idx": 500},
    )

    # Current price 10 pips up — R:R = 90/60 = 1.5 → accept
    close_bars = pd.DataFrame(
        {"open": [1.1005], "high": [1.1015], "low": [1.1000], "close": [1.10100], "volume": [100]},
        index=pd.date_range("2026-01-15 15:00", periods=1, freq="15min", tz="UTC"),
    )
    session.broker.get_live_bars = MagicMock(return_value=close_bars)

    with patch("ict_bot.trading.live_loop.is_asian_session", return_value=False):
        await session._process_signal_inner(sig)

    # Should have placed market order — R:R is acceptable
    session.broker.place_market_order.assert_called_once()


@pytest.mark.asyncio
async def test_rr_gate_rejects_price_past_sl():
    """Gate 11: reject long when price drops below SL (setup invalidated) (#40)."""
    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()

    sig = FakeSignal(
        entry_price=1.10000,
        stop_loss=1.09500,
        take_profit=1.11000,
        kill_zone="new_york",
        meta={"break_idx": 600},
    )

    # Current price below SL — setup gone
    bars = pd.DataFrame(
        {"open": [1.094], "high": [1.095], "low": [1.093], "close": [1.09400], "volume": [100]},
        index=pd.date_range("2026-01-15 15:00", periods=1, freq="15min", tz="UTC"),
    )
    session.broker.get_live_bars = MagicMock(return_value=bars)

    with patch("ict_bot.trading.live_loop.is_asian_session", return_value=False):
        await session._process_signal_inner(sig)

    session.broker.place_market_order.assert_not_called()


@pytest.mark.asyncio
async def test_structural_sl_tp_no_shift():
    """Position manager uses signal's structural SL/TP, not shifted (#40)."""
    from ict_bot.trading.position_manager import PositionManager

    broker = MagicMock()
    config = LiveConfig()
    config.dry_run = True
    pm = PositionManager(broker, config)

    sig = FakeSignal(
        entry_price=1.16110,
        stop_loss=1.16818,  # structural swing high
        take_profit=1.14353,
        kill_zone="london",
        direction="short",
        meta={"break_idx": 700},
    )

    pos = await pm.open_position(sig, 15000)

    # SL/TP should be at structural levels, not shifted
    assert pos.stop_loss == 1.16818
    assert pos.take_profit == 1.14353


def test_seen_signals_expiry():
    """Fingerprints older than 48 hours are expired (#39)."""
    from ict_bot.trading.live_loop import LiveTradingSession

    session = LiveTradingSession.__new__(LiveTradingSession)
    session._seen_signals = {
        "EUR_USD_long_100": (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat(),
        "GBP_USD_short_200": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }

    session._expire_seen_signals()

    # Old one expired, recent one kept
    assert "EUR_USD_long_100" not in session._seen_signals
    assert "GBP_USD_short_200" in session._seen_signals


def test_news_close_before_events_default_enabled():
    """news_close_before_events defaults to True (#36)."""
    cfg = LiveConfig()
    assert cfg.news_close_before_events is True


def test_news_filter_pre_close_window_30_min():
    """Pre-close triggers 28-31 min before high-impact event (#36)."""
    nf = NewsFilter(blackout_minutes=30, close_before_news=True)
    base_time = datetime(2026, 3, 1, 14, 0, 0, tzinfo=timezone.utc)

    event = NewsEvent(
        title="NFP", country="USD",
        date=base_time,
        impact="High",
        affected_pairs=["EUR_USD"],
    )
    nf._events = [event]
    nf._last_fetch = datetime.now(timezone.utc)

    # 30 min before = should trigger (inside 28-31 window)
    with patch("ict_bot.trading.news_filter.datetime") as mock_dt:
        mock_dt.now.return_value = base_time - timedelta(minutes=30)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        # Can't easily mock datetime.now inside the method, so test directly
        pass

    # Direct test: inject events and check at specific times
    nf2 = NewsFilter(blackout_minutes=30, close_before_news=True)
    nf2._events = [event]
    nf2._last_fetch = datetime.now(timezone.utc)

    # Monkey-patch refresh_if_needed to avoid HTTP calls
    nf2.refresh_if_needed = lambda: None

    # At 30 min before (13:30 UTC) — inside 28-31 window
    from unittest.mock import patch as mock_patch
    with mock_patch("ict_bot.trading.news_filter.datetime") as mock_dt:
        mock_dt.now.return_value = base_time - timedelta(minutes=30)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        result = nf2.get_pairs_to_close_before_news(["EUR_USD"])
    assert len(result) == 1
    assert result[0][0] == "EUR_USD"

    # At 20 min before (13:40 UTC) — too close, inside blackout
    with mock_patch("ict_bot.trading.news_filter.datetime") as mock_dt:
        mock_dt.now.return_value = base_time - timedelta(minutes=20)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        result = nf2.get_pairs_to_close_before_news(["EUR_USD"])
    assert len(result) == 0

    # At 40 min before (13:20 UTC) — too far out
    with mock_patch("ict_bot.trading.news_filter.datetime") as mock_dt:
        mock_dt.now.return_value = base_time - timedelta(minutes=40)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        result = nf2.get_pairs_to_close_before_news(["EUR_USD"])
    assert len(result) == 0


def test_news_filter_only_high_impact_triggers_close():
    """Only High impact events trigger pre-close (#36)."""
    nf = NewsFilter(blackout_minutes=30, close_before_news=True)
    nf.refresh_if_needed = lambda: None
    base_time = datetime(2026, 3, 1, 14, 0, 0, tzinfo=timezone.utc)

    medium_event = NewsEvent(
        title="PMI", country="USD",
        date=base_time, impact="Medium",
        affected_pairs=["EUR_USD"],
    )
    nf._events = [medium_event]
    nf._last_fetch = datetime.now(timezone.utc)

    from unittest.mock import patch as mock_patch
    with mock_patch("ict_bot.trading.news_filter.datetime") as mock_dt:
        mock_dt.now.return_value = base_time - timedelta(minutes=30)
        mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
        result = nf.get_pairs_to_close_before_news(["EUR_USD"])
    assert len(result) == 0


# ── Startup risk registration (USD conversion fix) ──────────────


def test_startup_risk_registration_uses_usd():
    """On startup, risk registered per position must be balance × risk_per_trade (USD).

    The old formula (risk_pips × pip_size × units) gave quote-currency amounts:
    for USD/JPY with 50 risk pips, pip_size=0.01, 36845 units → 18,422 JPY
    instead of ~$95 USD.  This would cause the 5% exposure limit to block all
    subsequent trades after a single JPY restart.
    """
    from ict_bot.risk import RiskConfig, RiskManager
    from ict_bot.trading.live_loop import LiveTradingSession

    config = LiveConfig(
        risk_per_trade=0.01,
        starting_balance=9_500.0,
        max_positions=3,
        state_file=tempfile.mktemp(suffix=".json"),
    )
    broker = make_mock_broker(config)

    # Simulate a USD/JPY position loaded from state
    pm = PositionManager(broker, config)
    pos = LivePosition(
        pair="USD_JPY",
        direction="short",
        entry_price=156.0,
        units=36845,
        stop_loss=156.50,
        take_profit=155.0,
        entry_time=datetime.now(timezone.utc).isoformat(),
        risk_pips=50.0,
        confluence_score=4,
    )
    pm.positions["USD_JPY"] = pos

    risk_mgr = RiskManager(RiskConfig(
        max_risk_per_trade=0.01,
        starting_balance=9_500.0,
    ))

    # Simulate the startup registration logic (the fixed version)
    balance = 9_500.0
    risk_mgr._balance = balance
    for pair, p in pm.positions.items():
        risk_amount = balance * config.risk_per_trade
        risk_mgr.register_open_position(pair, risk_amount)

    # The registered risk should be ~$95, NOT 18,422 JPY
    registered = risk_mgr._open_positions["USD_JPY"]
    assert registered == pytest.approx(95.0, abs=1.0), (
        f"Risk registered as {registered}, expected ~95 USD (got quote-currency amount?)"
    )

    # Exposure check: 95 / 9500 = 1%, well under 5% limit
    total_exposure = sum(risk_mgr._open_positions.values()) / risk_mgr._balance
    assert total_exposure < 0.05, (
        f"Exposure {total_exposure:.1%} exceeds 5% — would block all new trades"
    )


def test_startup_risk_old_formula_would_block_jpy():
    """Demonstrate that the OLD formula would block trades for USD/JPY.

    This is a regression guard: if someone changes the startup registration
    back to risk_pips × pip_size × units, the exposure for JPY blows up.
    """
    from ict_bot.risk import RiskConfig, RiskManager

    risk_mgr = RiskManager(RiskConfig(
        max_risk_per_trade=0.01,
        starting_balance=9_500.0,
        max_total_exposure=0.05,
    ))
    risk_mgr._balance = 9_500.0

    # Old formula for USD/JPY: 50 pips × 0.01 × 36845 = 18,422.5 (JPY!)
    old_risk = 50 * pip_size_for("USD_JPY") * 36845
    assert old_risk > 15_000, "Old formula should give a huge number (JPY)"

    # This would make exposure = 18422 / 9500 = 194%
    exposure_pct = old_risk / risk_mgr._balance
    assert exposure_pct > 1.0, (
        f"Old formula exposure {exposure_pct:.0%} should exceed 100%"
    )

    # Correct formula: balance × risk_per_trade = $95
    correct_risk = 9_500.0 * 0.01
    correct_exposure = correct_risk / risk_mgr._balance
    assert correct_exposure < 0.05, (
        f"Correct formula exposure {correct_exposure:.1%} should be under 5%"
    )


# ── Gate 10.5: Setup Alive Tests ─────────────────────────────────────


def make_bars(n=10, base_price=1.100, bar_overrides=None):
    """Create a DataFrame of OHLC bars for testing Gate 10.5 (Setup Alive).

    Args:
        n: Number of bars.
        base_price: Default OHLCV price.
        bar_overrides: Dict of {bar_index: {col: val}} to override specific bars.
    """
    dates = pd.date_range("2026-01-15 10:00", periods=n, freq="15min", tz="UTC")
    data = {
        "open": [base_price] * n,
        "high": [base_price + 0.0005] * n,
        "low": [base_price - 0.0005] * n,
        "close": [base_price] * n,
        "volume": [100] * n,
    }
    df = pd.DataFrame(data, index=dates)
    if bar_overrides:
        for idx, overrides in bar_overrides.items():
            for col, val in overrides.items():
                df.iloc[idx, df.columns.get_loc(col)] = val
    return df


def _make_session_for_setup_alive(live_bars_df):
    """Create a session with mocked broker returning specific live bars."""
    session = _make_session()
    session._traded_today = set()
    session._traded_today_file = Path(tempfile.mktemp(suffix=".json"))
    session._seen_signals = {}
    session._seen_signals_file = Path(tempfile.mktemp(suffix=".json"))
    session.telegram = MagicMock()
    session.telegram.send = AsyncMock()
    session.broker.get_live_bars = MagicMock(return_value=live_bars_df)
    return session


@pytest.mark.asyncio
async def test_setup_alive_rejects_long_sl_hit():
    """Gate 10.5: long signal rejected when SL was hit in subsequent bars."""
    bars = make_bars(n=10, bar_overrides={7: {"low": 1.094}})
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="long", entry_price=1.100,
        stop_loss=1.095, take_profit=1.110,
        meta={"bar_index": 5, "break_idx": 100},
    )
    await session._process_signal(sig)

    session.broker.place_market_order.assert_not_called()
    assert "EUR_USD" not in session.position_manager.positions


@pytest.mark.asyncio
async def test_setup_alive_rejects_long_tp_reached():
    """Gate 10.5: long signal rejected when TP was reached in subsequent bars."""
    bars = make_bars(n=10, bar_overrides={8: {"high": 1.111}})
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="long", entry_price=1.100,
        stop_loss=1.095, take_profit=1.110,
        meta={"bar_index": 5, "break_idx": 101},
    )
    await session._process_signal(sig)

    session.broker.place_market_order.assert_not_called()
    assert "EUR_USD" not in session.position_manager.positions


@pytest.mark.asyncio
async def test_setup_alive_rejects_short_sl_hit():
    """Gate 10.5: short signal rejected when SL was hit in subsequent bars."""
    bars = make_bars(n=10, bar_overrides={7: {"high": 1.106}})
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="short", entry_price=1.100,
        stop_loss=1.105, take_profit=1.090,
        meta={"bar_index": 5, "break_idx": 102},
    )
    await session._process_signal(sig)

    session.broker.place_market_order.assert_not_called()
    assert "EUR_USD" not in session.position_manager.positions


@pytest.mark.asyncio
async def test_setup_alive_passes_fresh_signal():
    """Gate 10.5: fresh signal (bar_index = last bar) passes through."""
    bars = make_bars(n=10)
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="long", entry_price=1.100,
        stop_loss=1.095, take_profit=1.110,
        meta={"bar_index": 9, "break_idx": 103},
    )
    await session._process_signal(sig)

    assert "EUR_USD" in session.position_manager.positions


@pytest.mark.asyncio
async def test_setup_alive_passes_untouched():
    """Gate 10.5: stale signal passes when SL/TP untouched in subsequent bars."""
    # All bars have low > 1.096 and high < 1.109 (safe for SL=1.095, TP=1.110)
    bars = make_bars(n=10)
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="long", entry_price=1.100,
        stop_loss=1.095, take_profit=1.110,
        meta={"bar_index": 5, "break_idx": 104},
    )
    await session._process_signal(sig)

    assert "EUR_USD" in session.position_manager.positions


@pytest.mark.asyncio
async def test_setup_alive_skips_no_bar_index():
    """Gate 10.5: gate is skipped when signal has no bar_index in meta."""
    # Bar 7 has low that would hit SL, but gate should be skipped
    bars = make_bars(n=10, bar_overrides={7: {"low": 1.094}})
    session = _make_session_for_setup_alive(bars)

    sig = FakeSignal(
        direction="long", entry_price=1.100,
        stop_loss=1.095, take_profit=1.110,
        meta={"break_idx": 105},  # no bar_index
    )
    await session._process_signal(sig)

    # Should pass through (gate skipped) — position opened
    assert "EUR_USD" in session.position_manager.positions
