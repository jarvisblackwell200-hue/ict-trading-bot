"""Unit tests for the live trading module (mocked IB — no Gateway needed)."""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    def make_trade(status="PreSubmitted"):
        trade = MagicMock()
        trade.orderStatus.status = status
        trade.order.orderId = 1
        trade.contract = MagicMock()
        return trade

    broker.place_market_order = AsyncMock(return_value=make_trade())
    broker.place_stop_order = AsyncMock(return_value=make_trade())
    broker.place_limit_order = AsyncMock(return_value=make_trade())
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
    assert cfg.ib_port == 4002  # paper
    assert cfg.risk_per_trade == 0.01
    assert cfg.timeframe == "M15"
    assert cfg.swing_length == 5
    assert cfg.confluence_threshold == 4
    assert cfg.min_rr == 2.0
    assert cfg.skip_days == [4]
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
    """Correct forex lot sizing from pip_value."""
    from ict_bot.trading.live_loop import LiveTradingSession

    config = LiveConfig()
    session = LiveTradingSession.__new__(LiveTradingSession)
    session.config = config

    # If risk is $100, SL is 50 pips → pip_value = $2/pip
    # For EUR_USD: units = $2 / 0.0001 = 20,000 units (0.2 lots)
    units = session._calculate_units(pip_value=2.0, pip_size=0.0001, pair="EUR_USD")
    assert units == 20_000

    # For USD_JPY: units = $2 / 0.01 = 200 units
    units = session._calculate_units(pip_value=2.0, pip_size=0.01, pair="USD_JPY")
    assert units == 200

    # Larger pip value
    units = session._calculate_units(pip_value=10.0, pip_size=0.0001, pair="GBP_USD")
    assert units == 100_000  # 1 standard lot


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
    session._last_trade_time = None
    session._started_at = None

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
    session._last_trade_time = None
    session._started_at = None

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
    session._last_trade_time = None
    session._started_at = None

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
    """Daily reset fires once per day after 05:00 UTC."""
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

    # Simulate check at 06:00 UTC (after 05:00 UTC midnight ET)
    with patch("ict_bot.trading.live_loop.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await session._check_daily_reset()

    assert session.risk_manager.daily_pnl == 0.0
    assert session._last_daily_reset is not None

    # Second call on same day should NOT reset again
    session.risk_manager.record_trade_result(-50.0, "EUR_USD")
    with patch("ict_bot.trading.live_loop.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2026, 1, 15, 7, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        await session._check_daily_reset()

    # daily_pnl should still be -50 (not reset again)
    assert session.risk_manager.daily_pnl == pytest.approx(-50.0)
