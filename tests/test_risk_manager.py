"""Unit tests for the RiskManager module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from ict_bot.risk import RiskConfig, RiskDecision, RiskManager


# ── Helpers ─────────────────────────────────────────────────────────


@dataclass
class FakeSignal:
    """Minimal signal stub for testing."""
    direction: str
    entry_price: float
    stop_loss: float
    pair: str = "EUR_USD"
    take_profit: float = 0.0
    confluence_score: int = 4
    confluences: dict = None
    kill_zone: Optional[str] = None

    def __post_init__(self):
        if self.confluences is None:
            self.confluences = {}


PIP = 0.0001  # EUR/USD pip size


def _long_signal(entry=1.1000, sl=1.0950, pair="EUR_USD"):
    return FakeSignal("long", entry, sl, pair=pair)


def _short_signal(entry=1.1000, sl=1.1050, pair="EUR_USD"):
    return FakeSignal("short", entry, sl, pair=pair)


# ── Tests ───────────────────────────────────────────────────────────


def test_approve_basic_signal():
    """1. A normal signal is approved with valid position sizing."""
    rm = RiskManager()
    sig = _long_signal()
    decision = rm.evaluate_signal(sig, PIP)

    assert decision.approved is True
    assert decision.reason == "approved"
    assert decision.position_size > 0
    assert decision.risk_amount > 0


def test_position_size_formula():
    """2. Position size matches the fixed-fractional formula."""
    config = RiskConfig(max_risk_per_trade=0.02, starting_balance=50_000.0)
    rm = RiskManager(config)

    sig = _long_signal(entry=1.1000, sl=1.0950)  # 50 pip SL
    decision = rm.evaluate_signal(sig, PIP)

    expected_risk = 50_000.0 * 0.02           # $1,000
    risk_pips = (1.1000 - 1.0950) / PIP       # 50 pips
    expected_pip_value = expected_risk / risk_pips  # $20/pip

    assert decision.approved
    assert decision.risk_amount == pytest.approx(expected_risk)
    assert decision.position_size == pytest.approx(expected_pip_value)


def test_daily_loss_circuit_breaker():
    """3. Daily loss exceeding threshold triggers circuit breaker and blocks."""
    config = RiskConfig(
        max_daily_loss=0.02,
        starting_balance=10_000.0,
    )
    rm = RiskManager(config)

    # Lose $200 (2% of starting balance) to trigger CB
    rm.record_trade_result(-200.0, "EUR_USD")

    assert rm.is_circuit_broken is True
    assert rm.daily_pnl == -200.0

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is False
    assert "circuit breaker" in decision.reason


def test_consecutive_losses_pause_and_reset():
    """4. Consecutive losses pause trading; a win resets the counter."""
    config = RiskConfig(max_consecutive_losses=2)
    rm = RiskManager(config)

    rm.record_trade_result(-50.0, "EUR_USD")
    rm.record_trade_result(-50.0, "GBP_USD")

    assert rm.consecutive_losses == 2

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is False
    assert "consecutive" in decision.reason

    # A win resets the counter
    rm.record_trade_result(100.0, "EUR_USD")
    assert rm.consecutive_losses == 0

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is True


def test_max_drawdown_kill_switch_latches():
    """5. Max drawdown triggers kill switch that latches (survives reset_daily)."""
    config = RiskConfig(
        max_drawdown=0.05,
        starting_balance=10_000.0,
        max_daily_loss=1.0,  # high so CB doesn't interfere
    )
    rm = RiskManager(config)

    # Lose 5% → $500
    rm.record_trade_result(-500.0, "EUR_USD")

    assert rm.is_killed is True
    assert rm.current_drawdown_pct == pytest.approx(0.05)

    # Kill switch blocks signals
    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is False
    assert "kill switch" in decision.reason

    # reset_daily does NOT clear kill switch
    rm.reset_daily()
    assert rm.is_killed is True

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is False


def test_exposure_limit_blocks():
    """6. Exposure limit blocks when total open risk is at the cap."""
    config = RiskConfig(
        max_total_exposure=0.03,   # 3%
        max_risk_per_trade=0.02,   # 2% per trade
        starting_balance=10_000.0,
    )
    rm = RiskManager(config)

    # Register one position that uses 2% exposure ($200)
    rm.register_open_position("EUR_USD", 200.0)

    # Next trade would add another 2% → 4% > 3% cap
    decision = rm.evaluate_signal(_long_signal(pair="GBP_USD"), PIP)
    assert decision.approved is False
    assert "exposure" in decision.reason


def test_exposure_includes_new_position():
    """7. Exposure check accounts for the new position's risk."""
    config = RiskConfig(
        max_total_exposure=0.03,
        max_risk_per_trade=0.02,
        starting_balance=10_000.0,
    )
    rm = RiskManager(config)

    # No existing positions, but one trade at 2% is fine (2% < 3%)
    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is True

    # Register it, then try another — would be 4% > 3%
    rm.register_open_position("EUR_USD", decision.risk_amount)
    decision2 = rm.evaluate_signal(_long_signal(pair="GBP_USD"), PIP)
    assert decision2.approved is False


def test_reset_daily_clears_cb_not_kill():
    """8. reset_daily() clears circuit breaker and daily P&L, not kill switch."""
    config = RiskConfig(
        max_daily_loss=0.01,
        max_drawdown=0.05,
        starting_balance=10_000.0,
    )
    rm = RiskManager(config)

    # Trigger CB
    rm.record_trade_result(-100.0, "EUR_USD")
    assert rm.is_circuit_broken is True
    assert rm.daily_pnl == -100.0

    rm.reset_daily()
    assert rm.is_circuit_broken is False
    assert rm.daily_pnl == 0.0
    # Balance is NOT restored
    assert rm.balance == pytest.approx(9_900.0)


def test_reset_all_restores_state():
    """9. reset_all() restores all state including kill switch."""
    config = RiskConfig(
        max_drawdown=0.05,
        starting_balance=10_000.0,
        max_daily_loss=1.0,
    )
    rm = RiskManager(config)

    rm.record_trade_result(-500.0)
    rm.record_trade_result(-500.0)
    assert rm.is_killed is True
    assert rm.balance == pytest.approx(9_000.0)

    rm.reset_all()
    assert rm.is_killed is False
    assert rm.balance == pytest.approx(10_000.0)
    assert rm.daily_pnl == 0.0
    assert rm.consecutive_losses == 0
    assert rm.current_drawdown_pct == 0.0


def test_zero_sl_distance_rejected():
    """10. Signal with SL at entry price is rejected."""
    rm = RiskManager()
    sig = FakeSignal("long", 1.1000, 1.1000)  # SL == entry
    decision = rm.evaluate_signal(sig, PIP)

    assert decision.approved is False
    assert "invalid SL" in decision.reason


def test_custom_config_thresholds():
    """11. Custom config values are respected."""
    config = RiskConfig(
        max_risk_per_trade=0.005,
        max_consecutive_losses=5,
        max_daily_loss=0.10,
        max_drawdown=0.20,
        starting_balance=100_000.0,
    )
    rm = RiskManager(config)

    # 5 consecutive losses should be fine (limit is 5, so 4 is ok)
    for _ in range(4):
        rm.record_trade_result(-100.0)
    assert rm.consecutive_losses == 4

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is True

    # 5th loss should block
    rm.record_trade_result(-100.0)
    assert rm.consecutive_losses == 5

    decision = rm.evaluate_signal(_long_signal(), PIP)
    assert decision.approved is False

    # Verify risk amount uses 0.5% of current balance
    rm.record_trade_result(100.0)  # reset consecutive
    decision = rm.evaluate_signal(_long_signal(), PIP)
    expected_risk = rm.balance * 0.005
    assert decision.risk_amount == pytest.approx(expected_risk)


def test_state_tracking_win_loss_sequence():
    """12. State tracks correctly across a realistic win/loss sequence."""
    config = RiskConfig(
        max_consecutive_losses=3,
        max_daily_loss=0.05,
        starting_balance=10_000.0,
    )
    rm = RiskManager(config)

    # Win → loss → loss → win → loss
    rm.record_trade_result(200.0, "EUR_USD")
    assert rm.consecutive_losses == 0
    assert rm.balance == pytest.approx(10_200.0)

    rm.record_trade_result(-100.0, "GBP_USD")
    assert rm.consecutive_losses == 1

    rm.record_trade_result(-100.0, "EUR_USD")
    assert rm.consecutive_losses == 2

    rm.record_trade_result(150.0, "GBP_USD")
    assert rm.consecutive_losses == 0
    assert rm.balance == pytest.approx(10_150.0)

    rm.record_trade_result(-50.0, "EUR_USD")
    assert rm.consecutive_losses == 1
    assert rm.balance == pytest.approx(10_100.0)

    # Daily P&L should be net of all trades
    assert rm.daily_pnl == pytest.approx(100.0)  # +200 -100 -100 +150 -50

    # Peak balance was 10,200 after first win
    assert rm.current_drawdown_pct == pytest.approx(
        (10_200.0 - 10_100.0) / 10_200.0
    )
