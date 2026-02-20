"""
Risk management module — circuit breakers, position sizing, and exposure limits.

Provides a stateful RiskManager that gates trade entry with:
  - Fixed-fractional position sizing
  - Daily loss circuit breaker
  - Consecutive loss pause
  - Max drawdown kill switch (latches until manual reset)
  - Total open exposure limit
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Thresholds for risk management."""
    max_risk_per_trade: float = 0.01        # 1% of balance per trade
    max_total_exposure: float = 0.05        # 5% total open risk
    max_daily_loss: float = 0.03            # 3% daily loss limit
    max_consecutive_losses: int = 3
    max_drawdown: float = 0.10              # 10% from peak
    starting_balance: float = 10_000.0


@dataclass
class RiskDecision:
    """Result of evaluating a signal through risk management."""
    approved: bool
    reason: str
    position_size: float    # pip value ($ per pip)
    risk_amount: float      # $ risked on this trade


class RiskManager:
    """
    Stateful risk manager that approves/rejects signals and tracks P&L.

    Kill switch latches — only reset_all() clears it (requires human intervention).
    Circuit breaker resets daily via reset_daily().
    Consecutive loss counter auto-resets on a win.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config = config or RiskConfig()
        self._balance = self._config.starting_balance
        self._peak_balance = self._balance
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._killed = False
        self._circuit_broken = False
        self._open_positions: dict[str, float] = {}  # pair -> risk_amount

    # ── Properties ──────────────────────────────────────────────────

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_balance == 0:
            return 0.0
        return (self._peak_balance - self._balance) / self._peak_balance

    @property
    def total_exposure_pct(self) -> float:
        if self._balance == 0:
            return 0.0
        return sum(self._open_positions.values()) / self._balance

    @property
    def is_killed(self) -> bool:
        return self._killed

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken

    # ── Core API ────────────────────────────────────────────────────

    def evaluate_signal(self, signal, pip_size: float) -> RiskDecision:
        """
        Check all risk gates and return a decision.

        Parameters
        ----------
        signal : Signal
            Must have .direction, .entry_price, .stop_loss, .pair attributes.
        pip_size : float
            Pip size for the instrument (e.g. 0.0001 for EUR/USD).
        """
        # 1. Kill switch (latches)
        if self._killed:
            return RiskDecision(False, "kill switch active", 0.0, 0.0)

        # 2. Circuit breaker (daily loss)
        if self._circuit_broken:
            return RiskDecision(False, "daily circuit breaker active", 0.0, 0.0)

        # 3. Consecutive losses
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            return RiskDecision(False, "consecutive loss limit reached", 0.0, 0.0)

        # 4. Calculate position size
        pip_value, risk_amount = self.calculate_position_size(signal, pip_size)
        if pip_value <= 0:
            return RiskDecision(False, "invalid SL distance", 0.0, 0.0)

        # 5. Exposure check (existing + new position)
        new_exposure = sum(self._open_positions.values()) + risk_amount
        if new_exposure / self._balance > self._config.max_total_exposure:
            return RiskDecision(False, "exposure limit exceeded", 0.0, 0.0)

        return RiskDecision(True, "approved", pip_value, risk_amount)

    def calculate_position_size(self, signal, pip_size: float) -> tuple[float, float]:
        """
        Fixed-fractional position sizing.

        Returns (pip_value, risk_amount).
        pip_value is 0.0 if SL distance is zero or negative.
        """
        if signal.direction == "long":
            risk_pips = (signal.entry_price - signal.stop_loss) / pip_size
        else:
            risk_pips = (signal.stop_loss - signal.entry_price) / pip_size

        if risk_pips <= 0:
            return 0.0, 0.0

        risk_amount = self._balance * self._config.max_risk_per_trade
        pip_value = risk_amount / risk_pips
        return pip_value, risk_amount

    def record_trade_result(self, pnl_amount: float, pair: str = "") -> None:
        """
        Update state after a trade closes.

        Parameters
        ----------
        pnl_amount : float
            Profit/loss in account currency (positive = win).
        pair : str
            Pair to remove from open positions (if tracked).
        """
        self._balance += pnl_amount

        # Track peak for drawdown
        if self._balance > self._peak_balance:
            self._peak_balance = self._balance

        # Daily P&L
        self._daily_pnl += pnl_amount

        # Consecutive losses
        if pnl_amount < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Remove from open positions
        if pair and pair in self._open_positions:
            del self._open_positions[pair]

        # Check daily loss circuit breaker
        if abs(self._daily_pnl) >= self._config.max_daily_loss * self._config.starting_balance:
            if self._daily_pnl < 0:
                self._circuit_broken = True

        # Check drawdown kill switch
        if self.current_drawdown_pct >= self._config.max_drawdown:
            self._killed = True

    def register_open_position(self, pair: str, risk_amount: float) -> None:
        """Track a newly opened position's risk."""
        self._open_positions[pair] = risk_amount

    def close_position(self, pair: str) -> None:
        """Remove a position from open tracking."""
        self._open_positions.pop(pair, None)

    def reset_daily(self) -> None:
        """Reset daily P&L and circuit breaker. Does NOT clear kill switch."""
        self._daily_pnl = 0.0
        self._circuit_broken = False

    def reset_all(self) -> None:
        """Full reset — restores all state including kill switch."""
        self._balance = self._config.starting_balance
        self._peak_balance = self._balance
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._killed = False
        self._circuit_broken = False
        self._open_positions.clear()

    # ── State Persistence ──────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Save risk manager state to JSON (atomic write)."""
        state = {
            "balance": self._balance,
            "peak_balance": self._peak_balance,
            "daily_pnl": self._daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "killed": self._killed,
            "circuit_broken": self._circuit_broken,
            "open_positions": self._open_positions,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def load_state(self, path: str) -> None:
        """Load risk manager state from JSON."""
        p = Path(path)
        if not p.exists():
            logger.info("No risk state file at %s", path)
            return
        try:
            data = json.loads(p.read_text())
            self._balance = data.get("balance", self._balance)
            self._peak_balance = data.get("peak_balance", self._peak_balance)
            self._daily_pnl = data.get("daily_pnl", self._daily_pnl)
            self._consecutive_losses = data.get("consecutive_losses", self._consecutive_losses)
            self._killed = data.get("killed", self._killed)
            self._circuit_broken = data.get("circuit_broken", self._circuit_broken)
            self._open_positions = data.get("open_positions", self._open_positions)
            logger.info(
                "Loaded risk state: balance=$%.2f, consecutive_losses=%d, "
                "killed=%s, circuit_broken=%s",
                self._balance, self._consecutive_losses,
                self._killed, self._circuit_broken,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load risk state from %s: %s", path, exc)
