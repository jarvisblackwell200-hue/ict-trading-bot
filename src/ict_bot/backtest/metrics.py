"""Performance metrics for backtesting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Summary metrics from a backtest run."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_rr: float
    expectancy: float         # avg profit per trade in R
    profit_factor: float
    total_pnl_pips: float
    max_drawdown_pips: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    avg_trade_duration_hours: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float

    def passed(self, min_win_rate: float = 0.40, min_rr: float = 1.2, min_trades: int = 30) -> bool:
        """Check if backtest meets minimum viability criteria."""
        return (
            self.total_trades >= min_trades
            and self.win_rate >= min_win_rate
            and self.avg_rr >= min_rr
            and self.expectancy > 0
        )

    def summary(self) -> str:
        verdict = "PASS" if self.passed() else "FAIL"
        return (
            f"\n{'='*60}\n"
            f"  BACKTEST RESULTS [{verdict}]\n"
            f"{'='*60}\n"
            f"  Total trades:      {self.total_trades}\n"
            f"  Wins / Losses:     {self.wins} / {self.losses}\n"
            f"  Win rate:          {self.win_rate:.1%}\n"
            f"  Avg R:R:           {self.avg_rr:.2f}\n"
            f"  Expectancy:        {self.expectancy:+.3f} R per trade\n"
            f"  Profit factor:     {self.profit_factor:.2f}\n"
            f"  Total P&L:         {self.total_pnl_pips:+.1f} pips\n"
            f"  Max drawdown:      {self.max_drawdown_pips:.1f} pips ({self.max_drawdown_pct:.1%})\n"
            f"  Sharpe ratio:      {self.sharpe_ratio:.2f}\n"
            f"  Calmar ratio:      {self.calmar_ratio:.2f}\n"
            f"  Avg duration:      {self.avg_trade_duration_hours:.1f} hours\n"
            f"  Max consec wins:   {self.max_consecutive_wins}\n"
            f"  Max consec losses: {self.max_consecutive_losses}\n"
            f"  Long:  {self.long_trades} trades ({self.long_win_rate:.1%} win)\n"
            f"  Short: {self.short_trades} trades ({self.short_win_rate:.1%} win)\n"
            f"{'='*60}"
        )


def calculate_metrics(trades: list[dict], pip_size: float = 0.0001) -> BacktestMetrics:
    """
    Calculate backtest metrics from a list of completed trades.

    Each trade dict should have: entry_price, exit_price, direction,
    stop_loss, take_profit, entry_time, exit_time, pnl_pips.
    """
    if not trades:
        return BacktestMetrics(
            total_trades=0, wins=0, losses=0, win_rate=0, avg_rr=0,
            expectancy=0, profit_factor=0, total_pnl_pips=0,
            max_drawdown_pips=0, max_drawdown_pct=0, sharpe_ratio=0,
            calmar_ratio=0, avg_trade_duration_hours=0,
            max_consecutive_wins=0, max_consecutive_losses=0,
            long_trades=0, short_trades=0, long_win_rate=0, short_win_rate=0,
        )

    df = pd.DataFrame(trades)

    wins = df[df["pnl_pips"] > 0]
    losses = df[df["pnl_pips"] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    total = len(df)

    win_rate = n_wins / total if total > 0 else 0

    # R:R realized
    df["r_multiple"] = df["pnl_pips"] / df["risk_pips"]
    avg_rr = df[df["r_multiple"] > 0]["r_multiple"].mean() if n_wins > 0 else 0
    avg_loss_r = abs(df[df["r_multiple"] <= 0]["r_multiple"].mean()) if n_losses > 0 else 1

    # Expectancy in R
    expectancy = df["r_multiple"].mean()

    # Profit factor
    gross_profit = wins["pnl_pips"].sum() if n_wins > 0 else 0
    gross_loss = abs(losses["pnl_pips"].sum()) if n_losses > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = df["pnl_pips"].sum()

    # Drawdown
    cumulative = df["pnl_pips"].cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    max_dd_pips = drawdown.max()
    max_dd_pct = max_dd_pips / running_max.max() if running_max.max() > 0 else 0

    # Sharpe (annualized, assuming ~250 trading days)
    if len(df) > 1:
        daily_returns = df["pnl_pips"]
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(250) if daily_returns.std() > 0 else 0
    else:
        sharpe = 0

    # Calmar
    annual_return = total_pnl * (250 / max(len(df), 1))
    calmar = annual_return / max_dd_pips if max_dd_pips > 0 else 0

    # Trade duration
    if "entry_time" in df.columns and "exit_time" in df.columns:
        durations = (pd.to_datetime(df["exit_time"]) - pd.to_datetime(df["entry_time"]))
        avg_duration_hours = durations.mean().total_seconds() / 3600
    else:
        avg_duration_hours = 0

    # Consecutive wins/losses
    is_win = (df["pnl_pips"] > 0).astype(int)
    max_consec_wins = _max_consecutive(is_win, 1)
    max_consec_losses = _max_consecutive(is_win, 0)

    # By direction
    longs = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]
    long_wr = (longs["pnl_pips"] > 0).mean() if len(longs) > 0 else 0
    short_wr = (shorts["pnl_pips"] > 0).mean() if len(shorts) > 0 else 0

    return BacktestMetrics(
        total_trades=total,
        wins=n_wins,
        losses=n_losses,
        win_rate=win_rate,
        avg_rr=avg_rr,
        expectancy=expectancy,
        profit_factor=profit_factor,
        total_pnl_pips=total_pnl,
        max_drawdown_pips=max_dd_pips,
        max_drawdown_pct=max_dd_pct,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        avg_trade_duration_hours=avg_duration_hours,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        long_trades=len(longs),
        short_trades=len(shorts),
        long_win_rate=long_wr,
        short_win_rate=short_wr,
    )


def _max_consecutive(series: pd.Series, value: int) -> int:
    """Count max consecutive occurrences of a value."""
    if len(series) == 0:
        return 0
    groups = (series != value).cumsum()
    counts = series.groupby(groups).sum()
    return int(counts.max()) if len(counts) > 0 else 0
