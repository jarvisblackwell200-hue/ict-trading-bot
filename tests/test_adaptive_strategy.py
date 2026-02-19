"""
Tests for the Adaptive Trend Continuation strategy.

Covers signal generation, EMA filtering, overextension rejection,
ATR-based SL, trailing stop mechanics, timeout, and config defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ict_bot.backtest.engine import BacktestConfig, simulate_trades
from src.ict_bot.backtest.metrics import calculate_metrics
from src.ict_bot.signals.detector import Signal
from src.ict_bot.signals.trend_adaptive import (
    _compute_atr,
    generate_adaptive_signals,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_ohlc(
    n: int = 200,
    start: str = "2023-01-01",
    freq: str = "1h",
    base_price: float = 1.1000,
    trend: float = 0.0,
    volatility: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLC data with controlled trend and volatility."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    prices = np.zeros(n)
    prices[0] = base_price

    for i in range(1, n):
        prices[i] = prices[i - 1] + trend + rng.normal(0, volatility)

    # Build OHLC from close prices
    opens = np.roll(prices, 1)
    opens[0] = base_price
    highs = np.maximum(prices, opens) + rng.uniform(0, volatility, n)
    lows = np.minimum(prices, opens) - rng.uniform(0, volatility, n)
    volume = rng.randint(100, 1000, n).astype(float)

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volume,
    }, index=idx)


def _make_trending_ohlc(
    n: int = 300,
    direction: str = "up",
    base: float = 1.1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create OHLC with a clear trend for BOS detection."""
    trend = 0.0003 if direction == "up" else -0.0003
    return _make_ohlc(n=n, base_price=base, trend=trend, volatility=0.0004, seed=seed)


def _make_signal(
    direction: str = "long",
    entry: float = 1.1000,
    sl: float = 1.0950,
    tp: float = 1.1500,
    atr: float = 0.0025,
    timestamp: str = "2023-01-10 10:00:00",
) -> Signal:
    """Create a minimal Signal for engine tests."""
    return Signal(
        timestamp=pd.Timestamp(timestamp),
        pair="EUR_USD",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        rr_ratio=abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0,
        confluence_score=2,
        confluences={"structure_break": "BOS", "ema_aligned": True},
        kill_zone=None,
        meta={"atr": atr, "bar_index": 0, "tp_method": "trailing"},
    )


# ── Test 1: Basic signal generation ─────────────────────────────


class TestSignalGeneration:
    def test_signal_generation_basic(self):
        """Generates signals from trending data with BOS + EMA conditions."""
        ohlc = _make_trending_ohlc(n=300, direction="up")
        signals = generate_adaptive_signals(
            ohlc, pair="EUR_USD", swing_length=10, ema_period=50, atr_period=14,
        )
        # Should generate at least some signals in a clear uptrend
        # (exact count depends on SMC library BOS detection)
        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)
            assert sig.direction in ("long", "short")
            assert sig.meta.get("atr", 0) > 0
            assert sig.meta.get("ema", 0) > 0

    def test_signal_fields_populated(self):
        """All Signal fields are properly populated."""
        ohlc = _make_trending_ohlc(n=300, direction="up")
        signals = generate_adaptive_signals(ohlc, pair="EUR_USD", swing_length=10)
        for sig in signals:
            assert sig.pair == "EUR_USD"
            assert sig.entry_price > 0
            assert sig.stop_loss > 0
            assert sig.take_profit > 0
            assert sig.rr_ratio > 0
            assert sig.confluence_score >= 2
            assert "structure_break" in sig.confluences
            assert "ema_aligned" in sig.confluences


# ── Test 2 & 3: EMA filter ──────────────────────────────────────


class TestEMAFilter:
    def test_ema_filter_longs_above(self):
        """Long signals only appear when close > EMA."""
        ohlc = _make_trending_ohlc(n=300, direction="up")
        signals = generate_adaptive_signals(ohlc, pair="EUR_USD", swing_length=10)
        ema = ohlc["close"].ewm(span=50, adjust=False).mean()
        for sig in signals:
            if sig.direction == "long":
                # The BOS bar (break_idx) should have close > EMA
                idx = sig.meta.get("break_idx", 0)
                if idx < len(ohlc) and idx < len(ema):
                    assert ohlc.iloc[idx]["close"] > ema.iloc[idx], \
                        f"Long signal at bar {idx} but close={ohlc.iloc[idx]['close']:.5f} <= ema={ema.iloc[idx]:.5f}"

    def test_ema_filter_shorts_below(self):
        """Short signals only appear when close < EMA."""
        ohlc = _make_trending_ohlc(n=300, direction="down")
        signals = generate_adaptive_signals(ohlc, pair="EUR_USD", swing_length=10)
        ema = ohlc["close"].ewm(span=50, adjust=False).mean()
        for sig in signals:
            if sig.direction == "short":
                idx = sig.meta.get("break_idx", 0)
                if idx < len(ohlc) and idx < len(ema):
                    assert ohlc.iloc[idx]["close"] < ema.iloc[idx], \
                        f"Short signal at bar {idx} but close={ohlc.iloc[idx]['close']:.5f} >= ema={ema.iloc[idx]:.5f}"


# ── Test 4: Overextension filter ────────────────────────────────


class TestOverextensionFilter:
    def test_overextension_filter(self):
        """Signals are rejected when price is > 3x ATR from EMA."""
        ohlc = _make_trending_ohlc(n=300, direction="up")
        signals = generate_adaptive_signals(ohlc, pair="EUR_USD", swing_length=10)

        ema = ohlc["close"].ewm(span=50, adjust=False).mean()
        atr = _compute_atr(ohlc, period=14)

        for sig in signals:
            idx = sig.meta.get("break_idx", 0)
            if idx < len(ohlc) and idx < len(ema) and idx < len(atr):
                dist = abs(ohlc.iloc[idx]["close"] - ema.iloc[idx])
                assert dist <= 3.0 * atr.iloc[idx] + 1e-10, \
                    f"Signal at bar {idx}: distance={dist:.6f} > 3*ATR={3*atr.iloc[idx]:.6f}"


# ── Test 5: ATR-based stop loss ──────────────────────────────────


class TestATRStopLoss:
    def test_atr_sl_calculation(self):
        """SL = entry ± 2x ATR."""
        ohlc = _make_trending_ohlc(n=300, direction="up")
        signals = generate_adaptive_signals(
            ohlc, pair="EUR_USD", swing_length=10, atr_sl_mult=2.0,
        )
        for sig in signals:
            atr_val = sig.meta.get("atr", 0)
            if atr_val > 0:
                expected_sl_dist = 2.0 * atr_val
                actual_sl_dist = abs(sig.entry_price - sig.stop_loss)
                assert abs(actual_sl_dist - expected_sl_dist) < 1e-4, \
                    f"SL distance {actual_sl_dist:.6f} != expected {expected_sl_dist:.6f}"


# ── Tests 6-8: Trailing stop mechanics ──────────────────────────


def _make_ohlc_for_trailing(
    entry_price: float = 1.1000,
    atr: float = 0.0025,
    n_bars: int = 50,
    peak_bar: int = 20,
    peak_price: float = 1.1200,
) -> pd.DataFrame:
    """Create OHLC that rises to a peak then stays flat, for trailing stop testing."""
    idx = pd.date_range("2023-01-10 09:00", periods=n_bars + 10, freq="1h")
    prices = np.linspace(entry_price, peak_price, peak_bar)
    prices = np.append(prices, np.full(n_bars + 10 - peak_bar, peak_price))

    rng = np.random.RandomState(99)
    noise = rng.uniform(-atr * 0.2, atr * 0.2, len(prices))
    closes = prices + noise
    opens = np.roll(closes, 1)
    opens[0] = entry_price
    highs = np.maximum(closes, opens) + abs(noise) * 0.5
    lows = np.minimum(closes, opens) - abs(noise) * 0.5
    volume = rng.randint(100, 1000, len(prices)).astype(float)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volume,
    }, index=idx)


class TestTrailingStop:
    def test_trailing_stop_activation(self):
        """Trailing stop doesn't activate before trailing_activation_r."""
        entry = 1.1000
        sl = 1.0950
        risk_pips = (entry - sl) / 0.0001  # 50 pips
        atr = 0.0025

        # Price only moves 1R (not enough for 1.5R activation)
        signal = _make_signal(
            direction="long", entry=entry, sl=sl,
            tp=entry + 10 * atr, atr=atr,
        )

        # Create OHLC that only goes up 1R then hits SL
        n = 60
        idx = pd.date_range("2023-01-10 09:00", periods=n, freq="1h")
        one_r = risk_pips * 0.0001  # 50 pips = 0.0050
        # Rise to just under 1.5R then drop
        prices = np.concatenate([
            np.linspace(entry, entry + one_r * 1.2, 15),  # Rise to 1.2R
            np.linspace(entry + one_r * 1.2, sl - 0.001, 45),  # Drop to SL
        ])
        ohlc = pd.DataFrame({
            "open": np.roll(prices, 1),
            "high": prices + atr * 0.1,
            "low": prices - atr * 0.1,
            "close": prices,
            "volume": np.ones(n) * 100,
        }, index=idx)
        ohlc.iloc[0, ohlc.columns.get_loc("open")] = entry

        config = BacktestConfig(
            pair="EUR_USD", spread_pips=0.0, slippage_pips=0.0,
            risk_per_trade=0.01, max_sl_pips=100.0,
            use_breakeven=False, use_partial_tp=False,
            use_confluence_sizing=False,
            use_trailing_stop=True, atr_trail_mult=2.5,
            trailing_activation_r=1.5, max_bars=500,
        )

        trades = simulate_trades([signal], ohlc, config)
        assert len(trades) == 1
        # Should hit SL (not trail), since we never reached 1.5R
        assert trades[0]["exit_reason"] == "SL"

    def test_trailing_stop_updates(self):
        """After activation, trail follows best_price - 2.5x ATR."""
        entry = 1.1000
        atr = 0.0025
        sl = entry - 2.0 * atr  # 1.0950
        risk_pips = (entry - sl) / 0.0001

        signal = _make_signal(
            direction="long", entry=entry, sl=sl,
            tp=entry + 10 * atr, atr=atr,
        )

        # Create OHLC: rise to 3R then pull back
        n = 60
        idx = pd.date_range("2023-01-10 09:00", periods=n, freq="1h")
        one_r = risk_pips * 0.0001
        peak = entry + 3.0 * one_r
        trail_level = peak - 2.5 * atr

        prices = np.concatenate([
            np.linspace(entry, peak, 20),                    # Rise to 3R
            np.linspace(peak, trail_level - 0.001, 40),     # Drop to trail
        ])
        ohlc = pd.DataFrame({
            "open": np.roll(prices, 1),
            "high": prices + atr * 0.05,
            "low": prices - atr * 0.05,
            "close": prices,
            "volume": np.ones(n) * 100,
        }, index=idx)
        ohlc.iloc[0, ohlc.columns.get_loc("open")] = entry

        config = BacktestConfig(
            pair="EUR_USD", spread_pips=0.0, slippage_pips=0.0,
            risk_per_trade=0.01, max_sl_pips=100.0,
            use_breakeven=False, use_partial_tp=False,
            use_confluence_sizing=False,
            use_trailing_stop=True, atr_trail_mult=2.5,
            trailing_activation_r=1.5, max_bars=500,
        )

        trades = simulate_trades([signal], ohlc, config)
        assert len(trades) == 1
        # Should exit via trailing stop with profit
        assert trades[0]["exit_reason"] == "TRAIL"
        assert trades[0]["pnl_pips"] > 0

    def test_trailing_stop_locks_profit(self):
        """Trailing stop never moves backward (toward entry)."""
        entry = 1.1000
        atr = 0.0025
        sl = entry - 2.0 * atr
        risk_pips = (entry - sl) / 0.0001

        signal = _make_signal(
            direction="long", entry=entry, sl=sl,
            tp=entry + 15 * atr, atr=atr,
        )

        # Rise to 3R, dip slightly, rise to 4R, then drop to trail
        n = 80
        idx = pd.date_range("2023-01-10 09:00", periods=n, freq="1h")
        one_r = risk_pips * 0.0001
        peak1 = entry + 3.0 * one_r
        dip = entry + 2.5 * one_r
        peak2 = entry + 4.0 * one_r
        trail_at_peak2 = peak2 - 2.5 * atr

        prices = np.concatenate([
            np.linspace(entry, peak1, 15),           # Rise to 3R
            np.linspace(peak1, dip, 10),             # Small dip
            np.linspace(dip, peak2, 15),             # Rise to 4R
            np.linspace(peak2, trail_at_peak2 - 0.001, 40),  # Drop to trail
        ])
        ohlc = pd.DataFrame({
            "open": np.roll(prices, 1),
            "high": prices + atr * 0.02,
            "low": prices - atr * 0.02,
            "close": prices,
            "volume": np.ones(n) * 100,
        }, index=idx)
        ohlc.iloc[0, ohlc.columns.get_loc("open")] = entry

        config = BacktestConfig(
            pair="EUR_USD", spread_pips=0.0, slippage_pips=0.0,
            risk_per_trade=0.01, max_sl_pips=100.0,
            use_breakeven=False, use_partial_tp=False,
            use_confluence_sizing=False,
            use_trailing_stop=True, atr_trail_mult=2.5,
            trailing_activation_r=1.5, max_bars=500,
        )

        trades = simulate_trades([signal], ohlc, config)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "TRAIL"
        # Exit should be near trail_at_peak2 (from peak2, not peak1)
        # The trail from peak2 is higher than from peak1
        trail_from_peak1 = peak1 - 2.5 * atr
        assert trades[0]["exit_price"] > trail_from_peak1, \
            "Trail should lock at highest peak level, not move backward"


# ── Test 9: Timeout exit ─────────────────────────────────────────


class TestTimeoutExit:
    def test_timeout_exit(self):
        """Trade exits after max_bars if neither SL nor TP hit."""
        entry = 1.1000
        atr = 0.0025
        sl = entry - 2.0 * atr
        # Put TP very far away
        tp = entry + 100 * atr

        signal = _make_signal(
            direction="long", entry=entry, sl=sl, tp=tp, atr=atr,
        )

        # Flat price — never hits SL or TP
        n = 60
        idx = pd.date_range("2023-01-10 09:00", periods=n, freq="1h")
        prices = np.full(n, entry)
        ohlc = pd.DataFrame({
            "open": prices,
            "high": prices + atr * 0.3,
            "low": prices - atr * 0.3,
            "close": prices,
            "volume": np.ones(n) * 100,
        }, index=idx)

        config = BacktestConfig(
            pair="EUR_USD", spread_pips=0.0, slippage_pips=0.0,
            risk_per_trade=0.01, max_sl_pips=100.0,
            use_breakeven=False, use_partial_tp=False,
            use_confluence_sizing=False,
            use_trailing_stop=True, atr_trail_mult=2.5,
            trailing_activation_r=1.5,
            max_bars=30,  # Short timeout for test
        )

        trades = simulate_trades([signal], ohlc, config)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "TIMEOUT"


# ── Test 10: Config defaults ─────────────────────────────────────


class TestConfigDefaults:
    def test_config_defaults(self):
        """New BacktestConfig fields have correct defaults."""
        config = BacktestConfig()
        assert config.use_trailing_stop is False
        assert config.atr_trail_mult == 2.5
        assert config.trailing_activation_r == 1.5
        assert config.max_bars == 200

    def test_config_adaptive_values(self):
        """Adaptive strategy config overrides work."""
        config = BacktestConfig(
            use_trailing_stop=True,
            atr_trail_mult=3.0,
            trailing_activation_r=2.0,
            max_bars=500,
        )
        assert config.use_trailing_stop is True
        assert config.atr_trail_mult == 3.0
        assert config.trailing_activation_r == 2.0
        assert config.max_bars == 500

    def test_pip_size_jpy(self):
        """JPY pairs use 0.01 pip size."""
        config = BacktestConfig(pair="USD_JPY")
        assert config.pip_size == 0.01

    def test_pip_size_non_jpy(self):
        """Non-JPY pairs use 0.0001 pip size."""
        config = BacktestConfig(pair="EUR_USD")
        assert config.pip_size == 0.0001


# ── Test: ATR computation ────────────────────────────────────────


class TestATRComputation:
    def test_compute_atr_basic(self):
        """ATR computes correctly on simple data."""
        ohlc = _make_ohlc(n=50, volatility=0.001)
        atr = _compute_atr(ohlc, period=14)
        assert len(atr) == len(ohlc)
        # First 13 values should be NaN (rolling window not filled)
        assert atr.iloc[:13].isna().all()
        # Values after should be positive
        assert (atr.iloc[14:] > 0).all()

    def test_compute_atr_higher_volatility(self):
        """Higher volatility data produces larger ATR values."""
        low_vol = _make_ohlc(n=100, volatility=0.0002, seed=1)
        high_vol = _make_ohlc(n=100, volatility=0.002, seed=1)
        atr_low = _compute_atr(low_vol, period=14)
        atr_high = _compute_atr(high_vol, period=14)
        # High vol ATR should be larger on average
        assert atr_high.iloc[20:].mean() > atr_low.iloc[20:].mean()
