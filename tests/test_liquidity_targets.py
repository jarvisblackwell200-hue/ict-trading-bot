"""Unit tests for liquidity targets and premium/discount filtering."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ict_bot.signals.detector import (
    _find_liquidity_target,
    _get_dealing_range,
    _is_premium_discount_valid,
    _score_entry,
)
from ict_bot.backtest.engine import BacktestConfig


# ── Helpers ─────────────────────────────────────────────────────────

def _make_swing_hl(n: int, swings: list[tuple[int, int, float]] | None = None) -> pd.DataFrame:
    """
    Build a swing_hl DataFrame.

    swings: list of (index, high_low, level) tuples.
        high_low: 1 = swing high, -1 = swing low
    """
    data = {
        "HighLow": np.full(n, np.nan),
        "Level": np.full(n, np.nan),
    }
    if swings:
        for idx, hl, level in swings:
            data["HighLow"][idx] = hl
            data["Level"][idx] = level
    return pd.DataFrame(data)


def _make_liquidity(n: int, pools: list[tuple[int, int, float, int, int]] | None = None) -> pd.DataFrame:
    """
    Build a liquidity DataFrame.

    pools: list of (index, liquidity_type, level, end_idx, swept) tuples.
        liquidity_type: 1 = equal highs, -1 = equal lows
        swept: 0 = not swept, >0 = swept at that bar index
    """
    data = {
        "Liquidity": np.full(n, np.nan),
        "Level": np.full(n, np.nan),
        "End": np.full(n, np.nan),
        "Swept": np.full(n, np.nan),
    }
    if pools:
        for idx, liq_type, level, end, swept in pools:
            data["Liquidity"][idx] = liq_type
            data["Level"][idx] = level
            data["End"][idx] = end
            data["Swept"][idx] = swept
    return pd.DataFrame(data)


def _make_pdh_pdl(n: int, rows: list[tuple[int, float, float, int, int]] | None = None) -> pd.DataFrame:
    """
    Build a previous_high_low DataFrame.

    rows: list of (index, prev_high, prev_low, broken_high, broken_low) tuples.
    """
    data = {
        "PreviousHigh": np.full(n, np.nan),
        "PreviousLow": np.full(n, np.nan),
        "BrokenHigh": np.zeros(n, dtype=np.int32),
        "BrokenLow": np.zeros(n, dtype=np.int32),
    }
    if rows:
        for idx, ph, pl, bh, bl in rows:
            data["PreviousHigh"][idx] = ph
            data["PreviousLow"][idx] = pl
            data["BrokenHigh"][idx] = bh
            data["BrokenLow"][idx] = bl
    return pd.DataFrame(data)


def _make_fvg(n: int) -> pd.DataFrame:
    """Build an empty FVG DataFrame."""
    return pd.DataFrame({
        "FVG": np.full(n, np.nan),
        "Top": np.full(n, np.nan),
        "Bottom": np.full(n, np.nan),
    })


def _make_primitives(
    n: int = 200,
    swings=None,
    pools=None,
    pdh_pdl_rows=None,
    pwh_pwl_rows=None,
) -> dict:
    """Build a minimal primitives dict for testing."""
    return {
        "swing_hl": _make_swing_hl(n, swings),
        "liquidity": _make_liquidity(n, pools),
        "pdh_pdl": _make_pdh_pdl(n, pdh_pdl_rows) if pdh_pdl_rows else None,
        "pwh_pwl": _make_pdh_pdl(n, pwh_pwl_rows) if pwh_pwl_rows else None,
        "fvg": _make_fvg(n),
        "ob": _make_fvg(n),  # reuse structure, won't be queried
    }


# ── _find_liquidity_target tests ─────────────────────────────────

class TestFindLiquidityTarget:

    def test_long_nearest(self):
        """Returns nearest equal-highs above entry for longs."""
        primitives = _make_primitives(
            pools=[
                (10, 1, 1.1100, 15, 0),  # equal highs at 1.1100, not swept
                (20, 1, 1.1200, 25, 0),  # equal highs at 1.1200, not swept
            ],
            swings=[(30, 1, 1.1300)],  # swing high as fallback
        )
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
        )
        assert price == 1.1100
        assert label == "equal_highs"

    def test_short_nearest(self):
        """Returns nearest equal-lows below entry for shorts."""
        primitives = _make_primitives(
            pools=[
                (10, -1, 1.0900, 15, 0),  # equal lows at 1.0900
                (20, -1, 1.0800, 25, 0),  # equal lows at 1.0800
            ],
        )
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=-1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
        )
        assert price == 1.0900
        assert label == "equal_lows"

    def test_min_rr_filter(self):
        """Skips targets that don't meet min_target_rr requirement."""
        primitives = _make_primitives(
            pools=[
                (10, 1, 1.1005, 15, 0),  # only 5 pips above entry (0.1R) — below 1.0R floor
                (20, 1, 1.1200, 25, 0),  # 200 pips above (4.0R) — valid
            ],
        )
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0, min_target_rr=1.0,
        )
        # 1.1005 - 1.1000 = 0.0005, 0.0005 / 0.005 = 0.1 < 1.0 → skip
        # 1.1200 - 1.1000 = 0.0200, 0.0200 / 0.005 = 4.0 >= 1.0 → valid
        assert price == 1.1200
        assert label == "equal_highs"

    def test_swept_excluded(self):
        """Swept liquidity pools are ignored."""
        primitives = _make_primitives(
            pools=[
                (10, 1, 1.1100, 15, 40),  # swept at bar 40
                (20, 1, 1.1200, 25, 0),   # not swept
            ],
        )
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
        )
        assert price == 1.1200
        assert label == "equal_highs"

    def test_pdh_target(self):
        """PDH picked up as valid long target."""
        primitives = _make_primitives(
            pdh_pdl_rows=[(50, 1.1150, 1.0850, 0, 0)],  # PDH=1.1150 unbroken
        )
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
            pdh_pdl=primitives["pdh_pdl"],
        )
        assert price == 1.1150
        assert label == "PDH"

    def test_fallback_none(self):
        """Returns (None, None) when no valid targets exist."""
        primitives = _make_primitives()  # no pools, no swings, no pdh/pdl
        price, label = _find_liquidity_target(
            primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
        )
        assert price is None
        assert label is None


# ── _get_dealing_range tests ─────────────────────────────────────

class TestDealingRange:

    def test_computation(self):
        """Correct high/low/equilibrium from swing data."""
        primitives = _make_primitives(
            swings=[
                (10, 1, 1.1200),   # swing high
                (20, -1, 1.0800),  # swing low
                (30, 1, 1.1300),   # another swing high
                (40, -1, 1.0900),  # another swing low
            ],
        )
        high, low, eq = _get_dealing_range(primitives, idx=50)
        assert high == 1.1300
        assert low == 1.0800
        assert eq == pytest.approx((1.1300 + 1.0800) / 2.0)


# ── Premium/Discount tests ──────────────────────────────────────

class TestPremiumDiscount:

    def test_long_in_discount(self):
        """Long below equilibrium → valid (discount)."""
        assert _is_premium_discount_valid(1.0900, direction=1, equilibrium=1.1000) is True

    def test_long_in_premium(self):
        """Long above equilibrium → invalid (premium)."""
        assert _is_premium_discount_valid(1.1100, direction=1, equilibrium=1.1000) is False

    def test_short_in_premium(self):
        """Short above equilibrium → valid (premium)."""
        assert _is_premium_discount_valid(1.1100, direction=-1, equilibrium=1.1000) is True

    def test_short_in_discount(self):
        """Short below equilibrium → invalid (discount)."""
        assert _is_premium_discount_valid(1.0900, direction=-1, equilibrium=1.1000) is False


# ── _score_entry with premium/discount ───────────────────────────

class TestScoreEntryPremiumDiscount:

    def _make_setup_and_zone(self):
        """Return a minimal setup and entry_zone for scoring tests."""
        setup = {
            "trigger": "BOS",
            "ote_top": None,
            "ote_bottom": None,
            "entry_zones": [{"type": "FVG", "top": 1.1010, "bottom": 1.0990}],
            "break_idx": 30,
        }
        entry_zone = {"type": "FVG", "top": 1.1010, "bottom": 1.0990}
        return setup, entry_zone

    def test_long_in_discount_bonus(self):
        """+1 confluence when long below equilibrium."""
        primitives = _make_primitives(
            swings=[
                (10, 1, 1.1200),
                (20, -1, 1.0800),
            ],
        )
        setup, entry_zone = self._make_setup_and_zone()
        # entry=1.0900 < equilibrium=1.1000 → discount → +1
        score, factors = _score_entry(
            idx=50, direction=1, primitives=primitives, setup=setup,
            entry_zone=entry_zone, current_kz=None, entry_price=1.0900,
            use_premium_discount=True,
        )
        assert "premium_discount" in factors
        assert factors["premium_discount"] == "discount"
        # Base: entry_zone(1) + structure(1) = 2, +1 for premium/discount = 3
        assert score == 3

    def test_long_in_premium_no_bonus(self):
        """No bonus when long above equilibrium."""
        primitives = _make_primitives(
            swings=[
                (10, 1, 1.1200),
                (20, -1, 1.0800),
            ],
        )
        setup, entry_zone = self._make_setup_and_zone()
        # entry=1.1100 > equilibrium=1.1000 → premium → no bonus
        score, factors = _score_entry(
            idx=50, direction=1, primitives=primitives, setup=setup,
            entry_zone=entry_zone, current_kz=None, entry_price=1.1100,
            use_premium_discount=True,
        )
        assert "premium_discount" not in factors
        assert score == 2  # entry_zone(1) + structure(1) only

    def test_short_in_premium_bonus(self):
        """+1 confluence when short above equilibrium."""
        primitives = _make_primitives(
            swings=[
                (10, 1, 1.1200),
                (20, -1, 1.0800),
            ],
        )
        setup, entry_zone = self._make_setup_and_zone()
        score, factors = _score_entry(
            idx=50, direction=-1, primitives=primitives, setup=setup,
            entry_zone=entry_zone, current_kz=None, entry_price=1.1100,
            use_premium_discount=True,
        )
        assert "premium_discount" in factors
        assert factors["premium_discount"] == "premium"
        assert score == 3


# ── Backward compatibility ───────────────────────────────────────

class TestBackwardCompat:

    def test_features_disabled(self):
        """Both flags False → max score 6 (no premium/discount), R:R TP."""
        primitives = _make_primitives(
            swings=[
                (10, 1, 1.1200),
                (20, -1, 1.0800),
            ],
        )
        setup = {
            "trigger": "BOS",
            "ote_top": None,
            "ote_bottom": None,
            "entry_zones": [{"type": "FVG", "top": 1.1010, "bottom": 1.0990}],
            "break_idx": 30,
        }
        entry_zone = {"type": "FVG", "top": 1.1010, "bottom": 1.0990}

        score, factors = _score_entry(
            idx=50, direction=1, primitives=primitives, setup=setup,
            entry_zone=entry_zone, current_kz=None, entry_price=1.0900,
            use_premium_discount=False,
        )
        # With use_premium_discount=False, factor 7 is skipped
        assert "premium_discount" not in factors
        # Max possible without premium_discount: 6
        assert score <= 6

        # _find_liquidity_target returns None with truly empty data (no pools, no swings)
        empty_primitives = _make_primitives()
        price, label = _find_liquidity_target(
            empty_primitives, idx=50, direction=1,
            entry_price=1.1000, sl_distance=0.0050, min_rr=2.0,
        )
        assert price is None
        assert label is None


# ── BacktestConfig defaults ──────────────────────────────────────

class TestBacktestConfigDefaults:

    def test_new_fields_default_true(self):
        """New BacktestConfig fields default to True."""
        config = BacktestConfig()
        assert config.use_liquidity_targets is True
        assert config.use_premium_discount is True

    def test_fields_can_be_disabled(self):
        """New fields can be set to False."""
        config = BacktestConfig(use_liquidity_targets=False, use_premium_discount=False)
        assert config.use_liquidity_targets is False
        assert config.use_premium_discount is False
