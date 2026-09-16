"""
tests/test_strategies.py

Tests unitarios de app/core/strategies/. Todos los datos son
sintéticos y deterministas (sin red, sin dependencias externas más
allá de numpy/pandas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.strategies import (
    CarryTrade,
    CrossSectionalMomentum,
    PCAStatArb,
    RiskParityStrategy,
    StrategyError,
    TrendFollowing,
    VolatilityTargeting,
    get_strategy,
    list_strategies,
)


def _trending_prices(n=150, start=100.0, drift=0.002, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    noise = rng.normal(0, 0.5, size=n)
    prices = start + np.cumsum(np.full(n, drift * start) + noise)
    return pd.DataFrame({"ASSET": prices}, index=idx)


def _multi_asset_prices(n=300, tickers=("A", "B", "C", "D"), seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0005 * (i - len(tickers) / 2)
        noise = rng.normal(0, 1.0, size=n)
        data[t] = 100 + np.cumsum(np.full(n, drift) + noise)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# get_strategy / list_strategies
# ---------------------------------------------------------------------------

class TestFactory:
    def test_list_strategies_returns_all_registered(self):
        names = list_strategies()
        assert names == sorted(names)
        assert "trend_following" in names
        assert "pca_statarb" in names

    def test_get_strategy_returns_instance(self):
        strat = get_strategy("trend_following", fast_window=5, slow_window=20)
        assert isinstance(strat, TrendFollowing)
        assert strat.fast_window == 5

    def test_get_strategy_case_insensitive(self):
        strat = get_strategy("  RISK_PARITY  ")
        assert isinstance(strat, RiskParityStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(StrategyError, match="desconocida"):
            get_strategy("no-existe")


# ---------------------------------------------------------------------------
# CrossSectionalMomentum
# ---------------------------------------------------------------------------

class TestCrossSectionalMomentum:
    def test_continuous_signal_sums_abs_to_one(self):
        prices = _multi_asset_prices()
        strat = CrossSectionalMomentum(lookback=60)
        signal = strat.generate_signals(prices)

        assert set(signal.index) == set(prices.columns)
        assert signal.abs().sum() == pytest.approx(1.0, abs=1e-9)

    def test_top_n_produces_equal_long_short_weights(self):
        prices = _multi_asset_prices()
        strat = CrossSectionalMomentum(lookback=60, top_n=1)
        signal = strat.generate_signals(prices)

        assert (signal == 1.0).sum() + (signal == -1.0).sum() == 2
        assert signal.abs().sum() == pytest.approx(2.0)

    def test_best_trending_asset_gets_positive_signal(self):
        prices = _multi_asset_prices()
        strat = CrossSectionalMomentum(lookback=200, top_n=1)
        signal = strat.generate_signals(prices)

        best_asset = (prices.iloc[-1] / prices.iloc[-201]).idxmax()
        assert signal[best_asset] > 0

    def test_single_asset_raises(self):
        prices = _trending_prices()
        strat = CrossSectionalMomentum(lookback=10)
        with pytest.raises(StrategyError, match="al menos 2 activos"):
            strat.generate_signals(prices)

    def test_insufficient_history_raises(self):
        prices = _multi_asset_prices(n=10)
        strat = CrossSectionalMomentum(lookback=60)
        with pytest.raises(StrategyError, match="observaciones"):
            strat.generate_signals(prices)

    def test_invalid_lookback_raises(self):
        with pytest.raises(StrategyError, match="lookback"):
            CrossSectionalMomentum(lookback=1)


# ---------------------------------------------------------------------------
# RiskParityStrategy
# ---------------------------------------------------------------------------

class TestRiskParityStrategy:
    def test_weights_sum_to_one(self):
        prices = _multi_asset_prices()
        strat = RiskParityStrategy(lookback=60)
        weights = strat.generate_signals(prices)

        assert weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert (weights >= 0).all()

    def test_less_volatile_asset_gets_more_weight(self):
        idx = pd.date_range("2023-01-01", periods=100, freq="B")
        rng = np.random.default_rng(3)
        low_vol = 100 + np.cumsum(rng.normal(0, 0.1, 100))
        high_vol = 100 + np.cumsum(rng.normal(0, 2.0, 100))
        prices = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol}, index=idx)

        strat = RiskParityStrategy(lookback=60)
        weights = strat.generate_signals(prices)

        assert weights["LOW"] > weights["HIGH"]

    def test_single_asset_raises(self):
        prices = _trending_prices()
        strat = RiskParityStrategy()
        with pytest.raises(StrategyError, match="al menos 2 activos"):
            strat.generate_signals(prices)


# ---------------------------------------------------------------------------
# VolatilityTargeting
# ---------------------------------------------------------------------------

class TestVolatilityTargeting:
    def test_leverage_is_non_negative_and_bounded(self):
        prices = _trending_prices(n=100)
        strat = VolatilityTargeting(target_vol=0.15, lookback=20, max_leverage=3.0)
        leverage = strat.generate_signals(prices)

        assert (leverage >= 0).all()
        assert (leverage <= 3.0).all()

    def test_multi_asset_prices_raises(self):
        prices = _multi_asset_prices()
        strat = VolatilityTargeting()
        with pytest.raises(StrategyError, match="único activo"):
            strat.generate_signals(prices)

    def test_invalid_target_vol_raises(self):
        with pytest.raises(StrategyError, match="target_vol"):
            VolatilityTargeting(target_vol=0)

    def test_insufficient_history_raises(self):
        prices = _trending_prices(n=5)
        strat = VolatilityTargeting(lookback=20)
        with pytest.raises(StrategyError, match="retornos"):
            strat.generate_signals(prices)


# ---------------------------------------------------------------------------
# TrendFollowing
# ---------------------------------------------------------------------------

class TestTrendFollowing:
    def test_returns_multiindex_series(self):
        prices = _multi_asset_prices(n=200)
        strat = TrendFollowing(fast_window=10, slow_window=50)
        signal = strat.generate_signals(prices)

        assert isinstance(signal.index, pd.MultiIndex)
        assert signal.index.names == ["date", "ticker"]
        assert set(signal.unique()).issubset({-1, 0, 1})

    def test_uptrend_gets_long_signal(self):
        idx = pd.date_range("2023-01-01", periods=120, freq="B")
        prices = pd.DataFrame({"ASSET": np.linspace(100, 200, 120)}, index=idx)

        strat = TrendFollowing(fast_window=5, slow_window=30)
        signal = strat.generate_signals(prices)

        assert signal.iloc[-1] == 1

    def test_fast_window_must_be_less_than_slow(self):
        with pytest.raises(StrategyError, match="fast_window"):
            TrendFollowing(fast_window=50, slow_window=20)

    def test_insufficient_history_raises(self):
        prices = _trending_prices(n=5)
        strat = TrendFollowing(fast_window=5, slow_window=8)
        with pytest.raises(StrategyError, match="observaciones"):
            strat.generate_signals(prices)


# ---------------------------------------------------------------------------
# CarryTrade
# ---------------------------------------------------------------------------

class TestCarryTrade:
    def test_positive_carry_gets_long_signal(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"carry_yield": [0.02] * 10}, index=idx)

        strat = CarryTrade()
        signal = strat.generate_signals(prices)

        assert (signal == 1).all()

    def test_negative_carry_gets_short_signal(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        prices = pd.DataFrame({"carry_yield": [-0.01] * 10}, index=idx)

        strat = CarryTrade()
        signal = strat.generate_signals(prices)

        assert (signal == -1).all()

    def test_deadband_flattens_small_carry(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"carry_yield": [0.001, -0.001, 0.05]}, index=idx)

        strat = CarryTrade(deadband=0.01)
        signal = strat.generate_signals(prices)

        assert list(signal) == [0, 0, 1]

    def test_missing_column_raises(self):
        prices = _trending_prices()
        strat = CarryTrade()
        with pytest.raises(StrategyError, match="carry_yield"):
            strat.generate_signals(prices)


# ---------------------------------------------------------------------------
# PCAStatArb
# ---------------------------------------------------------------------------

class TestPCAStatArb:
    def test_returns_series_with_valid_signals(self):
        prices = _multi_asset_prices(n=150)
        strat = PCAStatArb(lookback=60, entry_z=2.0, exit_z=0.5)
        signal = strat.generate_signals(prices)

        assert set(signal.unique()).issubset({-1, 0, 1})
        assert len(signal) == 60

    def test_single_asset_raises(self):
        prices = _trending_prices()
        strat = PCAStatArb()
        with pytest.raises(StrategyError, match="al menos 2 activos"):
            strat.generate_signals(prices)

    def test_entry_must_exceed_exit(self):
        with pytest.raises(StrategyError, match="entry_z"):
            PCAStatArb(entry_z=1.0, exit_z=1.0)

    def test_insufficient_history_raises(self):
        prices = _multi_asset_prices(n=10)
        strat = PCAStatArb(lookback=60)
        with pytest.raises(StrategyError, match="observaciones"):
            strat.generate_signals(prices)
