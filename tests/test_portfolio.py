"""
tests/test_portfolio.py

Tests unitarios de app/core/portfolio.py. Usa datos sintéticos
deterministas; requiere scipy (ya en el stack del proyecto) solo
para los tests de hrp_weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.portfolio import (
    PortfolioError,
    hrp_weights,
    markowitz_weights,
    rebalance_schedule,
    risk_parity_weights,
)


def _synthetic_returns(n=250, tickers=("A", "B", "C", "D"), seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    vols = np.linspace(0.005, 0.02, len(tickers))
    data = {t: rng.normal(0.0003, v, n) for t, v in zip(tickers, vols)}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# markowitz_weights
# ---------------------------------------------------------------------------

class TestMarkowitzWeights:
    def test_tangency_portfolio_sums_to_one(self):
        returns = _synthetic_returns()
        weights = markowitz_weights(returns)
        assert weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_target_return_portfolio_hits_target(self):
        returns = _synthetic_returns()
        target = returns.mean().mean()
        weights = markowitz_weights(returns, target_return=target)

        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        achieved = weights @ returns.mean()
        assert achieved == pytest.approx(target, abs=1e-8)

    def test_single_asset_raises(self):
        returns = _synthetic_returns(tickers=("A",))
        with pytest.raises(PortfolioError, match="al menos 2 activos"):
            markowitz_weights(returns)

    def test_singular_covariance_raises(self):
        idx = pd.date_range("2023-01-01", periods=50, freq="B")
        base = np.random.default_rng(1).normal(0, 0.01, 50)
        # B es un múltiplo exacto de A -> covarianza singular.
        returns = pd.DataFrame({"A": base, "B": base * 2.0}, index=idx)

        with pytest.raises(PortfolioError, match="singular"):
            markowitz_weights(returns)


# ---------------------------------------------------------------------------
# risk_parity_weights
# ---------------------------------------------------------------------------

class TestRiskParityWeights:
    def test_weights_sum_to_one_and_positive(self):
        returns = _synthetic_returns()
        weights = risk_parity_weights(returns)

        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights > 0).all()

    def test_equal_vol_assets_get_equal_weight(self):
        idx = pd.date_range("2023-01-01", periods=2000, freq="B")
        rng = np.random.default_rng(9)
        # Misma vol, sin correlación -> ERC debe repartir a partes iguales.
        returns = pd.DataFrame(
            {t: rng.normal(0.0, 0.01, 2000) for t in ["A", "B", "C"]}, index=idx
        )
        weights = risk_parity_weights(returns)

        assert weights["A"] == pytest.approx(1 / 3, abs=0.02)
        assert weights["B"] == pytest.approx(1 / 3, abs=0.02)
        assert weights["C"] == pytest.approx(1 / 3, abs=0.02)

    def test_custom_budget_is_respected_in_ranking(self):
        returns = _synthetic_returns(tickers=("A", "B"))
        # A debería recibir más peso que B si su presupuesto de riesgo es mayor.
        budget = pd.Series({"A": 0.8, "B": 0.2})
        weights = risk_parity_weights(returns, budget=budget)

        assert weights["A"] > weights["B"]

    def test_budget_not_summing_to_one_raises(self):
        returns = _synthetic_returns(tickers=("A", "B"))
        budget = pd.Series({"A": 0.5, "B": 0.6})
        with pytest.raises(PortfolioError, match="sumar 1.0"):
            risk_parity_weights(returns, budget=budget)

    def test_budget_mismatched_assets_raises(self):
        returns = _synthetic_returns(tickers=("A", "B"))
        budget = pd.Series({"A": 0.5, "X": 0.5})
        with pytest.raises(PortfolioError, match="mismos activos"):
            risk_parity_weights(returns, budget=budget)

    def test_single_asset_raises(self):
        returns = _synthetic_returns(tickers=("A",))
        with pytest.raises(PortfolioError, match="al menos 2 activos"):
            risk_parity_weights(returns)


# ---------------------------------------------------------------------------
# hrp_weights
# ---------------------------------------------------------------------------

class TestHRPWeights:
    def test_weights_sum_to_one_and_positive(self):
        returns = _synthetic_returns()
        weights = hrp_weights(returns)

        assert weights.sum() == pytest.approx(1.0, abs=1e-8)
        assert (weights > 0).all()
        assert set(weights.index) == set(returns.columns)

    def test_single_asset_raises(self):
        returns = _synthetic_returns(tickers=("A",))
        with pytest.raises(PortfolioError, match="al menos 2 activos"):
            hrp_weights(returns)

    def test_less_volatile_cluster_gets_more_weight(self):
        idx = pd.date_range("2023-01-01", periods=300, freq="B")
        rng = np.random.default_rng(11)
        low_vol = rng.normal(0, 0.005, 300)
        high_vol = rng.normal(0, 0.03, 300)
        returns = pd.DataFrame({"LOW": low_vol, "HIGH": high_vol}, index=idx)

        weights = hrp_weights(returns)
        assert weights["LOW"] > weights["HIGH"]


# ---------------------------------------------------------------------------
# rebalance_schedule
# ---------------------------------------------------------------------------

class TestRebalanceSchedule:
    def test_calendar_flags_first_day_of_each_month(self):
        idx = pd.date_range("2023-01-01", "2023-03-31", freq="B")
        weights = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)

        schedule = rebalance_schedule(weights, method="calendar", frequency="M")

        assert schedule["rebalance"].iloc[0]
        # Debe haber exactamente 3 rebalanceos (uno por mes: ene, feb, mar).
        assert schedule["rebalance"].sum() == 3

    def test_threshold_flags_large_drift(self):
        idx = pd.date_range("2023-01-01", periods=5, freq="B")
        weights = pd.DataFrame(
            {"A": [0.5, 0.51, 0.7, 0.71, 0.72], "B": [0.5, 0.49, 0.3, 0.29, 0.28]},
            index=idx,
        )

        schedule = rebalance_schedule(weights, method="threshold", threshold=0.05)

        assert list(schedule["rebalance"]) == [True, False, True, False, False]

    def test_empty_weights_raises(self):
        with pytest.raises(PortfolioError, match="vacío"):
            rebalance_schedule(pd.DataFrame())

    def test_unknown_method_raises(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="B")
        weights = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
        with pytest.raises(PortfolioError, match="method desconocido"):
            rebalance_schedule(weights, method="no-existe")
