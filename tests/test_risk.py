"""Tests para el módulo de riesgo."""
import numpy as np
import pandas as pd
import pytest

from app.core.risk import (
    drawdown_series,
    expected_shortfall,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)


def test_var_known_value():
    # 100 retornos: 95 positivos pequeños, 5 muy negativos
    returns = pd.Series([0.01] * 95 + [-0.05] * 5)
    var = value_at_risk(returns, confidence=0.95)
    assert var > 0
    assert pytest.approx(var, rel=0.01) == 0.05


def test_expected_shortfall_greater_than_var():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.02, 1000))
    var = value_at_risk(returns, 0.95)
    es = expected_shortfall(returns, 0.95)
    assert es >= var


def test_max_drawdown_known():
    prices = pd.Series([100, 110, 90, 95, 120])
    dd = max_drawdown(prices)
    # Caída máxima: de 110 a 90 = -18.18%
    assert pytest.approx(dd, rel=1e-3) == -0.1818


def test_drawdown_series_length():
    prices = pd.Series([100, 110, 90, 95, 120])
    dd = drawdown_series(prices)
    assert len(dd) == len(prices)
    assert dd.max() <= 0


def test_sharpe_ratio_positive_for_positive_returns():
    returns = pd.Series(np.full(252, 0.001))
    # Varianza cero → sharpe infinito o NaN; usar serie con algo de ruido
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))
    assert sharpe_ratio(returns) != 0


def test_sortino_ratio_handles_no_downside():
    returns = pd.Series(np.full(252, 0.001))
    # Sin downside → debería devolver NaN o inf, no crashear
    result = sortino_ratio(returns)
    assert np.isnan(result) or np.isinf(result)
