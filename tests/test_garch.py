"""Tests para el módulo garch."""
import numpy as np
import pandas as pd
import pytest

from app.core.garch import check_stationarity, fit_garch, is_stationary


def test_fit_garch_runs():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    result = fit_garch(returns, p=1, q=1)
    assert result.aic is not None
    assert len(result.conditional_volatility) == 500


def test_is_stationary_true():
    params = pd.Series({"alpha[1]": 0.1, "beta[1]": 0.8})
    assert is_stationary(params) is True


def test_is_stationary_false():
    params = pd.Series({"alpha[1]": 0.6, "beta[1]": 0.6})
    assert is_stationary(params) is False


def test_check_stationarity_gjr():
    # alpha + beta + gamma/2 = 0.1 + 0.7 + 0.15 = 0.95 < 1 → estacionario
    params = pd.Series({"alpha[1]": 0.1, "beta[1]": 0.7, "gamma[1]": 0.3})
    assert check_stationarity(params, "GJR-GARCH") is True


def test_check_stationarity_egarch():
    params = pd.Series({"beta[1]": 0.9})
    assert check_stationarity(params, "EGARCH") is True


def test_check_stationarity_unknown():
    params = pd.Series({"alpha[1]": 0.1})
    with pytest.raises(ValueError, match="no reconocido"):
        check_stationarity(params, "UNKNOWN")
