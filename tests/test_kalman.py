"""Tests para el módulo de Kalman filter."""
import numpy as np
import pandas as pd
import pytest

from app.core.kalman import (
    compare_static_dynamic,
    kalman_hedge_ratio,
    rolling_ols_hedge_ratio,
)


@pytest.fixture
def cointegrated_pair():
    """Par cointegrado sintético: y = 2*x + ruido estacionario."""
    np.random.seed(42)
    n = 500
    x = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 100,
                  index=pd.date_range("2020-01-01", periods=n, freq="B"))
    noise = pd.Series(np.random.normal(0, 0.5, n), index=x.index)
    y = 2 * x + noise
    return y, x


def test_kalman_hedge_ratio_runs(cointegrated_pair):
    y, x = cointegrated_pair
    result = kalman_hedge_ratio(y, x)
    assert len(result.beta) == len(y)
    assert result.beta.notna().all()
    assert result.spread.notna().all()


def test_kalman_beta_close_to_true(cointegrated_pair):
    """Con y = 2x + ruido, el beta debe converger cerca de 2."""
    y, x = cointegrated_pair
    result = kalman_hedge_ratio(y, x)
    beta_final = float(result.beta.iloc[-1])
    assert abs(beta_final - 2.0) < 0.2


def test_kalman_insufficient_data():
    y = pd.Series(np.random.randn(10))
    x = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError, match="al menos 30"):
        kalman_hedge_ratio(y, x)


def test_kalman_beta_std_positive(cointegrated_pair):
    y, x = cointegrated_pair
    result = kalman_hedge_ratio(y, x)
    assert (result.beta_std > 0).all()


def test_kalman_spread_mean_close_to_zero(cointegrated_pair):
    """Con un par cointegrado, el spread debe tener media ~0."""
    y, x = cointegrated_pair
    result = kalman_hedge_ratio(y, x)
    assert abs(result.spread.mean()) < 1.0


def test_kalman_high_delta_more_reactive(cointegrated_pair):
    """delta alto → beta con más varianza que delta bajo."""
    y, x = cointegrated_pair
    low = kalman_hedge_ratio(y, x, delta=1e-6)
    high = kalman_hedge_ratio(y, x, delta=1e-2)
    assert high.beta.std() > low.beta.std()


def test_kalman_log_likelihood_finite(cointegrated_pair):
    y, x = cointegrated_pair
    result = kalman_hedge_ratio(y, x)
    assert np.isfinite(result.log_likelihood)


def test_compare_static_dynamic(cointegrated_pair):
    y, x = cointegrated_pair
    df = compare_static_dynamic(y, x)
    assert "beta_ols" in df.columns
    assert "beta_kalman" in df.columns
    assert "diferencia" in df.columns
    assert len(df) == len(y)


def test_rolling_ols_hedge_ratio(cointegrated_pair):
    y, x = cointegrated_pair
    beta = rolling_ols_hedge_ratio(y, x, window=60)
    assert len(beta) == len(y)
    # Las primeras `window` observaciones deben ser NaN
    assert beta.iloc[:60].isna().all()
    # A partir de ahí, valores
    assert beta.iloc[60:].notna().all()


def test_rolling_ols_insufficient_window(cointegrated_pair):
    y, x = cointegrated_pair
    with pytest.raises(ValueError, match="window"):
        rolling_ols_hedge_ratio(y, x, window=5)


def test_rolling_ols_beta_close_to_true(cointegrated_pair):
    y, x = cointegrated_pair
    beta = rolling_ols_hedge_ratio(y, x, window=100)
    beta_final = float(beta.iloc[-1])
    assert abs(beta_final - 2.0) < 0.2


def test_rolling_and_kalman_correlated(cointegrated_pair):
    """Los dos estimadores deben parecerse en un par estable."""
    y, x = cointegrated_pair
    kalman = kalman_hedge_ratio(y, x)
    rolling = rolling_ols_hedge_ratio(y, x, window=100)
    common = kalman.beta.index.intersection(rolling.dropna().index)
    corr = kalman.beta.loc[common].corr(rolling.loc[common])
    assert corr > 0.8
