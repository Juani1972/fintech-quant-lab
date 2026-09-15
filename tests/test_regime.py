"""Tests para el módulo de detección de regímenes."""
import numpy as np
import pandas as pd
import pytest

from app.core.regime import (
    current_regime,
    fit_hmm,
    regime_stats_table,
    regime_summary,
)


@pytest.fixture
def synthetic_returns():
    """Serie con dos regímenes claros: baja vol y alta vol."""
    np.random.seed(42)
    n1, n2 = 300, 300
    low_vol = np.random.normal(0.0005, 0.005, n1)
    high_vol = np.random.normal(-0.001, 0.025, n2)
    values = np.concatenate([low_vol, high_vol])
    dates = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=dates)


def test_fit_hmm_two_states(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    assert result.n_states == 2
    assert len(result.states) == len(synthetic_returns)
    assert set(result.states.unique()).issubset({0, 1})


def test_states_ordered_by_volatility(synthetic_returns):
    """Régimen 0 debe tener menor varianza que Régimen 1."""
    result = fit_hmm(synthetic_returns, n_states=2)
    assert result.variances[0] < result.variances[1]


def test_fit_hmm_three_states(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=3)
    assert result.n_states == 3
    # varianzas ordenadas
    assert (np.diff(result.variances) >= 0).all()


def test_fit_hmm_insufficient_data():
    small = pd.Series(np.random.randn(10))
    with pytest.raises(ValueError, match="insuficientes"):
        fit_hmm(small, n_states=2)


def test_fit_hmm_invalid_states():
    returns = pd.Series(np.random.randn(100))
    with pytest.raises(ValueError, match="n_states"):
        fit_hmm(returns, n_states=1)


def test_transition_matrix_rows_sum_to_one(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    row_sums = result.transition_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_state_probs_sum_to_one(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    probs_sum = result.state_probs.sum(axis=1)
    assert np.allclose(probs_sum, 1.0, atol=1e-6)


def test_regime_summary_columns(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    summary = regime_summary(synthetic_returns, result)
    expected_cols = {
        "Nº obs", "% tiempo", "Retorno anual",
        "Volatilidad anual", "Sharpe", "Duración media (días)",
    }
    assert expected_cols.issubset(summary.columns)
    assert len(summary) == 2


def test_regime_summary_vol_ordering(synthetic_returns):
    """La volatilidad por régimen debe ser creciente."""
    result = fit_hmm(synthetic_returns, n_states=2)
    summary = regime_summary(synthetic_returns, result)
    vols = summary["Volatilidad anual"].values
    assert vols[0] < vols[1]


def test_current_regime_returns_int(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    cur = current_regime(result)
    assert isinstance(cur, int)
    assert 0 <= cur < 2


def test_regime_stats_table_shape(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    table = regime_stats_table(result)
    assert table.shape == (2, 2)


def test_state_volatilities(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    vols = result.state_volatilities()
    assert len(vols) == 2
    assert vols.iloc[0] < vols.iloc[1]


def test_state_means_annualized(synthetic_returns):
    result = fit_hmm(synthetic_returns, n_states=2)
    means = result.state_means_annualized()
    assert len(means) == 2
