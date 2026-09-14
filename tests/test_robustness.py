"""Tests para el módulo de robustez."""
import numpy as np
import pandas as pd
import pytest

from app.core.robustness import (
    MonteCarloResult,
    _block_bootstrap,
    _stability_score,
    monte_carlo_bootstrap,
    monte_carlo_gbm,
    robustness_score,
)


def test_monte_carlo_gbm_shape():
    paths = monte_carlo_gbm(
        initial_price=100, mu=0.08, sigma=0.2,
        horizon=252, n_simulations=100,
        random_state=42,
    )
    assert paths.shape == (100, 253)
    assert (paths[:, 0] == 100).all()
    assert (paths > 0).all()


def test_monte_carlo_gbm_invalid_price():
    with pytest.raises(ValueError, match="initial_price"):
        monte_carlo_gbm(0, 0.08, 0.2)


def test_monte_carlo_bootstrap_runs():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
    result = monte_carlo_bootstrap(returns, n_simulations=200, random_state=42)
    assert result.n_simulations == 200
    assert result.metric == "sharpe"
    assert np.isfinite(result.mean)
    assert result.percentiles["p05"] <= result.percentiles["p95"]


def test_monte_carlo_block_bootstrap():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500))
    result = monte_carlo_bootstrap(
        returns, n_simulations=100, block_size=10, random_state=42,
    )
    assert np.isfinite(result.mean)


def test_monte_carlo_insufficient_data():
    with pytest.raises(ValueError, match="al menos 2"):
        monte_carlo_bootstrap(pd.Series([0.01]))


def test_block_bootstrap_length():
    rng = np.random.default_rng(0)
    data = np.arange(100)
    sample = _block_bootstrap(data, horizon=50, block_size=5, rng=rng)
    assert len(sample) == 50


def test_stability_score_stable():
    values = np.array([1.0, 1.05, 0.98, 1.02, 1.01])
    score = _stability_score(values)
    assert score > 0.9


def test_stability_score_unstable():
    values = np.array([1.0, 5.0, -3.0, 10.0, -8.0])
    score = _stability_score(values)
    assert score < 0.5


def test_stability_score_insufficient():
    assert _stability_score(np.array([1.0])) == 0.0


def test_robustness_score_positive():
    mc = MonteCarloResult(
        n_simulations=100, horizon=252, metric="sharpe",
        distribution=np.random.normal(1.0, 0.3, 100),
        mean=1.0, std=0.3,
        percentiles={"p05": 0.5, "p25": 0.8, "p50": 1.0, "p75": 1.2, "p95": 1.5},
        var_95=-0.5, cvar_95=-0.7,
    )
    report = robustness_score(
        is_sharpe=1.5, oos_sharpe=1.2,
        mc_result=mc, sensitivity_score=0.8, n_trades=100,
    )
    assert 0 <= report.final_score <= 100
    assert report.final_score > 50
    assert isinstance(report.interpretation, str)


def test_robustness_score_fragile():
    mc = MonteCarloResult(
        n_simulations=100, horizon=252, metric="sharpe",
        distribution=np.random.normal(0.1, 0.5, 100),
        mean=0.1, std=0.5,
        percentiles={"p05": -0.7, "p25": -0.2, "p50": 0.1, "p75": 0.4, "p95": 0.8},
        var_95=0.7, cvar_95=1.0,
    )
    report = robustness_score(
        is_sharpe=2.0, oos_sharpe=0.3,
        mc_result=mc, sensitivity_score=0.2, n_trades=5,
    )
    assert report.final_score < 50
    assert "Frágil" in report.interpretation or "frágil" in report.interpretation
