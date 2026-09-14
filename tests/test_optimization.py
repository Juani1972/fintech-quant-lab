"""Tests para el módulo de optimización."""
import numpy as np
import pandas as pd
import pytest

from app.core.optimization import (
    _expand_grid,
    grid_search,
    heatmap_data,
)


@pytest.fixture
def prices_series():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    returns = np.random.normal(0.0003, 0.01, 500)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)


def _momentum_factory(prices, params):
    window = params["window"]
    ret = prices.pct_change(window)
    s = pd.Series(0, index=prices.index)
    s[ret > 0] = 1
    s[ret < 0] = -1
    return s


def test_expand_grid():
    grid = {"a": [1, 2], "b": [10, 20, 30]}
    combos = _expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": 10} in combos


def test_grid_search_runs(prices_series):
    result = grid_search(
        prices=prices_series,
        signal_factory=_momentum_factory,
        param_grid={"window": [10, 20, 30]},
        objective="sharpe",
    )
    assert len(result.grid) == 3
    assert "window" in result.param_names
    assert "window" in result.best_params
    assert result.best_params["window"] in [10, 20, 30]


def test_grid_search_empty_grid(prices_series):
    with pytest.raises(ValueError, match="vacío"):
        grid_search(prices_series, _momentum_factory, {})


def test_grid_search_invalid_objective(prices_series):
    with pytest.raises(ValueError, match="no encontrado"):
        grid_search(
            prices_series, _momentum_factory,
            {"window": [10, 20]},
            objective="nonexistent_metric",
        )


def test_heatmap_data(prices_series):
    result = grid_search(
        prices=prices_series,
        signal_factory=_momentum_factory,
        param_grid={"window": [10, 20, 30]},
        objective="sharpe",
    )
    # Solo un parámetro, así que heatmap de un solo eje
    # Añadimos otro parámetro artificial no usado
    grid = result.grid.copy()
    grid["dummy"] = [1, 2, 3]
    hm = heatmap_data(grid, x_param="window", y_param="dummy", metric="sharpe")
    assert hm.shape == (3, 3)
