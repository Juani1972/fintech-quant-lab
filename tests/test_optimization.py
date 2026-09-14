"""Tests para el módulo de optimización."""
import numpy as np
import pandas as pd
import pytest

from app.core.optimization import (
    _expand_grid,
    deflated_sharpe_ratio,
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


def test_expand_grid_empty():
    assert _expand_grid({}) == []


def test_expand_grid_empty_values():
    assert _expand_grid({"a": []}) == []


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


def test_grid_search_empty_values(prices_series):
    with pytest.raises(ValueError, match="vacío"):
        grid_search(prices_series, _momentum_factory, {"window": []})


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
    grid = result.grid.copy()
    grid["dummy"] = [1, 2, 3]
    hm = heatmap_data(grid, x_param="window", y_param="dummy", metric="sharpe")
    assert hm.shape == (3, 3)


def test_grid_search_preserves_int_param_dtype(prices_series):
    """Regresión: un parámetro entero (window) debe volver como `int` en
    best_params, no como `float`. Una fila de un DataFrame con columnas
    de distinto dtype se homogeneiza a un único dtype al extraerla como
    Series; extraer `best_params` de esa fila ya homogeneizada devolvía
    30.0 en vez de 30, lo que rompía cualquier código que usara ese
    parámetro para indexar (p.ej. `prices.pct_change(window)`).
    """
    result = grid_search(
        prices=prices_series,
        signal_factory=_momentum_factory,
        param_grid={"window": [10, 20, 30, 40, 50]},
        objective="sharpe",
    )
    assert isinstance(result.best_params["window"], int)


def test_deflated_sharpe_ratio_bounds(prices_series):
    """El DSR debe devolver una probabilidad válida en [0, 1] y ser
    consistente: n_trials debe coincidir con el nº de combinaciones
    válidas del grid.
    """
    from app.core.backtest import run_backtest

    result = grid_search(
        prices=prices_series,
        signal_factory=_momentum_factory,
        param_grid={"window": [10, 20, 30, 40, 50]},
        objective="sharpe",
    )
    best_signals = _momentum_factory(
        prices_series, result.best_params
    ).reindex(prices_series.index).fillna(0)
    bt = run_backtest(prices_series, best_signals, commission=0.0005, slippage=0.0002)

    dsr = deflated_sharpe_ratio(result, bt.returns)

    assert 0.0 <= dsr["dsr"] <= 1.0
    assert dsr["n_trials"] == 5
    assert dsr["sr_std"] >= 0.0


def test_deflated_sharpe_ratio_requires_sharpe_objective(prices_series):
    result = grid_search(
        prices=prices_series,
        signal_factory=_momentum_factory,
        param_grid={"window": [10, 20]},
        objective="total_return",
    )
    with pytest.raises(ValueError, match="objective='sharpe'"):
        deflated_sharpe_ratio(result, prices_series.pct_change().dropna())
