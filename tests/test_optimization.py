"""Tests para el módulo de optimización."""
import numpy as np
import pandas as pd
import pytest

from app.core.optimization import (
    _expand_grid,
    deflated_sharpe_ratio,
    grid_search,
    grid_search_walkforward,
    heatmap_data,
)


@pytest.fixture
def spread_series():
    """Spread sintético que cruza cero (como un spread de cointegración real)."""
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=700, freq="B")
    return pd.Series(np.cumsum(np.random.normal(0, 0.05, 700)), index=dates)


def _zscore_factory(prices, params):
    window = params["window"]
    roll = prices.rolling(window)
    z = (prices - roll.mean()) / roll.std()
    s = pd.Series(0, index=prices.index)
    s[z > 1.0] = -1
    s[z < -1.0] = 1
    return s


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


def test_deflated_sharpe_ratio_from_walkforward_result(prices_series):
    """Regresión: la página de Optimización usa grid_search_walkforward
    (no grid_search), cuyo resultado es un WalkForwardOptimizationResult
    con columnas 'oos_sharpe'/'is_sharpe' en el grid, no una columna
    'sharpe' plana como espera deflated_sharpe_ratio (pensada para
    OptimizationResult). La página adapta el resultado construyendo un
    OptimizationResult con la columna renombrada -- este test fija que
    esa adaptación produce un DSR válido, no solo que compile.
    """
    from app.core.backtest import run_backtest
    from app.core.optimization import OptimizationResult
    from app.core.walkforward import signal_from_momentum

    def factory(params):
        return signal_from_momentum(window=params["window"])

    objective = "sharpe"
    result = grid_search_walkforward(
        prices=prices_series,
        generator_factory=factory,
        param_grid={"window": [10, 20, 30, 45, 60]},
        objective=objective,
        train_size=200,
        test_size=63,
    )

    winning_generator = factory(result.best_params)
    winning_signal = winning_generator(prices_series, prices_series)
    winning_backtest = run_backtest(prices_series, winning_signal)

    dsr_input = OptimizationResult(
        grid=result.grid.rename(columns={f"oos_{objective}": objective}),
        best_params=result.best_params,
        best_metrics=result.best_oos_metrics,
        objective=objective,
        param_names=result.param_names,
    )
    dsr = deflated_sharpe_ratio(dsr_input, winning_backtest.returns)

    assert 0.0 <= dsr["dsr"] <= 1.0
    assert dsr["n_trials"] == 5
    assert dsr["sr_std"] >= 0.0


def test_grid_search_absolute_mode_on_spread(spread_series):
    """Regresión: grid_search debe poder optimizar sobre un spread que
    cruza cero (mode='absolute'), no solo sobre precios positivos.

    Antes de añadir el parámetro `mode`, esta llamada descartaba TODAS
    las combinaciones del grid con
    "ValueError: los precios deben ser estrictamente positivos",
    porque `grid_search` no tenía forma de pasarle `mode='absolute'`
    a `run_backtest` internamente.
    """
    result = grid_search(
        prices=spread_series,
        signal_factory=_zscore_factory,
        param_grid={"window": [20, 40, 60]},
        objective="sharpe",
        mode="absolute",
    )
    assert result.grid["sharpe"].notna().any(), (
        "Todas las combinaciones fallaron -- mode='absolute' no se está "
        "propagando a run_backtest."
    )


def test_grid_search_walkforward_absolute_mode_on_spread(spread_series):
    """Regresión equivalente a la anterior, pero para
    grid_search_walkforward -- que es la función que usa de verdad la
    página de Optimización para 'Pairs Trading (spread)'. Tenía el
    mismo bug: no reenviaba `mode` a `walk_forward_analysis`.
    """
    def factory(params):
        def gen(train, full):
            return _zscore_factory(full, params)
        return gen

    result = grid_search_walkforward(
        prices=spread_series,
        generator_factory=factory,
        param_grid={"window": [20, 40, 60]},
        objective="sharpe",
        train_size=300,
        test_size=100,
        mode="absolute",
    )
    assert result.grid["oos_sharpe"].notna().any(), (
        "Todas las combinaciones fallaron -- mode='absolute' no se está "
        "propagando a walk_forward_analysis."
    )
