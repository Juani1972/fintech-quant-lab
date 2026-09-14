"""Tests para el motor de walk-forward."""
import numpy as np
import pandas as pd
import pytest

from app.core.walkforward import (
    _compound_equity,
    signal_from_mean_reversion,
    signal_from_momentum,
    walk_forward_analysis,
)


@pytest.fixture
def prices_series():
    """Serie de precios de 1000 barras con tendencia leve."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    returns = np.random.normal(0.0003, 0.01, 1000)
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


def test_walk_forward_runs(prices_series):
    result = walk_forward_analysis(
        prices=prices_series,
        signal_generator=signal_from_momentum(window=30),
        train_size=200,
        test_size=50,
        step=50,
    )
    assert result.params["n_windows"] > 0
    assert "sharpe" in result.is_metrics_agg
    assert "sharpe" in result.oos_metrics_agg


def test_walk_forward_insufficient_data():
    prices = pd.Series(
        np.random.randn(50) + 100,
        index=pd.date_range("2024-01-01", periods=50, freq="B"),
    )
    with pytest.raises(ValueError, match="insuficientes"):
        walk_forward_analysis(
            prices=prices,
            signal_generator=signal_from_momentum(window=10),
            train_size=100,
            test_size=50,
        )


def test_walk_forward_windows_no_overlap():
    np.random.seed(0)
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(600) * 0.5),
        index=pd.date_range("2022-01-01", periods=600, freq="B"),
    )
    result = walk_forward_analysis(
        prices=prices,
        signal_generator=signal_from_momentum(window=20),
        train_size=200,
        test_size=50,
        step=50,
    )
    # Verificar que cada ventana de test no solapa con la siguiente
    for i in range(len(result.windows) - 1):
        assert result.windows[i].test_end < result.windows[i + 1].test_start


def test_walk_forward_oos_equity_monotonic_index():
    np.random.seed(0)
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(800) * 0.5),
        index=pd.date_range("2022-01-01", periods=800, freq="B"),
    )
    result = walk_forward_analysis(
        prices=prices,
        signal_generator=signal_from_mean_reversion(window=20, entry=1.5),
        train_size=200,
        test_size=50,
        step=50,
    )
    assert result.oos_equity_concat.index.is_monotonic_increasing


def test_compound_equity():
    """La composición debe empezar en capital inicial."""
    eq1 = pd.Series([100_000, 105_000, 110_000],
                    index=pd.date_range("2024-01-01", periods=3))
    eq2 = pd.Series([100_000, 95_000, 90_000],
                    index=pd.date_range("2024-01-10", periods=3))
    result = _compound_equity([eq1, eq2], initial_capital=100_000)
    assert result.iloc[0] == pytest.approx(100_000, rel=1e-6)


def test_walk_forward_is_and_oos_metrics_present(prices_series):
    result = walk_forward_analysis(
        prices=prices_series,
        signal_generator=signal_from_momentum(window=30),
        train_size=200,
        test_size=50,
        step=50,
    )
    for m in ["total_return", "sharpe", "max_drawdown"]:
        assert m in result.is_metrics_agg
        assert m in result.oos_metrics_agg
