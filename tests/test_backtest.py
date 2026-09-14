"""Tests del motor de backtesting."""
import numpy as np
import pandas as pd
import pytest

from app.core.backtest import (
    benchmark_metrics,
    buy_and_hold,
    compare_to_benchmark,
    run_backtest,
)


@pytest.fixture
def prices_up():
    """Serie de precios que sube un 1% diario durante 10 días."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.Series(100 * (1.01 ** np.arange(10)), index=dates)


@pytest.fixture
def prices_flat():
    """Serie de precios plana."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    return pd.Series([100.0] * 10, index=dates)


@pytest.fixture
def prices_zigzag():
    """Serie de precios que sube y baja."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    values = [100, 102, 101, 103, 104, 102, 105, 103, 106, 108]
    return pd.Series(values, index=dates, dtype=float)


def test_rejects_non_series():
    with pytest.raises(ValueError, match="pd.Series"):
        run_backtest([1, 2, 3], pd.Series([0, 1, 0]))


def test_rejects_negative_prices(prices_up):
    bad = prices_up.copy()
    bad.iloc[0] = -1
    signals = pd.Series(0, index=prices_up.index)
    with pytest.raises(ValueError, match="estrictamente positivos"):
        run_backtest(bad, signals)


def test_rejects_invalid_signals(prices_up):
    signals = pd.Series(2, index=prices_up.index)
    with pytest.raises(ValueError, match="solo puede contener"):
        run_backtest(prices_up, signals)


def test_rejects_negative_costs(prices_up):
    signals = pd.Series(0, index=prices_up.index)
    with pytest.raises(ValueError, match="commission"):
        run_backtest(prices_up, signals, commission=-0.01)


def test_signal_executed_next_bar(prices_up):
    """Una señal en t=0 debe producir posición 0 en t=0 y 1 en t=1."""
    signals = pd.Series(1, index=prices_up.index)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)

    assert result.positions.iloc[0] == 0
    assert (result.positions.iloc[1:] == 1).all()


def test_no_signal_no_return(prices_up):
    """Sin señales, el capital no debe moverse."""
    signals = pd.Series(0, index=prices_up.index)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)
    assert result.metrics["n_trades"] == 0


def test_long_flat_market_no_pnl(prices_flat):
    """Con precios planos y long, no debe haber ganancia."""
    signals = pd.Series(1, index=prices_flat.index)
    result = run_backtest(prices_flat, signals, commission=0, slippage=0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_long_uptrend_positive_return(prices_up):
    """Con precios subiendo 1% diario y long, debe haber ganancia."""
    signals = pd.Series(1, index=prices_up.index)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)
    expected = 100_000 * (1.01 ** 9)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_costs_reduce_equity(prices_up):
    """Los costes deben reducir el capital final."""
    signals = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], index=prices_up.index, dtype=float)

    no_costs = run_backtest(prices_up, signals, commission=0, slippage=0)
    with_costs = run_backtest(prices_up, signals, commission=0.01, slippage=0.005)

    assert with_costs.equity_curve.iloc[-1] < no_costs.equity_curve.iloc[-1]


def test_costs_proportional_to_turnover(prices_up):
    """Más cambios de posición → más coste total."""
    low_turnover = pd.Series([1] * 10, index=prices_up.index, dtype=float)
    high_turnover = pd.Series([1, -1] * 5, index=prices_up.index, dtype=float)

    r_low = run_backtest(prices_up, low_turnover, commission=0.001, slippage=0)
    r_high = run_backtest(prices_up, high_turnover, commission=0.001, slippage=0)

    assert r_high.metrics["turnover"] > r_low.metrics["turnover"]


def test_sharpe_zero_when_no_variance(prices_flat):
    signals = pd.Series(0, index=prices_flat.index)
    result = run_backtest(prices_flat, signals, commission=0, slippage=0)
    assert np.isnan(result.metrics["sharpe"])


def test_max_drawdown_negative(prices_zigzag):
    signals = pd.Series(1, index=prices_zigzag.index, dtype=float)
    result = run_backtest(prices_zigzag, signals, commission=0, slippage=0)
    assert result.metrics["max_drawdown"] <= 0


def test_metrics_keys_present(prices_up):
    signals = pd.Series(1, index=prices_up.index, dtype=float)
    result = run_backtest(prices_up, signals)
    expected_keys = {
        "total_return", "annual_return", "annual_volatility",
        "sharpe", "sortino", "max_drawdown", "calmar",
        "n_trades", "win_rate", "profit_factor",
        "avg_trade_pnl", "avg_bars_held", "exposure", "turnover",
    }
    assert expected_keys.issubset(result.metrics.keys())


def test_trades_extracted(prices_up):
    """Un ciclo long → flat debe generar exactamente 1 trade."""
    signals = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], index=prices_up.index, dtype=float)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)
    assert result.metrics["n_trades"] == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "long"


def test_trade_pnl_correct(prices_up):
    """Con long durante 4 barras al 1% diario, el PnL debe ser 1.01^4 - 1."""
    signals = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=prices_up.index, dtype=float)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)

    assert result.metrics["n_trades"] == 1
    trade = result.trades.iloc[0]

    expected_pnl = 1.01 ** 4 - 1
    assert trade["pnl_pct"] == pytest.approx(expected_pnl, rel=1e-6)
    assert trade["direction"] == "long"


def test_buy_and_hold(prices_up):
    bh = buy_and_hold(prices_up, initial_capital=100_000)
    expected = 100_000 * (prices_up.iloc[-1] / prices_up.iloc[0])
    assert bh.iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_compare_to_benchmark(prices_up):
    signals = pd.Series(1, index=prices_up.index, dtype=float)
    result = run_backtest(prices_up, signals, commission=0, slippage=0)
    bh = buy_and_hold(prices_up)
    df = compare_to_benchmark(result.equity_curve, bh)
    assert "strategy_norm" in df.columns
    assert "benchmark_norm" in df.columns
    assert df["strategy_norm"].iloc[0] == pytest.approx(1.0)


def test_multiple_trades_bars_held(prices_zigzag):
    """Varias operaciones (long -> short -> flat -> long) deben quedar bien
    emparejadas, con bars_held correcto en cada una.

    Regresión para la extracción vectorizada de trades: la versión
    original recalculaba la equity O(n) veces por iteración (O(n²) total)
    y usaba `index.get_loc` para el `bars_held`; este test fija el
    comportamiento esperado para detectar cualquier desajuste futuro.
    """
    signals = pd.Series(
        [1, 1, 1, -1, -1, 0, 0, 1, 1, 1], index=prices_zigzag.index, dtype=float,
    )
    result = run_backtest(prices_zigzag, signals, commission=0, slippage=0)

    assert result.metrics["n_trades"] == 3
    directions = list(result.trades["direction"])
    assert directions == ["long", "short", "long"]

    bars_held = list(result.trades["bars_held"])
    assert bars_held == [3, 2, 1]

    # Primer trade: long, entra en precio 102 (t=1), sale en 104 (t=4).
    trade0 = result.trades.iloc[0]
    assert trade0["entry_price"] == pytest.approx(102.0)
    assert trade0["exit_price"] == pytest.approx(104.0)
    assert trade0["pnl_pct"] == pytest.approx((104.0 - 102.0) / 102.0, rel=1e-9)


def test_benchmark_metrics_beta_matches_leverage():
    """Una 'estrategia' que es el benchmark apalancado 2x debe dar
    beta ~= 2.
    """
    np.random.seed(0)
    dates = pd.bdate_range("2020-01-01", periods=500)
    bench_log_returns = np.random.normal(0.0003, 0.01, 500)
    benchmark = pd.Series(100 * np.exp(np.cumsum(bench_log_returns)), index=dates)
    strategy = pd.Series(
        100 * np.exp(np.cumsum(2 * bench_log_returns)), index=dates,
    )

    m = benchmark_metrics(strategy, benchmark)
    assert m["beta"] == pytest.approx(2.0, abs=0.05)


def test_benchmark_metrics_zero_tracking_error_when_identical():
    """Si la estrategia es idéntica al benchmark, tracking_error y
    jensen_alpha deben ser (casi) cero y beta (casi) 1.
    """
    np.random.seed(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 300))), index=dates,
    )
    m = benchmark_metrics(prices, prices)
    assert m["beta"] == pytest.approx(1.0, abs=1e-6)
    assert m["tracking_error"] == pytest.approx(0.0, abs=1e-9)
    assert m["jensen_alpha"] == pytest.approx(0.0, abs=1e-9)


def test_benchmark_metrics_rejects_constant_benchmark():
    dates = pd.bdate_range("2020-01-01", periods=10)
    strategy = pd.Series(np.linspace(100, 110, 10), index=dates)
    flat_benchmark = pd.Series([100.0] * 10, index=dates)
    with pytest.raises(ValueError, match="varianza"):
        benchmark_metrics(strategy, flat_benchmark)
