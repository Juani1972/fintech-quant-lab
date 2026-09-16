"""Tests para el módulo de generación de informes HTML."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from app.core.report import (
    _fig_to_div,
    _format_metric,
    _meta_cards_html,
    _metrics_grid_html,
    _params_table_html,
    _trades_table_html,
    build_backtest_report,
    build_walkforward_report,
)


@pytest.fixture
def equity_curve():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    return pd.Series(100_000 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 100)),
                     index=dates)


@pytest.fixture
def positions():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    return pd.Series(np.random.choice([-1, 0, 1], 100), index=dates, dtype=int)


@pytest.fixture
def trades():
    return pd.DataFrame({
        "entry_date": pd.date_range("2020-01-01", periods=3),
        "exit_date": pd.date_range("2020-01-10", periods=3),
        "direction": ["long", "short", "long"],
        "entry_price": [100.0, 50.0, 200.0],
        "exit_price": [105.0, 48.0, 195.0],
        "pnl_pct": [0.05, 0.04, -0.025],
        "pnl_abs": [500.0, 400.0, -250.0],
        "bars_held": [5, 4, 6],
    })


# ============================================================
#  Helpers
# ============================================================
def test_format_metric_pct():
    assert _format_metric("total_return", 0.1234) == "12.34%"
    assert _format_metric("max_drawdown", -0.05) == "-5.00%"


def test_format_metric_float():
    assert _format_metric("sharpe", 1.2345) == "1.2345"


def test_format_metric_string():
    assert _format_metric("foo", "bar") == "bar"


def test_fig_to_div_returns_html():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
    html = _fig_to_div(fig)
    assert "<div" in html
    assert "plotly" in html.lower()


def test_metrics_grid_html_generates_cards():
    metrics = {"sharpe": 1.2, "total_return": 0.15, "max_drawdown": -0.08}
    html = _metrics_grid_html(metrics)
    assert "metric-card" in html
    assert "sharpe" in html
    assert "1.2000" in html


def test_meta_cards_html_escapes():
    html = _meta_cards_html({"Ticker": "<script>alert(1)</script>"})
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_params_table_html():
    html = _params_table_html({"window": 60, "entry": 2.0})
    assert "window" in html
    assert "60" in html
    assert "entry" in html


def test_trades_table_html_empty():
    html = _trades_table_html(pd.DataFrame())
    assert "No se generaron operaciones" in html


def test_trades_table_html_with_data(trades):
    html = _trades_table_html(trades)
    assert "long" in html
    assert "5.00%" in html


# ============================================================
#  Backtest report
# ============================================================
def test_build_backtest_report_runs(equity_curve, positions, trades):
    html = build_backtest_report(
        strategy="Momentum",
        tickers=["KO", "PEP"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={"window": 60},
        metrics={"sharpe": 1.2, "total_return": 0.15, "n_trades": 3},
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
    )
    assert "<!DOCTYPE html>" in html
    assert "Informe de Backtest" in html
    assert "Momentum" in html
    assert "KO" in html
    assert "Curva de capital" in html


def test_build_backtest_report_with_benchmark(equity_curve, positions, trades):
    benchmark = equity_curve * 1.1
    html = build_backtest_report(
        strategy="Momentum",
        tickers=["KO"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={},
        metrics={"sharpe": 1.2},
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        benchmark_equity=benchmark,
    )
    assert "Buy &amp; Hold" in html or "Buy & Hold" in html


def test_build_backtest_report_with_notes(equity_curve, positions, trades):
    html = build_backtest_report(
        strategy="Momentum",
        tickers=["KO"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={},
        metrics={},
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
        notes="Prueba con ventana más corta",
    )
    assert "Prueba con ventana más corta" in html


def test_build_backtest_report_escapes_strategy(equity_curve, positions, trades):
    html = build_backtest_report(
        strategy="<script>alert(1)</script>",
        tickers=["KO"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={},
        metrics={},
        equity_curve=equity_curve,
        positions=positions,
        trades=trades,
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ============================================================
#  Walk-forward report
# ============================================================
def test_build_walkforward_report_runs(equity_curve):
    html = build_walkforward_report(
        strategy="Pairs Trading",
        tickers=["KO", "PEP"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={"train_size": 504},
        is_metrics={"sharpe": 1.8, "total_return": 0.30},
        oos_metrics={"sharpe": 1.1, "total_return": 0.12},
        oos_equity=equity_curve,
        n_windows=5,
    )
    assert "<!DOCTYPE html>" in html
    assert "Walk-Forward" in html
    assert "In-Sample" in html
    assert "Out-of-Sample" in html


def test_build_walkforward_report_shows_degradation(equity_curve):
    html = build_walkforward_report(
        strategy="Pairs Trading",
        tickers=["KO"],
        start_date="2020-01-01",
        end_date="2024-12-31",
        params={},
        is_metrics={"sharpe": 2.0},
        oos_metrics={"sharpe": 0.5},
        oos_equity=equity_curve,
        n_windows=5,
    )
    assert "Degradación" in html
