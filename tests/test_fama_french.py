"""Tests para fama_french.py.

`load_factors` depende de una descarga de red (pandas_datareader ->
la librería de datos de Kenneth French), así que se mockea
`app.core.fama_french.web.DataReader` en vez de llamar a la red de
verdad -- igual que se hace con `yfinance` en test_data_loader.py.
"""
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.fama_french import load_factors, run_regression


def _fake_monthly_reader(n: int = 12) -> dict:
    """Simula la forma real que devuelve pandas_datareader para el
    dataset MENSUAL de Kenneth French: un dict con [0] = DataFrame con
    índice PeriodIndex (no DatetimeIndex) -- esto es precisamente lo
    que causaba el TypeError antes del fix (pd.to_datetime() no acepta
    PeriodDtype directamente en pandas >= 2.x).
    """
    idx = pd.period_range("2020-01", periods=n, freq="M")
    df = pd.DataFrame(
        {
            "Mkt-RF": np.random.default_rng(0).normal(0.5, 2, n),
            "SMB": np.random.default_rng(1).normal(0.1, 1, n),
            "HML": np.random.default_rng(2).normal(0.1, 1, n),
            "RF": np.full(n, 0.01),
        },
        index=idx,
    )
    return {0: df}


def _fake_daily_reader(n: int = 300) -> dict:
    """Simula el dataset DIARIO: índice ya convertible con pd.to_datetime
    directamente (DatetimeIndex o similar), sin el problema de PeriodDtype.
    """
    idx = pd.bdate_range("2020-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Mkt-RF": np.random.default_rng(3).normal(0.05, 1, n),
            "SMB": np.random.default_rng(4).normal(0.01, 0.5, n),
            "HML": np.random.default_rng(5).normal(0.01, 0.5, n),
            "RF": np.full(n, 0.0005),
        },
        index=idx,
    )
    return {0: df}


def test_load_factors_monthly_period_index_regression():
    """Regresión del bug real: pedir factores con frequency='monthly'
    devuelve un DataFrame con PeriodIndex desde pandas_datareader.
    Antes del fix, `load_factors` llamaba a `pd.to_datetime(ff.index)`
    sin distinguir el tipo de índice, lo que lanzaba:

        TypeError: Passing PeriodDtype data is invalid.
        Use `data.to_timestamp()` instead

    en cualquier pandas >= 2.x. El fix detecta PeriodIndex y usa
    `.to_timestamp()` en su lugar.
    """
    with patch(
        "app.core.fama_french.web.DataReader",
        return_value=_fake_monthly_reader(),
    ):
        ff = load_factors(date(2020, 1, 1), date(2020, 12, 31), frequency="monthly")

    assert isinstance(ff.index, pd.DatetimeIndex)
    assert len(ff) == 12
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    # Los valores se dividen por 100 (vienen en % desde la fuente)
    assert ff["RF"].iloc[0] == pytest.approx(0.0001)


def test_load_factors_daily_still_works():
    """El dataset diario no tenía el bug (su índice ya es convertible
    directamente), pero debe seguir funcionando igual tras el fix.
    """
    with patch(
        "app.core.fama_french.web.DataReader",
        return_value=_fake_daily_reader(),
    ):
        ff = load_factors(date(2020, 1, 1), date(2020, 12, 31), frequency="daily")

    assert isinstance(ff.index, pd.DatetimeIndex)
    assert len(ff) == 300


def test_load_factors_invalid_model_raises():
    with pytest.raises(ValueError, match="no soportado"):
        load_factors(date(2020, 1, 1), date(2020, 12, 31), model="7")


def test_load_factors_invalid_frequency_raises():
    with pytest.raises(ValueError, match="no soportada"):
        load_factors(date(2020, 1, 1), date(2020, 12, 31), frequency="weekly")


def test_load_factors_empty_response_raises():
    with patch(
        "app.core.fama_french.web.DataReader",
        return_value={0: pd.DataFrame()},
    ), pytest.raises(ValueError, match="Sin datos"):
        load_factors(date(2020, 1, 1), date(2020, 12, 31))


def test_load_factors_wraps_connection_errors():
    with patch(
        "app.core.fama_french.web.DataReader",
        side_effect=RuntimeError("timeout"),
    ), pytest.raises(ConnectionError, match="No se pudieron descargar"):
        load_factors(date(2020, 1, 1), date(2020, 12, 31))


def test_run_regression_recovers_known_beta():
    """Regresión con un solo factor sintético de beta conocido (2.0):
    la regresión debe recuperarlo con precisión razonable."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt_rf = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    rf = pd.Series(0.00005, index=dates)
    noise = pd.Series(rng.normal(0, 0.002, n), index=dates)

    returns = 2.0 * mkt_rf + rf + noise
    smb = pd.Series(rng.normal(0, 0.001, n), index=dates)
    hml = pd.Series(rng.normal(0, 0.001, n), index=dates)
    factors = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})

    result = run_regression(returns, factors, model="3")

    assert result.betas["Mkt-RF"] == pytest.approx(2.0, abs=0.1)
    assert result.n_obs == n
    assert 0 <= result.r_squared <= 1
