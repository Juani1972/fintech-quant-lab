"""Tests para data_loader."""
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.core.data_loader import compute_log_returns, load_prices


def test_log_returns_basic():
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.date_range("2024-01-01", periods=3))
    returns = compute_log_returns(prices)
    assert len(returns) == 2
    assert np.isclose(returns["A"].iloc[0], np.log(1.1))


def _fake_multiindex_download(tickers: list[str], n: int = 10) -> pd.DataFrame:
    """Simula la forma real de `yf.download` para varios tickers:
    columnas MultiIndex (campo, ticker).
    """
    dates = pd.bdate_range("2023-01-02", periods=n)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, tickers])
    data = np.random.default_rng(0).uniform(90, 110, size=(n, len(columns)))
    return pd.DataFrame(data, index=dates, columns=columns)


def _fake_flat_download(n: int = 10) -> pd.DataFrame:
    """Simula la forma real de `yf.download` para un único ticker:
    columnas planas (sin MultiIndex).
    """
    dates = pd.bdate_range("2023-01-02", periods=n)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = np.random.default_rng(0).uniform(90, 110, size=(n, len(fields)))
    return pd.DataFrame(data, index=dates, columns=fields)


def test_load_prices_multi_ticker(monkeypatch):
    """`load_prices` debe extraer correctamente el campo pedido cuando
    yfinance devuelve columnas MultiIndex (varios tickers).
    """
    fake = _fake_multiindex_download(["AAA", "BBB"])
    with patch("app.core.data_loader.yf.download", return_value=fake) as mock_dl:
        result = load_prices(
            ["AAA", "BBB"], date(2023, 1, 1), date(2023, 1, 31),
        )
    mock_dl.assert_called_once()
    assert list(result.columns) == ["AAA", "BBB"]
    assert len(result) == 10


def test_load_prices_single_ticker_flat_columns():
    """yfinance a veces devuelve columnas planas (sin MultiIndex) para
    un único ticker; `load_prices` debe manejarlo igualmente.
    """
    fake = _fake_flat_download()
    with patch("app.core.data_loader.yf.download", return_value=fake):
        result = load_prices(
            ["CCC"], date(2023, 2, 1), date(2023, 2, 28),
        )
    assert list(result.columns) == ["CCC"]
    assert len(result) == 10


def test_load_prices_rejects_empty_tickers():
    with pytest.raises(ValueError, match="al menos un ticker"):
        load_prices([], date(2023, 3, 1), date(2023, 3, 31))


def test_load_prices_rejects_invalid_dates():
    with pytest.raises(ValueError, match="anterior"):
        load_prices(["DDD"], date(2023, 4, 30), date(2023, 4, 1))


def test_load_prices_rejects_invalid_missing_policy():
    with pytest.raises(ValueError, match="missing_policy"):
        load_prices(
            ["EEE"], date(2023, 5, 1), date(2023, 5, 31), missing_policy="bogus",  # type: ignore[arg-type]
        )


def test_load_prices_wraps_yfinance_errors_as_connection_error():
    with (
        patch("app.core.data_loader.yf.download", side_effect=RuntimeError("timeout")),
        pytest.raises(ConnectionError, match="yfinance"),
    ):
        load_prices(["FFF"], date(2023, 6, 1), date(2023, 6, 30))


def test_load_prices_raises_on_empty_response():
    with (
        patch("app.core.data_loader.yf.download", return_value=pd.DataFrame()),
        pytest.raises(ValueError, match="No se obtuvieron datos"),
    ):
        load_prices(["GGG"], date(2023, 7, 1), date(2023, 7, 31))


def test_load_prices_ffill_fills_gaps():
    """missing_policy solo tiene efecto real cuando hay >1 ticker: con
    uno solo, dropna(how='all') ya elimina cualquier fila con NaN antes
    de que la política pueda actuar (una fila de 1 columna con NaN es
    'toda NaN' por definición).
    """
    fake = _fake_multiindex_download(["HHH", "III"], n=5)
    fake.loc[fake.index[2], ("Adj Close", "HHH")] = np.nan
    with patch("app.core.data_loader.yf.download", return_value=fake):
        result = load_prices(
            ["HHH", "III"], date(2023, 8, 1), date(2023, 8, 31), missing_policy="ffill",
        )
    assert not result["HHH"].isna().any()
    assert result["HHH"].iloc[2] == result["HHH"].iloc[1]  # relleno hacia delante


def test_load_prices_raise_policy_with_nans():
    fake = _fake_multiindex_download(["JJJ", "KKK"], n=5)
    fake.loc[fake.index[2], ("Adj Close", "JJJ")] = np.nan
    with (
        patch("app.core.data_loader.yf.download", return_value=fake),
        pytest.raises(ValueError, match="missing_policy='raise'"),
    ):
        load_prices(
            ["JJJ", "KKK"], date(2023, 9, 1), date(2023, 9, 30), missing_policy="raise",
        )
