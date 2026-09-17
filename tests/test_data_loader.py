"""Tests para data_loader."""
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.core.data_loader import compute_log_returns, load_prices, search_ticker


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


def test_search_ticker_returns_parsed_results():
    """search_ticker debe extraer symbol/name/exchange/type de los
    resultados de yf.Search, tal como los devuelve realmente (según
    el código fuente de yfinance: una lista de dicts con al menos
    'symbol', y opcionalmente 'shortname'/'longname'/'exchange'/
    'quoteType').
    """
    from unittest.mock import MagicMock, patch

    fake_quotes = [
        {"symbol": "AAPL", "shortname": "Apple Inc.", "exchange": "NMS", "quoteType": "EQUITY"},
        {"symbol": "APC.DE", "longname": "Apple Inc. (Alemania)", "exchange": "GER", "quoteType": "EQUITY"},
        {"exchange": "OPR", "quoteType": "OPTION"},  # sin "symbol" -> se descarta
    ]
    mock_search = MagicMock()
    mock_search.quotes = fake_quotes

    with patch("app.core.data_loader.yf.Search", return_value=mock_search) as m:
        results = search_ticker("Apple 001")  # query única para no chocar con caché

    m.assert_called_once()
    assert len(results) == 2  # el que no tiene "symbol" se descarta
    assert results[0] == {
        "symbol": "AAPL", "name": "Apple Inc.", "exchange": "NMS", "type": "EQUITY",
    }
    assert results[1]["name"] == "Apple Inc. (Alemania)"  # usa longname si no hay shortname


def test_search_ticker_rejects_short_query():
    with pytest.raises(ValueError, match="al menos 2 caracteres"):
        search_ticker("a")


def test_search_ticker_wraps_connection_errors():
    from unittest.mock import patch

    with (
        patch("app.core.data_loader.yf.Search", side_effect=RuntimeError("timeout")),
        pytest.raises(ConnectionError, match="No se pudo buscar"),
    ):
        search_ticker("Microsoft 003")  # query única para no chocar con caché


def test_load_prices_with_alternate_provider():
    """Regresión: load_prices(provider=...) debe delegar en
    app.core.providers.get_provider en vez de yfinance cuando se pide
    un proveedor distinto de 'yahoo', y aplicar la misma missing_policy
    sobre el resultado."""
    fake_df = pd.DataFrame(
        {"AAA": [100.0, np.nan, 102.0], "BBB": [50.0, 50.5, 51.0]},
        index=pd.bdate_range("2023-01-02", periods=3),
    )
    fake_provider = MagicMock()
    fake_provider.fetch.return_value = fake_df

    with patch("app.core.providers.get_provider", return_value=fake_provider) as m:
        result = load_prices(
            ["AAA", "BBB"], date(2023, 1, 1), date(2023, 1, 10),
            provider="stooq", missing_policy="ffill",
        )

    m.assert_called_once_with("stooq")
    assert list(result.columns) == ["AAA", "BBB"]
    assert not result["AAA"].isna().any()  # ffill aplicado igual que con yahoo


def test_load_prices_passes_provider_kwargs():
    fake_provider = MagicMock()
    fake_provider.fetch.return_value = pd.DataFrame(
        {"AAA": [100.0]}, index=pd.bdate_range("2023-01-02", periods=1),
    )
    with patch("app.core.providers.get_provider", return_value=fake_provider) as m:
        load_prices(
            ["AAA"], date(2023, 1, 1), date(2023, 1, 10),
            provider="alphavantage", provider_kwargs={"api_key": "clave-123"},
        )
    m.assert_called_once_with("alphavantage", api_key="clave-123")


def test_load_prices_wraps_provider_errors_as_connection_error():
    from app.core.providers import ProviderError

    with (
        patch("app.core.providers.get_provider", side_effect=ProviderError("sin datos")),
        pytest.raises(ConnectionError, match="Error descargando datos de stooq"),
    ):
        load_prices(["AAA"], date(2023, 1, 1), date(2023, 1, 10), provider="stooq")
