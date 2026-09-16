"""
tests/test_providers.py

Tests unitarios de app/core/providers.py.

No hacen ninguna llamada de red real: yfinance, pandas_datareader y
requests están mockeados. Pensados para correr en GitHub Actions sin
credenciales ni conexión a internet.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.providers import (
    AlphaVantageProvider,
    CSVProvider,
    ProviderError,
    StooqProvider,
    YahooProvider,
    get_provider,
)

START = date(2023, 1, 1)
END = date(2023, 1, 10)


def _dummy_index() -> pd.DatetimeIndex:
    return pd.date_range(START, END, freq="B")


# ---------------------------------------------------------------------------
# get_provider (factory)
# ---------------------------------------------------------------------------

class TestGetProvider:
    def test_returns_yahoo_provider(self):
        provider = get_provider("yahoo")
        assert isinstance(provider, YahooProvider)

    def test_is_case_insensitive_and_trims_whitespace(self):
        provider = get_provider("  YAHOO  ")
        assert isinstance(provider, YahooProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ProviderError, match="desconocido"):
            get_provider("no-existe")

    def test_passes_kwargs_to_constructor(self, tmp_path):
        provider = get_provider("csv", directory=tmp_path)
        assert isinstance(provider, CSVProvider)
        assert provider.directory == tmp_path


# ---------------------------------------------------------------------------
# YahooProvider
# ---------------------------------------------------------------------------

class TestYahooProvider:
    def test_fetch_single_ticker(self):
        idx = _dummy_index()
        fake_df = pd.DataFrame({"Close": range(len(idx))}, index=idx)

        with patch("yfinance.download", return_value=fake_df) as mocked:
            provider = YahooProvider()
            result = provider.fetch(["AAPL"], START, END)

        mocked.assert_called_once()
        assert list(result.columns) == ["AAPL"]
        assert len(result) == len(idx)

    def test_fetch_multi_ticker(self):
        idx = _dummy_index()
        columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Open"]])
        fake_df = pd.DataFrame(
            {(t, c): range(len(idx)) for t in ["AAPL", "MSFT"] for c in ["Close", "Open"]},
            index=idx,
            columns=columns,
        )

        with patch("yfinance.download", return_value=fake_df):
            provider = YahooProvider()
            result = provider.fetch(["AAPL", "MSFT"], START, END)

        assert set(result.columns) == {"AAPL", "MSFT"}

    def test_empty_response_raises(self):
        with patch("yfinance.download", return_value=pd.DataFrame()):
            provider = YahooProvider()
            with pytest.raises(ProviderError, match="no devolvió datos"):
                provider.fetch(["AAPL"], START, END)

    def test_empty_ticker_list_raises(self):
        provider = YahooProvider()
        with pytest.raises(ProviderError, match="no puede estar vacía"):
            provider.fetch([], START, END)


# ---------------------------------------------------------------------------
# StooqProvider
# ---------------------------------------------------------------------------

class TestStooqProvider:
    def test_fetch_success(self):
        idx = _dummy_index()
        fake_df = pd.DataFrame({"Close": range(len(idx))}, index=idx[::-1])  # orden descendente

        with patch("pandas_datareader.data.DataReader", return_value=fake_df):
            provider = StooqProvider()
            result = provider.fetch(["AAPL"], START, END)

        assert list(result.columns) == ["AAPL"]
        assert result.index.is_monotonic_increasing

    def test_provider_exception_is_wrapped(self):
        with patch("pandas_datareader.data.DataReader", side_effect=RuntimeError("boom")):
            provider = StooqProvider()
            with pytest.raises(ProviderError, match="Stooq falló"):
                provider.fetch(["AAPL"], START, END)

    def test_empty_response_raises(self):
        with patch("pandas_datareader.data.DataReader", return_value=pd.DataFrame()):
            provider = StooqProvider()
            with pytest.raises(ProviderError, match="no devolvió datos"):
                provider.fetch(["AAPL"], START, END)


# ---------------------------------------------------------------------------
# CSVProvider
# ---------------------------------------------------------------------------

class TestCSVProvider:
    def _write_csv(self, path: Path, filename: str) -> None:
        idx = _dummy_index()
        df = pd.DataFrame({"Date": idx, "Close": range(len(idx))})
        df.to_csv(path / filename, index=False)

    def test_fetch_reads_matching_files(self, tmp_path):
        self._write_csv(tmp_path, "AAPL.csv")
        provider = CSVProvider(directory=tmp_path)

        result = provider.fetch(["AAPL"], START, END)

        assert "AAPL" in result.columns
        assert not result.empty

    def test_missing_file_raises(self, tmp_path):
        provider = CSVProvider(directory=tmp_path)
        with pytest.raises(ProviderError, match="No se encontró el CSV"):
            provider.fetch(["AAPL"], START, END)

    def test_custom_ticker_files_mapping(self, tmp_path):
        self._write_csv(tmp_path, "apple_data.csv")
        provider = CSVProvider(directory=tmp_path, ticker_files={"AAPL": "apple_data.csv"})

        result = provider.fetch(["AAPL"], START, END)

        assert "AAPL" in result.columns

    def test_missing_close_column_raises(self, tmp_path):
        idx = _dummy_index()
        df = pd.DataFrame({"Date": idx, "Precio": range(len(idx))})
        df.to_csv(tmp_path / "AAPL.csv", index=False)

        provider = CSVProvider(directory=tmp_path)
        with pytest.raises(ProviderError, match="no tiene la columna"):
            provider.fetch(["AAPL"], START, END)


# ---------------------------------------------------------------------------
# AlphaVantageProvider
# ---------------------------------------------------------------------------

class TestAlphaVantageProvider:
    def test_requires_api_key(self):
        with pytest.raises(ProviderError, match="requiere una api_key"):
            AlphaVantageProvider(api_key="")

    def test_fetch_success(self):
        payload = {
            "Time Series (Daily)": {
                "2023-01-03": {"4. close": "100.0", "5. adjusted close": "99.5"},
                "2023-01-04": {"4. close": "101.0", "5. adjusted close": "100.4"},
            }
        }
        fake_response = MagicMock()
        fake_response.json.return_value = payload
        fake_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=fake_response):
            provider = AlphaVantageProvider(api_key="fake-key")
            result = provider.fetch(["AAPL"], START, END)

        assert "AAPL" in result.columns
        assert len(result) == 2

    def test_api_error_message_is_wrapped(self):
        fake_response = MagicMock()
        fake_response.json.return_value = {"Error Message": "clave inválida"}
        fake_response.raise_for_status.return_value = None

        with patch("requests.get", return_value=fake_response):
            provider = AlphaVantageProvider(api_key="fake-key")
            with pytest.raises(ProviderError, match="no devolvió serie temporal"):
                provider.fetch(["AAPL"], START, END)


# ---------------------------------------------------------------------------
# Validación común (_validate) a través de un proveedor real
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_ticker_column_raises(self, tmp_path):
        idx = _dummy_index()
        df = pd.DataFrame({"Date": idx, "Close": range(len(idx))})
        df.to_csv(tmp_path / "AAPL.csv", index=False)

        provider = CSVProvider(directory=tmp_path)
        with pytest.raises(ProviderError, match="No se encontró el CSV para MSFT"):
            provider.fetch(["AAPL", "MSFT"], START, END)

    def test_all_nan_raises(self, tmp_path):
        idx = _dummy_index()
        df = pd.DataFrame({"Date": idx, "Close": [None] * len(idx)})
        df.to_csv(tmp_path / "AAPL.csv", index=False)

        provider = CSVProvider(directory=tmp_path)
        with pytest.raises(ProviderError, match="Todos los valores son NaN"):
            provider.fetch(["AAPL"], START, END)
