"""
app/core/providers.py

Abstracción de fuentes de datos de mercado para Fintech Quant Lab.

Elimina el acoplamiento directo a `yfinance` y permite que el motor de
backtesting / análisis reciba datos de cualquier proveedor compatible,
incluyendo datos propios del cliente vía CSV.

Uso típico:

    from app.core.providers import get_provider
    from datetime import date

    provider = get_provider("yahoo")
    prices = provider.fetch(["AAPL", "MSFT"], date(2020, 1, 1), date(2023, 12, 31))

Todos los proveedores devuelven un DataFrame con:
    - índice: DatetimeIndex (fechas de cotización, ascendente)
    - columnas: un ticker por columna
    - valores: precio de cierre ajustado (float)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import pandas as pd


class ProviderError(Exception):
    """Error genérico al obtener datos de un proveedor de mercado."""


@runtime_checkable
class MarketDataProvider(Protocol):
    """Contrato que debe cumplir cualquier fuente de datos de mercado."""

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """
        Devuelve un DataFrame de precios de cierre ajustados.

        Args:
            tickers: lista de símbolos a descargar.
            start: fecha de inicio (inclusive).
            end: fecha de fin (inclusive).

        Returns:
            DataFrame indexado por fecha, una columna por ticker.

        Raises:
            ProviderError: si no se pudieron obtener datos válidos.
        """
        ...


class YahooProvider:
    """Proveedor basado en Yahoo Finance vía la librería `yfinance`."""

    name = "yahoo"

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "yfinance no está instalado. Añádelo a requirements.txt."
            ) from exc

        if not tickers:
            raise ProviderError("La lista de tickers no puede estar vacía.")

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )

        if raw is None or raw.empty:
            raise ProviderError(f"Yahoo Finance no devolvió datos para {tickers}.")

        df = _extract_close(raw, tickers)
        return _validate(df, tickers, self.name)


class StooqProvider:
    """Proveedor basado en Stooq vía `pandas_datareader`."""

    name = "stooq"

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        try:
            from pandas_datareader import data as pdr
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "pandas_datareader no está instalado. Añádelo a requirements.txt."
            ) from exc

        if not tickers:
            raise ProviderError("La lista de tickers no puede estar vacía.")

        series = {}
        for ticker in tickers:
            try:
                raw = pdr.DataReader(ticker, "stooq", start=start, end=end)
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"Stooq falló para {ticker}: {exc}") from exc

            if raw is None or raw.empty:
                raise ProviderError(f"Stooq no devolvió datos para {ticker}.")

            # Stooq devuelve orden descendente por fecha.
            series[ticker] = raw["Close"].sort_index()

        df = pd.DataFrame(series)
        return _validate(df, tickers, self.name)


class CSVProvider:
    """
    Proveedor a partir de uno o varios ficheros CSV locales del cliente.

    Cada CSV debe tener al menos dos columnas: una de fecha y una de precio
    de cierre. El nombre del fichero (sin extensión) se usa como ticker,
    salvo que se indique un mapeo explícito.
    """

    name = "csv"

    def __init__(
        self,
        directory: str | Path,
        date_column: str = "Date",
        close_column: str = "Close",
        ticker_files: dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            directory: carpeta donde están los CSV.
            date_column: nombre de la columna de fecha en los CSV.
            close_column: nombre de la columna de precio de cierre.
            ticker_files: mapeo opcional {ticker: nombre_de_fichero.csv}.
                Si no se indica, se asume "{ticker}.csv" dentro de `directory`.
        """
        self.directory = Path(directory)
        self.date_column = date_column
        self.close_column = close_column
        self.ticker_files = ticker_files or {}

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        if not tickers:
            raise ProviderError("La lista de tickers no puede estar vacía.")

        series = {}
        for ticker in tickers:
            filename = self.ticker_files.get(ticker, f"{ticker}.csv")
            path = self.directory / filename

            if not path.exists():
                raise ProviderError(f"No se encontró el CSV para {ticker}: {path}")

            raw = pd.read_csv(path, parse_dates=[self.date_column])

            if self.close_column not in raw.columns:
                raise ProviderError(
                    f"El CSV de {ticker} no tiene la columna '{self.close_column}'."
                )

            raw = raw.set_index(self.date_column).sort_index()
            mask = (raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))
            series[ticker] = raw.loc[mask, self.close_column]

        df = pd.DataFrame(series)
        return _validate(df, tickers, self.name)


class AlphaVantageProvider:
    """Proveedor basado en la API REST de Alpha Vantage."""

    name = "alphavantage"
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, request_timeout: float = 10.0) -> None:
        if not api_key:
            raise ProviderError("AlphaVantageProvider requiere una api_key.")
        self.api_key = api_key
        self.request_timeout = request_timeout

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "requests no está instalado. Añádelo a requirements.txt."
            ) from exc

        if not tickers:
            raise ProviderError("La lista de tickers no puede estar vacía.")

        series = {}
        for ticker in tickers:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "outputsize": "full",
                "apikey": self.api_key,
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=self.request_timeout)
            resp.raise_for_status()
            payload = resp.json()

            ts = payload.get("Time Series (Daily)")
            if not ts:
                note = payload.get("Note") or payload.get("Error Message") or payload
                raise ProviderError(
                    f"Alpha Vantage no devolvió serie temporal para {ticker}: {note}"
                )

            raw = pd.DataFrame.from_dict(ts, orient="index")
            raw.index = pd.to_datetime(raw.index)
            raw = raw.sort_index()
            close_col = "5. adjusted close" if "5. adjusted close" in raw.columns else "4. close"
            closes = raw[close_col].astype(float)

            mask = (closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end))
            series[ticker] = closes.loc[mask]

        df = pd.DataFrame(series)
        return _validate(df, tickers, self.name)


_PROVIDERS = {
    "yahoo": YahooProvider,
    "stooq": StooqProvider,
    "csv": CSVProvider,
    "alphavantage": AlphaVantageProvider,
}


def get_provider(name: str, **kwargs) -> MarketDataProvider:
    """
    Factory de proveedores de datos de mercado.

    Args:
        name: uno de 'yahoo', 'stooq', 'csv', 'alphavantage'.
        **kwargs: argumentos del constructor del proveedor elegido
            (p. ej. `api_key` para alphavantage, `directory` para csv).

    Returns:
        Una instancia de MarketDataProvider lista para usar.

    Raises:
        ProviderError: si `name` no es un proveedor conocido.
    """
    key = name.strip().lower()
    provider_cls = _PROVIDERS.get(key)

    if provider_cls is None:
        disponibles = ", ".join(sorted(_PROVIDERS))
        raise ProviderError(f"Proveedor desconocido '{name}'. Disponibles: {disponibles}")

    return cast(MarketDataProvider, provider_cls(**kwargs))


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normaliza la salida de yfinance (single o multi-ticker) a un DataFrame de cierres."""
    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            sub = raw[ticker]
            col = "Close" if "Close" in sub.columns else sub.columns[0]
            closes[ticker] = sub[col]
        return pd.DataFrame(closes)

    # Un solo ticker: columnas simples.
    col = "Close" if "Close" in raw.columns else raw.columns[0]
    return raw[[col]].rename(columns={col: tickers[0]})


def _validate(df: pd.DataFrame, tickers: list[str], provider_name: str) -> pd.DataFrame:
    """Comprueba que el DataFrame resultante tiene datos usables y lo ordena."""
    if df.empty:
        raise ProviderError(f"[{provider_name}] No se obtuvieron datos para {tickers}.")

    faltantes = [t for t in tickers if t not in df.columns]
    if faltantes:
        raise ProviderError(f"[{provider_name}] Faltan datos para: {faltantes}")

    df = df.sort_index()
    df = df.dropna(how="all")

    if df.empty:
        raise ProviderError(f"[{provider_name}] Todos los valores son NaN para {tickers}.")

    return df
