"""Descarga y gestión de datos de mercado."""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import CACHE_TTL
from app.core.cache import cached

MissingPolicy = Literal["ffill", "drop", "raise"]


@cached(ttl=CACHE_TTL, show_spinner=False)
def load_prices(
    tickers: Iterable[str],
    start: date,
    end: date,
    field: str = "Adj Close",
    missing_policy: MissingPolicy = "ffill",
    provider: str = "yahoo",
    provider_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Descarga precios ajustados, de yfinance u otra fuente alternativa.

    Args:
        tickers: Lista de tickers.
        start: Fecha de inicio.
        end: Fecha de fin.
        field: Columna a extraer ('Adj Close' o 'Close'). Solo aplica
            con provider='yahoo'; el resto de proveedores (ver
            `app.core.providers`) siempre devuelven un único precio de
            cierre por ticker.
        missing_policy: Cómo tratar los NaN tras alinear:
            - 'ffill': rellena hacia delante.
            - 'drop':  elimina cualquier fila con NaN.
            - 'raise': lanza ValueError si hay NaN.
        provider: 'yahoo' (por defecto, sin cambios de comportamiento
            respecto a versiones anteriores de esta función) u otro de
            los registrados en `app.core.providers.get_provider`
            ('stooq', 'csv', 'alphavantage').
        provider_kwargs: kwargs adicionales para el constructor del
            proveedor elegido cuando `provider != 'yahoo'` (p.ej.
            `{"api_key": "..."}` para alphavantage, `{"directory":
            "..."}` para csv). Ignorado con provider='yahoo'.

    Returns:
        DataFrame con precios, indexado por fecha.

    Raises:
        ValueError: Si no hay tickers, fechas inválidas, no hay datos, o
            hay NaN y `missing_policy='raise'`.
        ConnectionError: Si la descarga falla (yfinance u otro proveedor).
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("Debes proporcionar al menos un ticker.")

    if start >= end:
        raise ValueError(
            f"Fecha inicio ({start}) debe ser anterior a fecha fin ({end})."
        )

    if missing_policy not in ("ffill", "drop", "raise"):
        raise ValueError(
            f"missing_policy inválida: '{missing_policy}'. "
            "Opciones: 'ffill', 'drop', 'raise'."
        )

    if provider == "yahoo":
        try:
            raw = yf.download(
                tickers, start=start, end=end, progress=False, auto_adjust=False
            )
        except Exception as e:
            raise ConnectionError(f"Error descargando datos de yfinance: {e}") from e

        if raw is None or raw.empty:
            raise ValueError(
                f"No se obtuvieron datos para {tickers} entre {start} y {end}. "
                "Revisa los tickers y el rango de fechas."
            )

        if isinstance(raw.columns, pd.MultiIndex):
            if field not in raw.columns.get_level_values(0):
                raise ValueError(f"Campo '{field}' no disponible en los datos.")
            data = raw[field]
        else:
            if field not in raw.columns:
                raise ValueError(f"Campo '{field}' no disponible en los datos.")
            data = raw[[field]].rename(columns={field: tickers[0]})
    else:
        # El resto de proveedores (ver app.core.providers) ya devuelven
        # un DataFrame limpio de precios de cierre, una columna por
        # ticker -- no hay selección de "field" ni MultiIndex que tratar.
        from app.core.providers import ProviderError, get_provider

        try:
            prov = get_provider(provider, **(provider_kwargs or {}))
            data = prov.fetch(tickers, start, end)
        except ProviderError as e:
            raise ConnectionError(f"Error descargando datos de {provider}: {e}") from e

        if data is None or data.empty:
            raise ValueError(
                f"No se obtuvieron datos para {tickers} entre {start} y {end} "
                f"con el proveedor '{provider}'."
            )

    n_missing_before = int(data.isna().sum().sum())
    data = data.dropna(how="all")

    if missing_policy == "ffill":
        data = data.ffill()
    elif missing_policy == "drop":
        data = data.dropna()
    elif missing_policy == "raise" and data.isna().any().any():
        raise ValueError(
            f"Hay {int(data.isna().sum().sum())} NaN en los datos y "
            "missing_policy='raise'."
        )

    if data.empty:
        raise ValueError("Tras aplicar la política de NaNs no quedan datos válidos.")

    data.attrs["missing_policy"] = missing_policy
    data.attrs["missing_before"] = n_missing_before

    return data


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def data_quality_report(prices: pd.DataFrame) -> dict[str, object]:
    """Genera un informe de calidad de los datos cargados.

    Returns:
        Dict con métricas: filas, columnas, NaNs, duplicados, retornos extremos.
    """
    report: dict[str, object] = {
        "rows": int(len(prices)),
        "columns": int(prices.shape[1]),
        "missing": int(prices.isna().sum().sum()),
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "start": str(prices.index.min().date()) if len(prices) else None,
        "end": str(prices.index.max().date()) if len(prices) else None,
        "missing_policy": prices.attrs.get("missing_policy", "unknown"),
    }

    returns = compute_log_returns(prices)
    if not returns.empty:
        z = (returns - returns.mean()) / returns.std()
        report["extreme_returns"] = int((z.abs() > 5).sum().sum())
    else:
        report["extreme_returns"] = 0

    return report


@cached(ttl=CACHE_TTL, show_spinner=False)
def search_ticker(query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Busca tickers por nombre de empresa (o símbolo parcial) usando el
    buscador de Yahoo Finance.

    Pensado para el selector de tickers de la interfaz: el usuario
    escribe "Apple" o "Inditex" en vez de tener que saber de antemano
    que el ticker es "AAPL" o "ITX.MC" -- ver también docs/MERCADOS.md
    para buscar manualmente si esto no encuentra lo que buscas (algunos
    resultados menos comunes, ADRs, etc. pueden no aparecer aquí).

    Args:
        query: Nombre de empresa o símbolo a buscar. Cadenas muy cortas
            (menos de 2 caracteres) se rechazan para evitar búsquedas
            demasiado amplias/ruidosas.
        max_results: Nº máximo de resultados a devolver.

    Returns:
        Lista de dicts con claves "symbol", "name", "exchange",
        "type" (p. ej. "EQUITY", "ETF"...). Vacía si no hay resultados.

    Raises:
        ValueError: Si `query` está vacío o es demasiado corto.
        ConnectionError: Si falla la búsqueda contra Yahoo Finance.
    """
    query = query.strip()
    if len(query) < 2:
        raise ValueError("Escribe al menos 2 caracteres para buscar.")

    try:
        result = yf.Search(query, max_results=max_results, news_count=0, lists_count=0)
    except Exception as exc:
        raise ConnectionError(f"No se pudo buscar '{query}' en Yahoo Finance: {exc}") from exc

    quotes = result.quotes or []
    return [
        {
            "symbol": q.get("symbol", ""),
            "name": q.get("shortname") or q.get("longname") or "",
            "exchange": q.get("exchange", ""),
            "type": q.get("quoteType", ""),
        }
        for q in quotes
        if q.get("symbol")
    ]
