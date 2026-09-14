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
) -> pd.DataFrame:
    """Descarga precios ajustados de yfinance.

    Args:
        tickers: Lista de tickers.
        start: Fecha de inicio.
        end: Fecha de fin.
        field: Columna a extraer ('Adj Close' o 'Close').
        missing_policy: Cómo tratar los NaN tras alinear:
            - 'ffill': rellena hacia delante.
            - 'drop':  elimina cualquier fila con NaN.
            - 'raise': lanza ValueError si hay NaN.

    Returns:
        DataFrame con precios, indexado por fecha.

    Raises:
        ValueError: Si no hay tickers, fechas inválidas, no hay datos, o
            hay NaN y `missing_policy='raise'`.
        ConnectionError: Si yfinance falla.
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
