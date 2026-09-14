"""Descarga y gestión de datos de mercado."""
from __future__ import annotations

from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(
    tickers: Iterable[str],
    start: date,
    end: date,
    field: str = "Adj Close",
) -> pd.DataFrame:
    """Descarga precios ajustados de yfinance.

    Args:
        tickers: Lista de tickers.
        start: Fecha de inicio.
        end: Fecha de fin.
        field: Columna a extraer ('Adj Close' o 'Close').

    Returns:
        DataFrame con precios, indexado por fecha.

    Raises:
        ValueError: Si no hay tickers, fechas inválidas o no hay datos.
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("Debes proporcionar al menos un ticker.")

    if start >= end:
        raise ValueError(f"Fecha inicio ({start}) debe ser anterior a fecha fin ({end}).")

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

    data = data.dropna(how="all").ffill().dropna()
    if data.empty:
        raise ValueError("Tras limpiar NaNs no quedan datos válidos.")

    return data


@st.cache_data(ttl=3600, show_spinner=False)
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()
