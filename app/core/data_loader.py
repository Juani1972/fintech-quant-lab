"""Descarga y gestión de datos de mercado."""
from __future__ import annotations

from datetime import date
from typing import Iterable

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
    """
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        raise ValueError("Debes proporcionar al menos un ticker.")

    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        data = raw[field]
    else:
        data = raw[[field]].rename(columns={field: tickers[0]})

    return data.dropna(how="all").ffill().dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos logarítmicos."""
    return (prices / prices.shift(1)).apply(lambda x: x).pipe(lambda df: df.apply(lambda s: s))
