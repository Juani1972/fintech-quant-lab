"""Gestión centralizada del estado de sesión.

Todas las páginas deben leer los parámetros globales a través de
`get_global_params()` en lugar de acceder directamente a `st.session_state`.

Motivo: si un usuario navega directamente a una página por URL
sin pasar antes por `main.py`, las claves `global_start`/`global_end`
no existen. Sin defaults, `start`/`end` llegan como `None` y `load_prices`
revienta con `TypeError` en `start >= end`.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from app.config import DEFAULT_END, DEFAULT_START, DEFAULT_TICKERS


def get_global_params() -> tuple[list[str], date, date]:
    """Lee los parámetros globales con defaults seguros.

    Returns:
        (tickers, start, end) — siempre valores válidos, nunca None.
    """
    tickers_str = st.session_state.get("global_tickers", ", ".join(DEFAULT_TICKERS))
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    start = st.session_state.get("global_start", DEFAULT_START)
    end = st.session_state.get("global_end", DEFAULT_END)

    if start is None:
        start = DEFAULT_START
    if end is None:
        end = DEFAULT_END

    return tickers, start, end


def ensure_session_initialized() -> None:
    """Inicializa las claves de sesión si no existen.

    Llamar al principio de `main.py` y también de cada página
    (idempotente). Esto hace que el orden de carga de Streamlit
    deje de importar.
    """
    if "global_tickers" not in st.session_state:
        st.session_state["global_tickers"] = ", ".join(DEFAULT_TICKERS)
    if "global_start" not in st.session_state:
        st.session_state["global_start"] = DEFAULT_START
    if "global_end" not in st.session_state:
        st.session_state["global_end"] = DEFAULT_END
