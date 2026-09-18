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


def get_global_provider() -> str:
    """Lee la fuente de datos elegida globalmente (por defecto 'yahoo').

    Función separada de `get_global_params()` a propósito -- así no se
    cambia la firma que ya usan las 16 páginas existentes (todas hacen
    `tickers, start, end = get_global_params()`; añadir un cuarto valor
    ahí habría roto esa desestructuración en todas ellas). Las páginas
    que quieran respetar la fuente de datos elegida llaman a esta
    función aparte y se la pasan a `load_prices(..., provider=...)`.
    """
    return str(st.session_state.get("global_provider", "yahoo"))


def get_global_provider_kwargs() -> dict:
    """kwargs adicionales para `load_prices(..., provider_kwargs=...)`
    según el proveedor elegido (p.ej. la api_key de Alpha Vantage,
    introducida en el desplegable de `main.py`). Vacío para proveedores
    que no necesitan configuración extra (yahoo, stooq).
    """
    if get_global_provider() == "alphavantage":
        api_key = st.session_state.get("global_provider_api_key", "")
        if api_key:
            return {"api_key": api_key}
    return {}


def get_gemini_api_key() -> str:
    """Clave de Google AI Studio introducida por el usuario en el
    desplegable '🤖 Informes con IA' de main.py -- vacía si no se ha
    configurado ninguna. Las páginas que ofrezcan generar un informe
    con IA deben comprobar esto antes de mostrar el botón/sección
    correspondiente, y explicar cómo conseguir una clave si está vacía."""
    return str(st.session_state.get("gemini_api_key", ""))


def get_gemini_model() -> str:
    """Nombre del modelo de Gemini elegido por el usuario (por
    defecto, el de `app.core.ai_report.DEFAULT_MODEL`)."""
    from app.core.ai_report import DEFAULT_MODEL

    return str(st.session_state.get("gemini_model", DEFAULT_MODEL))


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
    if "global_provider" not in st.session_state:
        st.session_state["global_provider"] = "yahoo"
    if "gemini_model" not in st.session_state:
        from app.core.ai_report import DEFAULT_MODEL
        st.session_state["gemini_model"] = DEFAULT_MODEL
