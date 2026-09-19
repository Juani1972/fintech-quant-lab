"""Gestión centralizada del estado de sesión.

Todas las páginas deben leer los parámetros globales a través de
`get_global_params()` en lugar de acceder directamente a `st.session_state`.

La lista de tickers (`global_tickers`) la controla ENTERAMENTE `main.py`
a través del campo de texto de la barra lateral. `state.py` solo se
encarga de que la clave exista (aunque sea vacía) para que las páginas
no revienten si se navega directamente por URL sin pasar por la portada.

La clave de Gemini se resuelve con cascada en `app.credentials`:
secrets → env var → fichero local → campo manual de la barra lateral.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

from app.config import DEFAULT_END, DEFAULT_START


def get_global_params() -> tuple[list[str], date, date]:
    """Lee los parámetros globales con defaults seguros.

    Returns:
        (tickers, start, end) — siempre valores válidos, nunca None.
        `tickers` puede ser lista vacía si el usuario no ha escrito nada
        todavía en la barra lateral de la portada.
    """
    tickers_str = st.session_state.get("global_tickers", "")
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    start = st.session_state.get("global_start", DEFAULT_START)
    end = st.session_state.get("global_end", DEFAULT_END)

    if start is None:
        start = DEFAULT_START
    if end is None:
        end = DEFAULT_END

    return tickers, start, end


def get_global_provider() -> str:
    """Lee la fuente de datos elegida globalmente (por defecto 'yahoo')."""
    return str(st.session_state.get("global_provider", "yahoo"))


def get_global_provider_kwargs() -> dict:
    """kwargs adicionales para `load_prices(..., provider_kwargs=...)`
    según el proveedor elegido (p.ej. la api_key de Alpha Vantage).
    """
    if get_global_provider() == "alphavantage":
        api_key = st.session_state.get("global_provider_api_key", "")
        if api_key:
            return {"api_key": api_key}
    return {}


def get_gemini_api_key() -> str:
    """Clave de Google AI Studio.

    Delega en `app.credentials`, que busca en cascada:
    secrets → env var → fichero local → campo manual.
    """
    from app.credentials import get_gemini_key

    return get_gemini_key()


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

    Nota: `global_tickers` se crea VACÍO. Es `main.py` el que, si
    detecta que está vacío la primera vez, lo rellena con la semilla
    de `config.DEFAULT_TICKERS`. Las páginas no escriben nunca en
    esta clave.
    """
    if "global_tickers" not in st.session_state:
        st.session_state["global_tickers"] = ""
    if "global_start" not in st.session_state:
        st.session_state["global_start"] = DEFAULT_START
    if "global_end" not in st.session_state:
        st.session_state["global_end"] = DEFAULT_END
    if "global_provider" not in st.session_state:
        st.session_state["global_provider"] = "yahoo"
    if "gemini_model" not in st.session_state:
        from app.core.ai_report import DEFAULT_MODEL
        st.session_state["gemini_model"] = DEFAULT_MODEL
