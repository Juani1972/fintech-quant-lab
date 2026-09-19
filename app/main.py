"""Portada principal de Fintech Quant Lab."""
from __future__ import annotations

import os
import signal
from datetime import date

import streamlit as st

from app.config import (
    APP_ICON,
    DEFAULT_END,
    DEFAULT_START,
    DEFAULT_TICKERS,
    PAGES,
)
from app.core.ai_report import DEFAULT_MODEL as DEFAULT_GEMINI_MODEL
from app.core.data_loader import search_ticker
from app.core.universe import list_universes, universe_to_string
from app.styles import (
    callout,
    footer,
    hero,
    page_card,
    page_setup,
    section,
)

# ============================================================
#  Configuración + CSS
# ============================================================
page_setup("Fintech Quant Lab", APP_ICON)

# ============================================================
#  Estado de sesión
# ============================================================
# Semilla inicial: SOLO la primera vez que arranca la app y la clave
# no existe. Después, main.py es el único que escribe en esta clave.
if "global_tickers" not in st.session_state:
    st.session_state["global_tickers"] = ", ".join(DEFAULT_TICKERS)
if "global_start" not in st.session_state:
    st.session_state["global_start"] = DEFAULT_START
if "global_end" not in st.session_state:
    st.session_state["global_end"] = DEFAULT_END
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = DEFAULT_GEMINI_MODEL


# ============================================================
#  Callbacks (se ejecutan ANTES del rerun)
# ============================================================
def _on_universe_change() -> None:
    """Rellena `global_tickers` con los tickers del universo seleccionado."""
    name = st.session_state.get("_universe_select", "(personalizado)")
    if name == "(personalizado)":
        return
    tickers_str = universe_to_string(name)
    if tickers_str:
        st.session_state["global_tickers"] = tickers_str


def _on_search_result_click(symbol: str) -> None:
    """Añade `symbol` a la lista global de tickers. Se usa como
    `on_click` del botón de cada resultado del buscador -- así el
    cambio se aplica ANTES de que el resto del script se vuelva a
    renderizar, y el text_input de tickers ya lo ve actualizado.
    """
    current = [
        t.strip() for t in
        st.session_state.get("global_tickers", "").split(",")
        if t.strip()
    ]
    if symbol not in current:
        current.append(symbol)
    st.session_state["global_tickers"] = ", ".join(current)
    st.session_state["_last_searched_ticker"] = symbol


def _on_tickers_form_submit() -> None:
    """Aplica el valor del formulario de tickers a `global_tickers`."""
    st.session_state["global_tickers"] = st.session_state.get("_tickers_form_input", "")


def _on_reset_session() -> None:
    """Borra todas las claves de sesión y vuelve a arrancar limpio.
    Útil si quedan valores zombis que no se van ni reiniciando
    Streamlit (porque el navegador conserva la sesión)."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ============================================================
#  Sidebar: parámetros globales
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Parámetros globales")
    st.caption("Compartidos entre todas las páginas.")

    # --- Buscador de empresa por nombre ---
    with st.expander("🔍 Buscar empresa por nombre"):
        st.caption("Escribe el nombre y toca el resultado para añadirlo a Tickers.")
        search_query = st.text_input(
            "Nombre de la empresa",
            placeholder="Apple, Inditex, Toyota...",
            key="_ticker_search_query",
            label_visibility="collapsed",
        )
        if search_query and len(search_query.strip()) >= 2:
            try:
                results = search_ticker(search_query)
            except ConnectionError as e:
                results = []
                st.caption(f"⚠️ {e}")
            except ValueError:
                results = []

            if results:
                for r in results:
                    label = f"{r['symbol']} — {r['name']}" if r["name"] else r["symbol"]
                    if r["exchange"]:
                        label += f" ({r['exchange']})"
                    st.button(
                        label,
                        key=f"_add_ticker_{r['symbol']}",
                        use_container_width=True,
                        on_click=_on_search_result_click,
                        args=(r["symbol"],),
                    )
            elif len(search_query.strip()) >= 2:
                st.caption(
                    "Sin resultados. Prueba con otro nombre, o consulta "
                    "docs/MERCADOS.md para buscar manualmente."
                )

    # --- Selector de universo ---
    st.selectbox(
        "Universo rápido",
        ["(personalizado)", *list_universes()],
        key="_universe_select",
        on_change=_on_universe_change,
        help=(
            "Elige un universo predefinido para rellenar el campo de tickers. "
            "Después puedes editarlo libremente."
        ),
    )

    # --- Campo de tickers (FORMULARIO: solo se aplica al pulsar "Aplicar") ---
    st.markdown("**Tickers**")
    st.caption(
        "Separa por comas. Pulsa **Aplicar** para confirmar el cambio "
        "-- mientras escribes, el valor NO se aplica (así no se pierde "
        "si navegas antes de terminar)."
    )
    with st.form("_tickers_form", clear_on_submit=False):
        st.text_input(
            "Tickers (separados por coma)",
            value=st.session_state.get("global_tickers", ""),
            key="_tickers_form_input",
            label_visibility="collapsed",
            help="Ejemplo: AAPL, MSFT, KO, PEP",
        )
        st.form_submit_button(
            "✅ Aplicar tickers",
            use_container_width=True,
            on_click=_on_tickers_form_submit,
        )

    # Mostrar el valor efectivo actual
    st.caption(f"Actualmente aplicado: `{st.session_state.get('global_tickers', '')}`")

    # --- Rango de fechas ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.date_input(
            "Fecha inicio",
            key="global_start",
            min_value=date(1990, 1, 1),
            max_value=date.today(),
        )
    with col_b:
        st.date_input(
            "Fecha fin",
            key="global_end",
            min_value=date(1990, 1, 1),
            max_value=date.today(),
        )

    # --- Fuente de datos ---
    PROVIDER_LABELS = {
        "yahoo": "Yahoo Finance (por defecto)",
        "stooq": "Stooq",
        "alphavantage": "Alpha Vantage",
    }
    with st.expander("🌐 Fuente de datos"):
        provider_choice = st.selectbox(
            "Proveedor", list(PROVIDER_LABELS.keys()),
            format_func=lambda p: PROVIDER_LABELS[p],
            index=list(PROVIDER_LABELS.keys()).index(
                st.session_state.get("global_provider", "yahoo")
            ),
            label_visibility="collapsed",
        )
        st.session_state["global_provider"] = provider_choice
        if provider_choice == "alphavantage":
            st.text_input(
                "API key de Alpha Vantage", type="password",
                key="global_provider_api_key",
            )

    # --- Informes con IA ---
    with st.expander("🤖 Informes con IA (Google Gemini)"):
        st.text_input(
            "API key de Google AI Studio", type="password",
            key="gemini_api_key",
        )
        st.text_input("Modelo", key="gemini_model")

    st.divider()

    # --- Botones de control ---
    st.button(
        "🔄 Reiniciar sesión",
        use_container_width=True,
        on_click=_on_reset_session,
        help=(
            "Borra todo el estado de la sesión (tickers, fechas, "
            "configuraciones). Útil si algo se queda zombi y no se "
            "va ni reiniciando la app."
        ),
    )

    LOCAL_MODE = os.getenv("FQL_LOCAL_MODE", "true").lower() == "true"
    if LOCAL_MODE:
        if st.button(
            "🚪 Cerrar aplicación",
            type="secondary",
            use_container_width=True,
        ):
            st.warning("Cerrando Fintech Quant Lab...")
            os.kill(os.getpid(), signal.SIGTERM)


# ============================================================
#  Hero
# ============================================================
hero(
    title="Fintech Quant Lab",
    subtitle=(
        "Plataforma de análisis cuantitativo para series temporales financieras: "
        "volatilidad, cointegración, factores de riesgo, backtesting, walk-forward, "
        "optimización y análisis de robustez."
    ),
    icon=APP_ICON,
)

# ============================================================
#  Estado de la sesión
# ============================================================
section("📋 Estado de la sesión")

tickers_actuales = st.session_state.get("global_tickers", "")
fecha_ini = st.session_state.get("global_start", DEFAULT_START)
fecha_fin = st.session_state.get("global_end", DEFAULT_END)

n_tickers = len([t for t in tickers_actuales.split(",") if t.strip()])
dias = (
    (fecha_fin - fecha_ini).days
    if isinstance(fecha_ini, date) and isinstance(fecha_fin, date)
    else 0
)

cols = st.columns(4)
with cols[0]:
    st.metric("Tickers", n_tickers)
with cols[1]:
    st.metric("Fecha inicio", str(fecha_ini))
with cols[2]:
    st.metric("Fecha fin", str(fecha_fin))
with cols[3]:
    st.metric("Rango", f"{dias} días")

if n_tickers == 0:
    callout(
        "⚠️ No hay ningún ticker configurado. Escribe al menos uno en el "
        "campo <strong>Tickers</strong> de la barra lateral y pulsa "
        "<strong>✅ Aplicar tickers</strong> antes de usar cualquier página.",
        variant="warning",
    )

st.markdown("")

# ============================================================
#  Módulos disponibles
# ============================================================
section("🧭 Módulos disponibles")

st.caption(
    "Selecciona un módulo en el menú lateral. "
    "Los parámetros globales (tickers y fechas) se aplican a todos."
)

for i in range(0, len(PAGES), 2):
    cols = st.columns(2, gap="medium")
    with cols[0]:
        p = PAGES[i]
        page_card(p["icon"], p["name"], p["description"])
    if i + 1 < len(PAGES):
        with cols[1]:
            p = PAGES[i + 1]
            page_card(p["icon"], p["name"], p["description"])
    st.markdown("")

# ============================================================
#  Flujo recomendado
# ============================================================
section("🔀 Flujo de investigación recomendado")

callout(
    "<strong>1.</strong> <strong>GARCH</strong> — analiza la volatilidad del activo.<br>"
    "<strong>2.</strong> <strong>Cointegración</strong> — busca pares con relación estable.<br>"
    "<strong>3.</strong> <strong>Backtest</strong> — simula la estrategia con costes.<br>"
    "<strong>4.</strong> <strong>Walk-Forward</strong> — valida out-of-sample.<br>"
    "<strong>5.</strong> <strong>Optimización</strong> — ajusta parámetros con WF.<br>"
    "<strong>6.</strong> <strong>Robustez</strong> — Monte Carlo + sensitivity + score.",
    variant="info",
)

callout(
    "💡 <strong>Consejo</strong>: los datos se cachean durante 1 hora. "
    "Si cambias los tickers o las fechas, la descarga se realizará automáticamente.",
    variant="warning",
)

# ============================================================
#  Footer
# ============================================================
footer()