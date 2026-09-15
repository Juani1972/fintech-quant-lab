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
if "global_tickers" not in st.session_state:
    st.session_state["global_tickers"] = ", ".join(DEFAULT_TICKERS)
if "global_start" not in st.session_state:
    st.session_state["global_start"] = DEFAULT_START
if "global_end" not in st.session_state:
    st.session_state["global_end"] = DEFAULT_END


# ============================================================
#  Callback: al cambiar de universo, rellenar el campo de tickers
# ============================================================
def _on_universe_change() -> None:
    """Rellena `global_tickers` con los tickers del universo seleccionado."""
    name = st.session_state.get("_universe_select", "(personalizado)")
    if name == "(personalizado)":
        return
    tickers_str = universe_to_string(name)
    if tickers_str:
        st.session_state["global_tickers"] = tickers_str


# ============================================================
#  Sidebar: parámetros globales
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Parámetros globales")
    st.caption("Compartidos entre todas las páginas.")

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

    # --- Campo de tickers (siempre visible, editable) ---
    st.text_input(
        "Tickers (separados por coma)",
        key="global_tickers",
        help="Ejemplo: AAPL, MSFT, KO, PEP",
    )

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

    st.divider()

    # --- Botón de cierre (solo local) ---
    LOCAL_MODE = os.getenv("FQL_LOCAL_MODE", "true").lower() == "true"
    if LOCAL_MODE:
        if st.button(
            "🚪 Cerrar aplicación",
            type="secondary",
            use_container_width=True,
            help="Detiene el servidor Streamlit. Solo disponible en modo local.",
        ):
            st.warning("Cerrando Fintech Quant Lab...")
            os.kill(os.getpid(), signal.SIGTERM)
    else:
        st.caption("🔒 Botón de cierre deshabilitado (modo producción).")


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
