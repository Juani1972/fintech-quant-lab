"""Punto de entrada de la aplicación Streamlit - Fintech Quant Lab."""
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
    LAYOUT,
)

# ============================================================
#  Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Fintech Quant Lab",
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

# ============================================================
#  Estado inicial de sesión (parámetros globales)
# ============================================================
if "global_tickers" not in st.session_state:
    st.session_state["global_tickers"] = ", ".join(DEFAULT_TICKERS)
if "global_start" not in st.session_state:
    st.session_state["global_start"] = DEFAULT_START
if "global_end" not in st.session_state:
    st.session_state["global_end"] = DEFAULT_END

# ============================================================
#  Barra lateral: parámetros globales compartidos
# ============================================================
with st.sidebar:
    st.header("⚙️ Parámetros globales")

    st.text_input(
        "Tickers (separados por coma)",
        key="global_tickers",
        help="Ejemplo: AAPL, MSFT, KO, PEP",
    )

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

    st.caption("Estos parámetros se comparten entre todas las páginas.")

    st.markdown("---")

    # --------------------------------------------------------
    #  Botón de cierre (solo en modo local)
    # --------------------------------------------------------
    LOCAL_MODE = os.getenv("FQL_LOCAL_MODE", "true").lower() == "true"

    if LOCAL_MODE:
        if st.button(
            "🚪 Cerrar aplicación",
            type="secondary",
            use_container_width=True,
            help="Detiene el servidor Streamlit. Solo disponible en modo local.",
        ):
            st.warning("Cerrando Fintech Quant Lab... El servidor se detendrá en unos segundos.")
            os.kill(os.getpid(), signal.SIGTERM)
    else:
        st.caption("🔒 Botón de cierre deshabilitado (modo producción).")

# ============================================================
#  Contenido principal
# ============================================================
st.title(f"{APP_ICON} Fintech Quant Lab")

st.markdown(
    """
    ### Análisis cuantitativo de series temporales financieras

    Bienvenido a **Fintech Quant Lab**, una aplicación visual para realizar
    análisis avanzados sobre datos de mercado. Usa el menú lateral para
    navegar entre los distintos módulos:

    | Módulo | Descripción |
    |---|---|
    | **📈 GARCH** | Modelado de volatilidad condicional (GARCH, EGARCH, GJR-GARCH). |
    | **🔗 Cointegración** | Pairs trading, test de Engle-Granger, z-score, half-life. |
    | **📊 Fama-French** | Regresión de 3 y 5 factores con interpretación de alpha. |
    | **⚠️ Riesgo** | VaR, Expected Shortfall, drawdown, Sharpe, Sortino. |
    | **🧪 Backtest** | Motor de backtesting con anti-look-ahead y costes. |

    ---

    ### 🚀 Cómo empezar

    1. Introduce los **tickers** en la barra lateral (ej: `KO, PEP`).
    2. Selecciona el **rango de fechas**.
    3. Navega a la página del análisis que quieras realizar.
    4. Ajusta los parámetros específicos y pulsa **Ejecutar**.
    5. Descarga los resultados en CSV desde cada página.
    """
)

# ------------------------------------------------------------
#  Mostrar los parámetros activos en la portada
# ------------------------------------------------------------
tickers_actuales = st.session_state.get("global_tickers", "")
fecha_ini = st.session_state.get("global_start", DEFAULT_START)
fecha_fin = st.session_state.get("global_end", DEFAULT_END)

col1, col2, col3 = st.columns(3)
col1.metric("Tickers seleccionados", tickers_actuales or "—")
col2.metric("Fecha inicio", str(fecha_ini))
col3.metric("Fecha fin", str(fecha_fin))

st.markdown("---")

st.info(
    "💡 **Consejo**: Los datos se cachean durante 1 hora. "
    "Si cambias los tickers o las fechas, la descarga se realizará automáticamente."
)

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em; margin-top: 3em;'>
    Fintech Quant Lab · v0.2.0 · MIT License · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
