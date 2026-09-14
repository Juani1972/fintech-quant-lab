"""Punto de entrada de la aplicación Streamlit."""
import streamlit as st

from app.config import APP_ICON, APP_TITLE, LAYOUT

st.set_page_config(page_title="Fintech Quant Lab", page_icon=APP_ICON, layout=LAYOUT)

st.title(f"{APP_ICON} Fintech Quant Lab")
st.markdown(
    """
    ### Análisis cuantitativo de series temporales financieras

    Usa el menú lateral para navegar entre los módulos:

    - **📈 GARCH**: Modelado de volatilidad condicional.
    - **🔗 Cointegración**: Pairs trading y spread.
    - **📊 Fama-French**: Regresión de factores.
    - **⚠️ Riesgo**: VaR, Expected Shortfall, drawdown.

    Introduce los tickers y el rango de fechas en la barra lateral.
    """
)

with st.sidebar:
    st.header("⚙️ Parámetros globales")
    st.text_input("Tickers (separados por coma)", "KO, PEP", key="global_tickers")
    st.date_input("Fecha inicio", key="global_start")
    st.date_input("Fecha fin", key="global_end")
    st.caption("Estos parámetros se comparten entre páginas.")
