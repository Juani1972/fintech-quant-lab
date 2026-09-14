"""Página de modelado GARCH."""
import numpy as np
import streamlit as st

from app.core.data_loader import load_prices
from app.core.garch import fit_garch, forecast_volatility, is_stationary
from app.core.plotting import line_chart

st.set_page_config(page_title="GARCH", page_icon="📈", layout="wide")
st.header("📈 Modelado GARCH de Volatilidad")

# --- Leer parámetros globales de la sesión ---
tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if not tickers:
    st.error("Introduce al menos un ticker en la barra lateral.")
    st.stop()

# --- Parámetros específicos de esta página ---
with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ Parámetros GARCH")
    ticker = st.selectbox("Ticker a modelar", tickers)
    p = st.slider("Orden ARCH (p)", 1, 3, 1)
    q = st.slider("Orden GARCH (q)", 1, 3, 1)
    vol = st.selectbox("Tipo de modelo", ["Garch", "EGARCH", "GJR-GARCH"])
    dist = st.selectbox("Distribución", ["normal", "t", "skewt", "ged"])
    run = st.button("🚀 Ejecutar GARCH", type="primary", use_container_width=True)

if run:
    with st.spinner("Descargando datos..."):
        prices = load_prices(tickers, start, end)
        returns = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()

    with st.spinner("Ajustando modelo..."):
        result = fit_garch(returns, p=p, q=q, vol=vol, dist=dist)

    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{result.aic:.2f}")
    col2.metric("BIC", f"{result.bic:.2f}")
    col3.metric("Estacionario", "✅" if is_stationary(result.params) else "❌")

    st.subheader("Resumen del modelo")
    st.text(result.model_result.summary().as_text())

    st.subheader("Volatilidad condicional")
    st.plotly_chart(line_chart(result.conditional_volatility, f"Volatilidad condicional — {ticker}"))

    st.subheader("Pronóstico de volatilidad (30 días)")
    fc = forecast_volatility(result, horizon=30)
    st.plotly_chart(line_chart(fc, "Pronóstico de volatilidad"))

    st.download_button(
        "⬇️ Descargar parámetros (CSV)",
        result.params.to_csv().encode("utf-8"),
        file_name=f"garch_params_{ticker}.csv",
    )
