"""Página de cointegración y pairs trading."""
import streamlit as st

from app.config import DEFAULT_TICKERS
from app.core.cointegration import (
    engle_granger,
    generate_signals,
    half_life,
    rolling_zscore,
)
from app.core.data_loader import load_prices
from app.core.plotting import line_chart, zscore_chart

st.set_page_config(page_title="Cointegración", page_icon="🔗", layout="wide")
st.header("🔗 Cointegración y Pairs Trading")

with st.sidebar:
    tickers = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS)).split(",")
    tickers = [t.strip().upper() for t in tickers]
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Ticker 1", tickers, index=0)
    t2 = col2.selectbox("Ticker 2", tickers, index=1)
    window = st.slider("Ventana Z-score", 20, 120, 60)
    entry = st.slider("Umbral de entrada", 0.5, 3.0, 2.0, 0.1)
    run = st.button("🚀 Ejecutar análisis", type="primary")

if run:
    with st.spinner("Descargando datos..."):
        prices = load_prices(tickers, st.session_state.get("global_start"), st.session_state.get("global_end"))

    result = engle_granger(prices[t1], prices[t2])

    col1, col2, col3 = st.columns(3)
    col1.metric("p-valor Engle-Granger", f"{result.pvalue:.4f}")
    col2.metric("Beta (cobertura)", f"{result.beta:.4f}")
    col3.metric("Cointegradas", "✅" if result.is_cointegrated else "❌")

    st.subheader("Test ADF")
    st.write(f"ADF {t1}: p-valor = {result.adf_pvalue_1:.4f}")
    st.write(f"ADF {t2}: p-valor = {result.adf_pvalue_2:.4f}")

    st.subheader("Spread")
    st.plotly_chart(line_chart(result.spread, f"Spread {t1} - β·{t2}"))

    hl = half_life(result.spread)
    st.info(f"⏱️ Half-life de reversión: **{hl:.1f} días**" if hl != float("inf") else "Half-life no definida (no revierte).")

    st.subheader(f"Z-score (ventana={window})")
    z = rolling_zscore(result.spread, window)
    st.plotly_chart(zscore_chart(z, entry=entry))

    st.subheader("Señales de trading")
    signals = generate_signals(z, entry=entry)
    st.plotly_chart(line_chart(signals, "Señales (1=long, -1=short, 0=neutral)"))

    st.download_button(
        "⬇️ Descargar señales (CSV)",
        signals.to_csv().encode("utf-8"),
        file_name=f"signals_{t1}_{t2}.csv",
    )
