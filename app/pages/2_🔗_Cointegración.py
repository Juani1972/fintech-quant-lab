"""Página de cointegración y pairs trading."""
import streamlit as st

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

tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if len(tickers) < 2:
    st.error("Esta página requiere al menos 2 tickers. Añádelos en la barra lateral.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ Parámetros Pairs Trading")
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Ticker 1", tickers, index=0)
    t2 = col2.selectbox("Ticker 2", tickers, index=1)
    window = st.slider("Ventana Z-score", 20, 200, 60)
    entry = st.slider("Umbral de entrada", 0.5, 3.0, 2.0, 0.1)
    run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)

if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    result = engle_granger(prices[t1], prices[t2])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p-valor Engle-Granger", f"{result.pvalue:.4f}")
    c2.metric("ADF spread (p)", f"{result.adf_spread_pvalue:.4f}")
    c3.metric("Beta (cobertura)", f"{result.beta:.4f}")
    c4.metric("Cointegradas", "✅" if result.is_cointegrated else "❌")

    st.subheader("Modelo de cointegración")
    st.markdown(
        f"**Spread:** `{t1} - {result.alpha:.4f} - {result.beta:.4f} · {t2}`"
    )

    st.subheader("Test ADF del spread")
    st.write(f"**Estadístico:** {result.adf_spread_stat:.4f}")
    st.write(f"**p-valor:** {result.adf_spread_pvalue:.4f}")
    st.write("**Valores críticos:**")
    st.json(result.adf_spread_crit)

    if result.adf_spread_pvalue < 0.05:
        st.success("El spread es estacionario (ADF p < 0.05). Apto para pairs trading.")
    else:
        st.warning("El spread NO es estacionario según ADF. Cuidado con el pairs trading.")

    with st.expander("ADF individuales (contexto)"):
        st.write(f"ADF {t1}: p-valor = {result.adf_pvalue_1:.4f}")
        st.write(f"ADF {t2}: p-valor = {result.adf_pvalue_2:.4f}")
        st.caption(
            "Los ADF individuales suelen no rechazar la raíz unitaria. "
            "Lo relevante para pairs trading es el ADF del spread."
        )

    st.subheader("Spread")
    st.plotly_chart(
        line_chart(result.spread, f"Spread {t1} - α - β·{t2}"),
        use_container_width=True,
    )

    hl = half_life(result.spread)
    if hl != float("inf"):
        st.info(f"⏱️ Half-life de reversión: **{hl:.1f} días**")
    else:
        st.warning("Half-life no definida (el spread no revierte a la media).")

    st.subheader(f"Z-score (ventana={window})")
    z = rolling_zscore(result.spread, window)
    st.plotly_chart(zscore_chart(z, entry=entry), use_container_width=True)

    st.subheader("Señales de trading")
    signals = generate_signals(z, entry=entry)
    st.plotly_chart(
        line_chart(signals, "Señales (1=long, -1=short, 0=neutral)"),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Descargar señales (CSV)",
        signals.to_csv().encode("utf-8"),
        file_name=f"signals_{t1}_{t2}.csv",
    )
