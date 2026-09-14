"""Página de métricas de riesgo."""
import numpy as np
import streamlit as st

from app.config import DEFAULT_TICKERS
from app.core.data_loader import load_prices
from app.core.plotting import drawdown_chart, line_chart
from app.core.risk import (
    drawdown_series,
    expected_shortfall,
    max_drawdown,
    rolling_var,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

st.set_page_config(page_title="Riesgo", page_icon="⚠️", layout="wide")
st.header("⚠️ Medición de Riesgo")

with st.sidebar:
    tickers = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS)).split(",")
    tickers = [t.strip().upper() for t in tickers]
    ticker = st.selectbox("Ticker", tickers)
    confidence = st.slider("Nivel de confianza", 0.90, 0.99, 0.95, 0.01)
    window = st.slider("Ventana VaR rodante", 50, 500, 250)
    run = st.button("🚀 Calcular riesgo", type="primary")

if run:
    with st.spinner("Descargando datos..."):
        prices = load_prices(tickers, st.session_state.get("global_start"), st.session_state.get("global_end"))
        returns = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"VaR {confidence:.0%}", f"{value_at_risk(returns, confidence):.4%}")
    col2.metric(f"ES {confidence:.0%}", f"{expected_shortfall(returns, confidence):.4%}")
    col3.metric("Sharpe", f"{sharpe_ratio(returns):.3f}")
    col4.metric("Sortino", f"{sortino_ratio(returns):.3f}")

    st.metric("Máximo Drawdown", f"{max_drawdown(prices[ticker]):.2%}")

    st.subheader("VaR rodante")
    st.plotly_chart(line_chart(rolling_var(returns, window, confidence), f"VaR rodante ({window}d, {confidence:.0%})"))

    st.subheader("Drawdown")
    st.plotly_chart(drawdown_chart(drawdown_series(prices[ticker])))
