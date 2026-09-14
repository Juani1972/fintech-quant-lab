"""Página de regresión Fama-French."""
import numpy as np
import streamlit as st

from app.config import DEFAULT_TICKERS
from app.core.data_loader import load_prices
from app.core.fama_french import load_factors, run_regression

st.set_page_config(page_title="Fama-French", page_icon="📊", layout="wide")
st.header("📊 Regresión Fama-French")

with st.sidebar:
    tickers = st.text_input("Tickers", ", ".join(DEFAULT_TICKERS)).split(",")
    tickers = [t.strip().upper() for t in tickers]
    ticker = st.selectbox("Ticker", tickers)
    model = st.selectbox("Modelo", ["3", "5"], format_func=lambda x: f"{x} factores")
    run = st.button("🚀 Ejecutar regresión", type="primary")

if run:
    with st.spinner("Descargando datos..."):
        prices = load_prices(tickers, st.session_state.get("global_start"), st.session_state.get("global_end"))
        returns = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()

    with st.spinner("Descargando factores..."):
        factors = load_factors(st.session_state.get("global_start"), st.session_state.get("global_end"), model=model)

    result = run_regression(returns, factors, model=model)

    col1, col2, col3 = st.columns(3)
    col1.metric("Alpha", f"{result.alpha:.4%}")
    col2.metric("p-valor alpha", f"{result.alpha_pvalue:.4f}")
    col3.metric("R² ajustado", f"{result.adj_r_squared:.4f}")

    if result.alpha_pvalue < 0.05:
        st.success("Alpha estadísticamente significativo.")
    else:
        st.info("Alpha no significativo: el retorno se explica por los factores.")

    st.subheader("Betas de factores")
    st.dataframe(result.betas.rename("Beta").to_frame().assign(p_valor=result.betas_pvalues))

    st.subheader("Resumen completo")
    st.text(result.summary)
