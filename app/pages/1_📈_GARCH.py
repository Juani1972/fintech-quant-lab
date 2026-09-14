"""Página de modelado GARCH."""
import numpy as np
import streamlit as st

from app.core.data_loader import load_prices, compute_log_returns
from app.core.garch import (
    check_stationarity,
    fit_garch,
    forecast_volatility,
    residual_diagnostics,
)
from app.core.plotting import line_chart

st.set_page_config(page_title="GARCH", page_icon="📈", layout="wide")
st.header("📈 Modelado GARCH de Volatilidad")

# --- Leer parámetros globales ---
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
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    returns = compute_log_returns(prices)[ticker]

    with st.spinner("Ajustando modelo..."):
        try:
            result = fit_garch(returns, p=p, q=q, vol=vol, dist=dist)
        except Exception as e:
            st.error(f"Error ajustando GARCH: {e}")
            st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{result.aic:.2f}")
    col2.metric("BIC", f"{result.bic:.2f}")
    col3.metric(
        "Estacionario",
        "✅" if check_stationarity(result.params, result.model_type) else "❌",
    )

    st.subheader("Resumen del modelo")
    st.text(result.model_result.summary().as_text())

    st.subheader("Volatilidad condicional")
    st.plotly_chart(
        line_chart(result.conditional_volatility, f"Volatilidad condicional — {ticker}"),
        use_container_width=True,
    )

    st.subheader("Diagnósticos de residuos")
    try:
        diag = residual_diagnostics(result, lags=10)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Ljung-Box (residuos)", f"{diag['ljung_box_pvalue']:.4f}")
        d2.metric("Ljung-Box (residuos²)", f"{diag['ljung_box_squared_pvalue']:.4f}")
        d3.metric("ARCH-LM", f"{diag['arch_lm_pvalue']:.4f}")
        d4.metric("Jarque-Bera", f"{diag['jarque_bera_pvalue']:.4f}")

        if diag["ljung_box_squared_pvalue"] > 0.05 and diag["arch_lm_pvalue"] > 0.05:
            st.success("No queda evidencia de heterocedasticidad condicional en los residuos.")
        else:
            st.warning("Queda evidencia de estructura en los residuos. Considera otro orden o modelo.")
    except Exception as e:
        st.info(f"No se pudieron calcular diagnósticos: {e}")

    st.subheader("Pronóstico de volatilidad (30 días)")
    fc = forecast_volatility(result, horizon=30)
    st.plotly_chart(line_chart(fc, "Pronóstico de volatilidad"), use_container_width=True)

    st.download_button(
        "⬇️ Descargar parámetros (CSV)",
        result.params.to_csv().encode("utf-8"),
        file_name=f"garch_params_{ticker}.csv",
    )
