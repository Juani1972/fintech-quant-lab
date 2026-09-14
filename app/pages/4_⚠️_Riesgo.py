"""Página de métricas de riesgo."""
import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.plotting import drawdown_chart, line_chart
from app.core.risk import (
    calmar_ratio,
    drawdown_series,
    expected_shortfall,
    expected_shortfall_parametric,
    max_drawdown,
    rolling_var,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    value_at_risk_parametric,
)

st.set_page_config(page_title="Riesgo", page_icon="⚠️", layout="wide")
st.header("⚠️ Medición de Riesgo")

tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if not tickers:
    st.error("Introduce al menos un ticker en la barra lateral.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ Parámetros de riesgo")
    ticker = st.selectbox("Ticker", tickers)
    confidence = st.slider("Nivel de confianza", 0.90, 0.99, 0.95, 0.01)
    window = st.slider("Ventana VaR rodante", 50, 500, 250)
    run = st.button("🚀 Calcular riesgo", type="primary", use_container_width=True)

if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    returns = compute_log_returns(prices)[ticker]

    st.subheader("📉 VaR y Expected Shortfall")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR hist. {confidence:.0%}", f"{value_at_risk(returns, confidence):.4%}")
    c2.metric(
        f"VaR param. {confidence:.0%}",
        f"{value_at_risk_parametric(returns, confidence):.4%}",
    )
    c3.metric(f"ES hist. {confidence:.0%}", f"{expected_shortfall(returns, confidence):.4%}")
    c4.metric(
        f"ES param. {confidence:.0%}",
        f"{expected_shortfall_parametric(returns, confidence):.4%}",
    )

    st.subheader("📊 Ratios ajustados por riesgo")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sharpe", f"{sharpe_ratio(returns, periods_per_year=252):.3f}")
    r2.metric("Sortino", f"{sortino_ratio(returns, periods_per_year=252):.3f}")
    r3.metric("Calmar", f"{calmar_ratio(returns, periods_per_year=252):.3f}")
    r4.metric("Max Drawdown", f"{max_drawdown(prices[ticker]):.2%}")

    st.subheader("VaR rodante")
    st.plotly_chart(
        line_chart(
            rolling_var(returns, window, confidence),
            f"VaR rodante ({window}d, {confidence:.0%})",
        ),
        use_container_width=True,
    )

    st.subheader("Drawdown")
    st.plotly_chart(
        drawdown_chart(drawdown_series(prices[ticker])),
        use_container_width=True,
    )
