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
    value_at_risk_cornish_fisher,
    value_at_risk_parametric,
)
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Riesgo", "⚠️")

hero(
    title="Medición de Riesgo",
    subtitle=(
        "VaR histórico y paramétrico, Expected Shortfall, drawdown, "
        "Sharpe, Sortino y Calmar."
    ),
    icon="⚠️",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if not tickers:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros de riesgo")
    ticker = st.selectbox("Ticker", tickers)
    confidence = st.slider("Nivel de confianza", 0.90, 0.99, 0.95, 0.01)
    window = st.slider("Ventana VaR rodante", 50, 500, 250)
    run = st.button("🚀 Calcular riesgo", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    returns = compute_log_returns(prices)[ticker]

    section("📉 VaR y Expected Shortfall")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"VaR hist. {confidence:.0%}", f"{value_at_risk(returns, confidence):.4%}")
    c2.metric(
        f"VaR param. {confidence:.0%}",
        f"{value_at_risk_parametric(returns, confidence):.4%}",
    )
    c3.metric(
        f"VaR Cornish-Fisher {confidence:.0%}",
        f"{value_at_risk_cornish_fisher(returns, confidence):.4%}",
        help="VaR paramétrico ajustado por la asimetría y curtosis "
             "reales de los retornos, en vez de asumir normalidad pura.",
    )
    c4.metric(f"ES hist. {confidence:.0%}", f"{expected_shortfall(returns, confidence):.4%}")
    c5.metric(
        f"ES param. {confidence:.0%}",
        f"{expected_shortfall_parametric(returns, confidence):.4%}",
    )

    section("📊 Ratios ajustados por riesgo")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sharpe", f"{sharpe_ratio(returns, periods_per_year=252):.3f}")
    r2.metric("Sortino", f"{sortino_ratio(returns, periods_per_year=252):.3f}")
    r3.metric("Calmar", f"{calmar_ratio(returns, periods_per_year=252):.3f}")
    r4.metric("Max Drawdown", f"{max_drawdown(prices[ticker]):.2%}")

    section("VaR rodante")
    st.plotly_chart(
        line_chart(
            rolling_var(returns, window, confidence),
            f"VaR rodante ({window}d, {confidence:.0%})",
        ),
        use_container_width=True,
    )

    section("Drawdown")
    st.plotly_chart(
        drawdown_chart(drawdown_series(prices[ticker])),
        use_container_width=True,
    )

else:
    callout(
        "Configura el ticker y los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Calcular riesgo</strong>.",
        variant="info",
    )

footer()
