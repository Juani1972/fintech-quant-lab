"""Página de detección de regímenes con HMM."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.regime import (
    current_regime,
    fit_hmm,
    regime_stats_table,
    regime_summary,
)
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Regímenes", "📉")

hero(
    title="Detección de Regímenes (HMM)",
    subtitle=(
        "Hidden Markov Model gaussiano para identificar regímenes de mercado "
        "(alta/baja volatilidad, bull/bear). Los estados se ordenan por "
        "volatilidad ascendente: Régimen 0 = menor volatilidad."
    ),
    icon="📉",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if not tickers:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros HMM")
    ticker = st.selectbox("Ticker", tickers)
    n_states = st.slider("Nº de regímenes", 2, 5, 2)
    cov_type = st.selectbox("Tipo de covarianza", ["diag", "full", "tied", "spherical"])
    n_iter = st.slider("Iteraciones EM", 50, 1000, 200, 50)
    run = st.button("🚀 Detectar regímenes", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices([ticker], start, end)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    returns = compute_log_returns(prices)[ticker]

    with st.spinner("Ajustando HMM..."):
        try:
            result = fit_hmm(
                returns,
                n_states=n_states,
                covariance_type=cov_type,
                n_iter=n_iter,
            )
        except ValueError as e:
            callout(f"Error ajustando HMM: {e}", variant="danger")
            st.stop()

    # --- KPIs ---
    section("📊 Resultado del ajuste")
    cur = current_regime(result)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log-likelihood", f"{result.log_likelihood:.2f}")
    c2.metric("AIC", f"{result.aic:.2f}")
    c3.metric("BIC", f"{result.bic:.2f}")
    c4.metric("Régimen actual", f"Régimen {cur}")

    # --- Tabla resumen por régimen ---
    section("📋 Estadísticas por régimen")
    summary = regime_summary(returns, result)
    st.dataframe(
        summary.style.format({
            "% tiempo": "{:.2%}",
            "Retorno anual": "{:.2%}",
            "Volatilidad anual": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Duración media (días)": "{:.1f}",
        }),
        use_container_width=True,
    )

    # --- Matriz de transición ---
    section("🔄 Matriz de transición")
    trans = regime_stats_table(result)
    st.dataframe(
        trans.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
    )
    callout(
        "Cada fila indica la probabilidad de pasar del régimen actual al "
        "siguiente en un paso. La diagonal (alta) indica persistencia.",
        variant="info",
    )

    # --- Gráfico de precio con regímenes coloreados ---
    section("📈 Precio con regímenes")

    price_series = prices[ticker].reindex(result.states.index).ffill()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name="Precio", line={"color": "black", "width": 1.2},
    ))

    # Bandas por régimen
    colors = ["rgba(34,197,94,0.15)", "rgba(239,68,68,0.15)",
              "rgba(59,130,246,0.15)", "rgba(245,158,11,0.15)",
              "rgba(139,92,246,0.15)"]

    states = result.states
    changes = states.diff().fillna(0) != 0
    segment_start = states.index[0]
    segment_state = states.iloc[0]

    for i in range(1, len(states)):
        if states.iloc[i] != segment_state or i == len(states) - 1:
            fig.add_vrect(
                x0=segment_start, x1=states.index[i],
                fillcolor=colors[segment_state % len(colors)],
                layer="below", line_width=0,
            )
            segment_start = states.index[i]
            segment_state = states.iloc[i]

    fig.update_layout(
        title=f"{ticker} con regímenes HMM",
        yaxis_title="Precio",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Probabilidades por régimen ---
    section("🎲 Probabilidades posteriores")
    fig_p = go.Figure()
    for col in result.state_probs.columns:
        fig_p.add_trace(go.Scatter(
            x=result.state_probs.index,
            y=result.state_probs[col],
            mode="lines",
            name=col,
            stackgroup="one",
        ))
    fig_p.update_layout(
        title="Probabilidad posterior por régimen",
        yaxis_title="Probabilidad",
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig_p, use_container_width=True)

    # --- Descargas ---
    section("⬇️ Descargas")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Resumen por régimen (CSV)",
            summary.to_csv().encode("utf-8"),
            file_name=f"regime_summary_{ticker}.csv",
            use_container_width=True,
        )
    with c2:
        states_df = pd.DataFrame({
            "state": result.states,
            **{c: result.state_probs[c] for c in result.state_probs.columns},
        })
        st.download_button(
            "Secuencia de estados (CSV)",
            states_df.to_csv().encode("utf-8"),
            file_name=f"regime_states_{ticker}.csv",
            use_container_width=True,
        )

else:
    callout(
        "Configura los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Detectar regímenes</strong>.<br><br>"
        "<strong>Sugerencias:</strong><br>"
        "• 2 regímenes → alta/baja volatilidad o bull/bear.<br>"
        "• 3 regímenes → añade un régimen neutral o de transición.<br>"
        "• <code>diag</code> es más rápido y estable; "
        "<code>full</code> permite correlaciones entre estados.",
        variant="info",
    )

footer()
