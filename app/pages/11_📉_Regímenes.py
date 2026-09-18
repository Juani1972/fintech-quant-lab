"""Página de detección de regímenes con HMM."""
import json

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
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section, ticker_badge

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
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if not tickers:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros HMM")

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_regimenes_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_regimenes_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                ticker_val = loaded_cfg.get("ticker")
                if ticker_val is not None:
                    if ticker_val in tickers:
                        st.session_state["regimenes_ticker"] = ticker_val
                    else:
                        skipped.append(f"ticker ('{ticker_val}' no está en tus tickers actuales)")
                cov_val = loaded_cfg.get("cov_type")
                if cov_val is not None:
                    if cov_val in ["diag", "full", "tied", "spherical"]:
                        st.session_state["regimenes_cov_type"] = cov_val
                    else:
                        skipped.append("cov_type")
                for field, key, lo, hi in [
                    ("n_states", "regimenes_n_states", 2, 5),
                    ("n_iter", "regimenes_n_iter", 50, 1000),
                ]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if isinstance(val, int) and lo <= val <= hi:
                            st.session_state[key] = val
                        else:
                            skipped.append(field)
                st.session_state["_regimenes_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    ticker = st.selectbox("Ticker", tickers, key="regimenes_ticker")
    n_states = st.slider(
        "Nº de regímenes", 2, 5, 2,
        help=(
            "Nº de estados ocultos (p.ej. 'tranquilo' / 'agitado') que el "
            "modelo intenta distinguir. 2 es lo más habitual e "
            "interpretable; más estados capturan matices pero son más "
            "difíciles de etiquetar con sentido y de estimar de forma "
            "estable."
        ),
        key="regimenes_n_states",
    )
    cov_type = st.selectbox(
        "Tipo de covarianza", ["diag", "full", "tied", "spherical"],
        help=(
            "Cómo modela cada régimen la relación entre variables. "
            "'diag' (el más simple y robusto) asume que no hay "
            "correlación entre ellas dentro de cada régimen; 'full' "
            "captura toda la estructura de correlación pero necesita "
            "más datos para estimarse bien."
        ),
        key="regimenes_cov_type",
    )
    n_iter = st.slider(
        "Iteraciones EM", 50, 1000, 200, 50,
        help=(
            "Nº máximo de iteraciones del algoritmo Expectation-"
            "Maximization usado para ajustar el modelo oculto de Markov. "
            "Si no converge con el valor por defecto, subirlo rara vez "
            "ayuda -- suele indicar que hay pocos datos o demasiados "
            "regímenes para lo que muestran."
        ),
        key="regimenes_n_iter",
    )

    current_config = {
        "ticker": ticker, "n_states": n_states, "cov_type": cov_type, "n_iter": n_iter,
    }
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="regimenes_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Detectar regímenes", type="primary", use_container_width=True)

ticker_badge(ticker)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices([ticker], start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

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
