"""Página de hedge ratio dinámico con Kalman filter."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.data_loader import load_prices
from app.core.kalman import (
    compare_static_dynamic,
    kalman_hedge_ratio,
    rolling_ols_hedge_ratio,
)
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section

page_setup("Kalman", "🎛️")

hero(
    title="Hedge Ratio Dinámico (Kalman)",
    subtitle=(
        "Filtro de Kalman para estimar el hedge ratio (beta) de un par "
        "cointegrado de forma adaptativa, sin ventana móvil. Compara con "
        "OLS estático y OLS rodante."
    ),
    icon="🎛️",
)

ensure_session_initialized()
tickers, start, end = get_global_params()
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if len(tickers) < 2:
    callout("Esta página requiere al menos 2 tickers.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros Kalman")

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_kalman_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_kalman_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                for field, key in [("t1", "kalman_t1"), ("t2", "kalman_t2")]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if val in tickers:
                            st.session_state[key] = val
                        else:
                            skipped.append(f"{field} ('{val}' no está en tus tickers actuales)")
                for field, key, lo, hi in [
                    ("delta", "kalman_delta", 1e-5, 1e-2),
                    ("r_var", "kalman_r_var", 1e-5, 1e-2),
                    ("roll_window", "kalman_roll_window", 20, 200),
                ]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if isinstance(val, (int, float)) and lo <= val <= hi:
                            st.session_state[key] = val
                        else:
                            skipped.append(field)
                st.session_state["_kalman_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    t1 = st.selectbox("Ticker Y (dependiente)", tickers, index=0, key="kalman_t1")
    t2 = st.selectbox("Ticker X (independiente)", tickers, index=1, key="kalman_t2")
    delta = st.slider(
        "Delta (reactividad)", 1e-5, 1e-2, 1e-4, 1e-5,
        format="%.5f",
        help="Valores más altos → beta más reactivo a cambios recientes.",
        key="kalman_delta",
    )
    r_var = st.slider(
        "R (varianza observación)", 1e-5, 1e-2, 1e-3, 1e-4,
        format="%.4f",
        key="kalman_r_var",
    )
    roll_window = st.slider(
        "Ventana OLS rodante (comparación)", 20, 200, 60, key="kalman_roll_window",
    )

    current_config = {
        "t1": t1, "t2": t2, "delta": delta, "r_var": r_var, "roll_window": roll_window,
    }
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="kalman_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Calcular", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices([t1, t2], start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    y = prices[t1]
    x = prices[t2]

    with st.spinner("Aplicando filtro de Kalman..."):
        try:
            kalman = kalman_hedge_ratio(y, x, delta=delta, r_var=r_var)
        except ValueError as e:
            callout(f"Error en Kalman: {e}", variant="danger")
            st.stop()

    with st.spinner("Calculando OLS rodante..."):
        rolling_beta = rolling_ols_hedge_ratio(y, x, window=roll_window)

    comp = compare_static_dynamic(y, x, delta=delta, r_var=r_var)

    # --- KPIs ---
    section("📊 Resultado")
    beta_final = float(kalman.beta.iloc[-1])
    beta_ols = float(comp["beta_ols"].iloc[0])
    beta_std_final = float(kalman.beta_std.iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta Kalman (final)", f"{beta_final:.4f}")
    c2.metric("Beta OLS (estático)", f"{beta_ols:.4f}")
    c3.metric("Desv. std. beta", f"{beta_std_final:.4f}")
    c4.metric("Log-likelihood", f"{kalman.log_likelihood:.2f}")

    # --- Gráfico de betas ---
    section("📈 Evolución del hedge ratio")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=comp.index, y=comp["beta_ols"],
        mode="lines", name="OLS estático",
        line={"color": "gray", "width": 1.5, "dash": "dash"},
    ))
    fig.add_trace(go.Scatter(
        x=rolling_beta.index, y=rolling_beta.values,
        mode="lines", name=f"OLS rodante ({roll_window}d)",
        line={"color": "orange", "width": 1.5},
    ))
    fig.add_trace(go.Scatter(
        x=kalman.beta.index, y=kalman.beta.values,
        mode="lines", name="Kalman",
        line={"color": "#2563eb", "width": 2},
    ))
    # Banda ±2σ del Kalman
    fig.add_trace(go.Scatter(
        x=kalman.beta.index,
        y=(kalman.beta + 2 * kalman.beta_std).values,
        mode="lines", line={"width": 0}, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=kalman.beta.index,
        y=(kalman.beta - 2 * kalman.beta_std).values,
        mode="lines", line={"width": 0},
        fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Hedge ratio β(t) — {t1} vs {t2}",
        yaxis_title="β",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    callout(
        "La banda azul clara es <strong>±2σ</strong> del beta estimado. "
        "Si se ensancha, el filtro tiene menos certeza sobre el hedge ratio.",
        variant="info",
    )

    # --- Spread dinámico vs estático ---
    section("📉 Spread dinámico")

    from app.core.cointegration import engle_granger

    static = engle_granger(y, x)
    spread_static = static.spread.reindex(kalman.spread.index)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=spread_static.index, y=spread_static.values,
        mode="lines", name="Spread OLS estático",
        line={"color": "gray", "width": 1.2},
    ))
    fig_s.add_trace(go.Scatter(
        x=kalman.spread.index, y=kalman.spread.values,
        mode="lines", name="Spread Kalman",
        line={"color": "#2563eb", "width": 1.5},
    ))
    fig_s.update_layout(
        title="Spread estático vs dinámico",
        yaxis_title="Spread",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_s, use_container_width=True)

    # --- Comparación numérica ---
    section("📋 Comparación numérica")

    diff = (kalman.beta - comp["beta_ols"].iloc[0])
    stats = pd.DataFrame({
        "Métrica": [
            "Beta OLS (estático)",
            "Beta Kalman (final)",
            "Beta Kalman (media)",
            "Beta Kalman (std)",
            "Diferencia final OLS-Kalman",
            "Correlación OLS rodante vs Kalman",
        ],
        "Valor": [
            f"{beta_ols:.4f}",
            f"{beta_final:.4f}",
            f"{kalman.beta.mean():.4f}",
            f"{kalman.beta.std():.4f}",
            f"{float(diff.iloc[-1]):.4f}",
            f"{rolling_beta.corr(kalman.beta):.4f}",
        ],
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)

    # --- Descargas ---
    section("⬇️ Descargas")
    c1, c2 = st.columns(2)
    with c1:
        kalman_df = pd.DataFrame({
            "beta": kalman.beta,
            "alpha": kalman.alpha,
            "spread": kalman.spread,
            "beta_std": kalman.beta_std,
        })
        st.download_button(
            "Kalman (CSV)",
            kalman_df.to_csv().encode("utf-8"),
            file_name=f"kalman_{t1}_{t2}.csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Comparación betas (CSV)",
            comp.to_csv().encode("utf-8"),
            file_name=f"beta_comparison_{t1}_{t2}.csv",
            use_container_width=True,
        )

else:
    callout(
        "Configura los tickers y parámetros en la barra lateral y pulsa "
        "<strong>🚀 Calcular</strong>.<br><br>"
        "<strong>Parámetros del Kalman:</strong><br>"
        "• <code>delta</code>: reactividad del beta. Bajo (1e-5) → suave; "
        "alto (1e-2) → reactivo.<br>"
        "• <code>R</code>: varianza del ruido de observación. Bajo → "
        "el filtro confía más en los datos.<br><br>"
        "<strong>Cuándo usar Kalman:</strong> cuando esperas que la relación "
        "entre los dos activos cambie lentamente (fusiones, cambios de "
        "régimen, etc.).",
        variant="info",
    )

footer()
