"""Página de análisis de robustez."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core.cointegration import engle_granger
from app.core.data_loader import load_prices
from app.core.robustness import (
    monte_carlo_bootstrap,
    parameter_sensitivity,
    robustness_score,
)
from app.core.walkforward import (
    signal_from_mean_reversion,
    signal_from_momentum,
    signal_from_pairs_trading,
    walk_forward_analysis,
)
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Robustez", "🛡️")

hero(
    title="Análisis de Robustez",
    subtitle=(
        "Comprueba si la estrategia generaliza o es producto del overfitting: "
        "walk-forward, Monte Carlo, sensibilidad de parámetros y score agregado."
    ),
    icon="🛡️",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if len(tickers) < 1:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Configuración Robustez")
    strategy = st.selectbox(
        "Estrategia",
        ["Momentum", "Mean Reversion", "Pairs Trading (spread)"],
    )

    if strategy == "Pairs Trading (spread)":
        if len(tickers) < 2:
            callout("Pairs Trading requiere al menos 2 tickers.", variant="warning")
            st.stop()
        t1 = st.selectbox("Ticker 1", tickers, index=0)
        t2 = st.selectbox("Ticker 2", tickers, index=1)
    else:
        t1 = st.selectbox("Ticker", tickers, index=0)
        t2 = None

    st.markdown("**Parámetros base**")
    window = st.slider("Ventana", 20, 200, 60, 5)
    if strategy != "Momentum":
        entry = st.slider("Umbral entrada", 0.5, 3.0, 2.0, 0.1)
    else:
        entry = None

    st.markdown("**Walk-forward**")
    train_size = st.slider("Train", 200, 1000, 504, 21)
    test_size = st.slider("Test", 21, 250, 126, 21)

    st.markdown("**Monte Carlo**")
    n_sims = st.slider("Nº simulaciones", 200, 5000, 1000, 100)
    block_size = st.slider("Block size", 1, 20, 1, 1)

    run = st.button("🚀 Analizar robustez", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    # --- Preparar serie y factorías ---
    if strategy == "Pairs Trading (spread)":
        try:
            coint = engle_granger(prices[t1], prices[t2])
            series = coint.spread
        except Exception as e:
            callout(f"Error calculando cointegración: {e}", variant="danger")
            st.stop()

        def generator_factory(params):
            return signal_from_pairs_trading(
                window=params["window"],
                entry=params.get("entry", 2.0),
                exit_=0.5,
            )

        def simple_factory(p, params):
            from app.core.cointegration import generate_signals, rolling_zscore
            z = rolling_zscore(p, window=params["window"], shift=1)
            return generate_signals(z, entry=params.get("entry", 2.0), exit_=0.5)

        base_params = {"window": window, "entry": entry}
    elif strategy == "Momentum":
        series = prices[t1]

        def generator_factory(params):
            return signal_from_momentum(window=params["window"])

        def simple_factory(p, params):
            ret = p.pct_change(params["window"])
            s = pd.Series(0, index=p.index, dtype=int)
            s[ret > 0] = 1
            s[ret < 0] = -1
            return s

        base_params = {"window": window}
    else:
        series = prices[t1]

        def generator_factory(params):
            return signal_from_mean_reversion(
                window=params["window"],
                entry=params.get("entry", 1.5),
                exit_=0.5,
            )

        def simple_factory(p, params):
            mean = p.rolling(params["window"]).mean().shift(1)
            std = p.rolling(params["window"]).std().shift(1)
            z = (p - mean) / std
            s = pd.Series(0, index=p.index, dtype=int)
            pos = 0
            for i, zi in enumerate(z):
                if np.isnan(zi):
                    s.iloc[i] = pos
                    continue
                if pos == 0:
                    if zi > params.get("entry", 1.5):
                        pos = -1
                    elif zi < -params.get("entry", 1.5):
                        pos = 1
                elif abs(zi) < 0.5:
                    pos = 0
                s.iloc[i] = pos
            return s

        base_params = {"window": window, "entry": entry}

    with st.spinner("Ejecutando walk-forward..."):
        try:
            wf = walk_forward_analysis(
                prices=series,
                signal_generator=generator_factory(base_params),
                train_size=train_size,
                test_size=test_size,
            )
        except ValueError as e:
            callout(f"Error en walk-forward: {e}", variant="danger")
            st.stop()

    is_sharpe = wf.is_metrics_agg.get("sharpe", np.nan)
    oos_sharpe = wf.oos_metrics_agg.get("sharpe", np.nan)
    n_trades = wf.oos_metrics_agg.get("n_trades", 0)

    oos_returns = wf.oos_equity_concat.pct_change().dropna()
    if len(oos_returns) < 10:
        callout("Pocos retornos OOS para Monte Carlo fiable.", variant="warning")

    with st.spinner("Ejecutando Monte Carlo..."):
        try:
            mc = monte_carlo_bootstrap(
                oos_returns,
                n_simulations=n_sims,
                block_size=block_size,
                metric_name="sharpe",
            )
        except ValueError as e:
            callout(f"Error en Monte Carlo: {e}", variant="danger")
            st.stop()

    with st.spinner("Analizando sensibilidad..."):
        if strategy == "Momentum":
            variations = [max(5, window - 30), max(5, window - 15),
                          window, window + 15, window + 30]
            sens = parameter_sensitivity(
                series, simple_factory, base_params,
                param_name="window", variations=variations,
                metric="sharpe",
            )
        else:
            variations = [max(0.5, entry - 0.5), max(0.5, entry - 0.25),
                          entry, entry + 0.25, entry + 0.5]
            sens = parameter_sensitivity(
                series, simple_factory, base_params,
                param_name="entry", variations=variations,
                metric="sharpe",
            )

    report = robustness_score(
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        mc_result=mc,
        sensitivity_score=sens.stability_score,
        n_trades=int(n_trades),
    )

    section("🎯 Robustness Score")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Score final", f"{report.final_score:.1f} / 100")
    with c2:
        callout(report.interpretation, variant="info")

    comp_df = pd.DataFrame(
        list(report.components.items()),
        columns=["Componente", "Puntos"],
    ).set_index("Componente")
    st.dataframe(comp_df, use_container_width=True)

    section("📊 Walk-Forward (IS vs OOS)")
    wf_df = pd.DataFrame({
        "In-Sample": [
            f"{wf.is_metrics_agg.get('sharpe', np.nan):.2f}",
            f"{wf.is_metrics_agg.get('total_return', np.nan):.2%}",
            f"{wf.is_metrics_agg.get('max_drawdown', np.nan):.2%}",
        ],
        "Out-of-Sample": [
            f"{wf.oos_metrics_agg.get('sharpe', np.nan):.2f}",
            f"{wf.oos_metrics_agg.get('total_return', np.nan):.2%}",
            f"{wf.oos_metrics_agg.get('max_drawdown', np.nan):.2%}",
        ],
    }, index=["Sharpe", "Return total", "Max DD"])
    st.dataframe(wf_df, use_container_width=True)

    section("🎲 Monte Carlo (distribución del Sharpe OOS)")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc.distribution, nbinsx=50,
        marker_color="#2563eb", opacity=0.75,
        name="Sharpe simulado",
    ))
    fig.add_vline(x=mc.mean, line_dash="dash", line_color="green",
                  annotation_text=f"Media {mc.mean:.2f}")
    fig.add_vline(x=0, line_color="red", line_width=1)
    fig.update_layout(
        xaxis_title="Sharpe", yaxis_title="Frecuencia",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    mc_summary = pd.DataFrame({
        "Estadístico": ["Media", "Std", "p05", "p25", "p50", "p75", "p95", "VaR 95%", "CVaR 95%"],
        "Valor": [
            f"{mc.mean:.3f}", f"{mc.std:.3f}",
            f"{mc.percentiles['p05']:.3f}", f"{mc.percentiles['p25']:.3f}",
            f"{mc.percentiles['p50']:.3f}", f"{mc.percentiles['p75']:.3f}",
            f"{mc.percentiles['p95']:.3f}", f"{mc.var_95:.3f}", f"{mc.cvar_95:.3f}",
        ],
    })
    st.dataframe(mc_summary, use_container_width=True, hide_index=True)

    prob_positive = float((mc.distribution > 0).mean())
    st.metric("P(Sharpe simulado > 0)", f"{prob_positive:.1%}")

    section(f"📉 Sensibilidad de `{sens.param_name}`")
    fig_s = px.line(
        sens.variations, x="value", y="sharpe",
        markers=True,
        labels={"value": sens.param_name, "sharpe": "Sharpe"},
    )
    fig_s.add_vline(x=sens.base_value, line_dash="dash",
                    line_color="red", annotation_text="Base")
    fig_s.update_layout(template="plotly_white", height=350)
    st.plotly_chart(fig_s, use_container_width=True)
    callout(
        f"Score de estabilidad: <strong>{sens.stability_score:.2f}</strong> "
        "(1 = muy estable). Busca una meseta alrededor del valor base, "
        "no un pico aislado.",
        variant="info",
    )

    section("⬇️ Descargas")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Monte Carlo (CSV)",
            pd.DataFrame({"sharpe": mc.distribution}).to_csv(index=False).encode("utf-8"),
            file_name="monte_carlo.csv",
        )
    with c2:
        st.download_button(
            "Sensibilidad (CSV)",
            sens.variations.to_csv(index=False).encode("utf-8"),
            file_name=f"sensitivity_{sens.param_name}.csv",
        )
    with c3:
        st.download_button(
            "Robustness components (CSV)",
            comp_df.to_csv().encode("utf-8"),
            file_name="robustness_components.csv",
        )

else:
    callout(
        "Configura la estrategia y los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Analizar robustez</strong>.",
        variant="info",
    )

footer()
