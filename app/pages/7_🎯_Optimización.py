"""Página de optimización de parámetros con walk-forward."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.cointegration import engle_granger
from app.core.data_loader import load_prices
from app.core.optimization import (
    grid_search_walkforward,
    heatmap_data,
)
from app.core.walkforward import (
    signal_from_mean_reversion,
    signal_from_momentum,
    signal_from_pairs_trading,
)

st.set_page_config(page_title="Optimización", page_icon="🎯", layout="wide")
st.header("🎯 Optimización de Parámetros")
st.caption(
    "Grid search evaluado con walk-forward (OOS). Este método evita el "
    "overfitting del grid search clásico porque puntúa cada combinación "
    "por su rendimiento out-of-sample."
)

tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if len(tickers) < 1:
    st.error("Introduce al menos un ticker en la barra lateral.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ Configuración")

    strategy = st.selectbox(
        "Estrategia",
        ["Momentum", "Mean Reversion", "Pairs Trading (spread)"],
    )

    if strategy == "Pairs Trading (spread)":
        if len(tickers) < 2:
            st.error("Pairs Trading requiere al menos 2 tickers.")
            st.stop()
        t1 = st.selectbox("Ticker 1", tickers, index=0)
        t2 = st.selectbox("Ticker 2", tickers, index=1)
    else:
        t1 = st.selectbox("Ticker", tickers, index=0)
        t2 = None

    st.markdown("**Grid de parámetros**")
    if strategy == "Momentum":
        windows = st.multiselect(
            "Ventanas a probar", [10, 20, 30, 45, 60, 90, 120],
            default=[20, 40, 60],
        )
    else:
        windows = st.multiselect(
            "Ventanas Z-score", [20, 30, 45, 60, 90, 120],
            default=[30, 60, 90],
        )
        entries = st.multiselect(
            "Umbrales de entrada", [1.0, 1.5, 2.0, 2.5, 3.0],
            default=[1.5, 2.0, 2.5],
        )

    objective = st.selectbox(
        "Objetivo OOS", ["sharpe", "sortino", "calmar", "total_return"],
        index=0,
    )

    train_size = st.slider("Train size", 200, 1000, 504, 21)
    test_size = st.slider("Test size", 21, 250, 126, 21)

    run = st.button("🚀 Optimizar", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    # --- Serie a operar + factoría de generadores ---
    if strategy == "Pairs Trading (spread)":
        try:
            coint = engle_granger(prices[t1], prices[t2])
            series = coint.spread
        except Exception as e:
            st.error(f"Error calculando cointegración: {e}")
            st.stop()

        def factory(params):
            return signal_from_pairs_trading(
                window=params["window"],
                entry=params["entry"],
                exit_=params.get("exit_", 0.5),
            )
        param_grid = {
            "window": windows,
            "entry": entries,
            "exit_": [0.5],
        }
    elif strategy == "Momentum":
        series = prices[t1]

        def factory(params):
            return signal_from_momentum(window=params["window"])
        param_grid = {"window": windows}
    else:
        series = prices[t1]

        def factory(params):
            return signal_from_mean_reversion(
                window=params["window"],
                entry=params["entry"],
                exit_=0.5,
            )
        param_grid = {
            "window": windows,
            "entry": entries,
        }

    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        st.error("Configura al menos un valor en cada parámetro del grid.")
        st.stop()

    with st.spinner("Ejecutando grid search con walk-forward..."):
        try:
            result = grid_search_walkforward(
                prices=series,
                generator_factory=factory,
                param_grid=param_grid,
                objective=objective,
                train_size=train_size,
                test_size=test_size,
            )
        except ValueError as e:
            st.error(f"Error en la optimización: {e}")
            st.stop()

    st.success(f"Grid completado: {len(result.grid)} combinaciones evaluadas.")

    # --- Mejor combinación ---
    st.subheader("🏆 Mejor combinación (según OOS)")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Parámetros óptimos:**")
        st.json(result.best_params)
    with col2:
        st.markdown("**Métricas (OOS / IS):**")
        best_df = pd.DataFrame({
            "OOS": pd.Series(result.best_oos_metrics),
            "IS": pd.Series(result.best_is_metrics),
        })
        st.dataframe(best_df.style.format("{:.4f}"), use_container_width=True)

    # --- Tabla completa del grid ---
    st.subheader("📋 Grid completo")
    display_cols = result.param_names + [
        f"oos_{objective}", f"is_{objective}",
        "oos_sharpe", "oos_max_drawdown", "oos_n_trades",
    ]
    display_cols = [c for c in display_cols if c in result.grid.columns]
    st.dataframe(
        result.grid[display_cols].sort_values(f"oos_{objective}", ascending=False),
        use_container_width=True,
    )

    # --- Heatmap 2D (si hay exactamente 2 parámetros variables) ---
    if len(result.param_names) >= 2:
        st.subheader("🔥 Heatmap de sensibilidad")
        x_param = result.param_names[0]
        y_param = result.param_names[1]
        metric_col = f"oos_{objective}"
        try:
            hm = heatmap_data(result.grid, x_param, y_param, metric_col)
            fig = px.imshow(
                hm.values,
                x=hm.columns,
                y=hm.index,
                labels={"x": x_param, "y": y_param, "color": metric_col},
                color_continuous_scale="RdYlGn",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Busca una **meseta** verde alrededor del óptimo. "
                "Un pico aislado indica overfitting."
            )
        except Exception as e:
            st.info(f"No se pudo generar el heatmap: {e}")

    # --- Descarga ---
    st.download_button(
        "⬇️ Descargar grid completo (CSV)",
        result.grid.to_csv(index=False).encode("utf-8"),
        file_name=f"optimization_{strategy}.csv",
    )

else:
    st.info(
        "Configura la estrategia y el grid de parámetros en la barra lateral "
        "y pulsa **🚀 Optimizar**."
    )
