"""Página de walk-forward analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.cointegration import engle_granger
from app.core.data_loader import load_prices
from app.core.walkforward import (
    signal_from_mean_reversion,
    signal_from_momentum,
    signal_from_pairs_trading,
    walk_forward_analysis,
)

st.set_page_config(page_title="Walk-Forward", page_icon="🔬", layout="wide")
st.header("🔬 Walk-Forward Analysis")
st.caption(
    "Divide los datos en ventanas sucesivas de entrenamiento (IS) y prueba (OOS). "
    "Sirve para detectar overfitting: si OOS se degrada mucho respecto a IS, "
    "la estrategia no generaliza."
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
    st.subheader("🎛️ Configuración Walk-Forward")

    strategy = st.selectbox(
        "Estrategia",
        ["Pairs Trading (spread)", "Momentum", "Mean Reversion"],
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

    train_size = st.slider("Train size (barras)", 100, 1000, 504, 21)
    test_size = st.slider("Test size (barras)", 21, 500, 126, 21)
    step = st.slider("Step entre ventanas", 21, 250, test_size, 21)

    st.markdown("**Costes**")
    initial_capital = st.number_input("Capital inicial (€)", 1_000, value=100_000, step=10_000)
    commission = st.number_input("Comisión", 0.0, 0.05, 0.001, 0.0005, format="%.4f")
    slippage = st.number_input("Slippage", 0.0, 0.05, 0.0005, 0.0005, format="%.4f")

    run = st.button("🚀 Ejecutar walk-forward", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    # --- Preparar la serie sobre la que se opera ---
    if strategy == "Pairs Trading (spread)":
        try:
            coint = engle_granger(prices[t1], prices[t2])
            series = coint.spread
        except Exception as e:
            st.error(f"Error calculando cointegración: {e}")
            st.stop()
        generator = signal_from_pairs_trading(window=60, entry=2.0, exit_=0.5)
        series_name = f"Spread {t1} - α - β·{t2}"
    elif strategy == "Momentum":
        series = prices[t1]
        generator = signal_from_momentum(window=60)
        series_name = t1
    else:
        series = prices[t1]
        generator = signal_from_mean_reversion(window=30, entry=1.5, exit_=0.5)
        series_name = t1

    with st.spinner("Ejecutando walk-forward..."):
        try:
            result = walk_forward_analysis(
                prices=series,
                signal_generator=generator,
                train_size=train_size,
                test_size=test_size,
                step=step,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
            )
        except ValueError as e:
            st.error(f"Error en el walk-forward: {e}")
            st.stop()

    st.success(f"Walk-forward completado: {result.params['n_windows']} ventanas.")

    # --- Comparación IS vs OOS ---
    st.subheader("📊 Comparación In-Sample vs Out-of-Sample")
    is_m = result.is_metrics_agg
    oos_m = result.oos_metrics_agg

    comparison = pd.DataFrame({
        "In-Sample": [
            f"{is_m.get('total_return', np.nan):.2%}",
            f"{is_m.get('annual_return', np.nan):.2%}",
            f"{is_m.get('sharpe', np.nan):.2f}",
            f"{is_m.get('sortino', np.nan):.2f}",
            f"{is_m.get('max_drawdown', np.nan):.2%}",
            f"{is_m.get('win_rate', np.nan):.1%}",
        ],
        "Out-of-Sample": [
            f"{oos_m.get('total_return', np.nan):.2%}",
            f"{oos_m.get('annual_return', np.nan):.2%}",
            f"{oos_m.get('sharpe', np.nan):.2f}",
            f"{oos_m.get('sortino', np.nan):.2f}",
            f"{oos_m.get('max_drawdown', np.nan):.2%}",
            f"{oos_m.get('win_rate', np.nan):.1%}",
        ],
    }, index=["Retorno total", "Retorno anual", "Sharpe", "Sortino", "Max DD", "Win rate"])

    st.dataframe(comparison, use_container_width=True)

    # --- Alerta de degradación ---
    is_sharpe = is_m.get("sharpe", np.nan)
    oos_sharpe = oos_m.get("sharpe", np.nan)
    if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) and is_sharpe != 0:
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        if degradation > 0.5:
            st.error(
                f"⚠️ Degradación severa: Sharpe cae {degradation:.0%} de IS a OOS. "
                "Alta sospecha de overfitting."
            )
        elif degradation > 0.25:
            st.warning(f"Degradación moderada del Sharpe: {degradation:.0%}.")
        else:
            st.success(f"Degradación aceptable: {degradation:.0%}.")

    # --- Curva OOS concatenada ---
    st.subheader("📈 Curva de capital Out-of-Sample (compuesta)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.oos_equity_concat.index,
        y=result.oos_equity_concat.values,
        mode="lines", name="OOS",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.update_layout(
        yaxis_title="Capital", template="plotly_white", height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Detalle por ventana ---
    st.subheader("🔍 Detalle por ventana")
    rows = []
    for i, w in enumerate(result.windows):
        rows.append({
            "Ventana": i + 1,
            "Train start": w.train_start.date(),
            "Train end": w.train_end.date(),
            "Test start": w.test_start.date(),
            "Test end": w.test_end.date(),
            "IS Sharpe": f"{w.is_metrics.get('sharpe', np.nan):.2f}",
            "OOS Sharpe": f"{w.oos_metrics.get('sharpe', np.nan):.2f}",
            "IS Return": f"{w.is_metrics.get('total_return', np.nan):.2%}",
            "OOS Return": f"{w.oos_metrics.get('total_return', np.nan):.2%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- Descargas ---
    st.download_button(
        "⬇️ Descargar métricas IS/OOS (CSV)",
        comparison.to_csv().encode("utf-8"),
        file_name="walkforward_comparison.csv",
    )

else:
    st.info(
        "Configura los parámetros en la barra lateral y pulsa "
        "**🚀 Ejecutar walk-forward**.\n\n"
        "**Recomendaciones:**\n"
        "- `train_size` >= 2 años (504 barras diarias).\n"
        "- `test_size` ~ 6 meses (126 barras).\n"
        "- Step = test_size → ventanas sin solapamiento."
    )
