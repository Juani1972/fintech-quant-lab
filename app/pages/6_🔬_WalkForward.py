"""Página de walk-forward analysis."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.alerts import AlertDispatcher, check_backtest_rules, default_walkforward_rules
from app.core.backtest import BacktestMode
from app.core.cointegration import engle_granger
from app.core.data_loader import load_prices
from app.core.walkforward import (
    signal_from_mean_reversion,
    signal_from_momentum,
    signal_from_pairs_trading,
    walk_forward_analysis,
)
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Walk-Forward", "🔬")

hero(
    title="Walk-Forward Analysis",
    subtitle=(
        "Divide los datos en ventanas sucesivas de entrenamiento (IS) y prueba (OOS). "
        "Sirve para detectar overfitting: si OOS se degrada mucho respecto a IS, "
        "la estrategia no generaliza."
    ),
    icon="🔬",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if len(tickers) < 1:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Configuración Walk-Forward")

    strategy = st.selectbox(
        "Estrategia",
        ["Pairs Trading (spread)", "Momentum", "Mean Reversion"],
    )

    if strategy == "Pairs Trading (spread)":
        if len(tickers) < 2:
            callout("Pairs Trading requiere al menos 2 tickers.", variant="warning")
            st.stop()
        t1 = st.selectbox("Ticker 1", tickers, index=0)
        t2: str | None = st.selectbox("Ticker 2", tickers, index=1)
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
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    if strategy == "Pairs Trading (spread)":
        try:
            coint = engle_granger(prices[t1], prices[t2])
            series = coint.spread
        except Exception as e:
            callout(f"Error calculando cointegración: {e}", variant="danger")
            st.stop()
        generator = signal_from_pairs_trading(window=60, entry=2.0, exit_=0.5)
        wf_mode: BacktestMode = "absolute"
    elif strategy == "Momentum":
        series = prices[t1]
        generator = signal_from_momentum(window=60)
        wf_mode = "percent"
    else:
        series = prices[t1]
        generator = signal_from_mean_reversion(window=30, entry=1.5, exit_=0.5)
        wf_mode = "percent"

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
                mode=wf_mode,
            )
        except ValueError as e:
            callout(f"Error en el walk-forward: {e}", variant="danger")
            st.stop()

    callout(
        f"Walk-forward completado: <strong>{result.params['n_windows']} ventanas</strong>.",
        variant="success",
    )

    section("📊 Comparación In-Sample vs Out-of-Sample")
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

    is_sharpe = is_m.get("sharpe", np.nan)
    oos_sharpe = oos_m.get("sharpe", np.nan)
    if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) and is_sharpe != 0:
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        if degradation > 0.5:
            callout(
                f"⚠️ Degradación severa: Sharpe cae {degradation:.0%} de IS a OOS. "
                "Alta sospecha de overfitting.",
                variant="danger",
            )
        elif degradation > 0.25:
            callout(f"Degradación moderada del Sharpe: {degradation:.0%}.",
                    variant="warning")
        else:
            callout(f"Degradación aceptable: {degradation:.0%}.", variant="success")

    section("🔔 Alertas")
    triggered_alerts = check_backtest_rules(oos_m, rules=default_walkforward_rules())
    if triggered_alerts:
        for alert in triggered_alerts:
            callout(alert.format_text().replace("\n", "<br>"), variant=alert.severity.value)
        with st.form("dispatch_walkforward_alerts_form"):
            st.caption(
                "Evaluado contra `default_walkforward_rules()` (Sharpe OOS) "
                "sobre las métricas agregadas out-of-sample. Para reglas "
                "propias o configurar canales de envío, ve a 🔔 Alertas."
            )
            send_alerts = st.form_submit_button("📤 Enviar estas alertas por los canales configurados")
        if send_alerts:
            dispatcher = AlertDispatcher.from_env()
            n_sent = sum(
                1
                for alert in triggered_alerts
                for ok in dispatcher.dispatch(alert).values()
                if ok
            )
            if n_sent:
                st.success(f"{n_sent} envío(s) realizado(s) (consola/archivo siempre disponibles).")
            else:
                st.info(
                    "No hay canales configurados más allá de consola/archivo "
                    "-- configúralos en 🔔 Alertas para email/Telegram/Slack."
                )
    else:
        callout(
            "Ninguna regla por defecto de walk-forward se ha disparado.",
            variant="success",
        )

    section("📈 Curva de capital Out-of-Sample (compuesta)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.oos_equity_concat.index,
        y=result.oos_equity_concat.values,
        mode="lines", name="OOS",
        line={"color": "#2563eb", "width": 2},
    ))
    fig.update_layout(
        yaxis_title="Capital", template="plotly_white", height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("🔍 Detalle por ventana")
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

    st.download_button(
        "⬇️ Descargar métricas IS/OOS (CSV)",
        comparison.to_csv().encode("utf-8"),
        file_name="walkforward_comparison.csv",
    )

else:
    callout(
        "Configura los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar walk-forward</strong>.<br><br>"
        "<strong>Recomendaciones:</strong><br>"
        "• <code>train_size</code> ≥ 2 años (504 barras diarias).<br>"
        "• <code>test_size</code> ≈ 6 meses (126 barras).<br>"
        "• Step = test_size → ventanas sin solapamiento.",
        variant="info",
    )

footer()
