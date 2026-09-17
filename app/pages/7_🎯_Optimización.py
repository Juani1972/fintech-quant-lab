"""Página de optimización de parámetros con walk-forward."""
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core.backtest import BacktestMode, run_backtest
from app.core.cointegration import engle_granger
from app.core.data_loader import load_prices
from app.core.optimization import (
    OptimizationResult,
    deflated_sharpe_ratio,
    grid_search_walkforward,
    heatmap_data,
)
from app.core.walkforward import (
    signal_from_mean_reversion,
    signal_from_momentum,
    signal_from_pairs_trading,
)
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section

page_setup("Optimización", "🎯")

hero(
    title="Optimización de Parámetros",
    subtitle=(
        "Grid search evaluado con walk-forward (OOS). Este método evita el "
        "overfitting del grid search clásico porque puntúa cada combinación "
        "por su rendimiento out-of-sample."
    ),
    icon="🎯",
)

ensure_session_initialized()
tickers, start, end = get_global_params()
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if len(tickers) < 1:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Configuración")
    strategy = st.selectbox(
        "Estrategia",
        ["Momentum", "Mean Reversion", "Pairs Trading (spread)"],
    )

    t2: str | None

    if strategy == "Pairs Trading (spread)":
        if len(tickers) < 2:
            callout("Pairs Trading requiere al menos 2 tickers.", variant="warning")
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
            default=[20, 45, 60],
        )
        entries = None
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
        "Objetivo OOS", ["sharpe", "sortino", "calmar", "total_return"], index=0,
        help=(
            "Métrica out-of-sample que decide qué combinación de "
            "parámetros es 'la mejor' -- el grid search elige la que "
            "puntúe más alto en ESTA métrica, no en las demás. Sharpe es "
            "el estándar; Sortino penaliza solo la volatilidad a la baja; "
            "Calmar prioriza el ratio retorno/drawdown."
        ),
    )

    train_size = st.slider(
        "Train size", 200, 1000, 504, 21,
        help="Días usados para ajustar cada combinación del grid, antes de evaluarla out-of-sample.",
    )
    test_size = st.slider(
        "Test size", 21, 250, 126, 21,
        help="Días usados para evaluar cada combinación -- esto es lo que de verdad puntúa cada casilla del grid.",
    )

    run = st.button("🚀 Optimizar", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    if strategy == "Pairs Trading (spread)":
        assert entries is not None, "entries solo es None cuando strategy == 'Momentum'"
        try:
            coint = engle_granger(prices[t1], prices[t2])
            series = coint.spread
        except Exception as e:
            callout(f"Error calculando cointegración: {e}", variant="danger")
            st.stop()

        def factory(params):
            return signal_from_pairs_trading(
                window=params["window"],
                entry=params["entry"],
                exit_=0.5,
            )
        param_grid: dict[str, list] = {"window": windows, "entry": entries}
        opt_mode: BacktestMode = "absolute"
    elif strategy == "Momentum":
        series = prices[t1]

        def factory(params):
            return signal_from_momentum(window=params["window"])
        param_grid = {"window": windows}
        opt_mode = "percent"
    else:
        assert entries is not None, "entries solo es None cuando strategy == 'Momentum'"
        series = prices[t1]

        def factory(params):
            return signal_from_mean_reversion(
                window=params["window"],
                entry=params["entry"],
                exit_=0.5,
            )
        param_grid = {"window": windows, "entry": entries}
        opt_mode = "percent"

    if not param_grid or any(len(v) == 0 for v in param_grid.values()):
        callout("Configura al menos un valor en cada parámetro del grid.",
                variant="warning")
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
                mode=opt_mode,
            )
        except ValueError as e:
            callout(f"Error en la optimización: {e}", variant="danger")
            st.stop()

    callout(
        f"Grid completado: <strong>{len(result.grid)} combinaciones</strong> evaluadas.",
        variant="success",
    )

    section("🏆 Mejor combinación (según OOS)")
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

    if objective == "sharpe":
        section("🎲 Deflated Sharpe Ratio")
        callout(
            "Cuantas más combinaciones se prueban en el grid, más probable "
            "es que el Sharpe de la ganadora sea alto por puro azar de la "
            "selección múltiple, no porque tenga una ventaja real. El DSR "
            "corrige por esto: > 0.95 sugiere que el resultado probablemente "
            "no es solo el máximo esperable por azar entre tantos intentos.",
            variant="info",
        )
        try:
            winning_generator = factory(result.best_params)
            winning_signal = winning_generator(series, series)
            winning_backtest = run_backtest(series, winning_signal, mode=opt_mode)
            dsr_input = OptimizationResult(
                grid=result.grid.rename(columns={f"oos_{objective}": objective}),
                best_params=result.best_params,
                best_metrics=result.best_oos_metrics,
                objective=objective,
                param_names=result.param_names,
            )
            dsr = deflated_sharpe_ratio(dsr_input, winning_backtest.returns)
        except ValueError as e:
            callout(f"No se pudo calcular el DSR: {e}", variant="warning")
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Sharpe observado (anual)", f"{dsr['sr_observed']:.2f}")
            d2.metric("Sharpe máx. esperado por azar", f"{dsr['sr_expected_max']:.2f}")
            d3.metric("Nº combinaciones", f"{int(dsr['n_trials'])}")
            d4.metric("DSR", f"{dsr['dsr']:.2%}")
            if dsr["dsr"] < 0.95:
                callout(
                    f"DSR = {dsr['dsr']:.1%}, por debajo del umbral habitual "
                    "de 95% -- el Sharpe ganador podría deberse en buena "
                    "parte a haber probado muchas combinaciones, no a una "
                    "ventaja real. Trátalo con cautela.",
                    variant="warning",
                )

    section("📋 Grid completo")
    display_cols = result.param_names + [
        f"oos_{objective}", f"is_{objective}",
        "oos_sharpe", "oos_max_drawdown", "oos_n_trades",
    ]
    display_cols = [c for c in display_cols if c in result.grid.columns]
    st.dataframe(
        result.grid[display_cols].sort_values(f"oos_{objective}", ascending=False),
        use_container_width=True,
    )

    if len(result.param_names) >= 2:
        section("🔥 Heatmap de sensibilidad")
        x_param = result.param_names[0]
        y_param = result.param_names[1]
        metric_col = f"oos_{objective}"
        try:
            hm = heatmap_data(result.grid, x_param, y_param, metric_col)
            fig = px.imshow(
                hm.values,
                x=list(hm.columns),
                y=list(hm.index),
                labels={"x": x_param, "y": y_param, "color": metric_col},
                color_continuous_scale="RdYlGn",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)
            callout(
                "Busca una <strong>meseta</strong> verde alrededor del óptimo. "
                "Un pico aislado indica overfitting.",
                variant="info",
            )
        except Exception as e:
            callout(f"No se pudo generar el heatmap: {e}", variant="info")

    st.download_button(
        "⬇️ Descargar grid completo (CSV)",
        result.grid.to_csv(index=False).encode("utf-8"),
        file_name=f"optimization_{strategy}.csv",
    )

else:
    callout(
        "Configura la estrategia y el grid de parámetros en la barra lateral "
        "y pulsa <strong>🚀 Optimizar</strong>.",
        variant="info",
    )

footer()
