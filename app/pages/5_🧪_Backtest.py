"""Página de backtesting de estrategias cuantitativas."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.alerts import AlertDispatcher, check_backtest_rules
from app.core.backtest import (
    BacktestMode,
    BacktestResult,
    benchmark_metrics,
    buy_and_hold,
    run_backtest,
)
from app.core.cointegration import (
    engle_granger,
    rolling_zscore,
)
from app.core.cointegration import (
    generate_signals as signals_pairs,
)
from app.core.data_loader import load_prices
from app.core.experiments import ExperimentError, save_experiment
from app.core.history import init_db, save_run
from app.core.report import build_backtest_report
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import (
    callout,
    data_preview,
    footer,
    hero,
    named_config_manager,
    page_setup,
    section,
)

page_setup("Backtest", "🧪")

hero(
    title="Backtesting de Estrategias",
    subtitle=(
        "Motor vectorizado con anti-look-ahead, comisión, slippage, benchmark "
        "y métricas completas (Sharpe, Sortino, Calmar, Max DD, Win Rate). "
        "Guarda los resultados en el histórico y exporta un informe HTML."
    ),
    icon="🧪",
)

ensure_session_initialized()
tickers, start, end = get_global_params()
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if len(tickers) < 1:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()


# ============================================================
#  Estrategias
# ============================================================
def strategy_pairs_trading(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    entry: float,
    exit_: float,
) -> tuple[pd.Series, pd.Series]:
    """Pairs trading basado en cointegración. Opera sobre el spread."""
    coint_result = engle_granger(prices[ticker_a], prices[ticker_b])
    spread = coint_result.spread
    z = rolling_zscore(spread, window=window)
    signals = signals_pairs(z, entry=entry, exit_=exit_)
    return signals, spread


def strategy_momentum(
    prices: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Momentum simple."""
    ret = prices.pct_change(window)
    signals = pd.Series(0, index=prices.index, dtype=int)
    signals[ret > 0] = 1
    signals[ret < 0] = -1
    return signals, prices


def strategy_mean_reversion(
    prices: pd.Series,
    window: int,
    entry: float,
    exit_: float,
) -> tuple[pd.Series, pd.Series]:
    """Reversión a la media."""
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std

    signals = pd.Series(0, index=prices.index, dtype=int)
    position = 0
    for i, zi in enumerate(z):
        if np.isnan(zi):
            signals.iloc[i] = position
            continue
        if position == 0:
            if zi > entry:
                position = -1
            elif zi < -entry:
                position = 1
        elif abs(zi) < exit_:
            position = 0
        signals.iloc[i] = position
    return signals, prices


# ============================================================
#  Gráficos
# ============================================================
def plot_equity_curve(
    strategy: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Curva de capital",
) -> go.Figure:
    """Gráfico de curva de capital con benchmark opcional."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy.index, y=strategy.values, mode="lines",
        name="Estrategia", line={"color": "#2563eb", "width": 2},
    ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values, mode="lines",
            name="Buy & Hold", line={"color": "gray", "width": 1.5, "dash": "dash"},
        ))
    fig.update_layout(
        title=title, yaxis_title="Capital",
        template="plotly_white", height=450, hovermode="x unified",
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    """Gráfico de drawdown."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy",
        line={"color": "crimson"}, name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown (%)", yaxis_title="%",
        template="plotly_white", height=300,
    )
    return fig


def plot_positions(positions: pd.Series) -> go.Figure:
    """Gráfico de posiciones."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions.index, y=positions.values, mode="lines",
        line={"color": "#16a34a", "width": 1.5, "shape": "hv"},
        name="Posición",
    ))
    fig.update_layout(
        title="Posiciones (-1 short, 0 neutral, 1 long)",
        yaxis_title="Posición",
        template="plotly_white", height=250,
    )
    return fig


def format_metrics(result: BacktestResult) -> pd.DataFrame:
    """Convierte el dict de métricas en DataFrame formateado."""
    m = result.metrics
    pct_keys = {"total_return", "annual_return", "max_drawdown",
                "win_rate", "exposure"}
    rows = []
    for k, v in m.items():
        if k in pct_keys:
            rows.append((k, f"{v:.2%}"))
        elif isinstance(v, float):
            rows.append((k, f"{v:.4f}"))
        else:
            rows.append((k, str(v)))
    return pd.DataFrame(rows, columns=["Métrica", "Valor"])


# ============================================================
#  Sidebar
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🧪 Configuración del backtest")

    def _apply_bt_config(loaded_cfg: dict) -> list[str]:
        """Valida y aplica un dict de configuración al session_state de
        los widgets de esta página. Devuelve los campos ignorados
        (ticker ya no válido, valor fuera de rango...). Compartida
        entre el archivo JSON subido y las configuraciones nombradas
        de la sesión, para no duplicar la validación en dos sitios."""
        skipped = []
        if loaded_cfg.get("strategy") in ["Pairs Trading", "Momentum", "Mean Reversion"]:
            st.session_state["bt_strategy"] = loaded_cfg["strategy"]
        elif "strategy" in loaded_cfg:
            skipped.append("strategy")
        for field, key in [("ticker_a", "bt_ticker_a"), ("ticker_b", "bt_ticker_b")]:
            val = loaded_cfg.get(field)
            if val is not None:
                if val in tickers:
                    st.session_state[key] = val
                else:
                    skipped.append(f"{field} ('{val}' no está en tus tickers actuales)")
        for field, key, lo, hi in [
            ("window", "bt_window", 5, 250),
            ("entry", "bt_entry", 0.0, 3.0),
            ("exit_", "bt_exit", 0.0, 1.5),
            ("initial_capital", "bt_capital", 1_000, None),
            ("commission", "bt_commission", 0.0, 0.05),
            ("slippage", "bt_slippage", 0.0, 0.05),
        ]:
            val = loaded_cfg.get(field)
            if val is not None:
                if isinstance(val, (int, float)) and val >= lo and (hi is None or val <= hi):
                    st.session_state[key] = val
                else:
                    skipped.append(field)
        return skipped

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption(
            "Guarda los parámetros actuales como JSON para reutilizarlos "
            "luego, o carga un JSON guardado antes -- no incluye datos ni "
            "resultados, solo los valores de los controles de abajo."
        )
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_bt_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_bt_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = _apply_bt_config(loaded_cfg)
                st.session_state["_bt_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

        st.markdown("---")
        _bt_current_for_named = {
            "strategy": st.session_state.get("bt_strategy"),
            "ticker_a": st.session_state.get("bt_ticker_a"),
            "ticker_b": st.session_state.get("bt_ticker_b"),
            "window": st.session_state.get("bt_window"),
            "entry": st.session_state.get("bt_entry"),
            "exit_": st.session_state.get("bt_exit"),
            "initial_capital": st.session_state.get("bt_capital"),
            "commission": st.session_state.get("bt_commission"),
            "slippage": st.session_state.get("bt_slippage"),
        }
        _bt_loaded_named = named_config_manager(_bt_current_for_named, "bt")
        if _bt_loaded_named is not None:
            _apply_bt_config(_bt_loaded_named)
            st.rerun()

    strategy = st.selectbox(
        "Estrategia",
        ["Pairs Trading", "Momentum", "Mean Reversion"],
        key="bt_strategy",
    )

    ticker_b: str | None
    entry: float | None
    exit_: float | None

    if strategy == "Pairs Trading":
        if len(tickers) < 2:
            callout("Pairs Trading requiere al menos 2 tickers.", variant="warning")
            st.stop()
        ticker_a = st.selectbox("Ticker A (leg 1)", tickers, index=0, key="bt_ticker_a")
        ticker_b = st.selectbox("Ticker B (leg 2)", tickers, index=1, key="bt_ticker_b")
        window = st.slider("Ventana Z-score", 20, 200, 60, key="bt_window")
        entry = st.slider("Umbral de entrada (|z|)", 0.5, 3.0, 2.0, 0.1, key="bt_entry")
        exit_ = st.slider("Umbral de salida (|z|)", 0.0, 1.5, 0.5, 0.1, key="bt_exit")
    elif strategy == "Momentum":
        ticker_a = st.selectbox("Ticker", tickers, index=0, key="bt_ticker_a")
        ticker_b = None
        window = st.slider("Ventana de momentum (días)", 5, 250, 60, key="bt_window")
        entry = None
        exit_ = None
    else:  # Mean Reversion
        ticker_a = st.selectbox("Ticker", tickers, index=0, key="bt_ticker_a")
        ticker_b = None
        window = st.slider("Ventana media móvil", 10, 200, 30, key="bt_window")
        entry = st.slider("Umbral de entrada (|z|)", 0.5, 3.0, 1.5, 0.1, key="bt_entry")
        exit_ = st.slider("Umbral de salida (|z|)", 0.0, 1.5, 0.5, 0.1, key="bt_exit")

    st.markdown("**Costes y capital**")
    initial_capital = st.number_input(
        "Capital inicial (€)", min_value=1_000, value=100_000, step=10_000,
        key="bt_capital",
    )
    commission = st.number_input(
        "Comisión (fracción, ej. 0.001 = 10 bps)",
        min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f",
        help="Lo que cobra el bróker por cada operación, como fracción del importe.",
        key="bt_commission",
    )
    slippage = st.number_input(
        "Slippage (fracción, ej. 0.0005 = 5 bps)",
        min_value=0.0, max_value=0.05, value=0.0005, step=0.0005, format="%.4f",
        help=(
            "Diferencia entre el precio al que 'decides' operar y el "
            "precio al que realmente se ejecuta la orden (por retraso, "
            "liquidez insuficiente, etc.). Modela ese coste extra, "
            "aparte de la comisión, para que el backtest no sea "
            "optimista sobre lo que se conseguiría en la práctica."
        ),
        key="bt_slippage",
    )

    current_config = {
        "strategy": strategy, "ticker_a": ticker_a, "ticker_b": ticker_b,
        "window": window, "entry": entry, "exit_": exit_,
        "initial_capital": initial_capital, "commission": commission,
        "slippage": slippage,
    }
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="backtest_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Ejecutar backtest", type="primary", use_container_width=True)


# ============================================================
#  Ejecución
# ============================================================
if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    with st.spinner(f"Generando señales ({strategy})..."):
        try:
            if strategy == "Pairs Trading":
                assert ticker_b is not None and entry is not None and exit_ is not None, (
                    "ticker_b/entry/exit_ solo son None cuando strategy != 'Pairs Trading'"
                )
                signals, asset_series = strategy_pairs_trading(
                    prices, ticker_a, ticker_b, window, entry, exit_,
                )
                benchmark_prices = prices[ticker_a]
                bt_mode: BacktestMode = "absolute"
            elif strategy == "Momentum":
                signals, asset_series = strategy_momentum(prices[ticker_a], window)
                benchmark_prices = prices[ticker_a]
                bt_mode = "percent"
            else:
                assert entry is not None and exit_ is not None, (
                    "entry/exit_ solo son None cuando strategy == 'Momentum'"
                )
                signals, asset_series = strategy_mean_reversion(
                    prices[ticker_a], window, entry, exit_,
                )
                benchmark_prices = prices[ticker_a]
                bt_mode = "percent"
        except Exception as e:
            callout(f"Error generando señales: {e}", variant="danger")
            st.stop()

    signals = signals.reindex(asset_series.index).fillna(0)

    with st.spinner("Ejecutando backtest..."):
        try:
            result = run_backtest(
                prices=asset_series,
                signals=signals,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                mode=bt_mode,
            )
        except ValueError as e:
            callout(f"Error en el backtest: {e}", variant="danger")
            st.stop()

    callout(
        f"Backtest completado: <strong>{result.metrics['n_trades']} operaciones</strong>, "
        f"retorno total <strong>{result.metrics['total_return']:.2%}</strong>.",
        variant="success",
    )

    section("📊 KPIs principales")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retorno total", f"{result.metrics['total_return']:.2%}")
    c2.metric("Retorno anual", f"{result.metrics['annual_return']:.2%}")
    c3.metric("Sharpe", f"{result.metrics['sharpe']:.2f}")
    c4.metric("Max DD", f"{result.metrics['max_drawdown']:.2%}")
    c5.metric("Win rate", f"{result.metrics['win_rate']:.1%}")

    section("📈 Curva de capital")
    bh = buy_and_hold(benchmark_prices, initial_capital=initial_capital)
    bh = bh.reindex(result.equity_curve.index).ffill()
    st.plotly_chart(
        plot_equity_curve(
            result.equity_curve,
            benchmark=bh,
            title=f"{strategy} vs Buy & Hold ({ticker_a})",
        ),
        use_container_width=True,
    )

    section("🆚 Comparación contra el benchmark")
    callout(
        "El gráfico de arriba dibuja las dos curvas juntas, pero eso no dice "
        "si el exceso de retorno compensa el riesgo extra asumido -- estas "
        "métricas sí. Beta > 1 significa más sensible al mercado que el "
        "benchmark; alpha de Jensen positivo significa que bate lo que el "
        "CAPM predeciría dado ese beta.",
        variant="info",
    )
    try:
        bm = benchmark_metrics(result.equity_curve, bh)
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Beta", f"{bm['beta']:.2f}")
        bc2.metric("Alpha de Jensen (anual)", f"{bm['jensen_alpha']:.2%}")
        bc3.metric("Tracking error (anual)", f"{bm['tracking_error']:.2%}")
        bc4.metric("Information Ratio", f"{bm['information_ratio']:.2f}")
    except ValueError as e:
        callout(f"No se pudieron calcular las métricas de benchmark: {e}", variant="warning")

    col_dd, col_pos = st.columns(2)
    with col_dd:
        st.plotly_chart(plot_drawdown(result.equity_curve), use_container_width=True)
    with col_pos:
        st.plotly_chart(plot_positions(result.positions), use_container_width=True)

    section("📊 Métricas completas")
    col_m, col_p = st.columns([2, 1])
    with col_m:
        st.dataframe(
            format_metrics(result),
            use_container_width=True,
            hide_index=True,
        )
    with col_p:
        st.markdown("**Parámetros usados**")
        st.json(result.params)

    section("🔔 Alertas")
    triggered_alerts = check_backtest_rules(result.metrics)
    if triggered_alerts:
        for alert in triggered_alerts:
            callout(alert.format_text().replace("\n", "<br>"), variant=alert.severity.value)
        with st.form("dispatch_backtest_alerts_form"):
            st.caption(
                "Evaluado contra las reglas por defecto de "
                "`default_backtest_rules()` (Sharpe, Max Drawdown, nº de "
                "operaciones, win rate). Para reglas propias o configurar "
                "canales de envío, ve a la página 🔔 Alertas."
            )
            send_alerts = st.form_submit_button("📤 Enviar estas alertas por los canales configurados")
        if send_alerts:
            dispatcher = AlertDispatcher.from_env()
            n_sent = 0
            for alert in triggered_alerts:
                results = dispatcher.dispatch(alert)
                n_sent += sum(1 for ok in results.values() if ok)
            if n_sent:
                st.success(f"{n_sent} envío(s) realizado(s) (consola/archivo siempre disponibles).")
            else:
                st.info(
                    "No hay canales configurados más allá de consola/archivo "
                    "-- configúralos en 🔔 Alertas para email/Telegram/Slack."
                )
    else:
        callout(
            "Ninguna regla por defecto se ha disparado con estas métricas.",
            variant="success",
        )

    section("📋 Operaciones")
    if len(result.trades) > 0:
        trades_display = result.trades.copy()
        trades_display["pnl_pct"] = trades_display["pnl_pct"].map(lambda x: f"{x:.2%}")
        trades_display["pnl_abs"] = trades_display["pnl_abs"].map(lambda x: f"{x:,.2f} €")
        st.dataframe(trades_display, use_container_width=True, hide_index=True)

        section("⬇️ Descargas")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "Métricas (CSV)",
                format_metrics(result).to_csv(index=False).encode("utf-8"),
                file_name=f"metrics_{strategy}.csv",
            )
        with col_d2:
            st.download_button(
                "Trades (CSV)",
                result.trades.to_csv(index=False).encode("utf-8"),
                file_name=f"trades_{strategy}.csv",
            )
        with col_d3:
            equity_df = pd.DataFrame({
                "equity": result.equity_curve,
                "returns": result.returns,
                "positions": result.positions,
            })
            st.download_button(
                "Equity curve (CSV)",
                equity_df.to_csv().encode("utf-8"),
                file_name=f"equity_{strategy}.csv",
            )
    else:
        callout(
            "No se generaron operaciones. Prueba a relajar los umbrales "
            "de entrada o a ampliar el rango de fechas.",
            variant="info",
        )

    # ============================================================
    #  Informe HTML
    # ============================================================
    section("📄 Informe HTML")

    callout(
        "Descarga un informe HTML autocontenido con gráficos interactivos, "
        "métricas y tabla de operaciones. Se puede abrir en cualquier "
        "navegador y compartir por email.",
        variant="info",
    )

    try:
        report_html = build_backtest_report(
            strategy=strategy,
            tickers=tickers,
            start_date=str(start),
            end_date=str(end),
            params=result.params,
            metrics=result.metrics,
            equity_curve=result.equity_curve,
            positions=result.positions,
            trades=result.trades,
            benchmark_equity=bh,
        )
        st.download_button(
            "📄 Descargar informe HTML",
            report_html.encode("utf-8"),
            file_name=f"report_{strategy.replace(' ', '_').lower()}.html",
            mime="text/html",
            type="primary",
        )
    except Exception as e:
        callout(f"Error generando el informe: {e}", variant="warning")

    # ============================================================
    #  Guardar en histórico
    # ============================================================
    section("💾 Guardar en histórico")

    init_db()

    with st.form("save_run_form", clear_on_submit=True):
        notes = st.text_input(
            "Notas (opcional)",
            max_chars=200,
            placeholder="Ej: Prueba con ventana más corta",
        )
        submitted = st.form_submit_button(
            "💾 Guardar en histórico",
            type="primary",
            use_container_width=False,
        )

    if submitted:
        try:
            run_id = save_run(
                strategy=strategy,
                tickers=tickers,
                start_date=str(start),
                end_date=str(end),
                params={
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "window": window,
                    "entry": entry,
                    "exit_": exit_,
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "slippage": slippage,
                    "mode": bt_mode,
                },
                metrics=result.metrics,
                notes=notes or None,
            )
            callout(
                f"✅ Backtest guardado con ID <strong>#{run_id}</strong>. "
                f"Consúltalo en la página <strong>📚 Histórico</strong>.",
                variant="success",
            )
        except Exception as e:
            callout(f"Error al guardar: {e}", variant="danger")

    section("🧪 Guardar como experimento reproducible")
    callout(
        "Distinto del histórico de arriba: esto congela una copia exacta "
        "de los datos usados (no solo las métricas) junto con el commit "
        "de git actual, para poder reproducir este resultado exacto más "
        "adelante aunque los datos de mercado cambien entre tanto.",
        variant="info",
    )
    with st.form("save_experiment_form", clear_on_submit=True):
        exp_notes = st.text_input(
            "Descripción (opcional)", max_chars=200, key="exp_notes",
            placeholder="Ej: Baseline antes de ajustar comisión",
        )
        exp_submitted = st.form_submit_button("🧪 Guardar experimento")

    if exp_submitted:
        try:
            experiment = save_experiment(
                config={
                    "strategy": strategy,
                    "tickers": tickers,
                    "start_date": str(start),
                    "end_date": str(end),
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "window": window,
                    "entry": entry,
                    "exit_": exit_,
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "slippage": slippage,
                    "mode": bt_mode,
                    "notes": exp_notes or None,
                },
                data=prices,
                results=result.metrics,
            )
            callout(
                f"✅ Experimento guardado con ID <strong>{experiment.id}</strong> "
                f"(commit: {experiment.git_commit or 'sin repo git'}). "
                f"Consúltalo en <strong>🧪 Experimentos</strong>.",
                variant="success",
            )
        except ExperimentError as e:
            callout(f"No se pudo guardar el experimento: {e}", variant="danger")

else:
    callout(
        "Configura los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar backtest</strong>.<br><br>"
        "<strong>Estrategias disponibles:</strong><br>"
        "• <strong>Pairs Trading</strong>: cointegración + z-score del spread.<br>"
        "• <strong>Momentum</strong>: long si el retorno de <code>window</code> días es positivo.<br>"
        "• <strong>Mean Reversion</strong>: long/short según z-score del precio.<br><br>"
        "<strong>📄 Informe HTML:</strong> tras ejecutar, descarga un informe "
        "autocontenido con gráficos interactivos.<br>"
        "<strong>💾 Guardar en histórico:</strong> guarda la corrida para "
        "compararla después en la página 📚 Histórico.",
        variant="info",
    )

footer()
