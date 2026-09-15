"""Página de backtesting de estrategias cuantitativas."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.backtest import (
    BacktestResult,
    buy_and_hold,
    run_backtest,
)
from app.core.cointegration import (
    engle_granger,
    generate_signals as signals_pairs,
    rolling_zscore,
)
from app.core.data_loader import load_prices
from app.core.history import init_db, save_run
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Backtest", "🧪")

hero(
    title="Backtesting de Estrategias",
    subtitle=(
        "Motor vectorizado con anti-look-ahead, comisión, slippage, benchmark "
        "y métricas completas (Sharpe, Sortino, Calmar, Max DD, Win Rate). "
        "Guarda los resultados en el histórico para comparar corridas."
    ),
    icon="🧪",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

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
    """Estrategia de pairs trading basada en cointegración."""
    coint_result = engle_granger(prices[ticker_a], prices[ticker_b])
    spread = coint_result.spread
    z = rolling_zscore(spread, window=window)
    signals = signals_pairs(z, entry=entry, exit_=exit_)
    return signals, spread


def strategy_momentum(
    prices: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Momentum simple: long si el retorno de `window` días es positivo."""
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
    """Reversión a la media: short si el z-score es alto, long si bajo."""
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
        x=strategy.index,
        y=strategy.values,
        mode="lines",
        name="Estrategia",
        line={"color": "#2563eb", "width": 2},
    ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode="lines",
            name="Buy & Hold",
            line={"color": "gray", "width": 1.5, "dash": "dash"},
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Capital",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    return fig


def plot_drawdown(equity: pd.Series) -> go.Figure:
    """Gráfico de drawdown."""
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        line={"color": "crimson"},
        name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown (%)",
        yaxis_title="%",
        template="plotly_white",
        height=300,
    )
    return fig


def plot_positions(positions: pd.Series) -> go.Figure:
    """Gráfico de posiciones a lo largo del tiempo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=positions.index,
        y=positions.values,
        mode="lines",
        line={"color": "#16a34a", "width": 1.5, "shape": "hv"},
        name="Posición",
    ))
    fig.update_layout(
        title="Posiciones (-1 short, 0 neutral, 1 long)",
        yaxis_title="Posición",
        template="plotly_white",
        height=250,
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
#  Sidebar: parámetros del backtest
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🧪 Configuración del backtest")

    strategy = st.selectbox(
        "Estrategia",
        ["Pairs Trading", "Momentum", "Mean Reversion"],
    )

    if strategy == "Pairs Trading":
        if len(tickers) < 2:
            callout("Pairs Trading requiere al menos 2 tickers.", variant="warning")
            st.stop()
        ticker_a = st.selectbox("Ticker A (leg 1)", tickers, index=0)
        ticker_b = st.selectbox("Ticker B (leg 2)", tickers, index=1)
        window = st.slider("Ventana Z-score", 20, 200, 60)
        entry = st.slider("Umbral de entrada (|z|)", 0.5, 3.0, 2.0, 0.1)
        exit_ = st.slider("Umbral de salida (|z|)", 0.0, 1.5, 0.5, 0.1)
    elif strategy == "Momentum":
        ticker_a = st.selectbox("Ticker", tickers, index=0)
        ticker_b = None
        window = st.slider("Ventana de momentum (días)", 5, 250, 60)
        entry = None
        exit_ = None
    else:  # Mean Reversion
        ticker_a = st.selectbox("Ticker", tickers, index=0)
        ticker_b = None
        window = st.slider("Ventana media móvil", 10, 200, 30)
        entry = st.slider("Umbral de entrada (|z|)", 0.5, 3.0, 1.5, 0.1)
        exit_ = st.slider("Umbral de salida (|z|)", 0.0, 1.5, 0.5, 0.1)

    st.markdown("**Costes y capital**")
    initial_capital = st.number_input(
        "Capital inicial (€)",
        min_value=1_000, value=100_000, step=10_000,
    )
    commission = st.number_input(
        "Comisión (fracción, ej. 0.001 = 10 bps)",
        min_value=0.0, max_value=0.05, value=0.001, step=0.0005, format="%.4f",
    )
    slippage = st.number_input(
        "Slippage (fracción, ej. 0.0005 = 5 bps)",
        min_value=0.0, max_value=0.05, value=0.0005, step=0.0005, format="%.4f",
    )

    run = st.button("🚀 Ejecutar backtest", type="primary", use_container_width=True)


# ============================================================
#  Ejecución del backtest
# ============================================================
if run:
    # --- 1. Descargar datos ---
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    # --- 2. Generar señales ---
    with st.spinner(f"Generando señales ({strategy})..."):
        try:
            if strategy == "Pairs Trading":
                signals, asset_series = strategy_pairs_trading(
                    prices, ticker_a, ticker_b, window, entry, exit_,
                )
                benchmark_prices = prices[ticker_a]
            elif strategy == "Momentum":
                signals, asset_series = strategy_momentum(prices[ticker_a], window)
                benchmark_prices = prices[ticker_a]
            else:
                signals, asset_series = strategy_mean_reversion(
                    prices[ticker_a], window, entry, exit_,
                )
                benchmark_prices = prices[ticker_a]
        except Exception as e:
            callout(f"Error generando señales: {e}", variant="danger")
            st.stop()

    signals = signals.reindex(asset_series.index).fillna(0)

    # --- 3. Ejecutar backtest ---
    with st.spinner("Ejecutando backtest..."):
        try:
            result = run_backtest(
                prices=asset_series,
                signals=signals,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
            )
        except ValueError as e:
            callout(f"Error en el backtest: {e}", variant="danger")
            st.stop()

    callout(
        f"Backtest completado: <strong>{result.metrics['n_trades']} operaciones</strong>, "
        f"retorno total <strong>{result.metrics['total_return']:.2%}</strong>.",
        variant="success",
    )

    # --- KPIs ---
    section("📊 KPIs principales")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retorno total", f"{result.metrics['total_return']:.2%}")
    c2.metric("Retorno anual", f"{result.metrics['annual_return']:.2%}")
    c3.metric("Sharpe", f"{result.metrics['sharpe']:.2f}")
    c4.metric("Max DD", f"{result.metrics['max_drawdown']:.2%}")
    c5.metric("Win rate", f"{result.metrics['win_rate']:.1%}")

    # --- Curva de capital ---
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

    # --- Drawdown y posiciones ---
    col_dd, col_pos = st.columns(2)
    with col_dd:
        st.plotly_chart(plot_drawdown(result.equity_curve), use_container_width=True)
    with col_pos:
        st.plotly_chart(plot_positions(result.positions), use_container_width=True)

    # --- Métricas completas ---
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

    # --- Tabla de trades ---
    section("📋 Operaciones")
    if len(result.trades) > 0:
        trades_display = result.trades.copy()
        trades_display["pnl_pct"] = trades_display["pnl_pct"].map(lambda x: f"{x:.2%}")
        trades_display["pnl_abs"] = trades_display["pnl_abs"].map(lambda x: f"{x:,.2f} €")
        st.dataframe(trades_display, use_container_width=True, hide_index=True)

        # --- Descargas ---
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

else:
    callout(
        "Configura los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar backtest</strong>.<br><br>"
        "<strong>Estrategias disponibles:</strong><br>"
        "• <strong>Pairs Trading</strong>: cointegración + z-score del spread.<br>"
        "• <strong>Momentum</strong>: long si el retorno de <code>window</code> días es positivo.<br>"
        "• <strong>Mean Reversion</strong>: long/short según z-score del precio.<br><br>"
        "<strong>💾 Guardar en histórico:</strong> tras ejecutar, guarda la corrida "
        "para compararla después en la página 📚 Histórico.",
        variant="info",
    )

footer()
