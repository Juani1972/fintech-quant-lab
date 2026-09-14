"""Página de backtesting de estrategias cuantitativas.

Consume el motor `app.core.backtest` y permite al usuario:
    - Elegir estrategia (Pairs Trading, Momentum, Mean Reversion).
    - Configurar costes (comisión, slippage) y capital inicial.
    - Ver curva de capital, métricas, trades y comparación con buy & hold.
    - Descargar resultados en CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.backtest import (
    BacktestResult,
    buy_and_hold,
    compare_to_benchmark,
    run_backtest,
)
from app.core.cointegration import (
    engle_granger,
    generate_signals as signals_pairs,
    rolling_zscore,
)
from app.core.data_loader import load_prices


# ============================================================
#  Configuración de la página
# ============================================================
st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
st.header("🧪 Backtesting de Estrategias")


# ============================================================
#  Estrategias auxiliares
# ============================================================
def strategy_pairs_trading(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    entry: float,
    exit_: float,
) -> tuple[pd.Series, pd.Series]:
    """Estrategia de pairs trading basada en cointegración.

    Returns:
        (signals, spread) — señales sobre el spread y la serie del spread.
    """
    coint_result = engle_granger(prices[ticker_a], prices[ticker_b])
    spread = coint_result.spread
    z = rolling_zscore(spread, window=window)
    signals = signals_pairs(z, entry=entry, exit_=exit_)
    return signals, spread


def strategy_momentum(
    prices: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Momentum simple: long si el retorno de `window` días es positivo.

    Returns:
        (signals, price_series).
    """
    ret = prices.pct_change(window)
    signals = pd.Series(0, index=prices.index)
    signals[ret > 0] = 1
    signals[ret < 0] = -1
    return signals, prices


def strategy_mean_reversion(
    prices: pd.Series,
    window: int,
    entry: float,
    exit_: float,
) -> tuple[pd.Series, pd.Series]:
    """Reversión a la media: short si el z-score del precio es alto, long si bajo.

    Returns:
        (signals, price_series).
    """
    mean = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    z = (prices - mean) / std

    signals = pd.Series(0, index=prices.index)
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
    """Gráfico de curva de capital, opcionalmente con benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strategy.index, y=strategy.values,
        mode="lines", name="Estrategia",
        line=dict(color="#1f77b4", width=2),
    ))
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index, y=benchmark.values,
            mode="lines", name="Buy & Hold",
            line=dict(color="gray", width=1.5, dash="dash"),
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
        x=dd.index, y=dd.values,
        fill="tozeroy", line=dict(color="crimson"),
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
        x=positions.index, y=positions.values,
        mode="lines", line=dict(color="#2ca02c", width=1.5, shape="hv"),
        name="Posición",
    ))
    fig.update_layout(
        title="Posiciones (-1 short, 0 neutral, 1 long)",
        yaxis_title="Posición",
        template="plotly_white",
        height=250,
    )
    return fig


# ============================================================
#  Formateo de métricas
# ============================================================
def format_metrics(result: BacktestResult) -> pd.DataFrame:
    """Convierte el dict de métricas en un DataFrame formateado."""
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
#  Barra lateral: parámetros específicos del backtest
# ============================================================
tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if len(tickers) < 1:
    st.error("Introduce al menos un ticker en la barra lateral global.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🧪 Configuración del backtest")

    strategy = st.selectbox(
        "Estrategia",
        ["Pairs Trading", "Momentum", "Mean Reversion"],
    )

    if strategy == "Pairs Trading":
        if len(tickers) < 2:
            st.error("Pairs Trading requiere al menos 2 tickers.")
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
        entry = exit_ = None

    elif strategy == "Mean Reversion":
        ticker_a = st.selectbox("Ticker", tickers, index=0)
        ticker_b = None
        window = st.slider("Ventana media móvil", 10, 200, 30)
        entry = st.slider("Umbral de entrada (|z|)", 0.5, 3.0, 1.5, 0.1)
        exit_ = st.slider("Umbral de salida (|z|)", 0.0, 1.5, 0.5, 0.1)

    st.markdown("**Costes y capital**")
    initial_capital = st.number_input(
        "Capital inicial (€)", min_value=1_000, value=100_000, step=10_000
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
    # --- 1. Descarga de datos ---
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    # --- 2. Generar señales según estrategia ---
    with st.spinner(f"Generando señales ({strategy})..."):
        try:
            if strategy == "Pairs Trading":
                signals, asset_series = strategy_pairs_trading(
                    prices, ticker_a, ticker_b, window, entry, exit_
                )
                benchmark_prices = prices[ticker_a]
            elif strategy == "Momentum":
                signals, asset_series = strategy_momentum(prices[ticker_a], window)
                benchmark_prices = prices[ticker_a]
            else:  # Mean Reversion
                signals, asset_series = strategy_mean_reversion(
                    prices[ticker_a], window, entry, exit_
                )
                benchmark_prices = prices[ticker_a]
        except Exception as e:
            st.error(f"Error generando señales: {e}")
            st.stop()

    # --- 3. Alinear señales con precios del activo analizado ---
    signals = signals.reindex(asset_series.index).fillna(0)

    # --- 4. Ejecutar backtest ---
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
            st.error(f"Error en el backtest: {e}")
            st.stop()

    # ============================================================
    #  Resultados
    # ============================================================
    st.success(
        f"Backtest completado: {result.metrics['n_trades']} operaciones, "
        f"retorno total {result.metrics['total_return']:.2%}."
    )

    # --- KPIs principales ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Retorno total", f"{result.metrics['total_return']:.2%}")
    c2.metric("Retorno anual", f"{result.metrics['annual_return']:.2%}")
    c3.metric("Sharpe", f"{result.metrics['sharpe']:.2f}")
    c4.metric("Max DD", f"{result.metrics['max_drawdown']:.2%}")
    c5.metric("Win rate", f"{result.metrics['win_rate']:.1%}")

    # --- Curva de capital + benchmark ---
    st.subheader("📈 Curva de capital")
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
    st.subheader("📊 Métricas completas")
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
    st.subheader("📋 Operaciones")
    if len(result.trades) > 0:
        trades_display = result.trades.copy()
        trades_display["pnl_pct"] = trades_display["pnl_pct"].map(lambda x: f"{x:.2%}")
        trades_display["pnl_abs"] = trades_display["pnl_abs"].map(lambda x: f"{x:,.2f} €")
        st.dataframe(trades_display, use_container_width=True, hide_index=True)

        # --- Descargas ---
        st.subheader("⬇️ Descargas")
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
        st.info(
            "No se generaron operaciones. Prueba a relajar los umbrales "
            "de entrada o a ampliar el rango de fechas."
        )

else:
    st.info(
        "Configura los parámetros en la barra lateral y pulsa **🚀 Ejecutar backtest**.\n\n"
        "**Estrategias disponibles:**\n"
        "- **Pairs Trading**: cointegración + z-score del spread.\n"
        "- **Momentum**: long si el retorno de `window` días es positivo.\n"
        "- **Mean Reversion**: long/short según z-score del precio."
    )
