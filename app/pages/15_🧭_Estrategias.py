"""Página de exploración de estrategias (app/core/strategies)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import THEME
from app.core.data_loader import load_prices
from app.core.strategies import StrategyError, get_strategy, list_strategies
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Estrategias", "🧭")

hero(
    title="Explorador de Estrategias",
    subtitle=(
        "Genera y visualiza las señales de las 6 estrategias del registro "
        "app/core/strategies -- Trend Following, Volatility Targeting, "
        "PCA StatArb, Risk Parity, Cross-Sectional Momentum y Carry Trade."
    ),
    icon="🧭",
)

callout(
    "Esta página muestra la **señal generada** por cada estrategia (para "
    "qué apuesta, y cuándo), no un backtest de P&L completo -- cada una "
    "tiene una forma de señal distinta (por activo, por fecha, o ambas a "
    "la vez), así que no encajan todas en el mismo motor de backtest de "
    "una sola serie de precios que usa la página 🧪 Backtest. Conectarlas "
    "a un backtest multi-activo es la siguiente ampliación natural.",
    variant="info",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if not tickers:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

STRATEGY_LABELS = {
    "trend_following": "Trend Following (cruce de medias)",
    "volatility_targeting": "Volatility Targeting",
    "pca_statarb": "PCA StatArb",
    "risk_parity": "Risk Parity (inverse-vol)",
    "cross_sectional_momentum": "Cross-Sectional Momentum",
    "carry_trade": "Carry Trade",
}
MIN_TICKERS = {
    "trend_following": 1,
    "volatility_targeting": 1,
    "pca_statarb": 2,
    "risk_parity": 2,
    "cross_sectional_momentum": 2,
    "carry_trade": 1,
}

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🧭 Configuración de estrategia")

    available = list_strategies()
    strategy_name = st.selectbox(
        "Estrategia", available, format_func=lambda n: STRATEGY_LABELS.get(n, n),
    )

    params: dict = {}
    if strategy_name == "trend_following":
        params["fast_window"] = st.slider("Ventana rápida", 5, 60, 20)
        params["slow_window"] = st.slider("Ventana lenta", 20, 250, 100)
    elif strategy_name == "volatility_targeting":
        params["target_vol"] = st.slider("Vol. objetivo (anual)", 0.05, 0.60, 0.15, 0.01)
        params["lookback"] = st.slider("Ventana de vol. realizada", 5, 120, 20)
        params["max_leverage"] = st.slider("Apalancamiento máximo", 1.0, 5.0, 3.0, 0.5)
        single_ticker = st.selectbox("Ticker (activo único)", tickers)
    elif strategy_name == "pca_statarb":
        params["lookback"] = st.slider("Ventana", 10, 250, 60)
        params["entry_z"] = st.slider("Z-score entrada", 0.5, 4.0, 2.0, 0.1)
        params["exit_z"] = st.slider("Z-score salida", 0.0, 2.0, 0.5, 0.1)
    elif strategy_name == "risk_parity":
        params["lookback"] = st.slider("Ventana de volatilidad", 5, 250, 60)
    elif strategy_name == "cross_sectional_momentum":
        params["lookback"] = st.slider("Ventana de momentum", 20, 400, 252)
        top_n_enabled = st.checkbox("Limitar a top/bottom N", value=False)
        params["top_n"] = st.slider("N", 1, 10, 3) if top_n_enabled else None
    elif strategy_name == "carry_trade":
        params["deadband"] = st.slider("Banda muerta", 0.0, 0.05, 0.0, 0.005)

    run_clicked = st.button("🚀 Generar señales", type="primary")

if strategy_name == "carry_trade":
    callout(
        "Carry Trade necesita una columna `carry_yield` (diferencial de "
        "tipos, dividend yield...) calculada externamente -- no forma "
        "parte de los datos de precios que descarga la app. Esta "
        "estrategia no se puede ejecutar desde aquí todavía; su código "
        "está en `app/core/strategies/carry_trade.py` y ya tiene tests "
        "propios.",
        variant="warning",
    )
    footer()
    st.stop()

n_needed = MIN_TICKERS[strategy_name]
if len(tickers) < n_needed:
    callout(
        f"{STRATEGY_LABELS[strategy_name]} necesita al menos {n_needed} "
        f"ticker(s); tienes {len(tickers)}.",
        variant="warning",
    )
    st.stop()

if not run_clicked:
    callout("Configura los parámetros en la barra lateral y pulsa 'Generar señales'.", variant="info")
    st.stop()

try:
    prices = load_prices(tickers, start, end)
except (ValueError, ConnectionError) as e:
    callout(f"Error al cargar datos: {e}", variant="danger")
    st.stop()

price_input = prices[[single_ticker]] if strategy_name == "volatility_targeting" else prices

try:
    strat = get_strategy(strategy_name, **params)
    signals = strat.generate_signals(price_input)
except StrategyError as e:
    callout(f"No se pudo generar la señal: {e}", variant="danger")
    st.stop()

section("📡 Señal generada")

if isinstance(signals.index, pd.MultiIndex):
    # Multi-activo en el tiempo (trend_following): una línea por ticker.
    wide = signals.unstack(level=-1)
    fig = go.Figure()
    for col in wide.columns:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[col], mode="lines", name=str(col)))
    fig.update_layout(
        title="Señal por activo a lo largo del tiempo (-1 corto, 0 plano, 1 largo)",
        template="plotly_white", height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

elif pd.api.types.is_datetime64_any_dtype(signals.index):
    # Serie temporal sobre un único activo/spread (volatility_targeting, pca_statarb).
    fig = go.Figure(go.Scatter(
        x=signals.index, y=signals.values, mode="lines",
        line={"color": THEME["primary"], "width": 1.4},
    ))
    fig.update_layout(title="Señal a lo largo del tiempo", template="plotly_white", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Último valor: {signals.iloc[-1]:.3f}  ·  {len(signals)} observaciones")

else:
    # Snapshot cross-sectional (risk_parity, cross_sectional_momentum): un peso por ticker.
    colors = [THEME["danger"] if v < 0 else THEME["primary"] for v in signals.values]
    fig = go.Figure(go.Bar(
        x=signals.index, y=signals.values, marker_color=colors,
        text=[f"{v:.1%}" for v in signals.values], textposition="outside",
    ))
    fig.update_layout(
        title="Peso/señal por activo (última fecha disponible)",
        yaxis_tickformat=".0%", template="plotly_white", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(signals.rename("Peso").apply(lambda w: f"{w:.2%}"), use_container_width=True)

with st.expander("Ver datos en bruto"):
    st.dataframe(signals.rename("signal"), use_container_width=True)

footer()
