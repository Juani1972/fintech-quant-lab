"""Página de exploración de estrategias (app/core/strategies)."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import THEME
from app.core.data_loader import load_prices
from app.core.strategies import StrategyError, get_strategy, list_strategies
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section, ticker_badge

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
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

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

    STRATEGY_PARAM_RANGES: dict[str, dict[str, tuple[float, float]]] = {
        "trend_following": {"fast_window": (5, 60), "slow_window": (20, 250)},
        "volatility_targeting": {
            "target_vol": (0.05, 0.60), "lookback": (5, 120), "max_leverage": (1.0, 5.0),
        },
        "pca_statarb": {"lookback": (10, 250), "entry_z": (0.5, 4.0), "exit_z": (0.0, 2.0)},
        "risk_parity": {"lookback": (5, 250)},
        "cross_sectional_momentum": {"lookback": (20, 400), "top_n": (1, 10)},
        "carry_trade": {"deadband": (0.0, 0.05)},
    }

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_es_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_es_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                loaded_strategy = loaded_cfg.get("strategy_name")
                if loaded_strategy in STRATEGY_PARAM_RANGES:
                    st.session_state["es_strategy_name"] = loaded_strategy
                elif "strategy_name" in loaded_cfg:
                    skipped.append("strategy_name")

                effective_strategy = loaded_strategy or st.session_state.get("es_strategy_name")
                ranges: dict = {}
                if isinstance(effective_strategy, str) and effective_strategy in STRATEGY_PARAM_RANGES:
                    ranges = STRATEGY_PARAM_RANGES[effective_strategy]
                loaded_params = loaded_cfg.get("params", {})
                for field, val in loaded_params.items():
                    if field not in ranges:
                        continue  # no aplicable a esta estrategia
                    lo, hi = ranges[field]
                    if val is not None and isinstance(val, (int, float)) and lo <= val <= hi:
                        st.session_state[f"es_{field}"] = val
                    elif val is not None:
                        skipped.append(field)

                ticker_val = loaded_cfg.get("single_ticker")
                if ticker_val is not None:
                    if ticker_val in tickers:
                        st.session_state["es_single_ticker"] = ticker_val
                    else:
                        skipped.append(f"single_ticker ('{ticker_val}' no está en tus tickers actuales)")

                st.session_state["_es_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    available = list_strategies()
    strategy_name = st.selectbox(
        "Estrategia", available, format_func=lambda n: STRATEGY_LABELS.get(n, n),
        key="es_strategy_name",
    )

    params: dict = {}
    single_ticker = None
    if strategy_name == "trend_following":
        params["fast_window"] = st.slider(
            "Ventana rápida", 5, 60, 20,
            help="Media móvil corta. Cruza por encima de la lenta = señal larga; por debajo = corta.",
            key="es_fast_window",
        )
        params["slow_window"] = st.slider("Ventana lenta", 20, 250, 100, key="es_slow_window")
    elif strategy_name == "volatility_targeting":
        params["target_vol"] = st.slider(
            "Vol. objetivo (anual)", 0.05, 0.60, 0.15, 0.01,
            help=(
                "Volatilidad anualizada que se busca mantener en la "
                "cartera. La estrategia sube el apalancamiento cuando la "
                "volatilidad realizada está por debajo de este objetivo, "
                "y lo baja cuando está por encima."
            ),
            key="es_target_vol",
        )
        params["lookback"] = st.slider("Ventana de vol. realizada", 5, 120, 20, key="es_lookback")
        params["max_leverage"] = st.slider(
            "Apalancamiento máximo", 1.0, 5.0, 3.0, 0.5,
            help="Límite superior al apalancamiento, aunque el objetivo de volatilidad pida más.",
            key="es_max_leverage",
        )
        single_ticker = st.selectbox("Ticker (activo único)", tickers, key="es_single_ticker")
    elif strategy_name == "pca_statarb":
        params["lookback"] = st.slider(
            "Ventana", 10, 250, 60,
            help="Días usados para estimar los componentes principales y la media/desviación del spread resultante.",
            key="es_lookback",
        )
        params["entry_z"] = st.slider("Z-score entrada", 0.5, 4.0, 2.0, 0.1, key="es_entry_z")
        params["exit_z"] = st.slider("Z-score salida", 0.0, 2.0, 0.5, 0.1, key="es_exit_z")
    elif strategy_name == "risk_parity":
        params["lookback"] = st.slider(
            "Ventana de volatilidad", 5, 250, 60,
            help="Días usados para estimar la volatilidad de cada activo -- pesos inversamente proporcionales a ella.",
            key="es_lookback",
        )
    elif strategy_name == "cross_sectional_momentum":
        params["lookback"] = st.slider(
            "Ventana de momentum", 20, 400, 252,
            help="Días de retorno pasado usados para rankear los activos entre sí (252 ≈ 1 año).",
            key="es_lookback",
        )
        top_n_enabled = st.checkbox("Limitar a top/bottom N", value=False, key="es_top_n_enabled")
        params["top_n"] = st.slider(
            "N", 1, 10, 3,
            help="Solo los N activos con mejor y peor momentum entran en la cartera; el resto queda a peso 0.",
            key="es_top_n",
        ) if top_n_enabled else None
    elif strategy_name == "carry_trade":
        params["deadband"] = st.slider(
            "Banda muerta", 0.0, 0.05, 0.0, 0.005,
            help="Diferencial de carry mínimo para generar señal -- por debajo de esto, se considera ruido y no se opera.",
            key="es_deadband",
        )

    current_config = {
        "strategy_name": strategy_name, "params": params, "single_ticker": single_ticker,
    }
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="estrategias_config.json",
        mime="application/json",
    )

    run_clicked = st.button("🚀 Generar señales", type="primary")

if single_ticker:
    ticker_badge(single_ticker)
else:
    ticker_badge(*tickers)

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
    prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
except (ValueError, ConnectionError) as e:
    callout(f"Error al cargar datos: {e}", variant="danger")
    st.stop()
data_preview(prices)

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
