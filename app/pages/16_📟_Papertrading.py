"""Página de paper trading en vivo contra Alpaca."""
import os
from datetime import date, timedelta
from typing import Literal, cast

import numpy as np
import pandas as pd
import streamlit as st

from app.core.data_loader import load_prices
from app.core.papertrade import PaperAccount, PaperTradeError, StrategyRunner
from app.core.strategies import Strategy, StrategyError, get_strategy
from app.styles import callout, footer, hero, kpi_row, page_setup, section

page_setup("Papertrading", "📟")

hero(
    title="Paper Trading (Alpaca)",
    subtitle=(
        "Envía órdenes reales a un broker, pero sin arriesgar capital -- "
        "contra la cuenta de simulación (paper) de Alpaca."
    ),
    icon="📟",
)

API_KEY = os.environ.get("FQL_ALPACA_KEY_ID", "")
API_SECRET = os.environ.get("FQL_ALPACA_SECRET_KEY", "")

if not API_KEY or not API_SECRET:
    callout(
        "No hay credenciales de Alpaca configuradas -- esta página no puede "
        "conectar con ninguna cuenta todavía.",
        variant="warning",
    )
    with st.expander("🔐 Cómo configurarlo"):
        st.markdown(
            """
            1. Crea una cuenta gratuita en [alpaca.markets](https://alpaca.markets)
               (el plan de paper trading no requiere fondear nada real).
            2. En el panel de Alpaca, genera una **API Key** de la cuenta
               **Paper Trading** (no la de cuenta real).
            3. Define estas variables de entorno antes de lanzar la app:

            ```bash
            export FQL_ALPACA_KEY_ID=tu_key_id
            export FQL_ALPACA_SECRET_KEY=tu_secret_key
            ```

            O añádelas a `.streamlit/secrets.toml` (plantilla en
            `.streamlit/secrets.toml.example`) y léelas con
            `st.secrets["FQL_ALPACA_KEY_ID"]` si prefieres ese mecanismo.

            **Nunca subas estas claves a git** -- ambos archivos ya están
            en `.gitignore`.
            """
        )
    footer()
    st.stop()

try:
    account = PaperAccount(api_key=API_KEY, api_secret=API_SECRET)
except PaperTradeError as e:
    callout(f"No se pudo inicializar la cuenta: {e}", variant="danger")
    footer()
    st.stop()

# --- Estado de la cuenta ---
section("💰 Estado de la cuenta")
try:
    info = account.get_account()
except PaperTradeError as e:
    callout(f"No se pudo conectar con Alpaca: {e}", variant="danger")
    footer()
    st.stop()

kpi_row([
    ("Equity", f"${float(info.get('equity', 0)):,.2f}"),
    ("Cash", f"${float(info.get('cash', 0)):,.2f}"),
    ("Buying power", f"${float(info.get('buying_power', 0)):,.2f}"),
    ("Estado cuenta", str(info.get("status", "—"))),
])

# --- Posiciones abiertas ---
section("📊 Posiciones abiertas")
try:
    positions = account.get_positions()
except PaperTradeError as e:
    positions = []
    callout(f"No se pudieron cargar las posiciones: {e}", variant="warning")

if positions:
    df_pos = pd.DataFrame(positions)
    cols = [c for c in ["symbol", "qty", "side", "avg_entry_price",
                          "current_price", "unrealized_pl"] if c in df_pos.columns]
    st.dataframe(df_pos[cols] if cols else df_pos, use_container_width=True)
else:
    st.caption("Sin posiciones abiertas.")

# --- Enviar orden ---
section("🚀 Enviar orden")
callout(
    "Esto envía una orden real a la cuenta de PAPER de Alpaca (dinero "
    "simulado, no capital real) -- aun así, es una acción con efecto, "
    "no una simulación local.",
    variant="info",
)
with st.form("submit_order_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.text_input("Ticker", placeholder="AAPL").strip().upper()
    with col2:
        qty = st.number_input("Cantidad", min_value=0.0, value=1.0, step=1.0)
    with col3:
        side = st.selectbox("Sentido", ["buy", "sell"])

    order_type = st.selectbox("Tipo de orden", ["market", "limit"])
    limit_price = None
    if order_type == "limit":
        limit_price = st.number_input("Precio límite", min_value=0.01, value=100.0, step=0.01)

    confirm = st.checkbox("Confirmo que quiero enviar esta orden a Alpaca (paper).")
    submitted = st.form_submit_button("📤 Enviar orden", type="primary")

    if submitted:
        if not symbol:
            st.error("Indica un ticker.")
        elif not confirm:
            st.error("Marca la casilla de confirmación antes de enviar.")
        else:
            try:
                order = account.submit_order(
                    symbol=symbol, qty=qty,
                    side=cast(Literal["buy", "sell"], side),
                    order_type=order_type, limit_price=limit_price,
                )
                st.success(f"Orden enviada: {order.get('id', '(sin id)')} -- {side} {qty} {symbol}")
            except PaperTradeError as e:
                st.error(f"No se pudo enviar la orden: {e}")

# --- Acciones de emergencia ---
with st.expander("⚠️ Acciones de emergencia"):
    st.caption("Cancela todas las órdenes abiertas o cierra todas las posiciones de golpe.")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🛑 Cancelar todas las órdenes"):
            try:
                n = account.cancel_all()
                st.success(f"{n} orden(es) cancelada(s).")
            except PaperTradeError as e:
                st.error(f"No se pudo cancelar: {e}")
    with col_b:
        if st.button("🔻 Cerrar todas las posiciones"):
            try:
                n = account.close_all_positions()
                st.success(f"{n} posición(es) cerrada(s).")
            except PaperTradeError as e:
                st.error(f"No se pudo cerrar: {e}")

# --- Ejecutar una estrategia automáticamente (StrategyRunner) ---
section("🤖 Ejecutar una estrategia automáticamente")
callout(
    "Esto llama a `StrategyRunner.run_once()`: calcula la señal actual de "
    "cada símbolo y envía la orden necesaria para que la posición coincida "
    "con ella (comparado contra la posición ya abierta, así que no duplica "
    "órdenes si ya estás posicionado correctamente). Solo funciona bajo "
    "demanda con este botón -- `run_forever()` (el bucle continuo del "
    "propio módulo) no se expone aquí porque bloquearía el servidor de "
    "Streamlit; para ejecución desatendida real habría que correrlo aparte "
    "como un proceso independiente, no dentro de esta app.",
    variant="info",
)

STRATEGY_RUNNER_OPTIONS = {
    "trend_following": "Trend Following (cruce de medias)",
    "volatility_targeting": "Volatility Targeting",
}
with st.form("strategy_runner_form"):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        runner_strategy = st.selectbox(
            "Estrategia", list(STRATEGY_RUNNER_OPTIONS.keys()),
            format_func=lambda n: STRATEGY_RUNNER_OPTIONS[n],
            help=(
                "Solo estas dos encajan aquí: dan una señal -1/0/1 por "
                "símbolo individual con solo su propio histórico de "
                "precios. El resto del registro (PCA StatArb, Risk "
                "Parity, Cross-Sectional Momentum) necesita varios "
                "activos a la vez para decidir, así que no calzan con "
                "el diseño 'un símbolo, una señal' de StrategyRunner."
            ),
        )
    with col_s2:
        runner_symbols_raw = st.text_input(
            "Símbolos (separados por coma)", placeholder="AAPL, MSFT",
        )
    runner_capital = st.number_input(
        "Capital a repartir entre símbolos ($)",
        min_value=100.0, value=10_000.0, step=100.0,
    )
    confirm_runner = st.checkbox(
        "Confirmo que quiero calcular la señal y enviar la(s) orden(es) "
        "necesaria(s) a Alpaca (paper) para alinear la posición.",
    )
    run_strategy_clicked = st.form_submit_button("🔄 Ejecutar una pasada", type="primary")

if run_strategy_clicked:
    runner_symbols = [s.strip().upper() for s in runner_symbols_raw.split(",") if s.strip()]
    if not runner_symbols:
        st.error("Indica al menos un símbolo.")
    elif not confirm_runner:
        st.error("Marca la casilla de confirmación antes de ejecutar.")
    else:
        strat: Strategy | None
        try:
            strat = get_strategy(runner_strategy)
        except StrategyError as e:
            st.error(f"No se pudo cargar la estrategia: {e}")
            strat = None

        if strat is not None:
            def _price_fetcher(symbol: str, _strat=strat) -> pd.Series:
                end_date = date.today()
                start_date = end_date - timedelta(days=250)
                prices = load_prices([symbol], start_date, end_date)
                return prices[symbol]

            def _adapter(price_series: pd.Series, _strat=strat) -> int:
                df = price_series.to_frame(name="_symbol")
                signals = _strat.generate_signals(df)
                if isinstance(signals.index, pd.MultiIndex):
                    val = signals.xs("_symbol", level=-1).iloc[-1]
                else:
                    val = signals.iloc[-1]
                return int(np.sign(val)) if abs(val) > 1e-9 else 0

            try:
                runner = StrategyRunner(
                    strategy=_adapter,
                    symbols=runner_symbols,
                    capital=runner_capital,
                    account=account,
                    price_fetcher=_price_fetcher,
                )
                orders = runner.run_once()
            except (PaperTradeError, ValueError, ConnectionError, StrategyError) as e:
                st.error(f"No se pudo ejecutar la pasada: {e}")
            else:
                if orders:
                    st.success(f"{len(orders)} orden(es) enviada(s):")
                    st.json(orders)
                else:
                    st.info(
                        "Ninguna orden necesaria -- la posición actual ya "
                        "coincide con la señal de la estrategia."
                    )

footer()
