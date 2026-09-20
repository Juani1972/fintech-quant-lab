"""Página de métricas de riesgo."""
import json

import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.garch import fit_garch, forecast_volatility
from app.core.plotting import drawdown_chart, line_chart
from app.core.risk import (
    calmar_ratio,
    drawdown_series,
    expected_shortfall,
    expected_shortfall_parametric,
    max_drawdown,
    rolling_var,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
    value_at_risk_cornish_fisher,
    value_at_risk_filtered_historical,
    value_at_risk_parametric,
)
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section, ticker_badge

page_setup("Riesgo", "⚠️")

hero(
    title="Medición de Riesgo",
    subtitle=(
        "VaR histórico y paramétrico, Expected Shortfall, drawdown, "
        "Sharpe, Sortino y Calmar."
    ),
    icon="⚠️",
)

ensure_session_initialized()
tickers, start, end = get_global_params()
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if not tickers:
    callout("Introduce al menos un ticker en la barra lateral.", variant="warning")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros de riesgo")

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_riesgo_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_riesgo_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                ticker_val = loaded_cfg.get("ticker")
                if ticker_val is not None:
                    if ticker_val in tickers:
                        st.session_state["riesgo_ticker"] = ticker_val
                    else:
                        skipped.append(f"ticker ('{ticker_val}' no está en tus tickers actuales)")
                for field, key, lo, hi in [
                    ("confidence", "riesgo_confidence", 0.90, 0.99),
                    ("window", "riesgo_window", 50, 500),
                ]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if isinstance(val, (int, float)) and lo <= val <= hi:
                            st.session_state[key] = val
                        else:
                            skipped.append(field)
                st.session_state["_riesgo_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    # Consumir (una sola vez) la marca que deja el buscador de main.py.
    _last = st.session_state.pop("_last_searched_ticker", None)
    if _last and _last in tickers:
        st.session_state["riesgo_ticker"] = _last

    ticker = st.selectbox("Ticker", tickers, key="riesgo_ticker")
    confidence = st.slider("Nivel de confianza", 0.90, 0.99, 0.95, 0.01, key="riesgo_confidence")
    window = st.slider("Ventana VaR rodante", 50, 500, 250, key="riesgo_window")

    current_config = {"ticker": ticker, "confidence": confidence, "window": window}
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="riesgo_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Calcular riesgo", type="primary", use_container_width=True)

ticker_badge(ticker)

if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices([ticker], start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    returns = compute_log_returns(prices)[ticker]

    section("📉 VaR y Expected Shortfall")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"VaR hist. {confidence:.0%}", f"{value_at_risk(returns, confidence):.4%}")
    c2.metric(
        f"VaR param. {confidence:.0%}",
        f"{value_at_risk_parametric(returns, confidence):.4%}",
    )
    c3.metric(
        f"VaR Cornish-Fisher {confidence:.0%}",
        f"{value_at_risk_cornish_fisher(returns, confidence):.4%}",
        help="VaR paramétrico ajustado por la asimetría y curtosis "
             "reales de los retornos, en vez de asumir normalidad pura.",
    )
    try:
        garch_result = fit_garch(returns)
        forecast_vol = float(forecast_volatility(garch_result, horizon=1).iloc[0])
        var_fhs = value_at_risk_filtered_historical(
            returns, garch_result.conditional_volatility, forecast_vol, confidence,
        )
        c4.metric(
            f"VaR filtrado (GARCH) {confidence:.0%}", f"{var_fhs:.4%}",
            help="Simulación histórica filtrada: reescala los retornos "
                 "históricos por la volatilidad GARCH pronosticada para "
                 "hoy, en vez de asumir que la volatilidad futura será "
                 "igual al promedio de todo el histórico.",
        )
    except Exception as e:
        c4.metric(f"VaR filtrado (GARCH) {confidence:.0%}", "—", help=str(e))
    c5.metric(f"ES hist. {confidence:.0%}", f"{expected_shortfall(returns, confidence):.4%}")
    c6.metric(
        f"ES param. {confidence:.0%}",
        f"{expected_shortfall_parametric(returns, confidence):.4%}",
    )

    section("📊 Ratios ajustados por riesgo")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sharpe", f"{sharpe_ratio(returns, periods_per_year=252):.3f}")
    r2.metric("Sortino", f"{sortino_ratio(returns, periods_per_year=252):.3f}")
    r3.metric("Calmar", f"{calmar_ratio(returns, periods_per_year=252):.3f}")
    r4.metric("Max Drawdown", f"{max_drawdown(prices[ticker]):.2%}")

    section("VaR rodante")
    st.plotly_chart(
        line_chart(
            rolling_var(returns, window, confidence),
            f"VaR rodante ({window}d, {confidence:.0%})",
        ),
        use_container_width=True,
    )

    section("Drawdown")
    st.plotly_chart(
        drawdown_chart(drawdown_series(prices[ticker])),
        use_container_width=True,
    )

else:
    callout(
        "Configura el ticker y los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Calcular riesgo</strong>.",
        variant="info",
    )

footer()
