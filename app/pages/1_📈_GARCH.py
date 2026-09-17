"""Página de modelado GARCH."""
import json

import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.garch import (
    check_stationarity,
    fit_garch,
    forecast_volatility,
    residual_diagnostics,
)
from app.core.plotting import line_chart
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

page_setup("GARCH", "📈")

hero(
    title="Modelado GARCH",
    subtitle=(
        "Volatilidad condicional con soporte para GARCH, EGARCH y GJR-GARCH. "
        "Incluye diagnósticos de residuos y pronóstico a 30 días."
    ),
    icon="📈",
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
    st.markdown("## 🎛️ Parámetros GARCH")

    def _apply_garch_config(loaded_cfg: dict) -> list[str]:
        """Valida y aplica un dict de configuración al session_state de
        los widgets de esta página. Compartida entre el archivo JSON
        subido y las configuraciones nombradas de la sesión."""
        skipped = []
        ticker_val = loaded_cfg.get("ticker")
        if ticker_val is not None:
            if ticker_val in tickers:
                st.session_state["garch_ticker"] = ticker_val
            else:
                skipped.append(f"ticker ('{ticker_val}' no está en tus tickers actuales)")
        for field, key, lo, hi, options in [
            ("p", "garch_p", 1, 3, None),
            ("q", "garch_q", 1, 3, None),
            ("vol", "garch_vol", None, None, ["Garch", "EGARCH", "GJR-GARCH"]),
            ("dist", "garch_dist", None, None, ["normal", "t", "skewt", "ged"]),
        ]:
            val = loaded_cfg.get(field)
            if val is None:
                continue
            if options is not None:
                if val in options:
                    st.session_state[key] = val
                else:
                    skipped.append(field)
            elif lo is not None and hi is not None and isinstance(val, int) and lo <= val <= hi:
                st.session_state[key] = val
            else:
                skipped.append(field)
        return skipped

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_garch_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_garch_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = _apply_garch_config(loaded_cfg)
                st.session_state["_garch_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

        st.markdown("---")
        _garch_current_for_named = {
            "ticker": st.session_state.get("garch_ticker"),
            "p": st.session_state.get("garch_p"),
            "q": st.session_state.get("garch_q"),
            "vol": st.session_state.get("garch_vol"),
            "dist": st.session_state.get("garch_dist"),
        }
        _garch_loaded_named = named_config_manager(_garch_current_for_named, "garch")
        if _garch_loaded_named is not None:
            _apply_garch_config(_garch_loaded_named)
            st.rerun()

    ticker = st.selectbox("Ticker a modelar", tickers, key="garch_ticker")
    p = st.slider(
        "Orden ARCH (p)", 1, 3, 1,
        help=(
            "Nº de retornos al cuadrado pasados que influyen en la "
            "volatilidad de hoy. p=1 (el valor habitual) basta en la "
            "mayoría de series financieras diarias; subirlo rara vez "
            "mejora el ajuste y complica la estimación."
        ),
        key="garch_p",
    )
    q = st.slider(
        "Orden GARCH (q)", 1, 3, 1,
        help=(
            "Nº de volatilidades condicionales pasadas que influyen en la "
            "de hoy -- es lo que da memoria de largo plazo al modelo "
            "(clustering de volatilidad). GARCH(1,1) -- p=1, q=1 -- es el "
            "estándar de facto para retornos diarios."
        ),
        key="garch_q",
    )
    vol = st.selectbox(
        "Tipo de modelo", ["Garch", "EGARCH", "GJR-GARCH"],
        help=(
            "GARCH: simétrico, subidas y bajadas afectan igual a la "
            "volatilidad futura. EGARCH y GJR-GARCH: asimétricos, "
            "capturan el efecto apalancamiento (las caídas suelen elevar "
            "la volatilidad futura más que subidas de igual magnitud)."
        ),
        key="garch_vol",
    )
    dist = st.selectbox(
        "Distribución", ["normal", "t", "skewt", "ged"],
        help=(
            "Distribución asumida para los residuos estandarizados. "
            "'normal' es la más simple; 't' (t de Student) y 'skewt' "
            "capturan colas pesadas y asimetría, más realistas para "
            "retornos financieros pero con más parámetros que estimar."
        ),
        key="garch_dist",
    )

    current_config = {"ticker": ticker, "p": p, "q": q, "vol": vol, "dist": dist}
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="garch_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Ejecutar GARCH", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    returns = compute_log_returns(prices)[ticker]

    with st.spinner("Ajustando modelo..."):
        try:
            result = fit_garch(returns, p=p, q=q, vol=vol, dist=dist)
        except Exception as e:
            callout(f"Error ajustando GARCH: {e}", variant="danger")
            st.stop()

    section("📊 Resultado del ajuste")
    c1, c2, c3 = st.columns(3)
    c1.metric("AIC", f"{result.aic:.2f}")
    c2.metric("BIC", f"{result.bic:.2f}")
    c3.metric(
        "Estacionario",
        "✅ Sí" if check_stationarity(result.params, result.model_type) else "❌ No",
    )

    section("🧪 Diagnósticos de residuos")
    try:
        diag = residual_diagnostics(result, lags=10)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Ljung-Box", f"{diag['ljung_box_pvalue']:.4f}")
        d2.metric("Ljung-Box²", f"{diag['ljung_box_squared_pvalue']:.4f}")
        d3.metric("ARCH-LM", f"{diag['arch_lm_pvalue']:.4f}")
        d4.metric("Jarque-Bera", f"{diag['jarque_bera_pvalue']:.4f}")

        if diag["ljung_box_squared_pvalue"] > 0.05 and diag["arch_lm_pvalue"] > 0.05:
            callout(
                "No queda evidencia de heterocedasticidad condicional en los residuos.",
                variant="success",
            )
        else:
            callout(
                "Queda estructura en los residuos. Considera otro orden o modelo.",
                variant="warning",
            )
    except Exception as e:
        callout(f"No se pudieron calcular diagnósticos: {e}", variant="info")

    section("📈 Volatilidad condicional")
    st.plotly_chart(
        line_chart(
            result.conditional_volatility,
            f"Volatilidad condicional — {ticker}",
        ),
        use_container_width=True,
    )

    section("🔮 Pronóstico (30 días)")
    fc = forecast_volatility(result, horizon=30)
    st.plotly_chart(
        line_chart(fc, "Pronóstico de volatilidad"),
        use_container_width=True,
    )

    with st.expander("📋 Resumen completo del modelo"):
        st.text(result.model_result.summary().as_text())

    st.download_button(
        "⬇️ Descargar parámetros (CSV)",
        result.params.to_csv().encode("utf-8"),
        file_name=f"garch_params_{ticker}.csv",
    )

else:
    callout(
        "Configura los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar GARCH</strong>.",
        variant="info",
    )

footer()
