"""Página de regresión Fama-French con errores robustos HAC."""
import json

import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.fama_french import load_factors, run_regression
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, data_preview, footer, hero, page_setup, section, ticker_badge

page_setup("Fama-French", "📊")

hero(
    title="Regresión Fama-French",
    subtitle=(
        "Regresión de 3 y 5 factores con errores estándar HAC (Newey-West). "
        "Interpretación del alpha con t-estadístico robusto."
    ),
    icon="📊",
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
    st.markdown("## 🎛️ Parámetros Fama-French")

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_ff_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_ff_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                ticker_val = loaded_cfg.get("ticker")
                if ticker_val is not None:
                    if ticker_val in tickers:
                        st.session_state["ff_ticker"] = ticker_val
                    else:
                        skipped.append(f"ticker ('{ticker_val}' no está en tus tickers actuales)")
                model_val = loaded_cfg.get("model")
                if model_val is not None:
                    if model_val in ["3", "5"]:
                        st.session_state["ff_model"] = model_val
                    else:
                        skipped.append("model")
                cov_val = loaded_cfg.get("cov_type")
                if cov_val is not None:
                    if cov_val in ["HAC", "HC3", "nonrobust"]:
                        st.session_state["ff_cov_type"] = cov_val
                    else:
                        skipped.append("cov_type")
                maxlags_val = loaded_cfg.get("maxlags")
                if maxlags_val is not None:
                    if isinstance(maxlags_val, int) and 0 <= maxlags_val <= 50:
                        st.session_state["ff_maxlags"] = maxlags_val
                    else:
                        skipped.append("maxlags")
                st.session_state["_ff_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    # Consumir (una sola vez) la marca que deja el buscador de main.py.
    _last = st.session_state.pop("_last_searched_ticker", None)
    if _last and _last in tickers:
        st.session_state["ff_ticker"] = _last

    ticker = st.selectbox("Ticker", tickers, key="ff_ticker")
    model = st.selectbox(
        "Modelo", ["3", "5"], format_func=lambda x: f"{x} factores", key="ff_model",
    )
    cov_type = st.selectbox(
        "Errores estándar",
        ["HAC", "HC3", "nonrobust"],
        index=0,
        help="HAC = Newey-West (robusto a heterocedasticidad y autocorrelación)",
        key="ff_cov_type",
    )
    maxlags = st.number_input(
        "Lags HAC (0 = automático)",
        min_value=0, max_value=50, value=0, step=1,
        key="ff_maxlags",
    )

    current_config = {"ticker": ticker, "model": model, "cov_type": cov_type, "maxlags": maxlags}
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="famafrench_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Ejecutar regresión", type="primary", use_container_width=True)

ticker_badge(ticker)

if run:
    with st.spinner("Descargando datos del activo..."):
        try:
            prices = load_prices([ticker], start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

    returns = compute_log_returns(prices)[ticker]

    with st.spinner("Descargando factores Fama-French..."):
        try:
            factors = load_factors(start, end, model=model)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar factores: {e}", variant="danger")
            st.stop()

    try:
        result = run_regression(
            returns, factors,
            model=model,
            cov_type=cov_type,
            maxlags=maxlags if maxlags > 0 else None,
        )
    except ValueError as e:
        callout(f"Error en la regresión: {e}", variant="danger")
        st.stop()

    section("📊 Resultado de la regresión")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha", f"{result.alpha:.4%}")
    c2.metric("p-valor alpha", f"{result.alpha_pvalue:.4f}")
    c3.metric("R²", f"{result.r_squared:.4f}")
    c4.metric("R² ajustado", f"{result.adj_r_squared:.4f}")

    st.caption(
        f"Errores estándar: **{result.cov_type}** "
        + (f"(maxlags={result.maxlags})" if result.maxlags else "")
        + f" · N = {result.n_obs}"
    )

    if result.alpha_pvalue < 0.05:
        callout(
            f"Alpha estadísticamente significativo (t={result.alpha_tstat:.2f}).",
            variant="success",
        )
    else:
        callout(
            f"Alpha no significativo (t={result.alpha_tstat:.2f}). "
            "El retorno se explica por los factores.",
            variant="info",
        )

    section("Betas de factores")
    betas_df = result.betas.rename("Beta").to_frame()
    betas_df["t-stat"] = result.betas_tstats
    betas_df["p-valor"] = result.betas_pvalues
    st.dataframe(betas_df, use_container_width=True)

    with st.expander("📋 Resumen completo"):
        st.text(result.summary)

    st.download_button(
        "⬇️ Descargar betas (CSV)",
        betas_df.to_csv().encode("utf-8"),
        file_name=f"ff_betas_{ticker}.csv",
    )

else:
    callout(
        "Configura el ticker y el modelo en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar regresión</strong>.",
        variant="info",
    )

footer()