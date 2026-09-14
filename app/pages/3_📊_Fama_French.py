"""Página de regresión Fama-French con errores robustos HAC."""
import streamlit as st

from app.core.data_loader import compute_log_returns, load_prices
from app.core.fama_french import load_factors, run_regression

st.set_page_config(page_title="Fama-French", page_icon="📊", layout="wide")
st.header("📊 Regresión Fama-French")

tickers_str = st.session_state.get("global_tickers", "KO, PEP")
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
start = st.session_state.get("global_start")
end = st.session_state.get("global_end")

if not tickers:
    st.error("Introduce al menos un ticker en la barra lateral.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🎛️ Parámetros Fama-French")
    ticker = st.selectbox("Ticker", tickers)
    model = st.selectbox("Modelo", ["3", "5"], format_func=lambda x: f"{x} factores")
    cov_type = st.selectbox(
        "Errores estándar",
        ["HAC", "HC3", "nonrobust"],
        index=0,
        help="HAC = Newey-West (robusto a heterocedasticidad y autocorrelación)",
    )
    maxlags = st.number_input(
        "Lags HAC (0 = automático)",
        min_value=0, max_value=50, value=0, step=1,
    )
    run = st.button("🚀 Ejecutar regresión", type="primary", use_container_width=True)

if run:
    with st.spinner("Descargando datos del activo..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    returns = compute_log_returns(prices)[ticker]

    with st.spinner("Descargando factores Fama-French..."):
        try:
            factors = load_factors(start, end, model=model)
        except (ValueError, ConnectionError) as e:
            st.error(f"Error al cargar factores: {e}")
            st.stop()

    try:
        result = run_regression(
            returns, factors,
            model=model,
            cov_type=cov_type,
            maxlags=maxlags if maxlags > 0 else None,
        )
    except ValueError as e:
        st.error(f"Error en la regresión: {e}")
        st.stop()

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
        st.success(f"Alpha estadísticamente significativo (t={result.alpha_tstat:.2f}).")
    else:
        st.info(
            f"Alpha no significativo (t={result.alpha_tstat:.2f}). "
            "El retorno se explica por los factores."
        )

    st.subheader("Betas de factores")
    betas_df = result.betas.rename("Beta").to_frame()
    betas_df["t-stat"] = result.betas_tstats
    betas_df["p-valor"] = result.betas_pvalues
    st.dataframe(betas_df, use_container_width=True)

    st.subheader("Resumen completo")
    st.text(result.summary)

    st.download_button(
        "⬇️ Descargar betas (CSV)",
        betas_df.to_csv().encode("utf-8"),
        file_name=f"ff_betas_{ticker}.csv",
    )
