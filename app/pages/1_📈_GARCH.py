"""Página de modelado GARCH."""
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
from app.styles import callout, footer, hero, page_setup, section

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
    ticker = st.selectbox("Ticker a modelar", tickers)
    p = st.slider("Orden ARCH (p)", 1, 3, 1)
    q = st.slider("Orden GARCH (q)", 1, 3, 1)
    vol = st.selectbox("Tipo de modelo", ["Garch", "EGARCH", "GJR-GARCH"])
    dist = st.selectbox("Distribución", ["normal", "t", "skewt", "ged"])
    run = st.button("🚀 Ejecutar GARCH", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

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
