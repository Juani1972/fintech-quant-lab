"""Página de cointegración y pairs trading con corrección por múltiples tests."""
from itertools import combinations

import pandas as pd
import streamlit as st

from app.core.cointegration import (
    engle_granger,
    generate_signals,
    half_life,
    rolling_zscore,
)
from app.core.data_loader import load_prices
from app.core.multiple_testing import correct_pvalues
from app.core.plotting import line_chart, zscore_chart
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, page_setup, section

page_setup("Cointegración", "🔗")

hero(
    title="Cointegración y Pairs Trading",
    subtitle=(
        "Engle-Granger, ADF del spread, half-life y z-score causal. "
        "Corrección por múltiples tests (Bonferroni y Benjamini-Hochberg)."
    ),
    icon="🔗",
)

ensure_session_initialized()
tickers, start, end = get_global_params()

if len(tickers) < 2:
    callout(
        "Esta página requiere al menos 2 tickers. Añádelos en la barra lateral.",
        variant="warning",
    )
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros Pairs Trading")
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Ticker 1", tickers, index=0)
    t2 = col2.selectbox("Ticker 2", tickers, index=1)
    window = st.slider("Ventana Z-score", 20, 200, 60)
    entry = st.slider("Umbral de entrada", 0.5, 3.0, 2.0, 0.1)
    run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()

    # --- Multiple testing sobre todos los pares ---
    if len(tickers) > 2:
        section("🧪 Todos los pares (con corrección por múltiples tests)")
        pairs = list(combinations(tickers, 2))
        rows = []
        pvals = []
        for (a, b) in pairs:
            try:
                r = engle_granger(prices[a], prices[b])
                rows.append({
                    "Par": f"{a} / {b}",
                    "p-valor EG": r.pvalue,
                    "ADF spread p": r.adf_spread_pvalue,
                    "Beta": r.beta,
                })
                pvals.append(r.pvalue)
            except Exception:
                continue

        df_pairs = pd.DataFrame(rows)
        if len(df_pairs) > 0:
            pv_series = pd.Series(pvals, index=df_pairs.index)
            bonf = correct_pvalues(pv_series, method="bonferroni", alpha=0.05)
            bh = correct_pvalues(pv_series, method="bh", alpha=0.05)

            df_pairs["p Bonferroni"] = bonf.pvalues_adjusted
            df_pairs["p BH (FDR)"] = bh.pvalues_adjusted
            df_pairs["Rechaza BH"] = bh.rejected.map({True: "✅", False: "❌"})

            st.dataframe(
                df_pairs.style.format({
                    "p-valor EG": "{:.4f}",
                    "ADF spread p": "{:.4f}",
                    "Beta": "{:.4f}",
                    "p Bonferroni": "{:.4f}",
                    "p BH (FDR)": "{:.4f}",
                }),
                use_container_width=True,
            )
            callout(
                f"Se han realizado <strong>{len(df_pairs)} tests</strong>. "
                f"Rechazos brutos (p&lt;0.05): <strong>{(pv_series < 0.05).sum()}</strong>. "
                f"Tras Bonferroni: <strong>{bonf.n_rejected}</strong>. "
                f"Tras BH (FDR): <strong>{bh.n_rejected}</strong>.",
                variant="info",
            )

    # --- Detalle del par seleccionado ---
    section(f"📌 Detalle del par {t1} / {t2}")
    result = engle_granger(prices[t1], prices[t2])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p-valor Engle-Granger", f"{result.pvalue:.4f}")
    c2.metric("ADF spread (p)", f"{result.adf_spread_pvalue:.4f}")
    c3.metric("Beta (cobertura)", f"{result.beta:.4f}")
    c4.metric("Cointegradas", "✅" if result.is_cointegrated else "❌")

    st.markdown(f"**Spread:** `{t1} - {result.alpha:.4f} - {result.beta:.4f} · {t2}`")

    section("Test ADF del spread")
    st.write(f"**Estadístico:** {result.adf_spread_stat:.4f}")
    st.write(f"**p-valor:** {result.adf_spread_pvalue:.4f}")
    st.write("**Valores críticos:**")
    st.json(result.adf_spread_crit)

    if result.adf_spread_pvalue < 0.05:
        callout("El spread es estacionario (ADF p < 0.05). Apto para pairs trading.",
                variant="success")
    else:
        callout("El spread NO es estacionario según ADF. Cuidado con el pairs trading.",
                variant="warning")

    section("Spread")
    st.plotly_chart(
        line_chart(result.spread, f"Spread {t1} - α - β·{t2}"),
        use_container_width=True,
    )

    hl = half_life(result.spread)
    if hl != float("inf"):
        callout(f"⏱️ Half-life de reversión: <strong>{hl:.1f} días</strong>",
                variant="info")
    else:
        callout("Half-life no definida (el spread no revierte a la media).",
                variant="warning")

    section(f"Z-score (ventana={window})")
    z = rolling_zscore(result.spread, window)
    st.plotly_chart(zscore_chart(z, entry=entry), use_container_width=True)

    section("Señales de trading")
    signals = generate_signals(z, entry=entry)
    st.plotly_chart(
        line_chart(signals, "Señales (1=long, -1=short, 0=neutral)"),
        use_container_width=True,
    )

    st.download_button(
        "⬇️ Descargar señales (CSV)",
        signals.to_csv().encode("utf-8"),
        file_name=f"signals_{t1}_{t2}.csv",
    )

else:
    callout(
        "Configura el par y los parámetros en la barra lateral y pulsa "
        "<strong>🚀 Ejecutar análisis</strong>.",
        variant="info",
    )

footer()
