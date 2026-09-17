"""Página de cointegración y pairs trading con corrección por múltiples tests."""
import json
from itertools import combinations

import pandas as pd
import streamlit as st

from app.core.cointegration import (
    engle_granger,
    generate_signals,
    half_life,
    plain_language_summary,
    rolling_zscore,
)
from app.core.data_loader import load_prices
from app.core.multiple_testing import correct_pvalues
from app.core.plotting import line_chart, zscore_chart
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import callout, conclusion, data_preview, footer, hero, page_setup, section

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
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if len(tickers) < 2:
    callout(
        "Esta página requiere al menos 2 tickers. Añádelos en la barra lateral.",
        variant="warning",
    )
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Parámetros Pairs Trading")

    with st.expander("📂 Cargar / guardar configuración"):
        st.caption("Guarda estos parámetros como JSON, o carga unos guardados antes.")
        uploaded_config = st.file_uploader(
            "Cargar configuración (JSON)", type="json", key="_coint_config_upload",
        )
        if uploaded_config is not None and st.session_state.get("_coint_config_applied") != uploaded_config.name:
            try:
                loaded_cfg = json.load(uploaded_config)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                st.error(f"El archivo no es un JSON válido: {e}")
            else:
                skipped = []
                for field, key in [("t1", "coint_t1"), ("t2", "coint_t2")]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if val in tickers:
                            st.session_state[key] = val
                        else:
                            skipped.append(f"{field} ('{val}' no está en tus tickers actuales)")
                for field, key, lo, hi in [
                    ("window", "coint_window", 20, 200),
                    ("entry", "coint_entry", 0.5, 3.0),
                ]:
                    val = loaded_cfg.get(field)
                    if val is not None:
                        if isinstance(val, (int, float)) and lo <= val <= hi:
                            st.session_state[key] = val
                        else:
                            skipped.append(field)
                st.session_state["_coint_config_applied"] = uploaded_config.name
                if skipped:
                    st.warning(f"Cargado, salvo: {', '.join(skipped)} (fuera de rango o no aplicable ahora).")
                else:
                    st.success("Configuración cargada.")
                st.rerun()

    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Ticker 1", tickers, index=0, key="coint_t1")
    t2 = col2.selectbox("Ticker 2", tickers, index=1, key="coint_t2")
    window = st.slider(
        "Ventana Z-score", 20, 200, 60,
        help=(
            "Nº de días usados para calcular la media y desviación típica "
            "móviles del spread, sobre las que se mide el z-score. Una "
            "ventana corta reacciona más rápido a cambios de régimen pero "
            "da señales más ruidosas; una larga suaviza pero reacciona "
            "más lento."
        ),
        key="coint_window",
    )
    entry = st.slider(
        "Umbral de entrada", 0.5, 3.0, 2.0, 0.1,
        help=(
            "Z-score (nº de desviaciones típicas respecto a la media móvil "
            "del spread) a partir del cual se genera una señal de entrada. "
            "2.0 es un valor habitual: el spread se considera "
            "'suficientemente alejado' de su media como para apostar a "
            "que revierte."
        ),
        key="coint_entry",
    )

    current_config = {"t1": t1, "t2": t2, "window": window, "entry": entry}
    st.download_button(
        "💾 Guardar configuración actual (JSON)",
        json.dumps(current_config, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="cointegracion_config.json",
        mime="application/json",
    )

    run = st.button("🚀 Ejecutar análisis", type="primary", use_container_width=True)


if run:
    with st.spinner("Descargando datos..."):
        try:
            prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
        except (ValueError, ConnectionError) as e:
            callout(f"Error al cargar datos: {e}", variant="danger")
            st.stop()
        data_preview(prices)

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
    try:
        result = engle_granger(prices[t1], prices[t2])
    except ValueError as e:
        callout(f"No se pudo analizar este par: {e}", variant="danger")
        st.stop()

    callout(
        "⚠️ Alpha y beta se estiman con toda la muestra mostrada (incluido "
        "el propio periodo donde luego se generan señales) — es un ajuste "
        "in-sample habitual en el análisis exploratorio, pero optimista "
        "respecto a un uso en producción. Para una validación honesta, "
        "combina esto con Walk-Forward reestimando el spread solo con el "
        "tramo de entrenamiento.",
        variant="warning",
    )

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
        st.metric(
            "⏱️ Half-life de reversión", f"{hl:.1f} días",
            help=(
                "Tiempo medio estimado que tarda el spread en recorrer la "
                "mitad de la distancia hasta su media histórica tras "
                "desviarse -- se calcula ajustando un proceso AR(1) al "
                "spread (Ornstein-Uhlenbeck discreto). Un half-life corto "
                "(pocos días) indica reversión rápida, más aprovechable "
                "para pairs trading; uno muy largo indica que, aunque el "
                "par esté cointegrado, revertir a la media puede tardar "
                "demasiado para ser operable."
            ),
        )
    else:
        callout("Half-life no definida (el spread no revierte a la media).",
                variant="warning")

    conclusion_text, conclusion_variant = plain_language_summary(result, hl)
    conclusion(conclusion_text, variant=conclusion_variant)

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
