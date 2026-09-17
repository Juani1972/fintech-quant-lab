"""Página de experimentos reproducibles (config + datos + resultados congelados)."""
import streamlit as st

from app.core.experiments import (
    ExperimentError,
    diff_experiments,
    list_experiments,
    load_experiment,
)
from app.styles import callout, footer, hero, kpi_row, page_setup, section

page_setup("Experimentos", "🧪")

hero(
    title="Experimentos Reproducibles",
    subtitle=(
        "A diferencia del Histórico (solo métricas en SQLite), cada "
        "experimento congela una copia exacta de los datos usados y el "
        "commit de git, para poder reproducir un resultado concreto más "
        "adelante aunque los datos de mercado hayan cambiado entre tanto."
    ),
    icon="🧪",
)

experiment_ids = list_experiments()

if not experiment_ids:
    callout(
        "Todavía no hay ningún experimento guardado. Se guardan desde la "
        "sección <strong>🧪 Guardar como experimento reproducible</strong> "
        "al final de la página <strong>🧪 Backtest</strong>, tras ejecutar "
        "un backtest.",
        variant="info",
    )
    footer()
    st.stop()

section("📋 Experimentos guardados")
st.caption(f"{len(experiment_ids)} experimento(s) en disco (carpeta `experiments/`).")

selected_id = st.selectbox("Ver detalle de", experiment_ids)

try:
    exp = load_experiment(selected_id)
except ExperimentError as e:
    callout(f"No se pudo cargar el experimento: {e}", variant="danger")
    st.stop()

col_meta, col_config = st.columns([1, 2])
with col_meta:
    st.markdown("**Metadatos**")
    st.write(f"**ID:** `{exp.id}`")
    st.write(f"**Creado:** {exp.created_at}")
    st.write(f"**Commit git:** `{exp.git_commit or 'sin repo git'}`")
with col_config:
    st.markdown("**Configuración**")
    st.json(exp.config)

section("📊 Resultados")
try:
    results = exp.results
    kpi_row([
        (k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))
        for k, v in list(results.items())[:6]
    ])
    with st.expander("Ver todos los resultados"):
        st.json(results)
except Exception as e:
    callout(f"No se pudieron cargar los resultados: {e}", variant="warning")

section("📈 Datos congelados")
try:
    data = exp.data
    st.caption(f"{len(data)} filas × {data.shape[1]} columna(s), desde {data.index.min()} hasta {data.index.max()}.")
    st.dataframe(data.tail(10), use_container_width=True)
except Exception as e:
    callout(f"No se pudieron cargar los datos congelados: {e}", variant="warning")

# --- Comparar dos experimentos ---
if len(experiment_ids) >= 2:
    section("🆚 Comparar dos experimentos")
    col_a, col_b = st.columns(2)
    with col_a:
        exp_a_id = st.selectbox("Experimento A", experiment_ids, index=0, key="exp_a")
    with col_b:
        default_b = 1 if len(experiment_ids) > 1 else 0
        exp_b_id = st.selectbox("Experimento B", experiment_ids, index=default_b, key="exp_b")

    if exp_a_id == exp_b_id:
        callout("Elige dos experimentos distintos para compararlos.", variant="info")
    else:
        try:
            diff = diff_experiments(exp_a_id, exp_b_id)
        except ExperimentError as e:
            callout(f"No se pudo comparar: {e}", variant="danger")
        else:
            if diff.empty:
                callout(
                    "Config y resultados son idénticos entre ambos experimentos.",
                    variant="success",
                )
            else:
                st.caption(f"{len(diff)} diferencia(s) encontrada(s):")
                st.dataframe(diff, use_container_width=True, hide_index=True)

footer()
