"""Página de comparación de backtests guardados."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.history import (
    LOWER_IS_BETTER,
    compare_runs,
    init_db,
    list_runs,
)
from app.styles import callout, footer, hero, page_setup, section

page_setup("Comparar", "📊")

hero(
    title="Comparación de Backtests",
    subtitle=(
        "Selecciona varias corridas guardadas en el histórico y compáralas "
        "lado a lado: métricas, parámetros, mejor/peor por métrica y "
        "gráficos agregados."
    ),
    icon="📊",
)

init_db()

# ============================================================
#  Sidebar: filtros y selección
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Selección")

    strategy_filter = st.selectbox(
        "Filtrar por estrategia",
        ["Todas", "Pairs Trading", "Momentum", "Mean Reversion"],
    )

    limit = st.slider("Máximo de runs a listar", 10, 500, 100, 10)

strategy_param = None if strategy_filter == "Todas" else strategy_filter
available = list_runs(limit=limit, strategy=strategy_param)

if len(available) < 2:
    callout(
        "Necesitas al menos <strong>2 backtests guardados</strong> para comparar.<br><br>"
        "Ve a <strong>🧪 Backtest</strong>, ejecuta varios con distintos "
        "parámetros y guárdalos en el histórico.",
        variant="info",
    )
    footer()
    st.stop()

# --- Formatear opciones del multiselect ---
options_labels = {}
for entry_opt in available:
    assert entry_opt.id is not None, "entry_opt viene de list_runs(), siempre tiene id"
    label = (
        f"#{entry_opt.id} · {entry_opt.strategy} · {', '.join(entry_opt.tickers)} · "
        f"{entry_opt.created_at[:10]} · Sharpe={entry_opt.metrics.get('sharpe', float('nan')):.2f}"
    )
    options_labels[label] = entry_opt.id

with st.sidebar:
    st.markdown("---")
    st.markdown("## 📌 Runs a comparar")

    selected_labels = st.multiselect(
        "Selecciona 2 o más",
        list(options_labels.keys()),
        default=list(options_labels.keys())[:min(3, len(options_labels))],
        help="Elige entre 2 y 6 runs para una comparación legible.",
    )

if len(selected_labels) < 2:
    callout(
        "Selecciona al menos <strong>2 runs</strong> en la barra lateral.",
        variant="warning",
    )
    footer()
    st.stop()

if len(selected_labels) > 6:
    callout(
        "Has seleccionado más de 6 runs. La tabla puede volverse difícil "
        "de leer. Considera reducir la selección.",
        variant="warning",
    )

selected_ids = [options_labels[label] for label in selected_labels]

# ============================================================
#  Ejecutar comparación
# ============================================================
try:
    comparison = compare_runs(selected_ids)
except ValueError as e:
    callout(f"Error en la comparación: {e}", variant="danger")
    st.stop()

# ============================================================
#  Resumen de metadatos
# ============================================================
section("📌 Runs seleccionados")

summary_rows = []
for entry_row in comparison.entries:
    summary_rows.append({
        "ID": entry_row.id,
        "Estrategia": entry_row.strategy,
        "Tickers": ", ".join(entry_row.tickers),
        "Rango": f"{entry_row.start_date} → {entry_row.end_date}",
        "Creado": entry_row.created_at[:19].replace("T", " "),
        "Notas": entry_row.notes or "",
    })
st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
#  Tabla de métricas
# ============================================================
section("📊 Comparación de métricas")

# Formateo: porcentajes donde aplique
metrics_df = comparison.metrics_table.copy()

PCT_KEYS = {"total_return", "annual_return", "max_drawdown",
            "win_rate", "exposure", "annual_volatility"}

def _fmt_metric(val, metric_name: str) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v):
        return "—"
    if metric_name in PCT_KEYS:
        return f"{v:.2%}"
    return f"{v:.4f}"

# Renombrar columnas para claridad
metrics_df.columns = [f"#{cid}" for cid in metrics_df.columns]

# Añadir columnas "Mejor" y "Peor"
def _best_worst(row_name: str) -> tuple[str, str]:
    best = comparison.best_per_metric.get(row_name)
    worst = comparison.worst_per_metric.get(row_name)
    return (f"#{best}" if best else "—", f"#{worst}" if worst else "—")

metrics_df["Mejor"] = [_best_worst(m)[0] for m in metrics_df.index]
metrics_df["Peor"] = [_best_worst(m)[1] for m in metrics_df.index]

# Formatear valores
formatted = metrics_df.copy()
for col in [c for c in metrics_df.columns if c.startswith("#")]:
    formatted[col] = [
        _fmt_metric(metrics_df.at[idx, col], idx)
        for idx in metrics_df.index
    ]

st.dataframe(formatted, use_container_width=True)

callout(
    "Las columnas <strong>Mejor</strong> y <strong>Peor</strong> indican "
    "qué run tiene el mejor/peor valor en cada métrica, considerando la "
    "dirección correcta (Sharpe alto es bueno, Max DD menos negativo es "
    "mejor, turnover bajo es mejor, etc.).",
    variant="info",
)

# ============================================================
#  Tabla de parámetros
# ============================================================
section("⚙️ Comparación de parámetros")

params_df = comparison.params_table.copy()
params_df.columns = [f"#{cid}" for cid in params_df.columns]

# Convertir todo a string para evitar problemas de formato
params_df = params_df.astype(object).where(pd.notna(params_df), "—")
params_df = params_df.applymap(lambda x: str(x) if x != "—" else "—")

st.dataframe(params_df, use_container_width=True)

# ============================================================
#  Gráficos comparativos
# ============================================================
section("📈 Gráficos comparativos")

# --- Selección de métricas para el gráfico ---
METRIC_OPTIONS = [
    "sharpe", "sortino", "calmar", "total_return", "annual_return",
    "max_drawdown", "win_rate", "profit_factor", "turnover",
]

chart_metrics = st.multiselect(
    "Métricas a graficar (máx. 4 para legibilidad)",
    METRIC_OPTIONS,
    default=["sharpe", "max_drawdown", "total_return"],
)

if chart_metrics:
    if len(chart_metrics) > 4:
        callout(
            "Has seleccionado más de 4 métricas. El gráfico puede verse "
            "recargado.",
            variant="warning",
        )

    # --- Bar chart agrupado por métrica ---
    fig = go.Figure()
    for metric in chart_metrics:
        if metric not in comparison.metrics_table.index:
            continue
        row = comparison.metrics_table.loc[metric]
        fig.add_trace(go.Bar(
            name=metric,
            x=[f"#{cid}" for cid in row.index],
            y=row.values,
            text=[_fmt_metric(v, metric) for v in row.values],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        title="Comparación por métrica",
        template="plotly_white",
        height=450,
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 60, "r": 30, "t": 60, "b": 80},
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Radar chart (opcional) ---
if st.checkbox("Mostrar radar chart (normalizado 0-100)"):
    RADAR_METRICS = [
        "sharpe", "sortino", "calmar", "total_return",
        "win_rate", "profit_factor",
    ]
    present = [m for m in RADAR_METRICS if m in comparison.metrics_table.index]

    if len(present) >= 3:
        fig = go.Figure()
        for cid in comparison.metrics_table.columns:
            values = []
            for m in present:
                row = comparison.metrics_table.loc[m].dropna()
                if len(row) < 2:
                    values.append(50.0)
                    continue
                # Normalizar 0-100 en base a min/max de la fila
                vmin, vmax = row.min(), row.max()
                if vmax == vmin:
                    values.append(50.0)
                else:
                    # Cuanto más alto mejor en la mayoría; para otros
                    # invertimos
                    raw = comparison.metrics_table.at[m, cid]
                    if m in LOWER_IS_BETTER:
                        norm = 100 * (vmax - raw) / (vmax - vmin)
                    else:
                        norm = 100 * (raw - vmin) / (vmax - vmin)
                    values.append(float(norm))
            values.append(values[0])  # cerrar el polígono

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=present + [present[0]],
                fill="toself",
                name=f"#{cid}",
            ))

        fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            title="Radar normalizado (0-100, mejor = más exterior)",
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        callout(
            "Se necesitan al menos 3 métricas presentes para el radar.",
            variant="info",
        )

# ============================================================
#  Ranking agregado
# ============================================================
section("🏆 Ranking agregado")

st.caption(
    "Score compuesto a partir de las métricas clave, normalizado 0-100. "
    "No es una recomendación — sirve solo como resumen comparativo."
)

RANKING_METRICS = {
    "sharpe": "higher",
    "sortino": "higher",
    "calmar": "higher",
    "total_return": "higher",
    "max_drawdown": "higher",
    "win_rate": "higher",
    "profit_factor": "higher",
    "turnover": "lower",
}

ranking_rows = []
for cid in comparison.metrics_table.columns:
    score = 0.0
    n_used = 0
    for metric, direction in RANKING_METRICS.items():
        if metric not in comparison.metrics_table.index:
            continue
        row = comparison.metrics_table.loc[metric].dropna()
        if len(row) < 2 or cid not in row.index:
            continue
        vmin, vmax = row.min(), row.max()
        if vmax == vmin:
            norm = 50.0
        else:
            raw = comparison.metrics_table.at[metric, cid]
            if direction == "lower":
                norm = 100 * (vmax - raw) / (vmax - vmin)
            else:
                norm = 100 * (raw - vmin) / (vmax - vmin)
        score += float(norm)
        n_used += 1

    avg = score / n_used if n_used else 0.0
    entry = next(e for e in comparison.entries if e.id == cid)
    ranking_rows.append({
        "ID": f"#{cid}",
        "Estrategia": entry.strategy,
        "Tickers": ", ".join(entry.tickers),
        "Score (0-100)": round(avg, 1),
        "Nº métricas usadas": n_used,
    })

ranking_df = pd.DataFrame(ranking_rows).sort_values(
    "Score (0-100)", ascending=False
).reset_index(drop=True)

st.dataframe(ranking_df, use_container_width=True, hide_index=True)

# ============================================================
#  Exportar
# ============================================================
section("⬇️ Exportar comparación")

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.download_button(
        "Métricas (CSV)",
        formatted.to_csv().encode("utf-8"),
        file_name="comparison_metrics.csv",
        use_container_width=True,
    )

with col_e2:
    st.download_button(
        "Ranking (CSV)",
        ranking_df.to_csv(index=False).encode("utf-8"),
        file_name="comparison_ranking.csv",
        use_container_width=True,
    )

footer()
