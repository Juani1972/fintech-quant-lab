"""Página de histórico de backtests guardados."""
import pandas as pd
import streamlit as st

from app.core.history import (
    clear_all,
    delete_run,
    get_run,
    init_db,
    list_runs,
)
from app.styles import callout, footer, hero, page_setup, section

page_setup("Histórico", "📚")

hero(
    title="Histórico de Backtests",
    subtitle=(
        "Consulta los backtests guardados, compara parámetros y métricas, "
        "y elimina entradas obsoletas."
    ),
    icon="📚",
)

init_db()

# ============================================================
#  Sidebar
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎛️ Filtros")
    strategy_filter = st.selectbox(
        "Estrategia",
        ["Todas", "Pairs Trading", "Momentum", "Mean Reversion"],
    )
    limit = st.slider("Máximo de entradas", 10, 500, 100, 10)

    st.markdown("---")
    st.markdown("## 🧹 Gestión")
    if st.button("🗑️ Borrar todo el histórico", type="secondary",
                 use_container_width=True):
        st.session_state["_confirm_clear"] = True

    if st.session_state.get("_confirm_clear"):
        callout(
            "⚠️ Esto eliminará <strong>todas</strong> las entradas del histórico. "
            "¿Estás seguro?",
            variant="danger",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sí, borrar todo", type="primary",
                         use_container_width=True):
                n = clear_all()
                st.session_state["_confirm_clear"] = False
                st.success(f"Eliminadas {n} entradas.")
                st.rerun()
        with c2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state["_confirm_clear"] = False
                st.rerun()


# ============================================================
#  Listado
# ============================================================
strategy_param = None if strategy_filter == "Todas" else strategy_filter
entries = list_runs(limit=limit, strategy=strategy_param)

if not entries:
    callout(
        "No hay backtests guardados con los filtros actuales.<br><br>"
        "Ve a la página <strong>🧪 Backtest</strong>, ejecuta uno y pulsa "
        "<strong>💾 Guardar en histórico</strong>.",
        variant="info",
    )
    footer()
    st.stop()

section(f"📋 {len(entries)} entrada(s)")

# --- Tabla resumen ---
summary_rows = []
for e in entries:
    summary_rows.append({
        "ID": e.id,
        "Fecha": e.created_at,
        "Estrategia": e.strategy,
        "Tickers": ", ".join(e.tickers),
        "Sharpe": e.metrics.get("sharpe"),
        "Retorno total": e.metrics.get("total_return"),
        "Max DD": e.metrics.get("max_drawdown"),
        "Nº trades": e.metrics.get("n_trades"),
        "Notas": e.notes or "",
    })

df = pd.DataFrame(summary_rows)

# --- Formato de columnas numéricas ---
def _fmt_pct(x):
    try:
        return f"{float(x):.2%}"
    except (TypeError, ValueError):
        return "—"

def _fmt_num(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"

st.dataframe(
    df.style.format({
        "Sharpe": lambda x: _fmt_num(x, 3),
        "Retorno total": _fmt_pct,
        "Max DD": _fmt_pct,
        "Nº trades": lambda x: str(x) if x is not None else "—",
    }),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
#  Detalle de una entrada
# ============================================================
section("🔍 Detalle de una entrada")

available_ids = [e.id for e in entries if e.id is not None]
selected_id = st.selectbox("Selecciona un ID", available_ids)

if selected_id is not None:
    entry = get_run(selected_id)
    if entry is None:
        callout("La entrada seleccionada ya no existe.", variant="warning")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Parámetros**")
        st.json(entry.params)

    with col2:
        st.markdown("**Métricas**")
        st.json(entry.metrics)

    st.markdown(
        f"**Estrategia**: {entry.strategy}  \n"
        f"**Tickers**: {', '.join(entry.tickers)}  \n"
        f"**Rango**: {entry.start_date} → {entry.end_date}  \n"
        f"**Creado**: {entry.created_at}"
    )
    if entry.notes:
        st.markdown(f"**Notas**: {entry.notes}")

    if st.button(f"🗑️ Eliminar entrada #{entry.id}", type="secondary"):
        if delete_run(entry.id):
            st.success(f"Entrada #{entry.id} eliminada.")
            st.rerun()
        else:
            st.error("No se pudo eliminar la entrada.")

# ============================================================
#  Descarga
# ============================================================
section("⬇️ Exportar")
st.download_button(
    "Descargar histórico (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="history.csv",
)

footer()
