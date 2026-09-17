"""Estilos CSS y componentes UI reutilizables.

Uso:
    from app.styles import inject_css, hero, metric_card, section, kpi_row

    inject_css()
    hero("📈 Fintech Quant Lab", "Plataforma de análisis cuantitativo...")
    section("📊 Métricas principales")
    kpi_row([
        ("Sharpe", "1.42", "+0.15"),
        ("Max DD", "-12.3%", None),
    ])
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import THEME


# ============================================================
#  CSS global
# ============================================================
def inject_css() -> None:
    """Inyecta el CSS global de la aplicación.

    Llama a esta función al principio de cada página (main + pages).
    """
    st.markdown(
        f"""
        <style>
        /* ============================================
           Fondo y tipografía
           ============================================ */
        .stApp {{
            background: linear-gradient(180deg, #f7f8fb 0%, #ffffff 500px);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* ============================================
           Sidebar
           ============================================ */
        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {THEME["border"]};
        }}
        section[data-testid="stSidebar"] h2 {{
            font-size: 1.05rem;
            color: {THEME["muted"]};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }}

        /* ============================================
           Métricas
           ============================================ */
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid {THEME["border"]};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: all 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            transform: translateY(-2px);
            border-color: {THEME["primary"]}40;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {THEME["muted"]};
            font-size: 0.78rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        div[data-testid="stMetricValue"] {{
            color: {THEME["text"]};
            font-weight: 600;
            font-size: 1.5rem;
        }}

        /* ============================================
           Botones
           ============================================ */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.15s ease;
            border: 1px solid {THEME["border"]};
        }}
        .stButton > button:hover {{
            border-color: {THEME["primary"]};
            box-shadow: 0 2px 8px {THEME["primary"]}20;
        }}
        .stButton > button[kind="primary"] {{
            background: {THEME["primary"]};
            border-color: {THEME["primary"]};
        }}
        .stButton > button[kind="primary"]:hover {{
            background: {THEME["primary_dark"]};
            border-color: {THEME["primary_dark"]};
        }}

        /* ============================================
           Tabs
           ============================================ */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid {THEME["border"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 8px 18px;
            font-weight: 500;
        }}

        /* ============================================
           DataFrames y expanders
           ============================================ */
        .stDataFrame, div[data-testid="stExpander"] {{
            border-radius: 10px;
            border: 1px solid {THEME["border"]};
            overflow: hidden;
        }}
        div[data-testid="stExpander"] {{
            background: #ffffff;
            margin-bottom: 0.75rem;
        }}

        /* ============================================
           Componentes personalizados
           ============================================ */
        .fql-hero {{
            background: linear-gradient(135deg, {THEME["primary"]} 0%,
                                              {THEME["primary_dark"]} 100%);
            padding: 42px 36px;
            border-radius: 18px;
            color: white;
            margin-bottom: 32px;
            box-shadow: 0 10px 30px {THEME["primary"]}25;
        }}
        .fql-hero h1 {{
            color: white;
            margin: 0 0 10px 0;
            font-size: 2.15rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        .fql-hero p {{
            color: rgba(255,255,255,0.92);
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.5;
            max-width: 780px;
        }}

        .fql-section {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 2rem 0 1rem 0;
            padding-bottom: 8px;
            border-bottom: 2px solid {THEME["border"]};
        }}
        .fql-section h2 {{
            margin: 0;
            font-size: 1.35rem;
            font-weight: 600;
            color: {THEME["text"]};
        }}

        .fql-card {{
            background: #ffffff;
            border: 1px solid {THEME["border"]};
            border-radius: 12px;
            padding: 20px 22px;
            transition: all 0.15s ease;
            height: 100%;
        }}
        .fql-card:hover {{
            border-color: {THEME["primary"]}60;
            box-shadow: 0 6px 20px rgba(0,0,0,0.06);
            transform: translateY(-2px);
        }}
        .fql-card-icon {{
            font-size: 2rem;
            margin-bottom: 10px;
            display: block;
        }}
        .fql-card-title {{
            font-size: 1.05rem;
            font-weight: 600;
            color: {THEME["text"]};
            margin-bottom: 6px;
        }}
        .fql-card-desc {{
            font-size: 0.9rem;
            color: {THEME["muted"]};
            line-height: 1.45;
        }}

        .fql-callout {{
            padding: 14px 18px;
            border-radius: 10px;
            border-left: 4px solid;
            margin: 1rem 0;
            font-size: 0.95rem;
        }}
        .fql-callout-info    {{ background: {THEME["primary"]}10; border-color: {THEME["primary"]}; }}
        .fql-callout-success {{ background: {THEME["success"]}10; border-color: {THEME["success"]}; }}
        .fql-callout-warning {{ background: {THEME["warning"]}15; border-color: {THEME["warning"]}; }}
        .fql-callout-danger  {{ background: {THEME["danger"]}10;  border-color: {THEME["danger"]}; }}

        .fql-footer {{
            text-align: center;
            color: {THEME["muted"]};
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid {THEME["border"]};
        }}

        /* Ocultar footer por defecto de Streamlit */
        footer {{ visibility: hidden; }}
        #MainMenu {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
#  Componentes
# ============================================================
def hero(title: str, subtitle: str, icon: str = "📈") -> None:
    """Cabecera tipo hero con gradiente."""
    st.markdown(
        f"""
        <div class="fql-hero">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    """Encabezado de sección con línea inferior."""
    st.markdown(
        f'<div class="fql-section"><h2>{title}</h2></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str | None = None,
                help_text: str | None = None) -> None:
    """Tarjeta de métrica individual."""
    st.metric(label=label, value=value, delta=delta, help=help_text)


def kpi_row(metrics: Iterable[tuple]) -> None:
    """Fila de KPIs en columnas.

    Args:
        metrics: Iterable de tuplas `(label, value)` o `(label, value, delta)`.
    """
    metrics = list(metrics)
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics, strict=True):
        label, value = m[0], m[1]
        delta = m[2] if len(m) > 2 else None
        with col:
            st.metric(label=label, value=value, delta=delta)


def callout(text: str, variant: str = "info") -> None:
    """Caja de aviso con color según variante.

    Args:
        variant: 'info', 'success', 'warning', 'danger'.
    """
    if variant not in ("info", "success", "warning", "danger"):
        variant = "info"
    st.markdown(
        f'<div class="fql-callout fql-callout-{variant}">{text}</div>',
        unsafe_allow_html=True,
    )


def page_card(icon: str, title: str, description: str) -> None:
    """Tarjeta de navegación para el menú principal."""
    st.markdown(
        f"""
        <div class="fql-card">
            <span class="fql-card-icon">{icon}</span>
            <div class="fql-card-title">{title}</div>
            <div class="fql-card-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def data_preview(prices: pd.DataFrame) -> None:
    """Vista previa de los datos cargados, antes de ejecutar cualquier
    análisis: gráfico rápido de todas las series, estadísticas
    descriptivas y un informe de calidad (huecos, duplicados, retornos
    extremos). Pensado para que el usuario confirme de un vistazo que
    los datos son los que esperaba antes de lanzar un cálculo largo.

    Colapsado por defecto (dentro de un expander) para no ocupar
    espacio en pantallas donde el usuario ya sabe lo que está haciendo.
    """
    from app.core.data_loader import data_quality_report

    with st.expander("👁️ Vista previa de los datos cargados", expanded=False):
        report = data_quality_report(prices)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas", cast(int, report["rows"]))
        c2.metric("Columnas", cast(int, report["columns"]))
        c3.metric("Rango", f"{report['start']} → {report['end']}")
        c4.metric("Huecos rellenados", cast(int, report["missing"]))

        if report["duplicate_dates"]:
            st.warning(f"⚠️ {report['duplicate_dates']} fecha(s) duplicada(s) en el índice.")
        if report["extreme_returns"]:
            st.warning(
                f"⚠️ {report['extreme_returns']} retorno(s) diario(s) "
                "a más de 5 desviaciones típicas de la media -- revisa "
                "si son splits/dividendos mal ajustados o datos erróneos."
            )

        fig = go.Figure()
        for col in prices.columns:
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices[col], mode="lines", name=str(col),
            ))
        fig.update_layout(
            title="Precios cargados", template="plotly_white", height=320,
            margin={"l": 40, "r": 20, "t": 40, "b": 30},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Estadísticas descriptivas:")
        st.dataframe(prices.describe().T, use_container_width=True)

        st.markdown("**🔬 Validación avanzada**")
        st.caption(
            "Outliers y estacionariedad se calculan sobre los RETORNOS, no "
            "sobre los precios en nivel -- una serie de precios con "
            "tendencia marcaría como \"atípico\" cualquier valor alejado de "
            "su media histórica, sin que sea un error de datos real."
        )
        from app.core.data_loader import compute_log_returns, detect_outliers, stationarity_report

        returns = compute_log_returns(prices)
        if returns.empty:
            st.caption("No hay suficientes datos para calcular retornos.")
        else:
            outlier_mask = detect_outliers(returns, method="zscore", threshold=5.0)
            n_outliers = int(outlier_mask.sum().sum())
            if n_outliers:
                st.warning(
                    f"⚠️ {n_outliers} retorno(s) diario(s) marcado(s) como "
                    "atípico(s) (> 5 desviaciones típicas de la media, por "
                    "columna)."
                )
                outlier_dates = outlier_mask[outlier_mask.any(axis=1)]
                st.dataframe(
                    prices.reindex(outlier_dates.index).round(4),
                    use_container_width=True,
                )
            else:
                st.caption("Sin retornos atípicos detectados (z-score > 5).")

            try:
                stat_report = stationarity_report(returns)
                stat_report_display = stat_report.copy()
                stat_report_display["is_stationary"] = stat_report_display["is_stationary"].map(
                    {True: "✅ Sí", False: "❌ No"}
                )
                st.caption("Estacionariedad de los retornos (test ADF, p < 0.05 = estacionario):")
                st.dataframe(stat_report_display, use_container_width=True)
                if (~stat_report["is_stationary"]).any():
                    st.warning(
                        "⚠️ Al menos un ticker tiene retornos no estacionarios "
                        "-- inusual (los retornos financieros normalmente sí "
                        "lo son); revisa si hay tramos con comportamiento muy "
                        "distinto entre sí en la misma serie (p. ej. un "
                        "cambio de negocio, una fusión, o un error de datos)."
                    )
            except Exception as e:  # noqa: BLE001
                st.caption(f"No se pudo calcular la estacionariedad: {e}")


def named_config_manager(current_config: dict, page_key: str) -> dict | None:
    """Guardar/cargar varias configuraciones con nombre, dentro de la
    sesión del propio usuario -- NO en disco del servidor. Cada
    usuario ve solo las suyas y desaparecen al cerrar la sesión; es la
    forma segura de dar esta comodidad en una app que puede tener
    varios usuarios a la vez (guardar en una carpeta compartida del
    servidor, como haría un `os.listdir()` de un directorio local,
    dejaría ver -- y cargar -- las configuraciones de otros usuarios).

    Se usa junto al guardar/cargar por archivo JSON que ya tiene cada
    página (ese sigue siendo la forma de llevarte una configuración
    fuera de la sesión actual, p.ej. a otro ordenador); esto es solo
    para ir y venir rápido entre varias configuraciones sin tener que
    descargar/subir un archivo cada vez.

    Args:
        current_config: dict con los valores actuales de los widgets
            de la página, para ofrecer guardarlos.
        page_key: prefijo único de la página (p.ej. "bt", "garch"),
            para no mezclar los nombres guardados entre páginas.

    Returns:
        El dict de la configuración elegida, si el usuario pulsa
        "Cargar" en este rerun -- la propia página es responsable de
        aplicarlo con la misma validación que ya usa para el archivo
        JSON subido (comprobar que tickers/opciones siguen siendo
        válidos, valores dentro de rango, etc.). None si no se ha
        pulsado nada.
    """
    store_key = f"_named_configs_{page_key}"
    if store_key not in st.session_state:
        st.session_state[store_key] = {}
    store = st.session_state[store_key]

    st.caption(
        "Guarda varias configuraciones con nombre para ir cambiando entre "
        "ellas rápido -- solo dura mientras tengas esta pestaña abierta y "
        "solo tú las ves. Para guardarlas de verdad (o llevártelas a otro "
        "ordenador), usa el JSON de arriba."
    )
    name = st.text_input("Nombre para guardar", key=f"_{page_key}_named_config_input")
    if st.button("💾 Guardar con este nombre", key=f"_{page_key}_named_config_save"):
        clean_name = name.strip()
        if not clean_name:
            st.error("Indica un nombre.")
        else:
            store[clean_name] = current_config
            st.success(f"Guardada como '{clean_name}' (solo en esta sesión).")

    if not store:
        return None

    selected = st.selectbox(
        "Configuraciones guardadas en esta sesión", list(store.keys()),
        key=f"_{page_key}_named_config_select",
    )
    col_load, col_delete = st.columns(2)
    result = None
    with col_load:
        if st.button("📂 Cargar", key=f"_{page_key}_named_config_load"):
            result = store[selected]
    with col_delete:
        if st.button("🗑️ Borrar", key=f"_{page_key}_named_config_delete"):
            del store[selected]
            st.rerun()
    return result


def footer() -> None:
    """Pie de página."""
    st.markdown(
        '<div class="fql-footer">'
        'Fintech Quant Lab · v0.3.0 · MIT License · 2026'
        '</div>',
        unsafe_allow_html=True,
    )


def page_setup(page_title: str, page_icon: str = "📈") -> None:
    """Configuración estándar de página: set_page_config + inyectar CSS."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
