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
