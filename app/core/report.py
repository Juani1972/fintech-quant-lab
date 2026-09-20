"""Generación de informes HTML autocontenidos a partir de resultados.

El HTML incluye:
    - Cabecera con metadatos (estrategia, tickers, fechas, parámetros).
    - Tabla de métricas clave.
    - Gráficos interactivos de Plotly (curva de capital, drawdown).
    - Tabla de operaciones (primeras N).
    - Disclaimer.

El HTML se puede abrir directamente en el navegador y compartir por email.
Plotly se carga desde CDN, así que no requiere conexión para renderizar
más allá de la primera carga.
"""
from __future__ import annotations

import os
from datetime import datetime
from html import escape
from typing import Any

import markdown as _markdown
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from fpdf.fonts import TextStyle
from matplotlib import get_data_path as _mpl_data_path

# ============================================================
#  CSS embebido
# ============================================================
_CSS = """
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    background: #f7f8fb;
    margin: 0;
    padding: 40px 20px;
    line-height: 1.5;
}
.container {
    max-width: 1100px;
    margin: 0 auto;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    padding: 48px 56px;
}
h1 { color: #2563eb; margin: 0 0 8px 0; font-size: 2rem; }
h2 {
    color: #1e293b;
    margin: 36px 0 16px 0;
    font-size: 1.25rem;
    padding-bottom: 8px;
    border-bottom: 2px solid #e2e8f0;
}
.subtitle { color: #64748b; margin: 0 0 32px 0; font-size: 1rem; }
.meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 24px;
}
.meta-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
}
.meta-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.meta-value { font-size: 1rem; font-weight: 600; color: #1e293b; }
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 24px 0;
    font-size: 0.92rem;
}
th {
    text-align: left;
    background: #f1f5f9;
    padding: 10px 14px;
    border-bottom: 2px solid #e2e8f0;
    color: #475569;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
td {
    padding: 9px 14px;
    border-bottom: 1px solid #f1f5f9;
}
tr:hover td { background: #f8fafc; }
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0 28px 0;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e293b;
    margin-top: 4px;
}
.callout {
    padding: 14px 18px;
    border-radius: 8px;
    border-left: 4px solid;
    margin: 16px 0;
    font-size: 0.95rem;
}
.callout-info    { background: #eff6ff; border-color: #2563eb; }
.callout-warning { background: #fffbeb; border-color: #f59e0b; }
.footer {
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #e2e8f0;
    color: #94a3b8;
    font-size: 0.82rem;
    text-align: center;
}
.chart { margin: 8px 0 24px 0; }
"""


# ============================================================
#  Helpers
# ============================================================
def _fig_to_div(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    """Convierte una figura Plotly en un `<div>` HTML.

    Args:
        fig: figura de Plotly.
        include_plotlyjs: si True, embebe el script de plotly.js
            (usar solo en el primer gráfico del informe).

    Returns:
        HTML string listo para insertar.
    """
    return str(fig.to_html(
        full_html=False,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        config={"displayModeBar": False, "responsive": True},
        div_id=None,
    ))


_PCT_KEYS = {
    "total_return", "annual_return", "max_drawdown",
    "win_rate", "exposure",
}


def _format_metric(key: str, value: Any) -> str:
    """Formatea un valor de métrica para mostrar en HTML."""
    if isinstance(value, float):
        if key in _PCT_KEYS:
            return f"{value:.2%}"
        if abs(value) < 1000:
            return f"{value:.4f}"
        return f"{value:,.2f}"
    return escape(str(value))


def _metrics_grid_html(metrics: dict[str, Any], keys: list[str] | None = None) -> str:
    """Genera una grid de tarjetas con las métricas clave."""
    if keys is None:
        keys = [
            "total_return", "annual_return", "sharpe", "sortino",
            "max_drawdown", "calmar", "win_rate", "profit_factor",
            "n_trades", "avg_bars_held", "exposure", "turnover",
        ]
    cards = []
    for k in keys:
        if k not in metrics:
            continue
        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-label">{escape(k)}</div>'
            f'<div class="metric-value">{_format_metric(k, metrics[k])}</div>'
            f'</div>'
        )
    return f'<div class="metrics-grid">{"".join(cards)}</div>'


def _meta_cards_html(items: dict[str, Any]) -> str:
    """Genera las tarjetas de metadatos (tickers, fechas, etc.)."""
    cards = []
    for label, value in items.items():
        cards.append(
            f'<div class="meta-card">'
            f'<div class="meta-label">{escape(str(label))}</div>'
            f'<div class="meta-value">{escape(str(value))}</div>'
            f'</div>'
        )
    return f'<div class="meta">{"".join(cards)}</div>'


def _params_table_html(params: dict[str, Any]) -> str:
    """Tabla con los parámetros usados en el backtest."""
    rows = []
    for k, v in params.items():
        rows.append(
            f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Parámetro</th><th>Valor</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _trades_table_html(trades: pd.DataFrame, max_rows: int = 50) -> str:
    """Tabla con las primeras N operaciones."""
    if trades.empty:
        return "<p>No se generaron operaciones.</p>"

    df = trades.head(max_rows).copy()

    # Formateo de columnas si existen
    if "pnl_pct" in df.columns:
        df["pnl_pct"] = df["pnl_pct"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else "—"
        )
    if "pnl_abs" in df.columns:
        df["pnl_abs"] = df["pnl_abs"].map(
            lambda x: f"{x:,.2f} €" if pd.notna(x) else "—"
        )

    headers = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")

    caption = (
        f"<p style='color:#64748b;font-size:0.85rem;'>"
        f"Mostrando {len(df)} de {len(trades)} operaciones.</p>"
    )
    return (
        f"<table><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{caption}"
    )


def _pct_change(series: pd.Series) -> pd.Series:
    """pct_change robusto que no explota con ceros."""
    return series.pct_change().fillna(0)


# ============================================================
#  Construcción del informe
# ============================================================
def build_backtest_report(
    strategy: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    equity_curve: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    benchmark_equity: pd.Series | None = None,
    notes: str | None = None,
    title: str = "Informe de Backtest",
) -> str:
    """Construye un informe HTML autocontenido de un backtest.

    Args:
        strategy: Nombre de la estrategia.
        tickers: Lista de tickers usados.
        start_date: Fecha de inicio (str).
        end_date: Fecha de fin (str).
        params: Diccionario de parámetros.
        metrics: Diccionario de métricas del backtest.
        equity_curve: Curva de capital.
        positions: Serie de posiciones (-1/0/1).
        trades: DataFrame de trades.
        benchmark_equity: Curva de buy & hold (opcional).
        notes: Notas opcionales del usuario.
        title: Título del informe.

    Returns:
        HTML completo como string.
    """
    # --- Gráfico de curva de capital ---
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode="lines",
        name="Estrategia",
        line={"color": "#2563eb", "width": 2},
    ))
    if benchmark_equity is not None:
        bh = benchmark_equity.reindex(equity_curve.index).ffill()
        fig_equity.add_trace(go.Scatter(
            x=bh.index,
            y=bh.values,
            mode="lines",
            name="Buy & Hold",
            line={"color": "gray", "width": 1.5, "dash": "dash"},
        ))
    fig_equity.update_layout(
        title="Curva de capital",
        yaxis_title="Capital",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )

    # --- Gráfico de drawdown ---
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        line={"color": "crimson"},
        name="Drawdown",
    ))
    fig_dd.update_layout(
        title="Drawdown (%)",
        yaxis_title="%",
        template="plotly_white",
        height=280,
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )

    # --- Gráfico de posiciones ---
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=positions.index,
        y=positions.values,
        mode="lines",
        line={"color": "#16a34a", "width": 1.5, "shape": "hv"},
        name="Posición",
    ))
    fig_pos.update_layout(
        title="Posiciones (-1 short, 0 neutral, 1 long)",
        yaxis_title="Posición",
        template="plotly_white",
        height=220,
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )

    # --- Sección de notas (si las hay) ---
    notes_html = ""
    if notes:
        notes_html = (
            f'<h2>Notas</h2>'
            f'<div class="callout callout-info">{escape(notes)}</div>'
        )

    # --- Timestamp ---
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- HTML final ---
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} — {escape(strategy)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{escape(title)}</h1>
  <p class="subtitle">
    Estrategia: <strong>{escape(strategy)}</strong> ·
    Generado: {escape(ts)}
  </p>

  <h2>Metadatos</h2>
  {_meta_cards_html({
      "Tickers": ", ".join(tickers),
      "Fecha inicio": start_date,
      "Fecha fin": end_date,
      "Nº barras": len(equity_curve),
      "Nº operaciones": metrics.get("n_trades", "—"),
  })}

  <h2>Métricas principales</h2>
  {_metrics_grid_html(metrics)}

  {notes_html}

  <h2>Curva de capital</h2>
  <div class="chart">{_fig_to_div(fig_equity, include_plotlyjs=True)}</div>

  <h2>Drawdown</h2>
  <div class="chart">{_fig_to_div(fig_dd)}</div>

  <h2>Posiciones</h2>
  <div class="chart">{_fig_to_div(fig_pos)}</div>

  <h2>Parámetros</h2>
  {_params_table_html(params)}

  <h2>Operaciones</h2>
  {_trades_table_html(trades)}

  <div class="callout callout-warning">
    <strong>Disclaimer:</strong> Este informe es educativo y de investigación.
    No constituye asesoramiento financiero. Los resultados pasados no
    garantizan resultados futuros. Valida siempre los resultados antes de
    cualquier uso real.
  </div>

  <div class="footer">
    Fintech Quant Lab · v0.3.0 · MIT License · 2026
  </div>
</div>
</body>
</html>"""

    return html


# ============================================================
#  Informe de IA (Gemini) en PDF
# ============================================================
# fpdf2 en vez de un motor HTML->PDF completo (weasyprint, xhtml2pdf):
# el informe de IA es solo texto (sin gráficos Plotly que reproducir),
# y fpdf2 es una dependencia ligera y pura-Python -- coherente con el
# resto de conectores externos del proyecto (ver app/core/ai_report.py).
_PDF_BLUE = (37, 99, 235)  # #2563eb, mismo azul que los <h1> del HTML
_PDF_GRAY = (100, 116, 139)  # #64748b, mismo gris que .subtitle
_PDF_INK = (30, 41, 59)  # #1e293b, mismo color de texto que el HTML
_PDF_AMBER = (146, 64, 14)  # tono del disclaimer (.callout-warning)


def _register_unicode_font(pdf: FPDF) -> str:
    """Registra DejaVu Sans (viene incluida con matplotlib, ya una
    dependencia del proyecto) y devuelve el nombre de familia a usar.

    Los "core fonts" de PDF (helvetica, times...) solo soportan
    latin-1 -- Gemini devuelve con frecuencia rayas largas, comillas
    tipográficas u otros caracteres fuera de ese rango, y el PDF
    fallaba al generarse. DejaVu Sans cubre Unicode con normalidad y
    evita añadir una dependencia nueva solo para tipografías.
    """
    fonts_dir = os.path.join(_mpl_data_path(), "fonts", "ttf")
    pdf.add_font("DejaVu", "", os.path.join(fonts_dir, "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", os.path.join(fonts_dir, "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", os.path.join(fonts_dir, "DejaVuSans-Oblique.ttf"))
    return "DejaVu"


def build_ai_report_pdf(
    title: str,
    meta: dict[str, Any],
    report_text: str,
) -> bytes:
    """Construye un PDF del informe generado con IA (Gemini), con el
    mismo lenguaje visual que los informes HTML (cabecera azul,
    metadatos, disclaimer).

    Args:
        title: título del informe (p.ej. "Informe con IA — GARCH").
        meta: metadatos a mostrar (ticker, parámetros del modelo...).
        report_text: texto en markdown devuelto por Gemini.

    Returns:
        Contenido del PDF como bytes, listo para `st.download_button`.
    """
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    font = _register_unicode_font(pdf)

    pdf.set_font(font, "B", 18)
    pdf.set_text_color(*_PDF_BLUE)
    pdf.multi_cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font(font, "", 9)
    pdf.set_text_color(*_PDF_GRAY)
    pdf.multi_cell(0, 5, f"Generado: {ts}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.set_text_color(*_PDF_INK)
    for label, value in meta.items():
        pdf.set_font(font, "B", 9)
        pdf.write(6, f"{str(label)}: ")
        pdf.set_font(font, "", 9)
        pdf.write(6, f"{str(value)}\n")
    pdf.ln(4)

    pdf.set_font(font, "", 11)
    pdf.set_text_color(*_PDF_INK)
    heading_style = TextStyle(font_family=font, font_style="B", color=_PDF_BLUE)
    pdf.write_html(
        _markdown.markdown(report_text),
        font_family=font,
        li_prefix_color=_PDF_INK,
        tag_styles={
            "h1": heading_style, "h2": heading_style, "h3": heading_style,
            "h4": heading_style, "h5": heading_style, "h6": heading_style,
        },
    )
    pdf.ln(6)

    pdf.set_font(font, "I", 8)
    pdf.set_text_color(*_PDF_AMBER)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: este informe combina cálculos de la app con texto "
        "generado por un modelo de IA (Google Gemini) a partir de esos "
        "cálculos. Es educativo y de investigación, no constituye "
        "asesoramiento financiero. Verifica siempre los datos antes de "
        "cualquier uso real.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    return bytes(pdf.output())


def build_walkforward_report(
    strategy: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    params: dict[str, Any],
    is_metrics: dict[str, Any],
    oos_metrics: dict[str, Any],
    oos_equity: pd.Series,
    n_windows: int,
    notes: str | None = None,
    title: str = "Informe de Walk-Forward",
) -> str:
    """Construye un informe HTML de un walk-forward analysis."""
    # --- Gráfico de curva OOS ---
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(
        x=oos_equity.index,
        y=oos_equity.values,
        mode="lines",
        name="OOS",
        line={"color": "#2563eb", "width": 2},
    ))
    fig_equity.update_layout(
        title="Curva de capital Out-of-Sample (compuesta)",
        yaxis_title="Capital",
        template="plotly_white",
        height=420,
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 60, "b": 40},
    )

    # --- Tabla IS vs OOS ---
    keys = ["total_return", "annual_return", "sharpe", "sortino",
            "max_drawdown", "win_rate"]
    rows = []
    for k in keys:
        is_val = is_metrics.get(k)
        oos_val = oos_metrics.get(k)
        is_str = _format_metric(k, is_val) if is_val is not None else "—"
        oos_str = _format_metric(k, oos_val) if oos_val is not None else "—"
        rows.append(f"<tr><td>{escape(k)}</td><td>{is_str}</td><td>{oos_str}</td></tr>")

    table_is_oos = (
        "<table><thead><tr>"
        "<th>Métrica</th><th>In-Sample</th><th>Out-of-Sample</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    # --- Degradación ---
    is_sharpe = is_metrics.get("sharpe")
    oos_sharpe = oos_metrics.get("sharpe")
    degradation_html = ""
    if is_sharpe and oos_sharpe and is_sharpe != 0:
        deg = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        if deg > 0.5:
            variant = "callout-warning"
            msg = f"Degradación severa: Sharpe cae {deg:.1%} de IS a OOS."
        elif deg > 0.25:
            variant = "callout-warning"
            msg = f"Degradación moderada del Sharpe: {deg:.1%}."
        else:
            variant = "callout-info"
            msg = f"Degradación aceptable: {deg:.1%}."
        degradation_html = f'<div class="callout {variant}">{escape(msg)}</div>'

    notes_html = ""
    if notes:
        notes_html = (
            f'<h2>Notas</h2>'
            f'<div class="callout callout-info">{escape(notes)}</div>'
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} — {escape(strategy)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <h1>{escape(title)}</h1>
  <p class="subtitle">
    Estrategia: <strong>{escape(strategy)}</strong> ·
    Generado: {escape(ts)}
  </p>

  <h2>Metadatos</h2>
  {_meta_cards_html({
      "Tickers": ", ".join(tickers),
      "Fecha inicio": start_date,
      "Fecha fin": end_date,
      "Nº ventanas": n_windows,
  })}

  <h2>Comparación In-Sample vs Out-of-Sample</h2>
  {table_is_oos}
  {degradation_html}

  {notes_html}

  <h2>Curva de capital OOS</h2>
  <div class="chart">{_fig_to_div(fig_equity, include_plotlyjs=True)}</div>

  <h2>Parámetros</h2>
  {_params_table_html(params)}

  <div class="callout callout-warning">
    <strong>Disclaimer:</strong> Este informe es educativo y de investigación.
    No constituye asesoramiento financiero.
  </div>

  <div class="footer">
    Fintech Quant Lab · v0.3.0 · MIT License · 2026
  </div>
</div>
</body>
</html>"""

    return html
