"""Gráficos reutilizables con Plotly."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def line_chart(series: pd.Series, title: str, ylabel: str = "", color: str = "#1f77b4") -> go.Figure:
    """Gráfico de línea simple."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=color)))
    fig.update_layout(title=title, yaxis_title=ylabel, template="plotly_white", height=400)
    return fig


def zscore_chart(zscore: pd.Series, entry: float = 2.0) -> go.Figure:
    """Gráfico de z-score con umbrales."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zscore.index, y=zscore.values, mode="lines", name="Z-score"))
    fig.add_hline(y=entry, line_dash="dash", line_color="red")
    fig.add_hline(y=-entry, line_dash="dash", line_color="green")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(title="Z-score del Spread", template="plotly_white", height=400)
    return fig


def drawdown_chart(dd: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Gráfico de drawdown."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color="crimson")))
    fig.update_layout(title=title, yaxis_title="Drawdown", template="plotly_white", height=400)
    return fig
