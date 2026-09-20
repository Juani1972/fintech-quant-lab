"""Página de consulta rápida: serie histórica de precios y
rentabilidad de una o varias empresas en el periodo elegido en la
barra lateral -- sin configurar ningún modelo ni parámetro adicional,
solo los tickers y fechas ya globales de la portada.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.data_loader import load_prices
from app.core.plotting import line_chart
from app.state import (
    ensure_session_initialized,
    get_global_params,
    get_global_provider,
    get_global_provider_kwargs,
)
from app.styles import (
    callout,
    conclusion,
    data_preview,
    footer,
    hero,
    page_setup,
    section,
    ticker_badge,
)

page_setup("Rentabilidad", "💹")

hero(
    title="Precio y Rentabilidad",
    subtitle=(
        "La serie histórica de precios y la rentabilidad de tus empresas "
        "en el periodo que elijas en la barra lateral -- al instante, sin "
        "configurar nada más."
    ),
    icon="💹",
)

ensure_session_initialized()
tickers, start, end = get_global_params()
provider = get_global_provider()
provider_kwargs = get_global_provider_kwargs()

if not tickers:
    callout(
        "Introduce al menos un ticker en la barra lateral (\"Tickers\") y "
        "pulsa <strong>✅ Aplicar tickers</strong>.",
        variant="warning",
    )
    st.stop()

ticker_badge(*tickers)

with st.spinner("Descargando precios..."):
    try:
        prices = load_prices(tickers, start, end, provider=provider, provider_kwargs=provider_kwargs)
    except (ValueError, ConnectionError) as e:
        callout(f"Error al cargar datos: {e}", variant="danger")
        st.stop()
    data_preview(prices)

# ============================================================
#  Rentabilidad por empresa
# ============================================================
section("💰 Rentabilidad por empresa")


class _ReturnRow(NamedTuple):
    ticker: str
    first_price: float
    last_price: float
    total_return: float
    annual_return: float
    highest: float
    lowest: float


rows: list[_ReturnRow] = []
for ticker in tickers:
    series = prices[ticker].dropna()
    if len(series) < 2:
        continue
    first_price = float(series.iloc[0])
    last_price = float(series.iloc[-1])
    total_return = last_price / first_price - 1
    n_days = (series.index[-1] - series.index[0]).days
    annual_return = (1 + total_return) ** (365 / n_days) - 1 if n_days > 0 else float("nan")
    rows.append(_ReturnRow(
        ticker=ticker,
        first_price=first_price,
        last_price=last_price,
        total_return=total_return,
        annual_return=annual_return,
        highest=float(series.max()),
        lowest=float(series.min()),
    ))

if not rows:
    callout(
        "No hay suficientes datos en este periodo para calcular la "
        "rentabilidad. Prueba con un rango de fechas más amplio.",
        variant="warning",
    )
    footer()
    st.stop()

summary = pd.DataFrame([{
    "Ticker": r.ticker,
    "Precio inicial": r.first_price,
    "Precio final": r.last_price,
    "Rentabilidad del periodo": r.total_return,
    "Rentabilidad anualizada": r.annual_return,
    "Máximo": r.highest,
    "Mínimo": r.lowest,
} for r in rows])
st.dataframe(
    summary.style.format({
        "Precio inicial": "{:.2f}",
        "Precio final": "{:.2f}",
        "Rentabilidad del periodo": "{:.2%}",
        "Rentabilidad anualizada": "{:.2%}",
        "Máximo": "{:.2f}",
        "Mínimo": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

for r in rows:
    verb = "ganado" if r.total_return >= 0 else "perdido"
    conclusion(
        f"<strong>{r.ticker}</strong> ha {verb} un "
        f"<strong>{abs(r.total_return):.2%}</strong> "
        f"entre el {start} y el {end} (de {r.first_price:.2f} "
        f"a {r.last_price:.2f}).",
        variant="success" if r.total_return >= 0 else "danger",
    )

# ============================================================
#  Evolución del precio
# ============================================================
section("📈 Evolución del precio")
for ticker in tickers:
    st.plotly_chart(
        line_chart(prices[ticker].dropna(), f"Precio — {ticker}"),
        use_container_width=True,
    )

if len(tickers) > 1:
    section("📊 Comparación (todas empiezan en 100)")
    valid_prices = prices[tickers].dropna()
    normalized = (valid_prices / valid_prices.iloc[0]) * 100
    fig = go.Figure()
    for ticker in tickers:
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized[ticker], mode="lines", name=ticker,
        ))
    fig.update_layout(
        title="Evolución comparada",
        yaxis_title="Índice (inicio = 100)",
        template="plotly_white", height=420, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    callout(
        "Todas las líneas empiezan en 100 para poder comparar la "
        "rentabilidad relativa entre empresas, aunque tengan precios muy "
        "distintos entre sí.",
        variant="info",
    )

# ============================================================
#  Exportar
# ============================================================
section("⬇️ Exportar")
st.download_button(
    "Descargar precios (CSV)",
    prices.to_csv().encode("utf-8"),
    file_name="precios_historicos.csv",
)

footer()
