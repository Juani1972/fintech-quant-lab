"""Página de construcción de carteras (Markowitz, Risk Parity, HRP)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import THEME
from app.core.data_loader import compute_log_returns, load_prices
from app.core.history import init_db, save_run
from app.core.plotting import drawdown_chart
from app.core.portfolio import (
    PortfolioError,
    hrp_weights,
    markowitz_weights,
    rebalance_schedule,
    risk_parity_weights,
)
from app.core.risk import calmar_ratio, drawdown_series, max_drawdown, sharpe_ratio, sortino_ratio
from app.state import ensure_session_initialized, get_global_params
from app.styles import callout, footer, hero, kpi_row, page_setup, section

page_setup("Portfolio", "💼")

hero(
    title="Construcción de Carteras",
    subtitle=(
        "Pesos óptimos de cartera: Hierarchical Risk Parity (López de Prado, "
        "2016), Markowitz media-varianza y Equal Risk Contribution -- sin "
        "cvxpy ni scikit-learn, solo álgebra lineal y descenso cíclico."
    ),
    icon="💼",
)

init_db()
ensure_session_initialized()
tickers, start, end = get_global_params()

if len(tickers) < 2:
    callout(
        "Se necesitan al menos 2 tickers en la barra lateral para construir "
        "una cartera -- con uno solo no hay diversificación que optimizar.",
        variant="warning",
    )
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown("## 💼 Configuración de cartera")
    method = st.selectbox(
        "Método",
        [
            "Hierarchical Risk Parity (HRP)",
            "Risk Parity (ERC)",
            "Markowitz (cartera tangente)",
            "Markowitz (retorno objetivo)",
        ],
        help=(
            "HRP: robusto frente a matrices de covarianza mal condicionadas, "
            "sin necesitar invertir la matriz. Risk Parity: cada activo aporta "
            "el mismo riesgo. Markowitz: óptimo media-varianza clásico (puede "
            "dar posiciones cortas, al no restringir pesos ≥ 0)."
        ),
    )
    target_return_pct = None
    if method == "Markowitz (retorno objetivo)":
        target_return_pct = st.slider("Retorno anual objetivo (%)", 1.0, 60.0, 15.0, 0.5)

    initial_capital = st.number_input(
        "Capital inicial (€)", min_value=1000.0, value=100_000.0, step=1000.0,
    )

    st.markdown("**Rebalanceo**")
    rebalance_mode = st.selectbox(
        "Modo",
        ["Sin rebalanceo (buy & hold)", "Calendario", "Por desviación"],
        help=(
            "Sin rebalanceo: los pesos calculados se dejan derivar libremente "
            "según el rendimiento de cada activo. Calendario: se vuelve al "
            "peso objetivo cada N periodos. Por desviación: se rebalancea "
            "solo cuando algún peso se aleja demasiado del objetivo."
        ),
    )
    rebalance_freq = None
    rebalance_threshold = None
    if rebalance_mode == "Calendario":
        rebalance_freq = st.selectbox("Frecuencia", ["M", "Q", "W"], format_func=lambda f: {
            "M": "Mensual", "Q": "Trimestral", "W": "Semanal",
        }[f])
    elif rebalance_mode == "Por desviación":
        rebalance_threshold = st.slider("Desviación máxima tolerada", 0.01, 0.30, 0.05, 0.01)

    run_clicked = st.button("🚀 Calcular cartera", type="primary")

if not run_clicked:
    callout(
        "Configura el método en la barra lateral y pulsa 'Calcular cartera'.",
        variant="info",
    )
    st.stop()

try:
    prices = load_prices(tickers, start, end)
except (ValueError, ConnectionError) as e:
    callout(f"Error al cargar datos: {e}", variant="danger")
    st.stop()

returns = compute_log_returns(prices)

try:
    if method == "Hierarchical Risk Parity (HRP)":
        weights = hrp_weights(returns)
    elif method == "Risk Parity (ERC)":
        weights = risk_parity_weights(returns)
    elif method == "Markowitz (cartera tangente)":
        weights = markowitz_weights(returns)
    else:
        assert target_return_pct is not None
        target_daily = (1 + target_return_pct / 100) ** (1 / 252) - 1
        weights = markowitz_weights(returns, target_return=target_daily)
except PortfolioError as e:
    callout(f"No se pudo calcular la cartera: {e}", variant="danger")
    st.stop()

# --- Pesos ---
section("⚖️ Pesos de la cartera")

has_shorts = (weights < 0).any()
if has_shorts:
    callout(
        "Esta cartera incluye posiciones cortas (peso negativo) -- es un "
        "resultado válido de Markowitz sin restricción de no-negatividad, "
        "no un error. Revisa si es lo que buscabas.",
        variant="warning",
    )

col_chart, col_table = st.columns([2, 1])
with col_chart:
    colors = [THEME["danger"] if w < 0 else THEME["primary"] for w in weights.values]
    fig = go.Figure(go.Bar(
        x=weights.index, y=weights.values, marker_color=colors,
        text=[f"{w:.1%}" for w in weights.values], textposition="outside",
    ))
    fig.update_layout(
        title="Peso por activo", yaxis_tickformat=".0%",
        template="plotly_white", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        weights.rename("Peso").apply(lambda w: f"{w:.2%}"),
        use_container_width=True,
    )
    st.caption(f"Suma de pesos: {weights.sum():.4f}")

# --- Backtest de la cartera, con o sin rebalanceo ---
section("📈 Evolución de la cartera")


def _simulate_drifting_weights(returns: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Pesos día a día dejando que deriven libremente (sin rebalancear
    nunca), partiendo de `target`. Sirve como insumo para detectar
    fechas de rebalanceo por desviación con `rebalance_schedule`."""
    w = target.copy()
    rows = []
    for date in returns.index:
        rows.append(w.copy())
        grown = w * (1 + returns.loc[date])
        total = grown.sum()
        w = grown / total if total != 0 else w
    return pd.DataFrame(rows, index=returns.index)


def _simulate_rebalanced_portfolio(
    returns: pd.DataFrame, target: pd.Series, rebalance_dates: pd.Index,
) -> pd.Series:
    """Retorno diario de la cartera, reseteando a `target` en
    `rebalance_dates` y dejando derivar los pesos el resto de días."""
    w = target.copy()
    daily_returns = []
    for date in returns.index:
        if date in rebalance_dates:
            w = target.copy()
        day_ret = float((w * returns.loc[date]).sum())
        daily_returns.append(day_ret)
        grown = w * (1 + returns.loc[date])
        total = grown.sum()
        w = grown / total if total != 0 else w
    return pd.Series(daily_returns, index=returns.index)


n_rebalances = None
if rebalance_mode == "Sin rebalanceo (buy & hold)":
    callout(
        "Los pesos se calculan una vez con todo el histórico y se dejan "
        "derivar libremente según el rendimiento de cada activo -- sin "
        "volver nunca al peso objetivo.",
        variant="info",
    )
    portfolio_returns = (returns * weights).sum(axis=1)
else:
    if rebalance_mode == "Calendario":
        assert rebalance_freq is not None
        drifting = _simulate_drifting_weights(returns, weights)
        schedule = rebalance_schedule(drifting, method="calendar", frequency=rebalance_freq)
    else:
        assert rebalance_threshold is not None
        drifting = _simulate_drifting_weights(returns, weights)
        schedule = rebalance_schedule(
            drifting, method="threshold", threshold=rebalance_threshold,
        )
    rebalance_dates = schedule.index[schedule["rebalance"]]
    n_rebalances = len(rebalance_dates)
    portfolio_returns = _simulate_rebalanced_portfolio(returns, weights, rebalance_dates)
    callout(
        f"Cartera rebalanceada de vuelta al peso objetivo {n_rebalances} "
        f"veces en el periodo (modo: {rebalance_mode.lower()}).",
        variant="info",
    )

equity = initial_capital * (1 + portfolio_returns).cumprod()
equity.iloc[0] = initial_capital

sharpe = sharpe_ratio(portfolio_returns)
sortino = sortino_ratio(portfolio_returns)
calmar = calmar_ratio(portfolio_returns)
mdd = max_drawdown(equity)

kpi_row([
    ("Retorno total", f"{(equity.iloc[-1] / initial_capital - 1):.2%}"),
    ("Sharpe", f"{sharpe:.2f}"),
    ("Sortino", f"{sortino:.2f}"),
    ("Calmar", f"{calmar:.2f}"),
    ("Max Drawdown", f"{mdd:.2%}"),
])

fig_eq = go.Figure(go.Scatter(
    x=equity.index, y=equity.values, mode="lines",
    line={"color": THEME["primary"], "width": 1.6},
))
fig_eq.update_layout(
    title="Capital de la cartera", yaxis_title="Capital (€)",
    template="plotly_white", height=400,
)
st.plotly_chart(fig_eq, use_container_width=True)

st.plotly_chart(drawdown_chart(drawdown_series(equity)), use_container_width=True)

# --- Guardar en histórico ---
section("💾 Guardar en histórico")
with st.form("save_portfolio_run", clear_on_submit=True):
    notes = st.text_input("Notas (opcional)")
    if st.form_submit_button("💾 Guardar esta cartera"):
        run_id = save_run(
            strategy=f"Portfolio — {method}",
            tickers=list(tickers),
            start_date=str(start),
            end_date=str(end),
            params={
                "method": method, "weights": weights.round(4).to_dict(),
                "rebalance_mode": rebalance_mode,
                "n_rebalances": n_rebalances,
            },
            metrics={
                "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
                "max_drawdown": mdd,
                "total_return": float(equity.iloc[-1] / initial_capital - 1),
            },
            notes=notes or None,
        )
        st.success(f"Guardado como entrada #{run_id} -- consúltala en 📚 Histórico.")

footer()
