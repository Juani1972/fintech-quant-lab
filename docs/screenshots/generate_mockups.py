"""Genera mockups (matplotlib) para docs/screenshots/, a partir de
cálculos reales de app.core sobre datos sintéticos (no capturas de
pantalla reales de Streamlit -- eso requiere un navegador que no está
disponible en este entorno).

Genera:
    dashboard.png    -- maqueta de la portada (hero + KPIs + grid de módulos)
    walkforward.png  -- barras IS vs OOS por ventana + equity OOS concatenada
    robustness.png   -- histograma Monte Carlo + barras de componentes + pesos
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from app.config import PAGES, THEME  # noqa: E402
from app.core.backtest import buy_and_hold, run_backtest  # noqa: E402
from app.core.robustness import (  # noqa: E402
    DEFAULT_ROBUSTNESS_WEIGHTS,
    monte_carlo_bootstrap,
    parameter_sensitivity,
    robustness_score,
)
from app.core.walkforward import (  # noqa: E402
    signal_from_momentum,
    walk_forward_analysis,
)

OUT = REPO / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = THEME["primary"]        # "#2563eb"
PRIMARY_DARK = THEME["primary_dark"]
MUTED = THEME["muted"]            # "#64748b"
SUCCESS = THEME["success"]
WARNING = THEME["warning"]
DANGER = THEME["danger"]
TEXT = THEME["text"]
BORDER = THEME["border"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ============================================================
#  Datos sintéticos (par cointegrado realista, como en el resto de
#  la sesión: sin red disponible aquí para yfinance).
# ============================================================
def synthetic_pair(n: int = 1300, seed: int = 11) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 1, n))
    a = pd.Series(45 + common * 0.3 + rng.normal(0, 0.4, n), index=dates)
    b = pd.Series(150 + common * 0.9 + rng.normal(0, 1.2, n), index=dates)
    return a, b


# ============================================================
#  1. dashboard.png
# ============================================================
def make_dashboard() -> None:
    fig = plt.figure(figsize=(13, 9), dpi=150)
    fig.patch.set_facecolor("#f8fafc")

    # --- Hero ---
    hero_ax = fig.add_axes([0.04, 0.83, 0.92, 0.13])
    hero_ax.axis("off")
    hero_ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.04",
        transform=hero_ax.transAxes, linewidth=0,
        facecolor=PRIMARY,
    ))
    hero_ax.text(0.035, 0.62, "FINTECH QUANT LAB", transform=hero_ax.transAxes,
                 fontsize=22, fontweight="bold", color="white", va="center")
    hero_ax.text(0.035, 0.25, "Plataforma de análisis cuantitativo — GARCH, cointegración,"
                 " backtesting, walk-forward y robustez", transform=hero_ax.transAxes,
                 fontsize=11, color="#dbeafe", va="center")

    # --- KPIs de sesión ---
    kpis = [("TICKERS", "KO, PEP"), ("INICIO", "2021-01-04"),
            ("FIN", "2026-09-15"), ("RANGO", "~1430 días")]
    kpi_width = 0.92 / 4 - 0.012
    for i, (label, value) in enumerate(kpis):
        x = 0.04 + i * (kpi_width + 0.016)
        ax = fig.add_axes([x, 0.71, kpi_width, 0.085])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.06",
            transform=ax.transAxes, linewidth=1, edgecolor=BORDER, facecolor="white",
        ))
        ax.text(0.5, 0.62, label, transform=ax.transAxes, fontsize=8.5,
                color=MUTED, ha="center", fontweight="bold")
        ax.text(0.5, 0.28, value, transform=ax.transAxes, fontsize=13,
                color=TEXT, ha="center", fontweight="bold")

    # --- Título de sección ---
    fig.text(0.04, 0.665, "Módulos disponibles", fontsize=13, fontweight="bold", color=TEXT)
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.652, 0.652], color=BORDER, lw=1,
                               transform=fig.transFigure))

    # --- Grid de 8 módulos (2 columnas x 4 filas) ---
    ncols, nrows = 2, 4
    card_w = 0.92 / ncols - 0.016
    card_h = 0.58 / nrows - 0.014
    icon_colors = [PRIMARY, "#7c3aed", "#0891b2", WARNING,
                   "#059669", "#db2777", "#ea580c", DANGER]
    for i, page in enumerate(PAGES):
        row, col = divmod(i, ncols)
        x = 0.04 + col * (card_w + 0.016)
        y = 0.60 - (row + 1) * (card_h + 0.014) + 0.014
        ax = fig.add_axes([x, y, card_w, card_h])
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.05",
            transform=ax.transAxes, linewidth=1, edgecolor=BORDER, facecolor="white",
        ))
        # "icono" = marcador circular en vez de emoji (DejaVu Sans no
        # tiene glifos de emoji) y en vez de un Circle en coords de eje
        # (saldría ovalado, ya que las tarjetas no son cuadradas) --
        # un marker se mide en puntos físicos, así que es circular
        # siempre, sea cual sea el aspect ratio del eje.
        ax.plot(0.09, 0.72, marker="o", markersize=13, transform=ax.transAxes,
                color=icon_colors[i % len(icon_colors)], markeredgewidth=0)
        ax.text(0.17, 0.72, page["name"], transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=TEXT, va="center")
        desc = page["description"]
        desc = desc if len(desc) <= 78 else desc[:75] + "..."
        # envolver a ~40 caracteres por línea
        words, lines, cur = desc.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 42:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        ax.text(0.09, 0.42, "\n".join(lines[:3]), transform=ax.transAxes, fontsize=8.3,
                color=MUTED, va="top")

    fig.savefig(OUT / "01_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("dashboard.png OK")


# ============================================================
#  2. walkforward.png
# ============================================================
def make_walkforward() -> None:
    a, _b = synthetic_pair()
    wf = walk_forward_analysis(
        a, signal_from_momentum(window=60),
        train_size=400, test_size=100, step=100,
        initial_capital=100_000, commission=0.001, slippage=0.0005,
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), dpi=150, height_ratios=[1, 1.3],
    )
    fig.subplots_adjust(hspace=0.35)

    # --- Panel 1: Sharpe IS vs OOS por ventana ---
    labels = [f"V{i+1}" for i in range(len(wf.windows))]
    is_sharpe = [w.is_metrics.get("sharpe", np.nan) for w in wf.windows]
    oos_sharpe = [w.oos_metrics.get("sharpe", np.nan) for w in wf.windows]
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, is_sharpe, width, label="In-sample", color=MUTED, alpha=0.85)
    ax1.bar(x + width / 2, oos_sharpe, width, label="Out-of-sample", color=PRIMARY)
    ax1.axhline(0, color=BORDER, lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Sharpe")
    ax1.set_title("Walk-Forward: Sharpe por ventana (IS vs OOS)", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Panel 2: equity OOS concatenada vs buy & hold ---
    oos_eq = wf.oos_equity_concat.to_numpy()
    oos_eq_idx = wf.oos_equity_concat.index

    bh_full = buy_and_hold(a, initial_capital=100_000)
    bh_oos = bh_full.reindex(oos_eq_idx).to_numpy()

    ax2.plot(oos_eq_idx[:len(oos_eq)], oos_eq / 1000, color="#2563eb", lw=1.6,
             label="Estrategia (OOS concatenado)")
    ax2.plot(oos_eq_idx[:len(bh_oos)], bh_oos / 1000, color="#64748b", lw=1.2,
             ls="--", label="Buy & hold A (mismos días)")

    ax2.set_ylabel("Capital (miles €)")
    ax2.set_title("Equity curve OOS vs Buy & Hold", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left", frameon=True, facecolor="white",
               edgecolor=BORDER, framealpha=0.92)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()

    fig.savefig(OUT / "07_walkforward.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("walkforward.png OK")


# ============================================================
#  3. robustness.png
# ============================================================
def make_robustness() -> None:
    a, _b = synthetic_pair()
    signals = signal_from_momentum(window=60)(a, a)
    bt = run_backtest(a, signals, commission=0.001, slippage=0.0005)
    is_sharpe = bt.metrics["sharpe"]

    mc = monte_carlo_bootstrap(bt.returns, n_simulations=1000, block_size=5, random_state=42)

    def factory(prices, params):
        return signal_from_momentum(window=params["window"])(prices, prices)

    sens = parameter_sensitivity(
        a, factory, {"window": 60}, param_name="window",
        variations=[30, 45, 60, 75, 90], metric="sharpe",
    )

    report = robustness_score(
        is_sharpe=is_sharpe, oos_sharpe=mc.mean, mc_result=mc,
        sensitivity_score=sens.stability_score, n_trades=len(bt.trades),
    )

    fig = plt.figure(figsize=(11, 8), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.4, wspace=0.3)

    # --- Histograma Monte Carlo ---
    ax_mc = fig.add_subplot(gs[0, :])
    ax_mc.hist(mc.distribution, bins=50, color=PRIMARY, alpha=0.75, edgecolor="white")
    ax_mc.axvline(mc.mean, color=DANGER, lw=1.6, ls="--",
                  label=f"Media = {mc.mean:.2f}")
    ax_mc.axvline(0, color=TEXT, lw=1)
    ax_mc.set_title("Distribución Monte Carlo del Sharpe (bootstrap por bloques)",
                     fontsize=12, fontweight="bold")
    ax_mc.set_xlabel("Sharpe simulado")
    ax_mc.legend(frameon=False, fontsize=9)
    ax_mc.spines[["top", "right"]].set_visible(False)

    # --- Barras de componentes del robustness score ---
    display_names = {
        "degradacion_is_oos": "Degradación IS→OOS",
        "monte_carlo_sharpe": "Sharpe Monte Carlo",
        "estabilidad_parametros": "Estabilidad parámetros",
        "n_trades": "Nº operaciones",
    }
    ax_comp = fig.add_subplot(gs[1, 0])
    comp_names = list(report.components.keys())
    comp_vals = list(report.components.values())
    comp_max = [DEFAULT_ROBUSTNESS_WEIGHTS[k] for k in comp_names]
    y = np.arange(len(comp_names))
    ax_comp.barh(y, comp_max, color=BORDER, height=0.55, label="máximo posible")
    ax_comp.barh(y, comp_vals, color=SUCCESS, height=0.55, label="obtenido")
    ax_comp.set_yticks(y)
    ax_comp.set_yticklabels([display_names.get(n, n) for n in comp_names], fontsize=9)
    ax_comp.set_xlabel("Puntos")
    ax_comp.set_title(f"Robustness score: {report.final_score:.0f}/100",
                       fontsize=11, fontweight="bold")
    ax_comp.legend(frameon=False, fontsize=8, loc="lower right")
    ax_comp.spines[["top", "right"]].set_visible(False)

    # --- Tabla de pesos por defecto ---
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis("off")
    ax_tbl.set_title("Pesos por defecto (parametrizables)", fontsize=11,
                      fontweight="bold", loc="left")
    rows = [[display_names.get(k, k), f"{v:.0f}"]
            for k, v in DEFAULT_ROBUSTNESS_WEIGHTS.items()]
    tbl = ax_tbl.table(cellText=rows, colLabels=["Componente", "Peso"],
                        loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if r == 0:
            cell.set_facecolor(PRIMARY)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#f8fafc")

    fig.savefig(OUT / "09_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("robustness.png OK")


# ============================================================
#  4. garch.png
# ============================================================
def make_garch() -> None:
    from app.core.garch import fit_garch, forecast_volatility, residual_diagnostics

    a, _b = synthetic_pair()
    returns = np.log(a / a.shift(1)).dropna()
    result = fit_garch(returns, vol="Garch", rescale=True)
    diag = residual_diagnostics(result)
    fc = forecast_volatility(result, horizon=30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150,
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    ax1.plot(result.conditional_volatility.index, result.conditional_volatility.values,
              color=PRIMARY, lw=1.1)
    fc_idx = pd.bdate_range(result.conditional_volatility.index[-1], periods=len(fc) + 1)[1:]
    ax1.plot(fc_idx, fc.values, color=DANGER, lw=1.6, ls="--", label="Pronóstico 30d")
    ax1.set_title("Volatilidad condicional GARCH(1,1)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Volatilidad diaria")
    ax1.legend(frameon=False, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.axis("off")
    ax2.set_title("Diagnósticos de residuos", fontsize=12, fontweight="bold", loc="left")
    rows = [
        ["Ljung-Box (resid)", f"{diag['ljung_box_pvalue']:.3f}"],
        ["Ljung-Box (resid²)", f"{diag['ljung_box_squared_pvalue']:.3f}"],
        ["ARCH-LM", f"{diag['arch_lm_pvalue']:.3f}"],
        ["Jarque-Bera", f"{diag['jarque_bera_pvalue']:.3f}"],
        ["Convergió", "Sí" if result.converged else "No"],
        ["AIC / BIC", f"{result.aic:.0f} / {result.bic:.0f}"],
    ]
    tbl = ax2.table(cellText=rows, colLabels=["Test", "p-valor"],
                     loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(PRIMARY if r == 0 else ("white" if r % 2 else "#f8fafc"))
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")

    fig.savefig(OUT / "02_garch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("02_garch.png OK")


# ============================================================
#  5. cointegration.png
# ============================================================
def make_cointegration() -> None:
    from app.core.cointegration import engle_granger, half_life, rolling_zscore

    a, b = synthetic_pair()
    coint = engle_granger(a, b)
    hl = half_life(coint.spread)
    z = rolling_zscore(coint.spread, window=60)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True,
                                    height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.15)

    ax1.plot(coint.spread.index, coint.spread.values, color=PRIMARY, lw=0.9)
    ax1.axhline(coint.spread.mean(), color=MUTED, lw=1, ls="--")
    ax1.set_title(
        f"Spread A/B  (Engle-Granger p={coint.pvalue:.4f}, "
        f"half-life≈{hl:.0f}d)", fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Spread")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(z.index, z.values, color="#7c3aed", lw=0.9)
    ax2.axhline(2.0, color=DANGER, lw=1, ls="--", label="Entrada ±2.0")
    ax2.axhline(-2.0, color=DANGER, lw=1, ls="--")
    ax2.axhline(0.5, color=SUCCESS, lw=1, ls=":", label="Salida ±0.5")
    ax2.axhline(-0.5, color=SUCCESS, lw=1, ls=":")
    ax2.set_title("Z-score causal del spread (ventana 60d)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Z-score")
    ax2.legend(frameon=False, fontsize=9, ncol=2)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()

    fig.savefig(OUT / "03_cointegration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("03_cointegration.png OK")


# ============================================================
#  6. famafrench.png
# ============================================================
def make_famafrench() -> None:
    from app.core.fama_french import run_regression

    a, _b = synthetic_pair()
    rng = np.random.default_rng(3)
    returns = np.log(a / a.shift(1)).dropna()
    n = len(returns)
    dates = returns.index

    mkt_rf = pd.Series(rng.normal(0.0003, 0.008, n), index=dates)
    smb = pd.Series(rng.normal(0.0, 0.004, n), index=dates)
    hml = pd.Series(rng.normal(0.0, 0.004, n), index=dates)
    rf = pd.Series(0.00005, index=dates)
    factors = pd.DataFrame({"Mkt-RF": mkt_rf, "SMB": smb, "HML": hml, "RF": rf})

    res = run_regression(returns, factors, model="3", cov_type="HAC")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150,
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    names = list(res.betas.index)
    vals = res.betas.values
    pvals = res.betas_pvalues.values
    colors = [SUCCESS if p < 0.05 else MUTED for p in pvals]
    ax1.barh(names, vals, color=colors)
    ax1.axvline(0, color=TEXT, lw=1)
    ax1.set_title("Betas (verde = significativo p<0.05)", fontsize=11, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.axis("off")
    ax2.set_title(f"Modelo de 3 factores — R²={res.r_squared:.3f}",
                  fontsize=12, fontweight="bold", loc="left")
    rows = [["Alpha (diario)", f"{res.alpha:.5f}", f"{res.alpha_pvalue:.3f}"]]
    for name in names:
        rows.append([name, f"{res.betas[name]:.3f}", f"{res.betas_pvalues[name]:.3f}"])
    rows.append(["Nº obs.", str(res.n_obs), ""])
    rows.append(["Cov. type", res.cov_type, ""])
    tbl = ax2.table(cellText=rows, colLabels=["Coef.", "Valor", "p-valor"],
                     loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(PRIMARY if r == 0 else ("white" if r % 2 else "#f8fafc"))
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")

    fig.savefig(OUT / "04_famafrench.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("04_famafrench.png OK (factores sintéticos -- run_regression es real)")


# ============================================================
#  7. risk.png
# ============================================================
def make_risk() -> None:
    from app.core.risk import (
        drawdown_series,
        expected_shortfall,
        max_drawdown,
        sharpe_ratio,
        sortino_ratio,
        value_at_risk,
        value_at_risk_cornish_fisher,
        value_at_risk_parametric,
    )

    a, _b = synthetic_pair()
    returns = a.pct_change().dropna()
    dd = drawdown_series(a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150,
                                    gridspec_kw={"width_ratios": [1, 1.2]})

    ax1.fill_between(dd.index, dd.values * 100, 0, color=DANGER, alpha=0.55)
    ax1.set_title(f"Drawdown  (máx={max_drawdown(a):.2%})",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Drawdown (%)")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.axis("off")
    ax2.set_title("Métricas de riesgo (95%)", fontsize=12, fontweight="bold", loc="left")
    rows = [
        ["VaR histórico", f"{value_at_risk(returns):.2%}"],
        ["VaR paramétrico", f"{value_at_risk_parametric(returns):.2%}"],
        ["VaR Cornish-Fisher", f"{value_at_risk_cornish_fisher(returns):.2%}"],
        ["Expected Shortfall", f"{expected_shortfall(returns):.2%}"],
        ["Sharpe", f"{sharpe_ratio(returns):.2f}"],
        ["Sortino", f"{sortino_ratio(returns):.2f}"],
    ]
    tbl = ax2.table(cellText=rows, colLabels=["Métrica", "Valor"],
                     loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(PRIMARY if r == 0 else ("white" if r % 2 else "#f8fafc"))
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")

    fig.savefig(OUT / "05_risk.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("05_risk.png OK")


# ============================================================
#  8. backtest.png
# ============================================================
def make_backtest() -> None:
    from app.core.backtest import benchmark_metrics

    a, _b = synthetic_pair()
    signals = signal_from_momentum(window=45)(a, a)
    bt = run_backtest(a, signals, commission=0.001, slippage=0.0005)
    bh = buy_and_hold(a, initial_capital=100_000)
    bm = benchmark_metrics(bt.equity_curve, bh)

    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax1.plot(bt.equity_curve.index, bt.equity_curve.values / 1000,
              color=PRIMARY, lw=1.4, label="Estrategia (Momentum)")
    ax1.plot(bh.index, bh.values / 1000, color=MUTED, lw=1.1, ls="--",
              label="Buy & Hold")
    ax1.set_ylabel("Capital (miles €)")
    ax1.set_title(
        f"Backtest — Sharpe={bt.metrics['sharpe']:.2f}  "
        f"MaxDD={bt.metrics['max_drawdown']:.1%}  "
        f"Beta={bm['beta']:.2f}  α anual={bm['jensen_alpha']:.1%}",
        fontsize=11.5, fontweight="bold",
    )
    ax1.legend(frameon=True, facecolor="white", edgecolor=BORDER, fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()

    fig.savefig(OUT / "06_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("06_backtest.png OK")


# ============================================================
#  9. optimization.png
# ============================================================
def make_optimization() -> None:
    from app.core.optimization import grid_search, heatmap_data

    a, _b = synthetic_pair()

    def factory(prices, params):
        ret = prices.pct_change(int(params["window"]))
        s = pd.Series(0, index=prices.index, dtype=int)
        thr = params["threshold"] / 100
        s[ret > thr] = 1
        s[ret < -thr] = -1
        return s

    grid = {"window": [20, 30, 45, 60, 75, 90], "threshold": [1, 2, 3, 4]}
    result = grid_search(a, factory, grid, objective="sharpe")
    hm = heatmap_data(result.grid, x_param="window", y_param="threshold", metric="sharpe")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    im = ax.imshow(hm.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(hm.columns)))
    ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(len(hm.index)))
    ax.set_yticklabels(hm.index)
    ax.set_xlabel("window")
    ax.set_ylabel("threshold (%)")
    ax.set_title(
        f"Heatmap de Sharpe — mejor: window={result.best_params['window']}, "
        f"threshold={result.best_params['threshold']}",
        fontsize=11.5, fontweight="bold",
    )
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, label="Sharpe")

    fig.savefig(OUT / "08_optimization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("07_optimization.png OK")


# ============================================================
#  10. historico.png
# ============================================================
def make_historico() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=150)
    ax.axis("off")
    ax.set_title("Histórico de backtests guardados (SQLite)", fontsize=12,
                  fontweight="bold", loc="left", pad=14)

    rows = [
        ["3", "2026-09-14", "Pairs Trading", "KO, PEP", "1.42", "14.2%", "-12.4%", "45"],
        ["2", "2026-09-10", "Momentum", "AAPL, MSFT", "0.87", "9.1%", "-18.6%", "112"],
        ["1", "2026-09-05", "Mean Reversion", "XOM, CVX", "1.15", "11.8%", "-9.3%", "68"],
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=["ID", "Fecha", "Estrategia", "Tickers", "Sharpe",
                   "Retorno anual", "Max DD", "Nº trades"],
        colWidths=[0.05, 0.13, 0.16, 0.14, 0.10, 0.14, 0.11, 0.11],
        loc="center", cellLoc="center", colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.1)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(PRIMARY if r == 0 else ("white" if r % 2 else "#f8fafc"))
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")

    fig.savefig(OUT / "10_historico.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("08_historico.png OK")


if __name__ == "__main__":
    make_dashboard()
    make_garch()
    make_cointegration()
    make_famafrench()
    make_risk()
    make_backtest()
    make_walkforward()
    make_optimization()
    make_robustness()
    make_historico()
