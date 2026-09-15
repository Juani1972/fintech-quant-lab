# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🧪 Pipeline cuantitativo completo
#
# Este notebook encadena **todos los módulos** de `app.core` en un único
# flujo reproducible, sin necesidad de Streamlit:
#
# 1. **Descarga de datos** — `data_loader`
# 2. **GARCH** — volatilidad condicional del activo
# 3. **Cointegración** — búsqueda de par cointegrado y spread
# 4. **Señales** — z-score causal + umbrales
# 5. **Backtest** — motor vectorizado con costes
# 6. **Walk-forward** — validación IS/OOS
# 7. **Monte Carlo** — distribución del Sharpe OOS
# 8. **Robustness score** — agregado 0-100
# 9. **Resumen final** — decisión sobre la estrategia
#
# Este es el tipo de análisis que se haría antes de considerar operar
# cualquier estrategia en real. **No es asesoramiento financiero.**

# %%
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.core.backtest import run_backtest
from app.core.cointegration import (
    engle_granger,
    generate_signals,
    half_life,
    rolling_zscore,
)
from app.core.data_loader import compute_log_returns, load_prices
from app.core.garch import (
    check_stationarity,
    fit_garch,
    forecast_volatility,
    residual_diagnostics,
)
from app.core.robustness import (
    monte_carlo_bootstrap,
    parameter_sensitivity,
    robustness_score,
)
from app.core.walkforward import (
    signal_from_pairs_trading,
    walk_forward_analysis,
)

# %% [markdown]
# ---
# ## 0. Configuración
#
# Todo lo parametrizable está aquí arriba. Cambia estos valores y vuelve
# a ejecutar el notebook.

# %%
TICKER_A = "KO"
TICKER_B = "PEP"
START = date.today() - timedelta(days=5 * 365)
END = date.today()

# Parámetros de la estrategia
WINDOW = 60          # ventana del z-score
ENTRY = 2.0          # umbral de entrada (|z|)
EXIT = 0.5           # umbral de salida (|z|)

# Costes del backtest
INITIAL_CAPITAL = 100_000.0
COMMISSION = 0.001
SLIPPAGE = 0.0005

# Walk-forward
TRAIN_SIZE = 504     # 2 años diarios
TEST_SIZE = 126      # 6 meses

# Monte Carlo
N_SIMS = 1000
BLOCK_SIZE = 1

print(f"Activos:      {TICKER_A} / {TICKER_B}")
print(f"Periodo:      {START} → {END}")
print(f"Ventana:      {WINDOW} días")
print(f"Entrada:      |z| > {ENTRY}")
print(f"Salida:       |z| < {EXIT}")

# %% [markdown]
# ---
# ## 1. Descarga de datos

# %%
prices = load_prices([TICKER_A, TICKER_B], START, END)

print(f"Observaciones: {len(prices)}")
print(f"Rango:         {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"Política NaN:  {prices.attrs.get('missing_policy')}")
prices.tail()

# %% [markdown]
# ---
# ## 2. GARCH — volatilidad condicional
#
# Ajustamos un GARCH(1,1) sobre los retornos de `TICKER_A` y verificamos:
#
# - Estacionariedad del modelo.
# - Diagnósticos de residuos (Ljung-Box, ARCH-LM, Jarque-Bera).
# - Pronóstico de volatilidad a 30 días.

# %%
returns_a = compute_log_returns(prices)[TICKER_A]

garch_result = fit_garch(returns_a, p=1, q=1, vol="Garch", dist="t")

print(f"AIC:           {garch_result.aic:.2f}")
print(f"BIC:           {garch_result.bic:.2f}")
print(f"Estacionario:  {check_stationarity(garch_result.params, garch_result.model_type)}")
print()

diag = residual_diagnostics(garch_result, lags=10)
print("Diagnósticos de residuos (p-valores):")
for k, v in diag.items():
    print(f"  {k:<35} {v:.4f}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

axes[0].plot(returns_a.index, returns_a.values, linewidth=0.8)
axes[0].set_title(f"Retornos diarios — {TICKER_A}")
axes[0].set_ylabel("Retorno log")

axes[1].plot(garch_result.conditional_volatility.index,
             garch_result.conditional_volatility.values,
             color="darkorange", linewidth=1)
axes[1].set_title(f"Volatilidad condicional (GARCH) — {TICKER_A}")
axes[1].set_ylabel("Volatilidad")

plt.tight_layout()
plt.show()

# %%
forecast = forecast_volatility(garch_result, horizon=30)
print(f"Volatilidad esperada día 1:  {forecast.iloc[0]:.4f}")
print(f"Volatilidad esperada día 30: {forecast.iloc[-1]:.4f}")

# %% [markdown]
# ---
# ## 3. Cointegración — buscar par estable
#
# Test de Engle-Granger + ADF del spread. Lo que importa para pairs
# trading es que **el spread sea estacionario**, no los precios
# individuales.

# %%
coint = engle_granger(prices[TICKER_A], prices[TICKER_B])

print(f"p-valor Engle-Granger:  {coint.pvalue:.4f}")
print(f"ADF spread (p-valor):   {coint.adf_spread_pvalue:.4f}")
print(f"ADF spread (estadístico): {coint.adf_spread_stat:.4f}")
print(f"Alpha:                  {coint.alpha:.4f}")
print(f"Beta (hedge ratio):     {coint.beta:.4f}")
print(f"Cointegradas:           {coint.is_cointegrated}")

hl = half_life(coint.spread)
if np.isfinite(hl):
    print(f"Half-life:              {hl:.1f} días")
else:
    print("Half-life:              inf (no revierte)")

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(coint.spread.index, coint.spread.values, linewidth=1)
ax.axhline(coint.spread.mean(), color="red", linestyle="--", linewidth=0.8)
ax.set_title(f"Spread: {TICKER_A} − α − β·{TICKER_B}")
ax.set_ylabel("Spread")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Señales — z-score causal
#
# El z-score usa `shift(1)` para que la media y la desviación se calculen
# con información disponible **antes** de la observación actual. Después
# el motor de backtesting ejecutará en `t+1`.

# %%
z = rolling_zscore(coint.spread, window=WINDOW)
signals = generate_signals(z, entry=ENTRY, exit_=EXIT)

print(f"Observaciones con señal long:  {(signals == 1).sum()}")
print(f"Observaciones con señal short: {(signals == -1).sum()}")
print(f"Observaciones neutral:         {(signals == 0).sum()}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(z.index, z.values, label="Z-score", color="steelblue")
axes[0].axhline(ENTRY, color="r", linestyle="--", linewidth=0.8)
axes[0].axhline(-ENTRY, color="g", linestyle="--", linewidth=0.8)
axes[0].axhline(0, color="gray", linewidth=0.5)
axes[0].set_title("Z-score del spread")
axes[0].legend()

axes[1].plot(signals.index, signals.values, color="purple", linewidth=1)
axes[1].set_title("Señales (-1 short, 0 neutral, 1 long)")
axes[1].set_ylabel("Posición")

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5. Backtest — simulación con costes
#
# Ejecutamos el backtest sobre el spread con las señales generadas.
# El motor aplica `signals.shift(1)` internamente para evitar look-ahead.

# %%
bt = run_backtest(
    prices=coint.spread,
    signals=signals,
    initial_capital=INITIAL_CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
)

print(bt.summary())

# %%
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(bt.equity_curve.index, bt.equity_curve.values,
             linewidth=1.5, label="Estrategia")
axes[0].axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_title("Curva de capital")
axes[0].set_ylabel("Capital (€)")
axes[0].legend()

running_max = bt.equity_curve.cummax()
dd = (bt.equity_curve - running_max) / running_max * 100
axes[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.5)
axes[1].set_title("Drawdown (%)")
axes[1].set_ylabel("%")

plt.tight_layout()
plt.show()

# %%
if len(bt.trades) > 0:
    print(f"Total operaciones: {len(bt.trades)}")
    print()
    print(bt.trades.head(10).to_string(index=False))
else:
    print("Sin operaciones. Prueba a relajar los umbrales o ampliar el rango.")

# %% [markdown]
# ---
# ## 6. Walk-forward — validación IS/OOS
#
# Dividimos los datos en ventanas sucesivas de entrenamiento (IS) y
# prueba (OOS). Si el Sharpe OOS se degrada mucho respecto a IS, la
# estrategia no generaliza.

# %%
generator = signal_from_pairs_trading(
    window=WINDOW, entry=ENTRY, exit_=EXIT,
)

wf = walk_forward_analysis(
    prices=coint.spread,
    signal_generator=generator,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    initial_capital=INITIAL_CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
)

print(f"Ventanas ejecutadas: {wf.params['n_windows']}")

# %%
is_m = wf.is_metrics_agg
oos_m = wf.oos_metrics_agg

comparison = pd.DataFrame({
    "In-Sample": {
        "Retorno total":   f"{is_m.get('total_return', np.nan):.2%}",
        "Sharpe":          f"{is_m.get('sharpe', np.nan):.2f}",
        "Sortino":         f"{is_m.get('sortino', np.nan):.2f}",
        "Max DD":          f"{is_m.get('max_drawdown', np.nan):.2%}",
        "Win rate":        f"{is_m.get('win_rate', np.nan):.1%}",
    },
    "Out-of-Sample": {
        "Retorno total":   f"{oos_m.get('total_return', np.nan):.2%}",
        "Sharpe":          f"{oos_m.get('sharpe', np.nan):.2f}",
        "Sortino":         f"{oos_m.get('sortino', np.nan):.2f}",
        "Max DD":          f"{oos_m.get('max_drawdown', np.nan):.2%}",
        "Win rate":        f"{oos_m.get('win_rate', np.nan):.1%}",
    },
})

comparison

# %%
is_sharpe = is_m.get("sharpe", np.nan)
oos_sharpe = oos_m.get("sharpe", np.nan)

if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) and is_sharpe != 0:
    degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
    print(f"Degradación IS → OOS: {degradation:.1%}")
    if degradation > 0.5:
        print("⚠️  Degradación severa. Alta sospecha de overfitting.")
    elif degradation > 0.25:
        print("⚠️  Degradación moderada. Revisar.")
    else:
        print("✅ Degradación aceptable.")

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(wf.oos_equity_concat.index, wf.oos_equity_concat.values,
        linewidth=1.8, color="#2563eb")
ax.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("Curva de capital Out-of-Sample (compuesta)")
ax.set_ylabel("Capital (€)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 7. Monte Carlo — distribución del Sharpe OOS
#
# Hacemos bootstrap sobre los retornos OOS de la estrategia para
# construir una distribución del Sharpe. Si el percentil 5 es negativo,
# el resultado podría ser fruto del azar.

# %%
oos_returns = wf.oos_equity_concat.pct_change().dropna()

mc = monte_carlo_bootstrap(
    oos_returns,
    n_simulations=N_SIMS,
    block_size=BLOCK_SIZE,
    metric_name="sharpe",
)

print(f"Media:      {mc.mean:.3f}")
print(f"Std:        {mc.std:.3f}")
print(f"Percentil 5:  {mc.percentiles['p05']:.3f}")
print(f"Percentil 95: {mc.percentiles['p95']:.3f}")
print(f"P(Sharpe > 0): {(mc.distribution > 0).mean():.1%}")

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(mc.distribution, bins=50, color="#2563eb", alpha=0.75, edgecolor="white")
ax.axvline(mc.mean, color="green", linestyle="--", linewidth=1.5,
           label=f"Media {mc.mean:.2f}")
ax.axvline(0, color="red", linewidth=1)
ax.set_title("Distribución del Sharpe simulado (bootstrap OOS)")
ax.set_xlabel("Sharpe")
ax.set_ylabel("Frecuencia")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 8. Sensibilidad de parámetros
#
# Variamos el umbral de entrada ±0.5 y vemos cómo cambia el Sharpe.
# Buscamos una **meseta** alrededor del valor base, no un pico aislado.

# %%
variations = [max(0.5, ENTRY - 0.5), max(0.5, ENTRY - 0.25),
              ENTRY, ENTRY + 0.25, ENTRY + 0.5]


def _pairs_factory(prices_series, params):
    z_local = rolling_zscore(prices_series, window=params["window"], shift=1)
    return generate_signals(z_local,
                            entry=params.get("entry", 2.0),
                            exit_=params.get("exit", 0.5))


sens = parameter_sensitivity(
    prices=coint.spread,
    signal_factory=_pairs_factory,
    base_params={"window": WINDOW, "entry": ENTRY, "exit": EXIT},
    param_name="entry",
    variations=variations,
    metric="sharpe",
    initial_capital=INITIAL_CAPITAL,
    commission=COMMISSION,
    slippage=SLIPPAGE,
)

print(f"Estabilidad: {sens.stability_score:.2f} (1 = muy estable)")
print()
print(sens.variations.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sens.variations["value"], sens.variations["sharpe"],
        marker="o", linewidth=2)
ax.axvline(sens.base_value, color="red", linestyle="--", linewidth=1,
           label=f"Base = {sens.base_value}")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_title("Sensibilidad del Sharpe al umbral de entrada")
ax.set_xlabel("Umbral de entrada (|z|)")
ax.set_ylabel("Sharpe")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 9. Robustness score — agregado 0-100

# %%
report = robustness_score(
    is_sharpe=is_sharpe,
    oos_sharpe=oos_sharpe,
    mc_result=mc,
    sensitivity_score=sens.stability_score,
    n_trades=int(oos_m.get("n_trades", 0)),
)

print(f"Score final: {report.final_score:.1f} / 100")
print(f"Interpretación: {report.interpretation}")
print()
print("Componentes:")
for k, v in report.components.items():
    print(f"  {k:<25} {v:>6.1f}")

# %% [markdown]
# ---
# ## 10. Resumen final
#
# Este es el veredicto agregado sobre la estrategia.

# %%
print("=" * 60)
print("  RESUMEN FINAL DEL PIPELINE")
print("=" * 60)
print()
print(f"Par analizado:        {TICKER_A} / {TICKER_B}")
print(f"Periodo:              {START} → {END}")
print(f"Cointegración (EG):   p = {coint.pvalue:.4f} "
      f"({'✅' if coint.is_cointegrated else '❌'})")
print(f"ADF del spread:       p = {coint.adf_spread_pvalue:.4f} "
      f"({'✅' if coint.adf_spread_pvalue < 0.05 else '❌'})")
print(f"Half-life:            {hl:.1f} días" if np.isfinite(hl) else "Half-life: inf")
print()
print(f"Backtest (IS):")
print(f"  Retorno total:      {bt.metrics['total_return']:.2%}")
print(f"  Sharpe:             {bt.metrics['sharpe']:.2f}")
print(f"  Max DD:             {bt.metrics['max_drawdown']:.2%}")
print(f"  Nº operaciones:     {bt.metrics['n_trades']}")
print()
print(f"Walk-forward (OOS):")
print(f"  Sharpe IS:          {is_sharpe:.2f}")
print(f"  Sharpe OOS:         {oos_sharpe:.2f}")
if np.isfinite(is_sharpe) and is_sharpe != 0:
    print(f"  Degradación:        {degradation:.1%}")
print()
print(f"Monte Carlo:")
print(f"  Sharpe medio:       {mc.mean:.2f}")
print(f"  P(Sharpe > 0):      {(mc.distribution > 0).mean():.1%}")
print()
print(f"Sensibilidad:         {sens.stability_score:.2f} (1 = muy estable)")
print()
print(f"ROBUSTNESS SCORE:     {report.final_score:.1f} / 100")
print(f"  → {report.interpretation}")
print("=" * 60)
print()
print("⚠️  Este análisis es educativo. No constituye asesoramiento")
print("    financiero. Valida siempre los resultados antes de operar.")
