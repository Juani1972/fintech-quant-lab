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
# # Walk-forward + Monte Carlo + robustness score
#
# Flujo completo para validar si una estrategia generaliza fuera de
# muestra, en vez de quedarnos con un único backtest in-sample:
#
# 1. Descargar precios y definir un generador de señales (momentum).
# 2. Walk-forward analysis: IS vs OOS en ventanas sucesivas.
# 3. Monte Carlo bootstrap sobre los retornos OOS.
# 4. Sensibilidad del Sharpe a la ventana del parámetro.
# 5. Robustness score agregado (0-100).

# %%
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np

from app.core.backtest import run_backtest
from app.core.data_loader import load_prices
from app.core.robustness import (
    monte_carlo_bootstrap,
    parameter_sensitivity,
    robustness_score,
)
from app.core.walkforward import signal_from_momentum, walk_forward_analysis

# %% [markdown]
# ## 1. Datos

# %%
ticker = "SPY"
end = date.today()
start = end - timedelta(days=8 * 365)

prices = load_prices([ticker], start, end)[ticker]
print(prices.tail())

# %% [markdown]
# ## 2. Walk-forward analysis (momentum, ventana=60)
#
# Cada ventana entrena (in-sample) y evalúa después en el tramo
# siguiente que el generador de señales nunca vio (out-of-sample).

# %%
wf = walk_forward_analysis(
    prices,
    signal_generator=signal_from_momentum(window=60),
    train_size=504,   # ~2 años de bolsa
    test_size=126,    # ~6 meses
    initial_capital=100_000,
    commission=0.001,
    slippage=0.0005,
)

print(f"Ventanas evaluadas: {len(wf.windows)}")
print("\nMétricas IS (medias agregadas):")
for k, v in wf.is_metrics_agg.items():
    print(f"  {k:20s} {v:.4f}")

print("\nMétricas OOS (medias agregadas):")
for k, v in wf.oos_metrics_agg.items():
    print(f"  {k:20s} {v:.4f}")

is_sharpe = wf.is_metrics_agg.get("sharpe", np.nan)
oos_sharpe = wf.oos_metrics_agg.get("sharpe", np.nan)
print(f"\nDegradación IS -> OOS: {is_sharpe:.2f} -> {oos_sharpe:.2f}")

# %% [markdown]
# ## 3. Curva de capital OOS concatenada

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(wf.oos_equity_concat.index, wf.oos_equity_concat.values, linewidth=2)
ax.set_title(f"{ticker} — Equity OOS concatenada (walk-forward)")
ax.set_ylabel("Capital (€)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Monte Carlo bootstrap sobre los retornos OOS
#
# Remuestrea los retornos observados para estimar una distribución
# del Sharpe, en vez de fiarnos de un único resultado puntual.

# %%
oos_returns = wf.oos_equity_concat.pct_change().dropna()

mc = monte_carlo_bootstrap(
    oos_returns,
    n_simulations=1000,
    block_size=5,  # bloques de 5 días: preserva algo de autocorrelación
    random_state=42,
)

print(f"Sharpe medio simulado:  {mc.mean:.3f}")
print(f"Desviación:             {mc.std:.3f}")
print(f"VaR 95%:                {mc.var_95:.3f}")
print(f"CVaR 95%:               {mc.cvar_95:.3f}")
print("Percentiles:", {k: round(v, 3) for k, v in mc.percentiles.items()})

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mc.distribution, bins=50, alpha=0.75, color="steelblue")
ax.axvline(mc.mean, color="black", linestyle="--", label=f"Media = {mc.mean:.2f}")
ax.axvline(0, color="red", linewidth=0.8)
ax.set_title("Distribución Monte Carlo del Sharpe (bootstrap por bloques)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Sensibilidad de parámetros
#
# Si el Sharpe cambia bruscamente al mover ligeramente la ventana,
# es una señal de overfitting; una meseta estable es buena señal.


# %%
def factory(series, params):
    signals = signal_from_momentum(window=params["window"])(series, series)
    return signals


base_params = {"window": 60}
variations = [30, 45, 60, 75, 90, 120]

sens = parameter_sensitivity(
    prices, factory, base_params,
    param_name="window", variations=variations,
    metric="sharpe",
)

print(sens.variations.to_string(index=False))
print(f"\nStability score: {sens.stability_score:.3f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sens.variations["value"], sens.variations["sharpe"], marker="o")
ax.axvline(base_params["window"], color="gray", linestyle="--", label="Valor base")
ax.set_xlabel("Ventana (días)")
ax.set_ylabel("Sharpe")
ax.set_title("Sensibilidad del Sharpe a la ventana de momentum")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Robustness score agregado

# %%
bt_full = run_backtest(
    prices, signal_from_momentum(window=60)(prices, prices),
    initial_capital=100_000, commission=0.001, slippage=0.0005,
)

report = robustness_score(
    is_sharpe=is_sharpe,
    oos_sharpe=oos_sharpe,
    mc_result=mc,
    sensitivity_score=sens.stability_score,
    n_trades=len(bt_full.trades),
)

print("Componentes:")
for k, v in report.components.items():
    print(f"  {k:25s} {v:5.1f} / puntos")
print(f"\nScore final: {report.final_score:.1f} / 100")
print(f"Interpretación: {report.interpretation}")
