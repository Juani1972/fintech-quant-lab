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
# # Pairs Trading: cointegración → señales → backtest
#
# Flujo completo:
#
# 1. Descargar precios de dos activos.
# 2. Test de cointegración (Engle-Granger) y ADF del spread.
# 3. Construir el z-score causal.
# 4. Generar señales de entrada/salida.
# 5. Ejecutar el backtest con comisión y slippage (`mode="absolute"`
#    porque operamos sobre un spread que puede cruzar cero).
# 6. Comparar con buy & hold.

# %%
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np

from app.core.backtest import buy_and_hold, compare_to_benchmark, run_backtest
from app.core.cointegration import (
    engle_granger,
    generate_signals,
    half_life,
    rolling_zscore,
)
from app.core.data_loader import load_prices

# %% [markdown]
# ## 1. Datos

# %%
tickers = ["KO", "PEP"]
end = date.today()
start = end - timedelta(days=5 * 365)

prices = load_prices(tickers, start, end)
print(prices.tail())

# %% [markdown]
# ## 2. Test de cointegración

# %%
result = engle_granger(prices["KO"], prices["PEP"])

print(f"p-valor Engle-Granger:  {result.pvalue:.4f}")
print(f"ADF spread (p-valor):   {result.adf_spread_pvalue:.4f}")
print(f"Alpha:                  {result.alpha:.4f}")
print(f"Beta (hedge ratio):     {result.beta:.4f}")
print(f"Cointegradas:           {result.is_cointegrated}")

hl = half_life(result.spread)
if np.isfinite(hl):
    print(f"Half-life:              {hl:.1f} días")
else:
    print("Half-life:              inf (no revierte)")

# %% [markdown]
# ## 3. Z-score y señales

# %%
z = rolling_zscore(result.spread, window=60)
signals = generate_signals(z, entry=2.0, exit_=0.5)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(result.spread.index, result.spread.values, label="Spread")
axes[0].set_title("Spread (KO - α - β·PEP)")
axes[0].legend()

axes[1].plot(z.index, z.values, label="Z-score", color="steelblue")
axes[1].axhline(2, color="r", linestyle="--")
axes[1].axhline(-2, color="g", linestyle="--")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_title("Z-score del spread")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Backtest sobre el spread
#
# **Nota**: `mode="absolute"` es necesario porque el spread de una
# cointegración oscila alrededor de cero y puede tomar valores negativos.
# El modo `"percent"` (default) exige precios estrictamente positivos y
# falla con cualquier spread realista.

# %%
bt = run_backtest(
    prices=result.spread,
    signals=signals,
    initial_capital=100_000,
    commission=0.001,
    slippage=0.0005,
    mode="absolute",
)

print(bt.summary())

# %% [markdown]
# ## 5. Curva de capital vs Buy & Hold

# %%
bh = buy_and_hold(prices["KO"], initial_capital=100_000)
bh = bh.reindex(bt.equity_curve.index).ffill()

comparison = compare_to_benchmark(bt.equity_curve, bh)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(comparison.index, comparison["strategy_norm"],
        label="Pairs Trading", linewidth=2)
ax.plot(comparison.index, comparison["benchmark_norm"],
        label="Buy & Hold KO", linestyle="--")
ax.axhline(1.0, color="gray", linewidth=0.5)
ax.set_title("Curva de capital normalizada")
ax.set_ylabel("Capital (base 1.0)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Tabla de trades

# %%
if len(bt.trades) > 0:
    trades = bt.trades.copy()
    trades["pnl_pct"] = trades["pnl_pct"].map(lambda x: f"{x:.2%}")
    trades["pnl_abs"] = trades["pnl_abs"].map(lambda x: f"{x:,.0f} €")
    print(trades.head(10).to_string(index=False))
else:
    print("Sin operaciones.")
