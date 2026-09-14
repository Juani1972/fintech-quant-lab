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
# # Análisis GARCH de volatilidad
#
# Este notebook muestra cómo usar el módulo `app.core.garch` para:
#
# 1. Descargar precios de un activo.
# 2. Ajustar un modelo GARCH(1,1) a los retornos.
# 3. Comprobar la estacionariedad del modelo.
# 4. Ejecutar diagnósticos de residuos (Ljung-Box, ARCH-LM, Jarque-Bera).
# 5. Pronosticar la volatilidad a 30 días.

# %%
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np

from app.core.data_loader import compute_log_returns, load_prices
from app.core.garch import (
    check_stationarity,
    fit_garch,
    forecast_volatility,
    residual_diagnostics,
)

# %% [markdown]
# ## 1. Descarga de datos

# %%
ticker = "AAPL"
end = date.today()
start = end - timedelta(days=3 * 365)

prices = load_prices([ticker], start, end)
returns = compute_log_returns(prices)[ticker]

print(f"Observaciones: {len(returns)}")
print(f"Rango: {returns.index[0].date()} → {returns.index[-1].date()}")
print(f"Volatilidad anualizada: {returns.std() * np.sqrt(252):.2%}")

# %% [markdown]
# ## 2. Ajuste de GARCH(1,1)

# %%
result = fit_garch(returns, p=1, q=1, vol="Garch", dist="t")

print(f"AIC:  {result.aic:.2f}")
print(f"BIC:  {result.bic:.2f}")
print(f"Estacionario: {check_stationarity(result.params, result.model_type)}")
print()
print(result.model_result.summary())

# %% [markdown]
# ## 3. Volatilidad condicional

# %%
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(returns.index, np.sqrt(result.model_result.conditional_volatility ** 2) / 100)
ax.set_title(f"Volatilidad condicional — {ticker}")
ax.set_ylabel("Volatilidad diaria")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Diagnósticos de residuos
#
# - **Ljung-Box (residuos)**: ¿queda autocorrelación?
# - **Ljung-Box (residuos²)**: ¿queda heterocedasticidad?
# - **ARCH-LM**: contraste específico de efectos ARCH remanentes.
# - **Jarque-Bera**: normalidad de los residuos.

# %%
diag = residual_diagnostics(result, lags=10)
for k, v in diag.items():
    print(f"{k:<35} {v:.4f}")

if diag["ljung_box_squared_pvalue"] > 0.05 and diag["arch_lm_pvalue"] > 0.05:
    print("\n✅ No queda evidencia de heterocedasticidad condicional.")
else:
    print("\n⚠️ Queda estructura en los residuos. Prueba otro orden o modelo.")

# %% [markdown]
# ## 5. Pronóstico de volatilidad a 30 días

# %%
fc = forecast_volatility(result, horizon=30)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(fc.values, marker="o")
ax.set_title(f"Pronóstico de volatilidad — {ticker} (30 días)")
ax.set_xlabel("Día")
ax.set_ylabel("Volatilidad pronosticada")
plt.tight_layout()
plt.show()

print(f"Volatilidad esperada día 1:  {fc.iloc[0]:.4f}")
print(f"Volatilidad esperada día 30: {fc.iloc[-1]:.4f}")
