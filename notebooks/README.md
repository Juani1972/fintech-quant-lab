# Notebooks de ejemplo

Flujos reproducibles que usan directamente los módulos de `app.core`,
sin necesidad de Streamlit.

## Formato

Los notebooks están escritos en formato **jupytext** (`.py` con celdas `# %%`).
Esto permite:

- Verlos y ejecutarlos como scripts Python normales.
- Convertirlos a `.ipynb` cuando quieras abrirlos con Jupyter.

## Conversión a notebook

```bash
# Instalar jupytext
pip install jupytext

# Convertir todos los notebooks
jupytext --to notebook notebooks/*.py

# O uno solo
jupytext --to notebook notebooks/01_garch_analysis.py
```

Después abre el `.ipynb` con Jupyter:

```bash
jupyter lab notebooks/
```

## Notebooks disponibles

| Notebook | Contenido |
|---|---|
| `01_garch_analysis.py` | GARCH + diagnósticos + pronóstico de volatilidad |
| `02_pairs_trading_backtest.py` | Cointegración + señales + backtest con costes |
| `03_walkforward_robustness.py` | Walk-forward + Monte Carlo + robustness score |

## Ejecución como script

```bash
python notebooks/01_garch_analysis.py
```

Los resultados se imprimen en consola.
