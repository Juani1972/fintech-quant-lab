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
jupytext --to notebook notebooks/04_full_pipeline.py
