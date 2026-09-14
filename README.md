# 📈 Fintech Quant Lab

[![CI](https://github.com/Juani1972/fintech-quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Juani1972/fintech-quant-lab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Juani1972/fintech-quant-lab/branch/main/graph/badge.svg)](https://codecov.io/gh/Juani1972/fintech-quant-lab)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Plataforma de análisis cuantitativo** para series temporales financieras:
volatilidad, cointegración, factores, riesgo, backtesting, walk-forward,
optimización y análisis de robustez.

> **Nota**: Este proyecto es educativo y de investigación. No constituye
> asesoramiento financiero. Ver [Disclaimer](#-disclaimer).

---

## 🚀 Demo

> **Demo pública**: *(en preparación — `fintech-quant-lab.streamlit.app`)*

Mientras se despliega, puedes ejecutarlo localmente con Docker en un comando:

```bash
docker-compose up --build
```

Accede en **http://localhost:8501**.

---

## 📸 Capturas

### Dashboard principal
![Dashboard](docs/screenshots/01_dashboard.png)

### Análisis GARCH con diagnósticos de residuos
![GARCH](docs/screenshots/02_garch.png)

### Cointegración y pairs trading
![Cointegración](docs/screenshots/03_cointegration.png)

### Backtesting con curva de capital
![Backtest](docs/screenshots/04_backtest.png)

### Walk-forward (IS vs OOS)
![Walk-Forward](docs/screenshots/05_walkforward.png)

### Robustness score y Monte Carlo
![Robustez](docs/screenshots/06_robustness.png)

> 📌 Para añadir tus propias capturas, guarda los PNG en
> `docs/screenshots/` y actualiza los enlaces anteriores.

---

## ✨ Características

### Análisis estadístico
- **📈 GARCH / EGARCH / GJR-GARCH**: volatilidad condicional con
  diagnósticos de residuos (Ljung-Box, ARCH-LM, Jarque-Bera) y
  condiciones de estacionariedad por modelo.
- **🔗 Cointegración**: Engle-Granger, ADF del spread, half-life, z-score
  causal. **Corrección por múltiples tests** (Bonferroni, Benjamini-Hochberg).
- **📊 Fama-French**: regresión de 3 y 5 factores con **errores estándar
  HAC (Newey-West)** y interpretación de alpha.

### Backtesting y validación
- **🧪 Backtesting**: motor vectorizado con anti-look-ahead, comisión,
  slippage, benchmark y métricas completas.
- **🔬 Walk-forward**: validación IS/OOS por ventanas sucesivas.
- **🎯 Optimización**: grid search evaluado con walk-forward (evita overfitting).
- **🛡️ Robustez**: Monte Carlo, block bootstrap, sensibilidad de parámetros,
  robustness score 0-100.

### Interfaz
- **Streamlit multipágina** con gráficos interactivos (Plotly).
- **Exportación CSV** en cada página.
- **Caché de datos** con política de NaNs configurable.

---

## 🧪 Tests y cobertura

```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html   # reporte HTML en htmlcov/
```

La cobertura mínima exigida es **60%**. El CI ejecuta los tests en
Python 3.10, 3.11 y 3.12.

---

## 🗂️ Estructura

```
fintech-quant-lab/
├── app/
│   ├── main.py                 # Punto de entrada
│   ├── config.py               # Configuración global
│   ├── core/                   # Lógica de negocio (independiente de UI)
│   │   ├── data_loader.py      # Descarga y calidad de datos
│   │   ├── garch.py            # Modelos de volatilidad
│   │   ├── cointegration.py    # Pairs trading
│   │   ├── fama_french.py      # Factores con HAC
│   │   ├── risk.py             # VaR, ES, drawdown, ratios
│   │   ├── backtest.py         # Motor de backtesting
│   │   ├── walkforward.py      # Walk-forward analysis
│   │   ├── optimization.py     # Grid search + WF
│   │   ├── robustness.py       # Monte Carlo, bootstrap, score
│   │   ├── multiple_testing.py # Bonferroni, BH
│   │   └── plotting.py         # Gráficos reutilizables
│   └── pages/                  # 8 páginas Streamlit
├── notebooks/                  # Notebooks de ejemplo
├── tests/                      # Tests unitarios
├── docs/screenshots/           # Capturas para el README
├── .github/workflows/          # CI/CD
├── Dockerfile
└── docker-compose.yml
```

---

## 🛠️ Instalación

### Opción 1: pip (local)

```bash
git clone https://github.com/Juani1972/fintech-quant-lab.git
cd fintech-quant-lab
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

### Opción 2: scripts incluidos

```bash
# Linux / macOS
chmod +x install.sh launch.sh
./install.sh && ./launch.sh

# Windows
install.bat
launch.bat
```

### Opción 3: Docker

```bash
docker-compose up --build
```

### Opción 4: desarrollo (con tests y notebooks)

```bash
pip install -e ".[dev]"
pytest tests/ -v
jupyter lab notebooks/
```

---

## 📓 Notebooks de ejemplo

En `notebooks/` encontrarás tres flujos reproducibles que usan directamente
los módulos de `app.core` sin Streamlit:

1. **`01_garch_analysis.py`** — GARCH con diagnósticos y pronóstico.
2. **`02_pairs_trading_backtest.py`** — Cointegración + señales + backtest.
3. **`03_walkforward_robustness.py`** — Walk-forward + Monte Carlo.

Se pueden abrir como notebooks con:

```bash
jupytext --to notebook notebooks/01_garch_analysis.py
```

---

## 📖 Uso

1. Introduce los **tickers** en la barra lateral (ej: `KO, PEP`).
2. Selecciona el **rango de fechas**.
3. Navega entre páginas:
   - **📈 GARCH** — modelado de volatilidad.
   - **🔗 Cointegración** — pairs trading + multiple testing.
   - **📊 Fama-French** — regresión de factores.
   - **⚠️ Riesgo** — VaR, ES, drawdown, ratios.
   - **🧪 Backtest** — backtesting de estrategias.
   - **🔬 Walk-Forward** — validación IS/OOS.
   - **🎯 Optimización** — grid search con WF.
   - **🛡️ Robustez** — Monte Carlo + robustness score.
4. Ajusta parámetros y pulsa **Ejecutar**.
5. Descarga resultados en CSV.

---

## 🤝 Contribuir

Lee [CONTRIBUTING.md](CONTRIBUTING.md) y el [Código de Conducta](CODE_OF_CONDUCT.md).
Los PRs son bienvenidos.

---

## 🔒 Seguridad

Para reportar vulnerabilidades, consulta [SECURITY.md](SECURITY.md).

---

## 📜 Licencia

MIT © 2026 Juan Orozco — ver [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

Los datos provienen de **Yahoo Finance** vía `yfinance` y pueden contener
errores, huecos o ajustes. Este proyecto es educativo; **no constituye
asesoramiento financiero**. Valida siempre los resultados antes de
cualquier uso real.

**Ninguna estrategia aquí implementada debe considerarse rentable ni
adecuada para operar en mercados reales sin una validación exhaustiva
independiente.**

---

## 📚 Referencias

- **GARCH**: Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*.
- **Cointegración**: Engle & Granger (1987), *Co-integration and Error Correction*.
- **Fama-French**: Fama & French (1993, 2015), *Common risk factors*.
- **Newey-West**: Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*.
- **Benjamini-Hochberg**: Benjamini & Hochberg (1995), *Controlling the False Discovery Rate*.
