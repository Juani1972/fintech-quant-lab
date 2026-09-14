# 📈 Fintech Quant Lab

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-pytest-blueviolet)

Plataforma de análisis cuantitativo de series temporales financieras:
volatilidad, cointegración, factores, riesgo y **backtesting de estrategias**.

## ✨ Características

- **📈 GARCH / EGARCH / GJR-GARCH**: modelado de volatilidad condicional.
- **🔗 Cointegración y Pairs Trading**: Engle-Granger, ADF, z-score, half-life.
- **📊 Fama-French**: regresión de 3 y 5 factores con interpretación de alpha.
- **⚠️ Riesgo**: VaR, Expected Shortfall, drawdown, Sharpe, Sortino.
- **🧪 Backtesting**: motor vectorizado con anti-look-ahead, costes y slippage.
- **Interfaz multipágina** con gráficos interactivos (Plotly) y exportación CSV.
- **Caché de datos** para acelerar consultas repetidas.

## 🚀 Instalación rápida

### Opción 1: Local (pip)

```bash
git clone https://github.com/Juani1972/fintech-quant-lab.git
cd fintech-quant-lab
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

### Opción 2: Scripts incluidos

```bash
# Linux / macOS
chmod +x install.sh launch.sh
./install.sh
./launch.sh

# Windows
install.bat
launch.bat
```

### Opción 3: Docker

```bash
docker-compose up --build
```

Accede en **http://localhost:8501**.

## 🧪 Tests

```bash
pytest tests/ -v
pytest tests/ --cov=app --cov-report=term-missing   # con cobertura
```

## 🗂️ Estructura

```
app/
├── main.py              # Punto de entrada
├── config.py            # Configuración global
├── core/                # Lógica de negocio (independiente de UI)
│   ├── data_loader.py   # Descarga y caché de datos
│   ├── garch.py         # Modelos de volatilidad
│   ├── cointegration.py # Pairs trading
│   ├── fama_french.py   # Factores Fama-French
│   ├── risk.py          # VaR, ES, drawdown
│   ├── backtest.py      # Motor de backtesting
│   └── plotting.py      # Gráficos reutilizables
└── pages/               # Páginas Streamlit
    ├── 1_📈_GARCH.py
    ├── 2_🔗_Cointegración.py
    ├── 3_📊_Fama_French.py
    ├── 4_⚠️_Riesgo.py
    └── 5_🧪_Backtest.py
```

## 📖 Uso

1. Introduce los **tickers** en la barra lateral (ej: `KO, PEP`).
2. Selecciona el **rango de fechas**.
3. Navega entre páginas con el menú lateral.
4. Ajusta parámetros y pulsa **Ejecutar**.
5. Descarga resultados en CSV desde cada página.

## 🤝 Contribuir

1. Haz fork del repositorio.
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`.
3. Instala dependencias: `pip install -r requirements.txt`.
4. Ejecuta tests: `pytest tests/ -v`.
5. Formatea el código: `ruff check app/ tests/`.
6. Haz commit y push.
7. Abre un Pull Request.

## 📜 Licencia

MIT © 2026 Juan Orozco

## ⚠️ Disclaimer

Los datos provienen de **Yahoo Finance** vía `yfinance` y pueden contener
errores, huecos o ajustes. Este proyecto es educativo; **no constituye
asesoramiento financiero**. Valida siempre los resultados antes de
cualquier uso real.
