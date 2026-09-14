# 📈 Fintech Quant Lab

Aplicación visual (GUI) para análisis cuantitativo de series temporales financieras.
Construida con **Streamlit** + **statsmodels** + **arch** + **yfinance**.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Características

- **Modelado GARCH / EGARCH / GJR-GARCH** de volatilidad condicional.
- **Cointegración y Pairs Trading** (Engle-Granger, ADF, z-score, half-life).
- **Regresión Fama-French** (3 y 5 factores) con interpretación de alpha.
- **Medición de riesgo**: VaR, Expected Shortfall, drawdown, cambios de régimen.
- **Interfaz multipágina** con gráficos interactivos y exportación a CSV.
- **Caché de datos** para acelerar consultas repetidas.

## 🚀 Instalación rápida

### Opción 1: Local (pip)

```bash
git clone https://github.com/Juani1972/fintech-quant-lab.git
cd fintech-quant-lab
python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

### Opción 2: Docker

```bash
docker-compose up --build
```

Accede en tu navegador: **http://localhost:8501**

## 🐳 Docker

```bash
docker build -t fintech-quant-lab .
docker run -p 8501:8501 fintech-quant-lab
```

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📖 Uso

1. En la barra lateral introduce los **tickers** (ej: `KO, PEP`), el **rango de fechas** y el **tipo de análisis**.
2. Navega entre páginas con el menú lateral.
3. Ajusta parámetros (ventanas, umbrales, orden del modelo) y pulsa **Ejecutar**.
4. Exporta resultados a CSV desde cada página.

## 🗂️ Estructura

```
app/
├── main.py              # Punto de entrada
├── config.py            # Configuración global
├── core/                # Lógica de negocio (independiente de UI)
│   ├── data_loader.py   # Descarga y caché de datos
│   ├── garch.py         # Modelos de volatilidad
│   ├── cointegration.py # Pairs trading
│   ├── fama_french.py   # Factores
│   ├── risk.py          # VaR, ES, drawdown
│   └── plotting.py      # Gráficos reutilizables
└── pages/               # Páginas Streamlit
```

## 🤝 Contribuir

Lee [CONTRIBUTING.md](CONTRIBUTING.md) antes de abrir un PR.

## 📜 Licencia

MIT © 2026
