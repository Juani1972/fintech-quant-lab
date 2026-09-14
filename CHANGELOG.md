# Changelog

Todos los cambios notables de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Planned
- Despliegue público en Streamlit Cloud.
- Capturas de pantalla en el README.

## [0.3.0] — 2026-09-14

### Added
- **Motor de backtesting** vectorizado con anti-look-ahead (`app/core/backtest.py`).
- **Walk-forward analysis** para validación IS/OOS (`app/core/walkforward.py`).
- **Optimización de parámetros** con grid search y walk-forward (`app/core/optimization.py`).
- **Análisis de robustez**: Monte Carlo, block bootstrap, sensibilidad
  de parámetros y robustness score 0-100 (`app/core/robustness.py`).
- **Corrección por múltiples tests**: Bonferroni y Benjamini-Hochberg
  (`app/core/multiple_testing.py`).
- Páginas Streamlit: **Backtest**, **Walk-Forward**, **Optimización**, **Robustez**.
- **Notebooks reproducibles** en formato jupytext (`notebooks/`).
- **CI con cobertura** en Python 3.10, 3.11 y 3.12 (`.github/workflows/ci.yml`).
- Plantillas de issues y PRs.
- `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`.

### Changed
- **Fama-French**: uso de factores **diarios** (antes mensuales) para
  evitar pérdida masiva de observaciones al cruzar con retornos diarios.
- **Fama-French**: errores estándar **HAC (Newey-West)** por defecto.
- **GARCH**: `check_stationarity(params, model_type)` con condiciones
  específicas para GARCH, GJR-GARCH y EGARCH.
- **GARCH**: `check_stationarity` devuelve `bool` nativo de Python.
- **Cointegración**: `spread = y - alpha - beta*x` (antes ignoraba alpha).
- **Cointegración**: ADF del spread expuesto en `CointegrationResult`.
- **Z-score**: `shift=1` por defecto para evitar look-ahead.
- **Riesgo**: `sharpe_ratio` y `sortino_ratio` aceptan `periods_per_year`
  para coherencia con el motor de backtest.
- **Riesgo**: validaciones de `confidence`, `periods_per_year` y retornos finitos.
- **Data loader**: `missing_policy` configurable ('ffill', 'drop', 'raise').
- **Data loader**: `data_quality_report()` para diagnóstico.
- **Optimización**: validación correcta de grid vacío.

### Fixed
- **Crítico**: `IndentationError` en `data_loader.py`.
- Tests de estacionariedad GARCH (numpy bool vs bool nativo).
- Tests de VaR con percentil explícito.
- Test de Benjamini-Hochberg sin dependencia de semilla frágil.

### Removed
- Dependencia no usada de `matplotlib` en páginas de optimización/robustez.

## [0.2.0] — 2026-08-01

### Added
- Módulo Fama-French.
- Página de Riesgo con VaR, ES, Sharpe, Sortino.
- Docker y docker-compose.
- Scripts de instalación/lanzamiento (Linux/macOS/Windows).
- Tests unitarios con pytest.

### Changed
- Estructura modular: `app/core` separado de `app/pages`.

## [0.1.0] — 2026-06-01

### Added
- Versión inicial: GARCH, cointegración, Streamlit.
