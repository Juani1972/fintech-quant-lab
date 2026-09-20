# Changelog

Todos los cambios notables de este proyecto se documentan aquí.

El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/)
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added
- **Interpretación automática de resultados (sin IA)**: las 8 páginas
  del núcleo cuantitativo (GARCH, Cointegración, Fama-French, Riesgo,
  Backtest, Walk-Forward, Optimización, Robustez) muestran ahora una
  sección "📝 Interpretación" con reglas fijas que traducen cada
  métrica a español llano -- instantánea, gratuita y sin depender de
  ningún proveedor de IA (`app/core/interpretation/`). La ampliación
  narrativa con IA (Gemini/Groq) pasa a ser una capa opcional encima
  de esa base, con descarga en PDF de lo que haya en pantalla
  (`app/report_ui.py`, `build_interpretation_pdf` en
  `app/core/report.py`).
- **Informes con IA (Google Gemini)**: nuevas páginas GARCH y Backtest
  pueden generar un informe interpretando los resultados con Gemini
  (`app/core/ai_report.py`), con gestión de credenciales en cascada
  (`app/credentials.py`).
- **Exportar informe de IA a PDF**: junto al informe generado con
  Gemini (GARCH, Backtest), botón para descargarlo en PDF con el
  mismo lenguaje visual que los informes HTML existentes
  (`build_ai_report_pdf` en `app/core/report.py`).
- **Ver modelos disponibles**: botón en la barra lateral que consulta
  a la API del proveedor de IA elegido qué modelos admite la clave
  ahora mismo -- los modelos gratuitos cambian con el tiempo (Google
  retira los suyos de vez en cuando) y así no hace falta esperar a
  una actualización del código para saber cuál usar.
- **Groq como alternativa a Gemini**: segundo proveedor de IA para
  los informes (`app/core/groq_report.py`), con la misma clave
  gratuita sin tarjeta y el mismo flujo que Gemini. Selector de
  proveedor en la barra lateral ('🤖 Informes con IA'); cada uno
  guarda su propia clave y modelo por separado, así que cambiar de
  uno a otro no pierde la configuración del primero.
- **Reintentos automáticos en Gemini/Groq**: los errores 500/503
  (sobrecarga temporal del modelo gratuito) se reintentan solos con
  espera creciente (2s/4s/8s) antes de propagar el error -- suelen
  resolverse sin intervención del usuario.

### Fixed
- **Sidebar**: elegir un universo predefinido o una empresa por el
  buscador actualizaba `global_tickers` pero no el campo del
  formulario de tickers -- al pulsar "✅ Aplicar tickers" ese campo,
  con el valor viejo, sobrescribía el cambio recién hecho y revertía
  la selección.
- **Informe con IA**: la clave de Gemini pegada en la portada no se
  veía desde GARCH/Backtest (Streamlit borra el session_state de un
  widget en cualquier página donde no se vuelva a crear), y pulsar
  "Generar informe con IA" reiniciaba toda la sección de resultados
  de la página (el botón "Ejecutar GARCH/backtest" solo es `True` en
  su propio rerun).

### Planned
- Despliegue público en Streamlit Cloud.
- Capturas de pantalla en el README.

## [0.4.0] — 2026-09-14

### Added
- **`app/core/cache.py`**: desacopla la capa de negocio (`app/core`) de
  Streamlit — un único punto de acoplamiento con `st.cache_data`, con
  fallback si Streamlit no está disponible.
- **Benchmark**: `benchmark_metrics()` con beta, alpha de Jensen,
  tracking error e information ratio (antes solo se normalizaban curvas).
- **Optimización**: `deflated_sharpe_ratio()` (Bailey & López de Prado,
  2014) para corregir el sesgo de selección múltiple (winner's curse)
  del grid search.
- **Walk-forward**: parámetro `embargo` para dejar un hueco entre el
  tramo de entrenamiento y el de test de cada ventana.
- **Riesgo**: `value_at_risk_cornish_fisher()` (ajustado por asimetría
  y curtosis) y `value_at_risk_filtered_historical()` (FHS, usa
  volatilidad condicional GARCH).
- **Robustez**: pesos de `robustness_score` parametrizables +
  `robustness_score_weight_sensitivity()` para ver cuánto depende el
  score final del esquema de pesos elegido.
- **Notebook** `03_walkforward_robustness.py`.
- Logging (`logging.warning`) en las excepciones antes silenciadas de
  `walkforward.py`, `optimization.py` y `robustness.py`; `n_windows_skipped`
  expuesto en `WalkForwardResult`.
- **Cointegración**: `significance` y `fit_until` parametrizables en
  `engle_granger` (antes 0.05 fijo y siempre ajuste in-sample), con
  aviso visible en la página de Cointegración.
- **GARCH**: `GarchResult.converged` y `GarchResult.rescale_factor`.
- Tests: `load_prices` con yfinance mockeado (antes sin cobertura),
  `test_half_life_positive` reescrito con un caso de valor teórico
  conocido (antes casi imposible de fallar), y tests de regresión para
  cada punto anterior. Suite: 90 → 119 tests.

### Changed
- **GARCH**: `GJR-GARCH` ahora se traduce correctamente a
  `vol="GARCH", o=1` en la llamada a `arch_model` (antes lanzaba
  `ValueError: Unknown model type in vol` en cualquier ajuste GJR-GARCH).
- **GARCH**: `conditional_volatility` y `forecast_volatility` quedan en
  las unidades originales de los retornos — antes, con `rescale=True`
  (el valor por defecto), estaban 100x infladas de forma inconsistente.
- **Backtest**: `_extract_trades` ya no recalcula la curva de capital
  completa en cada iteración del bucle (O(n²) → O(n)); ~20x más rápido
  en backtests largos.
- **Optimización**: `best_params` se extrae columna a columna en vez de
  desde la fila ya homogeneizada del DataFrame — corrige que parámetros
  enteros (p.ej. `window`) volvían como `float`.
- **Dockerfile**: usa `requirements-lock.txt` (build reproducible);
  ya no instala `build-essential` (innecesario, hay wheels precompiladas
  para todas las dependencias); corre como usuario sin privilegios en
  vez de root; healthcheck sin dependencia de `curl` y con timeout.

### Fixed
- **Crítico**: `launch.bat (Windows)` renombrado a `launch.bat`.
- **Crítico**: `pyproject.toml` con configuración completa (se había
  perdido `[project]`, `[tool.pytest.ini_options]`, `[tool.ruff]`, etc.
  en un commit anterior).
- **Crítico**: `app/__init__.py` y `app/pages/__init__.py` (faltaban,
  rompían la resolución de módulos de mypy).
- **Crítico**: `app/state.py` separado de `app/styles.py` (un refactor
  anterior había sobrescrito las funciones de UI de `styles.py` con la
  gestión de sesión, rompiendo toda la app).

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
