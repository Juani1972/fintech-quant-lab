# Nuevos módulos — Fintech Quant Lab

Los 6 módulos acordados, completos y con tests, listos para subir al
repo `Juani1972/fintech-quant-lab` usando "Upload files" en el editor
web de GitHub. **119 tests, todos en verde.**

## Contenido

```
app/
  __init__.py                          (vacío, no sobrescribir si ya existe)
  core/
    __init__.py                         (vacío, no sobrescribir si ya existe)
    providers.py                         1. Abstracción de fuentes de datos
    experiments.py                       2. Reproducibilidad de experimentos
    strategies/                          3. Estrategias nuevas
      __init__.py                          registro central + get_strategy()
      base.py                              contrato Strategy común
      cross_sectional_momentum.py
      risk_parity_strategy.py
      volatility_targeting.py
      trend_following.py
      carry_trade.py
      pca_statarb.py
    portfolio.py                         4. Markowitz, ERC y HRP
    license.py                           5. Sistema de licencias
    papertrade.py                        6. Paper trading vía Alpaca (REST)
tests/
  __init__.py
  test_providers.py                      20 tests
  test_experiments.py                    17 tests
  test_strategies.py                     29 tests
  test_portfolio.py                      17 tests
  test_license.py                        15 tests
  test_papertrade.py                     21 tests
```

## Cómo subirlo

1. Si `app/__init__.py` y `app/core/__init__.py` ya existen en tu repo
   con contenido propio, NO los sobrescribas — sube solo los demás
   ficheros dentro de `app/core/` (incluida la carpeta `strategies/`
   completa).
2. Sube los 6 ficheros de test dentro de `tests/`.
3. Añade `requests` a `requirements.txt` si no lo tienes ya (lo usan
   `AlphaVantageProvider`, `license.py` y `papertrade.py`). `scipy` lo
   usa `hrp_weights` — según la tabla de priorización original ya
   estaba en tu stack.
4. Deja que el CI corra los 119 tests. No necesitas instalar nada
   distinto a lo anterior; ni Alpaca, ni un servidor de licencias real
   — todo lo que habla por red está mockeado en los tests.

## Qué hace cada módulo

### 1. `providers.py`
Contrato `MarketDataProvider` + `YahooProvider`, `StooqProvider`,
`CSVProvider`, `AlphaVantageProvider`, y la factory `get_provider()`.

### 2. `experiments.py`
`save_experiment`, `load_experiment`, `diff_experiments`,
`list_experiments` — congela config + datos + resultados de cada
corrida para reproducibilidad total.

### 3. `strategies/`
6 estrategias nuevas bajo el contrato común `Strategy.generate_signals`,
más `get_strategy()` / `list_strategies()`. **Pendiente**: migrar tus 3
estrategias actuales (Pairs, Momentum, MeanRev) a este mismo patrón —
necesitas hacerlo tú sobre tu código real, no lo tengo para adaptarlo
sin riesgo de romper algo que ya funciona.

### 4. `portfolio.py`
- `markowitz_weights(returns, target_return=None, risk_free=0.0)` —
  forma cerrada (cartera tangente o de varianza mínima a un retorno dado).
- `risk_parity_weights(returns, budget=None)` — Equal Risk Contribution
  vía descenso cíclico de coordenadas (no es solo inverse-vol).
- `hrp_weights(returns)` — Hierarchical Risk Parity (López de Prado 2016):
  clustering + quasi-diagonalización + bisección recursiva, con
  `scipy.cluster.hierarchy`.
- `rebalance_schedule(weights, method="calendar"|"threshold", ...)`.

**Pendiente** (no incluido, fuera del top-6 acordado): `black_litterman_weights`.

### 5. `license.py`
- `machine_fingerprint()` — hash SHA-256 de MAC + hostname + CPU.
- `verify_license(key, server_url)` — verifica contra tu servidor de
  licencias (tú tendrás que montar el endpoint `POST {server_url}/verify`
  que devuelva `{"valid": bool, "expires_at": "YYYY-MM-DD", "plan": str}`).
- `is_license_valid(info, grace_days=7)` — válida offline durante el
  margen de gracia desde la última verificación.
- `cache_license` / `load_cached_license` — caché en disco.

### 6. `papertrade.py`
- `PaperAccount` — cliente REST directo de Alpaca (sin el SDK
  `alpaca-py`, para no añadir esa dependencia): `get_account`,
  `get_positions`, `submit_order`, `cancel_all`, `close_all_positions`.
- `StrategyRunner` — ejecuta una estrategia (cualquier callable que
  devuelva señal -1/0/1) contra `PaperAccount`, ajustando la posición
  real a la señal en cada pasada (`run_once`) o en bucle
  (`run_forever` / `stop`).

## Ya no queda ningún módulo pendiente de los 6 acordados

Los siguientes de la lista original (no acordados en el top-6, quedan
como ideas para más adelante si te interesan): `report_pdf.py`,
`api.py` (FastAPI), `attribution.py`, `visualization.py`,
`machine_learning.py`, `multi_asset_backtest.py`, `johansen.py`, `breaks.py`.
