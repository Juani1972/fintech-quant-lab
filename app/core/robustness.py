"""Análisis de robustez de estrategias.

Incluye:
    - Monte Carlo: simulaciones bootstrap de retornos para distribuciones
      de métricas (Sharpe, Max DD, etc.).
    - Bootstrap: remuestreo con reemplazo.
    - Block bootstrap: remuestreo por bloques (preserva autocorrelación).
    - Parameter sensitivity: variación de métricas al perturbar parámetros.
    - Robustness score: agregado interpretable 0-100.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.core.backtest import run_backtest


# ============================================================
#  Monte Carlo / Bootstrap
# ============================================================
@dataclass
class MonteCarloResult:
    """Resultado de una simulación Monte Carlo sobre retornos."""
    n_simulations: int
    horizon: int
    metric: str
    distribution: np.ndarray          # Valores simulados
    mean: float
    std: float
    percentiles: dict[str, float]
    var_95: float
    cvar_95: float


def monte_carlo_bootstrap(
    returns: pd.Series,
    n_simulations: int = 1000,
    horizon: int | None = None,
    metric_fn: Callable[[pd.Series], float] | None = None,
    metric_name: str = "sharpe",
    block_size: int = 1,
    random_state: int | None = 42,
) -> MonteCarloResult:
    """Monte Carlo por bootstrap sobre los retornos observados.

    Args:
        returns: Serie de retornos (diarios) observados.
        n_simulations: Número de simulaciones.
        horizon: Nº de barras por simulación. Si None, usa len(returns).
        metric_fn: Función que recibe una serie de retornos y devuelve
            una métrica. Si None, usa Sharpe anualizado.
        metric_name: Nombre de la métrica (para reportes).
        block_size: Tamaño del bloque para block bootstrap. Si 1, bootstrap
            iid. Si > 1, preserva autocorrelación.
        random_state: Semilla.

    Returns:
        MonteCarloResult con la distribución de la métrica.
    """
    clean = returns.dropna().values
    if len(clean) < 2:
        raise ValueError("Se necesitan al menos 2 retornos.")

    if horizon is None:
        horizon = len(clean)

    if metric_fn is None:
        metric_fn = _default_sharpe

    rng = np.random.default_rng(random_state)
    results = np.zeros(n_simulations)

    for i in range(n_simulations):
        if block_size <= 1:
            sample = rng.choice(clean, size=horizon, replace=True)
        else:
            sample = _block_bootstrap(clean, horizon, block_size, rng)
        results[i] = metric_fn(pd.Series(sample))

    return _summarize_mc(results, n_simulations, horizon, metric_name)


def monte_carlo_gbm(
    initial_price: float,
    mu: float,
    sigma: float,
    horizon: int = 252,
    n_simulations: int = 1000,
    random_state: int | None = 42,
) -> np.ndarray:
    """Monte Carlo con Movimiento Browniano Geométrico.

    Args:
        initial_price: Precio inicial S0.
        mu: Deriva anual (ej. 0.08).
        sigma: Volatilidad anual (ej. 0.20).
        horizon: Nº de pasos (días de trading).
        n_simulations: Nº de trayectorias.

    Returns:
        Array (n_simulations, horizon+1) con las trayectorias.
    """
    if initial_price <= 0:
        raise ValueError("initial_price debe ser > 0.")
    if sigma < 0:
        raise ValueError("sigma debe ser >= 0.")

    dt = 1 / 252
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal((n_simulations, horizon))
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_returns, axis=1)
    paths = initial_price * np.exp(log_paths)
    return np.column_stack([np.full(n_simulations, initial_price), paths])


# ============================================================
#  Análisis de sensibilidad de parámetros
# ============================================================
@dataclass
class SensitivityResult:
    """Resultado del análisis de sensibilidad de parámetros."""
    param_name: str
    base_value: float
    variations: pd.DataFrame       # columnas: value, metric
    stability_score: float         # 0-1 (1 = muy estable)


def parameter_sensitivity(
    prices: pd.Series,
    signal_factory: Callable[[pd.Series, dict], pd.Series],
    base_params: dict,
    param_name: str,
    variations: list,
    metric: str = "sharpe",
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> SensitivityResult:
    """Analiza cómo cambia `metric` al variar un único parámetro.

    Un cambio abrupto indica fragilidad (overfitting). Una meseta
    alrededor del valor base indica robustez.
    """
    rows = []
    for val in variations:
        params = {**base_params, param_name: val}
        try:
            signals = signal_factory(prices, params).reindex(prices.index).fillna(0)
            result = run_backtest(
                prices, signals,
                initial_capital=initial_capital,
                commission=commission, slippage=slippage,
            )
            rows.append({"value": val, metric: result.metrics.get(metric, np.nan)})
        except Exception:
            rows.append({"value": val, metric: np.nan})

    df = pd.DataFrame(rows).sort_values("value")
    stability = _stability_score(df[metric].values)

    return SensitivityResult(
        param_name=param_name,
        base_value=base_params[param_name],
        variations=df,
        stability_score=stability,
    )


# ============================================================
#  Robustness score
# ============================================================
@dataclass
class RobustnessReport:
    """Reporte agregado de robustez."""
    components: dict[str, float]
    final_score: float
    interpretation: str


def robustness_score(
    is_sharpe: float,
    oos_sharpe: float,
    mc_result: MonteCarloResult,
    sensitivity_score: float,
    n_trades: int,
) -> RobustnessReport:
    """Calcula un score agregado de robustez 0-100.

    Componentes:
        - Degradación IS → OOS.
        - Sharpe Monte Carlo medio (positivo).
        - Estabilidad de parámetros.
        - Nº suficiente de trades.
    """
    components: dict[str, float] = {}

    # 1. Degradación IS → OOS (0-30 puntos)
    if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) and abs(is_sharpe) > 1e-6:
        degradation = max(0.0, (is_sharpe - oos_sharpe) / abs(is_sharpe))
        components["degradacion_is_oos"] = float(np.clip(30 * (1 - min(degradation, 1)), 0, 30))
    else:
        components["degradacion_is_oos"] = 0.0

    # 2. Sharpe MC medio (0-30 puntos)
    mc_sharpe = mc_result.mean
    components["monte_carlo_sharpe"] = float(np.clip(30 * (mc_sharpe / 2), 0, 30))

    # 3. Estabilidad de parámetros (0-25 puntos)
    components["estabilidad_parametros"] = float(np.clip(25 * sensitivity_score, 0, 25))

    # 4. Nº de trades (0-15 puntos)
    components["n_trades"] = float(np.clip(15 * min(n_trades / 100, 1), 0, 15))

    final = sum(components.values())
    final = float(np.clip(final, 0, 100))

    if final >= 80:
        interp = "Robusta — la estrategia generaliza bien."
    elif final >= 60:
        interp = "Aceptable — vigilar la degradación OOS."
    elif final >= 40:
        interp = "Frágil — alta probabilidad de overfitting."
    else:
        interp = "Muy frágil — no recomendada en producción."

    return RobustnessReport(
        components=components,
        final_score=final,
        interpretation=interp,
    )


# ============================================================
#  Utilidades internas
# ============================================================
def _default_sharpe(returns: pd.Series) -> float:
    """Sharpe anualizado simple."""
    std = returns.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(np.sqrt(252) * returns.mean() / std)


def _block_bootstrap(
    data: np.ndarray,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap preservando autocorrelación."""
    n = len(data)
    n_blocks = int(np.ceil(horizon / block_size))
    starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
    chunks = [data[s : s + block_size] for s in starts]
    concatenated = np.concatenate(chunks)[:horizon]
    return concatenated


def _summarize_mc(
    results: np.ndarray,
    n_simulations: int,
    horizon: int,
    metric_name: str,
) -> MonteCarloResult:
    """Resume la distribución de la métrica simulada."""
    valid = results[np.isfinite(results)]
    if len(valid) == 0:
        raise ValueError("Todas las simulaciones dieron métricas no finitas.")

    pcts = {
        "p05": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
    }
    var_95 = float(-np.percentile(valid, 5))
    tail = valid[valid <= np.percentile(valid, 5)]
    cvar_95 = float(-tail.mean()) if len(tail) else np.nan

    return MonteCarloResult(
        n_simulations=n_simulations,
        horizon=horizon,
        metric=metric_name,
        distribution=valid,
        mean=float(valid.mean()),
        std=float(valid.std(ddof=1)),
        percentiles=pcts,
        var_95=var_95,
        cvar_95=cvar_95,
    )


def _stability_score(values: np.ndarray) -> float:
    """Score 0-1: 1 = muy estable (baja dispersión relativa)."""
    valid = values[np.isfinite(values)]
    if len(valid) < 2:
        return 0.0
    mean = np.mean(valid)
    std = np.std(valid, ddof=1)
    if abs(mean) < 1e-9:
        return 0.0
    cv = std / abs(mean)          # coeficiente de variación
    return float(np.clip(1 - cv, 0, 1))
