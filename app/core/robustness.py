"""Análisis de robustez de estrategias.

Incluye:
    - Monte Carlo: simulaciones bootstrap de retornos.
    - Bootstrap: remuestreo con reemplazo.
    - Block bootstrap: remuestreo por bloques (preserva autocorrelación).
    - Parameter sensitivity: variación de métricas al perturbar parámetros.
    - Robustness score: agregado interpretable 0-100.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.core.backtest import BacktestMode, run_backtest


# ============================================================
#  Monte Carlo / Bootstrap
# ============================================================
@dataclass
class MonteCarloResult:
    """Resultado de una simulación Monte Carlo sobre retornos."""
    n_simulations: int
    horizon: int
    metric: str
    distribution: np.ndarray
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
    """Monte Carlo por bootstrap sobre los retornos observados."""
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
    """Monte Carlo con Movimiento Browniano Geométrico."""
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
    variations: pd.DataFrame
    stability_score: float


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
    mode: BacktestMode = "percent",
) -> SensitivityResult:
    """Analiza cómo cambia `metric` al variar un único parámetro.

    Args:
        prices: Serie de precios o spread.
        signal_factory: Función `(prices, params) -> signals`.
        base_params: Parámetros base.
        param_name: Nombre del parámetro a variar.
        variations: Lista de valores a probar.
        metric: Métrica del backtest a usar.
        initial_capital, commission, slippage: Parámetros del backtest.
        mode: 'percent' para precios, 'absolute' para spreads.
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
                mode=mode,
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


# Pesos por defecto (puntos máximos de cada componente, suman 100).
# Son una elección heurística del autor -- no están derivados de ninguna
# teoría formal de cuánto "debería" pesar cada factor. Se exponen como
# parámetro de `robustness_score` precisamente para que se pueda evaluar
# qué tan sensible es el score final a esta elección (ver
# `robustness_score_weight_sensitivity`).
DEFAULT_ROBUSTNESS_WEIGHTS: dict[str, float] = {
    "degradacion_is_oos": 30.0,
    "monte_carlo_sharpe": 30.0,
    "estabilidad_parametros": 25.0,
    "n_trades": 15.0,
}


def robustness_score(
    is_sharpe: float,
    oos_sharpe: float,
    mc_result: MonteCarloResult,
    sensitivity_score: float,
    n_trades: int,
    weights: dict[str, float] | None = None,
) -> RobustnessReport:
    """Calcula un score agregado de robustez 0-100.

    Los pesos por defecto (30/30/25/15, ver `DEFAULT_ROBUSTNESS_WEIGHTS`)
    son una elección razonada pero arbitraria del autor, no un resultado
    derivado formalmente. Pasa `weights` para explorar otras
    ponderaciones, o usa `robustness_score_weight_sensitivity` para ver
    de un vistazo cuánto cambia el score final entre varios esquemas de
    pesos razonables.
    """
    w = weights if weights is not None else DEFAULT_ROBUSTNESS_WEIGHTS
    missing = set(DEFAULT_ROBUSTNESS_WEIGHTS) - set(w)
    if missing:
        raise ValueError(f"Faltan pesos para: {sorted(missing)}")

    components: dict[str, float] = {}

    if np.isfinite(is_sharpe) and np.isfinite(oos_sharpe) and abs(is_sharpe) > 1e-6:
        degradation = max(0.0, (is_sharpe - oos_sharpe) / abs(is_sharpe))
        components["degradacion_is_oos"] = float(
            np.clip(w["degradacion_is_oos"] * (1 - min(degradation, 1)),
                    0, w["degradacion_is_oos"])
        )
    else:
        components["degradacion_is_oos"] = 0.0

    mc_sharpe = mc_result.mean
    components["monte_carlo_sharpe"] = float(
        np.clip(w["monte_carlo_sharpe"] * (mc_sharpe / 2), 0, w["monte_carlo_sharpe"])
    )

    components["estabilidad_parametros"] = float(
        np.clip(w["estabilidad_parametros"] * sensitivity_score, 0, w["estabilidad_parametros"])
    )

    components["n_trades"] = float(
        np.clip(w["n_trades"] * min(n_trades / 100, 1), 0, w["n_trades"])
    )

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


def robustness_score_weight_sensitivity(
    is_sharpe: float,
    oos_sharpe: float,
    mc_result: MonteCarloResult,
    sensitivity_score: float,
    n_trades: int,
) -> pd.DataFrame:
    """Recalcula el robustness score bajo varios esquemas de pesos, para
    ver de un vistazo cuánto depende el número final de la ponderación
    30/30/25/15 elegida por defecto (ver `robustness_score`).
    """
    schemes = {
        "default (30/30/25/15)": DEFAULT_ROBUSTNESS_WEIGHTS,
        "equitativo (25/25/25/25)": {
            "degradacion_is_oos": 25.0, "monte_carlo_sharpe": 25.0,
            "estabilidad_parametros": 25.0, "n_trades": 25.0,
        },
        "prioriza OOS (45/30/15/10)": {
            "degradacion_is_oos": 45.0, "monte_carlo_sharpe": 30.0,
            "estabilidad_parametros": 15.0, "n_trades": 10.0,
        },
        "prioriza estabilidad (15/20/50/15)": {
            "degradacion_is_oos": 15.0, "monte_carlo_sharpe": 20.0,
            "estabilidad_parametros": 50.0, "n_trades": 15.0,
        },
    }

    rows = []
    for name, w in schemes.items():
        report = robustness_score(
            is_sharpe, oos_sharpe, mc_result, sensitivity_score, n_trades, weights=w,
        )
        rows.append({"esquema": name, "score_final": report.final_score,
                      "interpretacion": report.interpretation})

    return pd.DataFrame(rows)


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
    concatenated: np.ndarray = np.concatenate(chunks)[:horizon]
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
    cv = std / abs(mean)
    return float(np.clip(1 - cv, 0, 1))
