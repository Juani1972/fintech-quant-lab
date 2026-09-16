"""Optimización de parámetros de estrategias."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from app.core.backtest import BacktestMode, run_backtest
from app.core.walkforward import SignalGenerator, walk_forward_analysis

logger = logging.getLogger(__name__)

GeneratorFactory = Callable[[dict], SignalGenerator]
SimpleSignalFactory = Callable[[pd.Series, dict], pd.Series]


@dataclass
class OptimizationResult:
    """Resultado de un grid search."""
    grid: pd.DataFrame
    best_params: dict
    best_metrics: dict
    objective: str
    param_names: list[str]


@dataclass
class WalkForwardOptimizationResult:
    """Resultado de un grid search evaluado con walk-forward."""
    grid: pd.DataFrame
    best_params: dict
    best_oos_metrics: dict
    best_is_metrics: dict
    objective: str
    param_names: list[str]


def grid_search(
    prices: pd.Series,
    signal_factory: SimpleSignalFactory,
    param_grid: dict[str, list],
    objective: str = "sharpe",
    minimize: bool = False,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    mode: BacktestMode = "percent",
) -> OptimizationResult:
    """Ejecuta un grid search sobre `param_grid`.

    Args:
        prices: Serie de precios.
        signal_factory: Función `(prices, params) -> signals`.
        param_grid: Dict `{nombre: [valores]}`.
        objective: Métrica a optimizar.
        minimize: Si True, minimiza el objetivo.
        initial_capital, commission, slippage: Parámetros del backtest.
        mode: 'percent' para precios, 'absolute' para spreads que
            cruzan cero (p. ej. pairs trading).

    Returns:
        OptimizationResult con el grid completo y la mejor combinación.

    Raises:
        ValueError: Si el grid está vacío o el objetivo no existe.
    """
    if not param_grid or not any(param_grid.values()):
        raise ValueError("El grid de parámetros está vacío.")

    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("El grid de parámetros está vacío.")

    rows = []
    for params in combos:
        try:
            signals = signal_factory(prices, params)
            signals = signals.reindex(prices.index).fillna(0)
            result = run_backtest(
                prices, signals,
                initial_capital=initial_capital,
                commission=commission, slippage=slippage,
                mode=mode,
            )
            row = {**params, **result.metrics}
            rows.append(row)
        except Exception:
            logger.warning(
                "Combinación de parámetros descartada en grid_search: %s", params,
                exc_info=True,
            )
            row = {**params}
            row[objective] = np.nan
            rows.append(row)

    grid = pd.DataFrame(rows)

    if objective not in grid.columns:
        raise ValueError(f"Objetivo '{objective}' no encontrado en las métricas.")

    valid = grid[grid[objective].notna()]
    if valid.empty:
        raise ValueError("Ninguna combinación produjo métricas válidas.")

    idx = valid[objective].idxmin() if minimize else valid[objective].idxmax()
    best = grid.loc[idx]

    param_names = list(param_grid.keys())
    # best_params se extrae columna a columna (grid[k].loc[idx]) y no de
    # `best[k]` (la fila ya extraída): una fila de un DataFrame con
    # columnas de distinto dtype (window:int64, sharpe:float64...) se
    # homogeneiza a un único dtype (float64) al extraerla como Series,
    # así que un parámetro entero como una ventana volvería siempre como
    # float (30.0 en vez de 30) y rompería cualquier código que lo use
    # para indexar (p.ej. `prices.pct_change(window)`).
    best_params = {k: _coerce(grid[k].loc[idx]) for k in param_names}
    best_metrics = {k: float(best[k]) for k in grid.columns if k not in param_names}

    return OptimizationResult(
        grid=grid,
        best_params=best_params,
        best_metrics=best_metrics,
        objective=objective,
        param_names=param_names,
    )


def deflated_sharpe_ratio(
    result: OptimizationResult,
    strategy_returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    `grid_search` elige la combinación de mayor Sharpe entre N intentos.
    Cuantas más combinaciones se prueban, más probable es que el Sharpe
    de la "ganadora" sea alto por puro azar de la selección múltiple
    (winner's curse), no porque la estrategia tenga una ventaja real.

    El DSR responde: dado que se probaron N combinaciones (con la
    dispersión de Sharpes observada en `result.grid`), ¿qué tan
    probable es que el Sharpe de la ganadora sea genuinamente positivo,
    y no solo el máximo esperable por azar entre N intentos?

    Es la contrapartida, para la selección de parámetros, de la
    corrección por múltiples tests (Bonferroni/BH) que ya se aplica en
    el módulo de cointegración.

    Args:
        result: Resultado de `grid_search` con `objective='sharpe'`
            (el DSR está definido específicamente para el Sharpe ratio).
        strategy_returns: Retornos (por barra) de la estrategia ganadora
            (`result.best_params`), para estimar su asimetría y curtosis.
        periods_per_year: Periodicidad de `strategy_returns` (252 para
            retornos diarios de mercados bursátiles).

    Returns:
        Dict con:
            - sr_observed: Sharpe anualizado de la combinación ganadora.
            - sr_expected_max: Sharpe máximo esperado por puro azar,
              dado el nº de combinaciones probadas.
            - sr_std: desviación estándar estimada del Sharpe ganador
              (ajustada por asimetría/curtosis de los retornos).
            - n_trials: nº de combinaciones válidas usadas para estimar
              `sr_expected_max`.
            - dsr: probabilidad en [0, 1] de que el Sharpe verdadero sea
              > 0 una vez descontado el sesgo de selección. Como regla
              práctica, DSR > 0.95 sugiere que el resultado probablemente
              no es solo un artefacto de haber probado muchas combinaciones.

    Nota: la fórmula asume trials aproximadamente independientes. Si el
    grid varía un único parámetro de forma continua (p.ej. ventanas de
    5 en 5), combinaciones vecinas producen estrategias muy
    correlacionadas entre sí, lo que puede hacer que el DSR sea más
    conservador de lo necesario. Es una limitación conocida del método
    (Bailey & López de Prado, 2014), no un error de esta implementación.

    Raises:
        ValueError: Si `result.objective != 'sharpe'`, si hay menos de
            10 observaciones de retornos, o menos de 2 combinaciones
            válidas en el grid.
    """
    from scipy.stats import kurtosis, norm, skew

    if result.objective != "sharpe":
        raise ValueError(
            "deflated_sharpe_ratio requiere un grid_search con "
            f"objective='sharpe' (se usó '{result.objective}')."
        )

    returns = strategy_returns.dropna().to_numpy()
    n_obs = len(returns)
    if n_obs < 10:
        raise ValueError(
            f"Se necesitan al menos 10 observaciones de retornos (hay {n_obs})."
        )

    ret_std = returns.std(ddof=1)
    sr_period = float(returns.mean() / ret_std) if ret_std > 0 else 0.0
    sr_annual = sr_period * np.sqrt(periods_per_year)

    trial_sharpes = result.grid[result.objective].dropna().to_numpy()
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        raise ValueError(
            f"Se necesitan al menos 2 combinaciones válidas en el grid (hay {n_trials})."
        )

    # Sharpe máximo esperado por azar entre n_trials intentos con
    # varianza sr_std_trials (aproximación de Bailey & López de Prado).
    sr_std_trials = float(np.std(trial_sharpes, ddof=1)) if n_trials > 1 else 0.0
    euler_mascheroni = 0.5772156649015329
    if sr_std_trials > 0:
        sr_expected_max = sr_std_trials * (
            (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
        )
    else:
        sr_expected_max = 0.0

    # Desviación del estimador del Sharpe, ajustada por asimetría y
    # curtosis de los retornos de la estrategia ganadora.
    skew_r = float(skew(returns))
    kurt_r = float(kurtosis(returns, fisher=False))  # normal -> 3.0
    denom = max(1 - skew_r * sr_period + (kurt_r - 1) / 4 * sr_period ** 2, 1e-12)
    sigma_sr_annual = float(np.sqrt(denom / (n_obs - 1)) * np.sqrt(periods_per_year))

    dsr = float(norm.cdf((sr_annual - sr_expected_max) / sigma_sr_annual)) \
        if sigma_sr_annual > 0 else float("nan")

    return {
        "sr_observed": sr_annual,
        "sr_expected_max": float(sr_expected_max),
        "sr_std": sigma_sr_annual,
        "n_trials": n_trials,
        "dsr": dsr,
    }


def grid_search_walkforward(
    prices: pd.Series,
    generator_factory: GeneratorFactory,
    param_grid: dict[str, list],
    objective: str = "sharpe",
    minimize: bool = False,
    train_size: int = 504,
    test_size: int = 126,
    step: int | None = None,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    mode: BacktestMode = "percent",
) -> WalkForwardOptimizationResult:
    """Grid search evaluado con walk-forward (OOS).

    Args:
        mode: 'percent' para precios, 'absolute' para spreads que
            cruzan cero (p. ej. pairs trading). Se reenvía a cada
            `walk_forward_analysis()` del grid.
    """
    if not param_grid or not any(param_grid.values()):
        raise ValueError("El grid de parámetros está vacío.")

    combos = _expand_grid(param_grid)
    if not combos:
        raise ValueError("El grid de parámetros está vacío.")

    rows = []
    for params in combos:
        try:
            generator = generator_factory(params)
            wf = walk_forward_analysis(
                prices=prices,
                signal_generator=generator,
                train_size=train_size,
                test_size=test_size,
                step=step,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                mode=mode,
            )
            row = {
                **params,
                **{f"is_{k}": v for k, v in wf.is_metrics_agg.items()},
                **{f"oos_{k}": v for k, v in wf.oos_metrics_agg.items()},
            }
            rows.append(row)
        except Exception:
            logger.warning(
                "Combinación de parámetros descartada en grid_search_walkforward: %s",
                params, exc_info=True,
            )
            row = {**params}
            row[f"oos_{objective}"] = np.nan
            rows.append(row)

    grid = pd.DataFrame(rows)
    oos_col = f"oos_{objective}"

    if oos_col not in grid.columns:
        raise ValueError(f"Objetivo '{oos_col}' no encontrado.")

    valid = grid[grid[oos_col].notna()]
    if valid.empty:
        raise ValueError("Ninguna combinación produjo métricas OOS válidas.")

    idx = valid[oos_col].idxmin() if minimize else valid[oos_col].idxmax()
    best = grid.loc[idx]

    param_names = list(param_grid.keys())
    # Ver el comentario equivalente en grid_search(): se extrae columna a
    # columna para no perder el dtype entero de los parámetros.
    best_params = {k: _coerce(grid[k].loc[idx]) for k in param_names}
    best_is = {k.replace("is_", ""): float(best[k]) for k in grid.columns if k.startswith("is_")}
    best_oos = {k.replace("oos_", ""): float(best[k]) for k in grid.columns if k.startswith("oos_")}

    return WalkForwardOptimizationResult(
        grid=grid,
        best_params=best_params,
        best_oos_metrics=best_oos,
        best_is_metrics=best_is,
        objective=objective,
        param_names=param_names,
    )


def _expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Convierte un dict de listas en una lista de combinaciones.

    Devuelve [] si el dict está vacío o si alguna lista de valores está vacía.
    """
    if not param_grid:
        return []
    if any(not values for values in param_grid.values()):
        return []
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in product(*values)]


def _coerce(value):
    """Convierte valores numpy a tipos nativos."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def heatmap_data(
    grid: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
) -> pd.DataFrame:
    """Prepara una matriz 2D para visualizar como heatmap."""
    if x_param not in grid.columns or y_param not in grid.columns:
        raise ValueError(f"Parámetros '{x_param}' o '{y_param}' no están en el grid.")
    if metric not in grid.columns:
        raise ValueError(f"Métrica '{metric}' no está en el grid.")
    return grid.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
