"""Optimización de parámetros de estrategias.

Diseñado para evitar overfitting:
    - Grid search explícito (no random).
    - Objetivo configurable (Sharpe, Sortino, Calmar, return).
    - Con la opción de evaluar cada combinación con walk-forward.
    - Reporte de la superficie de parámetros (para visualizar estabilidad).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from app.core.backtest import run_backtest
from app.core.walkforward import SignalGenerator, walk_forward_analysis


# Tipo: función que recibe un dict de parámetros y devuelve un SignalGenerator
GeneratorFactory = Callable[[dict], SignalGenerator]

# Tipo: función que recibe un dict de parámetros y devuelve señales sobre la serie
SimpleSignalFactory = Callable[[pd.Series, dict], pd.Series]


@dataclass
class OptimizationResult:
    """Resultado de un grid search."""
    grid: pd.DataFrame               # Todas las combinaciones con sus métricas
    best_params: dict                # Mejor combinación
    best_metrics: dict               # Métricas de la mejor combinación
    objective: str                   # Métrica optimizada
    param_names: list[str]


@dataclass
class WalkForwardOptimizationResult:
    """Resultado de un grid search evaluado con walk-forward."""
    grid: pd.DataFrame               # Una fila por combinación, métricas IS/OOS
    best_params: dict
    best_oos_metrics: dict
    best_is_metrics: dict
    objective: str
    param_names: list[str]


# ============================================================
#  Grid search simple (backtest único)
# ============================================================
def grid_search(
    prices: pd.Series,
    signal_factory: SimpleSignalFactory,
    param_grid: dict[str, list],
    objective: str = "sharpe",
    minimize: bool = False,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> OptimizationResult:
    """Ejecuta un grid search sobre `param_grid`.

    Args:
        prices: Serie de precios.
        signal_factory: Función `(prices, params) -> signals`.
        param_grid: Dict `{nombre: [valores]}`.
        objective: Métrica a optimizar ('sharpe', 'sortino', 'calmar',
            'total_return', 'max_drawdown').
        minimize: Si True, minimiza el objetivo (útil para max_drawdown).
        initial_capital, commission, slippage: Parámetros del backtest.

    Returns:
        OptimizationResult con el grid completo y la mejor combinación.
    """
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
            )
            row = {**params, **result.metrics}
            rows.append(row)
        except Exception:
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
    best_params = {k: _coerce(best[k]) for k in param_names}
    best_metrics = {k: float(best[k]) for k in grid.columns if k not in param_names}

    return OptimizationResult(
        grid=grid,
        best_params=best_params,
        best_metrics=best_metrics,
        objective=objective,
        param_names=param_names,
    )


# ============================================================
#  Grid search con walk-forward (recomendado)
# ============================================================
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
) -> WalkForwardOptimizationResult:
    """Grid search evaluado con walk-forward (OOS).

    Este es el método robusto: para cada combinación de parámetros,
    ejecuta un walk-forward y usa la métrica OOS promedio como objetivo.
    Evita el overfitting del grid search simple.
    """
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
            )
            row = {
                **params,
                **{f"is_{k}": v for k, v in wf.is_metrics_agg.items()},
                **{f"oos_{k}": v for k, v in wf.oos_metrics_agg.items()},
            }
            rows.append(row)
        except Exception:
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
    best_params = {k: _coerce(best[k]) for k in param_names}
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


# ============================================================
#  Utilidades
# ============================================================
def _expand_grid(param_grid: dict[str, list]) -> list[dict]:
    """Convierte un dict de listas en una lista de combinaciones."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


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
