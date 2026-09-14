"""Correcciones por múltiples contrastes.

Cuando se realizan N tests de hipótesis simultáneamente (por ejemplo,
buscar pares cointegrados entre muchos tickers), el p-value individual
deja de ser interpretable sin corrección.

Implementa:
    - Bonferroni: controla el Family-Wise Error Rate (FWER).
    - Benjamini-Hochberg: controla el False Discovery Rate (FDR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MultipleTestingResult:
    """Resultado de una corrección por múltiples tests."""
    method: str
    alpha: float
    pvalues_original: pd.Series
    pvalues_adjusted: pd.Series
    rejected: pd.Series
    n_tests: int
    n_rejected: int


def bonferroni(
    pvalues: pd.Series | list[float],
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Corrección de Bonferroni.

    Ajusta cada p-value multiplicándolo por el número de tests y
    lo trunca a 1. Rechaza H0 si p_adj < alpha.

    Args:
        pvalues: Serie o lista de p-values.
        alpha: Nivel de significación global.

    Returns:
        MultipleTestingResult.
    """
    pv = _to_series(pvalues)
    n = len(pv)
    if n == 0:
        raise ValueError("Se necesita al menos un p-value.")

    pv_adj = (pv * n).clip(upper=1.0)
    rejected = pv_adj < alpha

    return MultipleTestingResult(
        method="bonferroni",
        alpha=alpha,
        pvalues_original=pv,
        pvalues_adjusted=pv_adj,
        rejected=rejected,
        n_tests=n,
        n_rejected=int(rejected.sum()),
    )


def benjamini_hochberg(
    pvalues: pd.Series | list[float],
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Corrección de Benjamini-Hochberg (FDR).

    Ordena los p-values, aplica p_adj = p * n / rank, y fuerza
    monotonicidad. Rechaza H0 si p_adj < alpha.

    Args:
        pvalues: Serie o lista de p-values.
        alpha: Nivel de FDR objetivo.

    Returns:
        MultipleTestingResult.
    """
    pv = _to_series(pvalues)
    n = len(pv)
    if n == 0:
        raise ValueError("Se necesita al menos un p-value.")

    # Orden ascendente
    order = pv.sort_values().index
    ranked = pv.loc[order]
    ranks = np.arange(1, n + 1)

    # p_adj = p * n / rank
    pv_adj_sorted = (ranked.values * n / ranks)
    # Monotonicidad: p_adj[i] = min(p_adj[i], p_adj[i+1]) desde el final
    pv_adj_sorted = np.minimum.accumulate(pv_adj_sorted[::-1])[::-1]
    pv_adj_sorted = np.clip(pv_adj_sorted, 0, 1)

    # Reconstruir en orden original
    pv_adj = pd.Series(index=pv.index, dtype=float)
    pv_adj.loc[order] = pv_adj_sorted
    pv_adj = pv_adj.reindex(pv.index)

    rejected = pv_adj < alpha

    return MultipleTestingResult(
        method="benjamini-hochberg",
        alpha=alpha,
        pvalues_original=pv,
        pvalues_adjusted=pv_adj,
        rejected=rejected,
        n_tests=n,
        n_rejected=int(rejected.sum()),
    )


def correct_pvalues(
    pvalues: pd.Series | list[float],
    method: str = "bh",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Despachador por método.

    Args:
        method: 'bonferroni' | 'bh' (Benjamini-Hochberg).
        alpha: Nivel de significación.
    """
    method_norm = method.lower().replace("-", "").replace("_", "")
    if method_norm == "bonferroni":
        return bonferroni(pvalues, alpha=alpha)
    if method_norm in ("bh", "benjaminihochberg", "fdr"):
        return benjamini_hochberg(pvalues, alpha=alpha)
    raise ValueError(
        f"Método '{method}' no soportado. Usa 'bonferroni' o 'bh'."
    )


def _to_series(pvalues: pd.Series | list[float]) -> pd.Series:
    """Convierte a pd.Series con validación."""
    if isinstance(pvalues, pd.Series):
        pv = pvalues.dropna().astype(float)
    else:
        pv = pd.Series([float(p) for p in pvalues if p is not None])
    if pv.isna().any():
        raise ValueError("Los p-values no pueden contener NaN.")
    if ((pv < 0) | (pv > 1)).any():
        raise ValueError("Todos los p-values deben estar en [0, 1].")
    return pv
