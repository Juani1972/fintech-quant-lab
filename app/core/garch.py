"""Modelos GARCH para volatilidad condicional."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult


@dataclass
class GarchResult:
    """Contenedor de resultados GARCH."""
    model_result: ARCHModelResult
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    params: pd.Series
    pvalues: pd.Series
    aic: float
    bic: float


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    vol: str = "Garch",
    dist: str = "normal",
    rescale: bool = True,
) -> GarchResult:
    """Ajusta un modelo GARCH a una serie de retornos.

    Args:
        returns: Serie de retornos (en % idealmente).
        p: Orden ARCH.
        q: Orden GARCH.
        vol: Tipo de modelo ('Garch', 'EGARCH', 'GJR-GARCH').
        dist: Distribución de errores ('normal', 't', 'skewt', 'ged').
        rescale: Si True, multiplica por 100 para estabilidad numérica.

    Returns:
        GarchResult con el modelo ajustado y métricas.
    """
    series = returns.dropna().copy()
    if rescale:
        series = series * 100

    model = arch_model(series, vol=vol, p=p, q=q, dist=dist)
    res = model.fit(disp="off")

    return GarchResult(
        model_result=res,
        conditional_volatility=res.conditional_volatility,
        standardized_residuals=res.resid / res.conditional_volatility,
        params=res.params,
        pvalues=res.pvalues,
        aic=res.aic,
        bic=res.bic,
    )


def is_stationary(params: pd.Series) -> bool:
    """Verifica condición de estacionariedad alpha + beta < 1."""
    alpha = params.filter(like="alpha").sum()
    beta = params.filter(like="beta").sum()
    return (alpha + beta) < 1.0


def forecast_volatility(result: GarchResult, horizon: int = 30) -> pd.Series:
    """Genera pronóstico de volatilidad a `horizon` días."""
    fc = result.model_result.forecast(horizon=horizon, reindex=False)
    var = fc.variance.iloc[-1]
    return np.sqrt(var)
