"""Modelos GARCH para volatilidad condicional."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

VolType = Literal["Garch", "EGARCH", "GJR-GARCH"]
ArchVolType = Literal["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"]
DistType = Literal["normal", "t", "skewt", "ged"]


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
    model_type: str


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

    # GJR-GARCH no es un valor de `vol` propio en la librería `arch`: se
    # consigue con vol="GARCH" + o=1 (término de asimetría/leverage).
    if vol == "GJR-GARCH":
        arch_vol: ArchVolType = "GARCH"
        o = 1
    else:
        arch_vol = cast(ArchVolType, "GARCH" if vol == "Garch" else vol)
        o = 0

    model = arch_model(
        series,
        vol=arch_vol,
        p=p,
        o=o,
        q=q,
        dist=cast(DistType, dist),
    )
    res = model.fit(disp="off")

    return GarchResult(
        model_result=res,
        conditional_volatility=res.conditional_volatility,
        standardized_residuals=res.resid / res.conditional_volatility,
        params=res.params,
        pvalues=res.pvalues,
        aic=float(res.aic),
        bic=float(res.bic),
        model_type=vol,
    )


def check_stationarity(params: pd.Series, model_type: str = "Garch") -> bool:
    """Comprueba la condición de estacionariedad según el tipo de modelo.

    Returns:
        `bool` nativo de Python.

    Raises:
        ValueError: Si el modelo no es reconocido.
    """
    model_type_norm = model_type.upper().replace("-", "").replace("_", "")

    if model_type_norm == "GARCH":
        alpha = params.filter(like="alpha").sum()
        beta = params.filter(like="beta").sum()
        return bool((alpha + beta) < 1.0)

    if model_type_norm in ("GJRGARCH", "GJR"):
        alpha = params.filter(like="alpha").sum()
        beta = params.filter(like="beta").sum()
        gamma = params.filter(like="gamma").sum()
        return bool((alpha + beta + gamma / 2) < 1.0)

    if model_type_norm == "EGARCH":
        beta = params.filter(like="beta").sum()
        return bool(abs(beta) < 1.0)

    raise ValueError(f"Modelo '{model_type}' no reconocido para test de estacionariedad.")


def is_stationary(params: pd.Series, model_type: str = "Garch") -> bool:
    """Alias de `check_stationarity`. Devuelve `bool` nativo."""
    return check_stationarity(params, model_type)


def residual_diagnostics(result: GarchResult, lags: int = 10) -> dict[str, float]:
    """Diagnósticos de residuos estandarizados."""
    from scipy.stats import jarque_bera
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    resid = result.standardized_residuals.dropna()

    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(resid ** 2, lags=[lags], return_df=True)
    arch_test = het_arch(resid, nlags=lags)
    _, jb_pval = jarque_bera(resid)

    return {
        "ljung_box_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "ljung_box_squared_pvalue": float(lb2["lb_pvalue"].iloc[0]),
        "arch_lm_pvalue": float(arch_test[1]),
        "jarque_bera_pvalue": float(jb_pval),
    }


def forecast_volatility(result: GarchResult, horizon: int = 30) -> pd.Series:
    """Genera pronóstico de volatilidad a `horizon` días."""
    fc = result.model_result.forecast(horizon=horizon, reindex=False)
    var = fc.variance.iloc[-1]
    return np.sqrt(var)
