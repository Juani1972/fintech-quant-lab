"""Modelos GARCH para volatilidad condicional."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)

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
    converged: bool
    rescale_factor: float


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
        GarchResult con el modelo ajustado y métricas. `converged` indica
        si el optimizador de máxima verosimilitud convergió — si es
        `False`, los parámetros pueden no ser fiables (ver el log de
        advertencia). `rescale_factor` guarda si se aplicó el ×100 de
        `rescale`, para poder revertirlo (p.ej. en `forecast_volatility`).
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

    converged = res.convergence_flag == 0
    if not converged:
        logger.warning(
            "El modelo GARCH (vol=%s, p=%d, q=%d) no convergió "
            "(convergence_flag=%d). Los parámetros pueden no ser fiables.",
            vol, p, q, res.convergence_flag,
        )

    rescale_factor = 100.0 if rescale else 1.0

    # `res.conditional_volatility`/`res.resid` están en la escala interna
    # del ajuste (×100 si rescale=True). El ratio para los residuos
    # estandarizados es invariante a esa escala; el resto de la salida
    # (conditional_volatility) se devuelve ya en las unidades originales
    # de `returns`, para que sea comparable con volatilidades calculadas
    # en otros módulos (p.ej. `risk.py`) y con `forecast_volatility`.
    standardized_residuals = res.resid / res.conditional_volatility

    return GarchResult(
        model_result=res,
        conditional_volatility=res.conditional_volatility / rescale_factor,
        standardized_residuals=standardized_residuals,
        params=res.params,
        pvalues=res.pvalues,
        aic=float(res.aic),
        bic=float(res.bic),
        model_type=vol,
        converged=converged,
        rescale_factor=rescale_factor,
    )


def check_stationarity(params: pd.Series, model_type: str = "Garch") -> bool:
    """Comprueba la condición de estacionariedad según el tipo de modelo.

    Para GARCH(1,1) y GJR-GARCH(1,1) la condición es exacta. Para
    EGARCH(1,1) también lo es (Nelson, 1991: la persistencia del
    proceso de log-varianza depende únicamente de beta, no de los
    términos en |ε|/asimetría). Para órdenes superiores (p>1 o q>1)
    en cualquiera de los tres modelos, sumar los coeficientes es una
    condición NECESARIA pero no suficiente (habría que comprobar las
    raíces del polinomio AR correspondiente) — en ese caso el resultado
    es una heurística conservadora, no una prueba formal.

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
    """Genera pronóstico de volatilidad a `horizon` días.

    Devuelve la volatilidad en las mismas unidades que la serie de
    retornos original pasada a `fit_garch` — es decir, si se ajustó
    con `rescale=True` (el valor por defecto, que multiplica por 100
    para estabilidad numérica del optimizador), este pronóstico se
    divide por ese mismo factor antes de devolverlo. Sin este ajuste,
    comparar esta salida con una volatilidad en escala decimal (p.ej.
    la de `risk.py`) daría un resultado 100x mayor de lo real.
    """
    fc = result.model_result.forecast(horizon=horizon, reindex=False)
    var = fc.variance.iloc[-1]
    return np.sqrt(var) / result.rescale_factor


def plain_language_summary(result: GarchResult, forecast: pd.Series) -> tuple[str, str]:
    """Traduce un GarchResult + su pronóstico a una frase de
    conclusión en español llano, sin jerga ("orden p/q", "AIC",
    "estacionariedad del proceso"), para quien no esté familiarizado
    con GARCH.

    Args:
        result: resultado de `fit_garch()`.
        forecast: resultado de `forecast_volatility(result, ...)`.

    Returns:
        (texto, variant) listo para pasar a `app.styles.conclusion()`.
        variant es 'danger' si el modelo no es fiable (no convergió o
        no es estacionario), 'warning' si converge pero anticipa una
        subida notable de volatilidad, 'success' en el resto de casos.
    """
    if not result.converged:
        return (
            "El optimizador <strong>no convergió</strong> al ajustar este "
            "modelo -- los parámetros y cualquier lectura que se haga de "
            "ellos no son fiables. Prueba con otro ticker, otro rango de "
            "fechas, u otra combinación de p/q antes de confiar en este "
            "resultado.",
            "danger",
        )

    if not check_stationarity(result.params, result.model_type):
        return (
            "El modelo ajustado <strong>no es estacionario</strong> -- "
            "según sus propios parámetros, la volatilidad no tendería a "
            "estabilizarse en el largo plazo, sino a crecer sin límite. "
            "Es una señal de que el modelo no encaja bien con estos "
            "datos concretos; trátalo con cautela.",
            "danger",
        )

    current_vol = float(result.conditional_volatility.iloc[-1])
    historical_avg = float(result.conditional_volatility.mean())
    forecast_end = float(forecast.iloc[-1])

    vs_historical = current_vol / historical_avg if historical_avg > 0 else 1.0
    if vs_historical > 1.3:
        level_text = "por encima de lo habitual para este activo"
    elif vs_historical < 0.7:
        level_text = "por debajo de lo habitual para este activo"
    else:
        level_text = "en línea con lo habitual para este activo"

    direction = forecast_end / current_vol if current_vol > 0 else 1.0
    if direction > 1.15:
        return (
            f"La volatilidad actual está {level_text}, y el modelo "
            f"anticipa que <strong>siga subiendo</strong> en los próximos "
            f"días -- no dice en qué dirección se moverá el precio, solo "
            f"que los movimientos (al alza o a la baja) podrían ser más "
            f"bruscos de lo normal.",
            "warning",
        )
    if direction < 0.85:
        return (
            f"La volatilidad actual está {level_text}, y el modelo "
            f"anticipa que <strong>vaya bajando</strong> hacia niveles más "
            f"tranquilos en los próximos días.",
            "success",
        )
    return (
        f"La volatilidad actual está {level_text}, y el modelo no "
        f"anticipa un cambio notable en los próximos días -- se espera "
        f"que el nivel actual de agitación del precio se mantenga "
        f"parecido.",
        "success",
    )
