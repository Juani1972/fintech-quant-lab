"""Regresiones Fama-French con descarga de factores diarios y errores HAC."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm

from app.config import CACHE_TTL_FACTORS
from app.core.cache import cached

FACTOR_COLS = {
    "3": ["Mkt-RF", "SMB", "HML"],
    "5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}

DATASETS = {
    "3": "F-F_Research_Data_Factors_daily",
    "5": "F-F_Research_Data_5_Factors_2x3_daily",
}


@dataclass
class FamaFrenchResult:
    """Resultado de la regresión Fama-French con HAC."""
    alpha: float
    alpha_pvalue: float
    alpha_tstat: float
    betas: pd.Series
    betas_pvalues: pd.Series
    betas_tstats: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    cov_type: str
    maxlags: int | None
    summary: str


@cached(ttl=CACHE_TTL_FACTORS, show_spinner=False)
def load_factors(
    start: date,
    end: date,
    model: str = "3",
    frequency: str = "daily",
) -> pd.DataFrame:
    """Descarga los factores Fama-French del data library de Kenneth French.

    Cacheado 24 h (los factores se actualizan a lo sumo diariamente).

    Args:
        start: Fecha de inicio.
        end: Fecha de fin.
        model: '3' o '5' factores.
        frequency: 'daily' o 'monthly'.

    Returns:
        DataFrame con factores en decimal, indexado por fecha.

    Raises:
        ValueError: Si el modelo o la frecuencia no son soportados.
        ConnectionError: Si falla la descarga.
    """
    if model not in FACTOR_COLS:
        raise ValueError(f"Modelo '{model}' no soportado. Usa '3' o '5'.")

    if frequency == "daily":
        dataset = DATASETS[model]
    elif frequency == "monthly":
        dataset = (
            "F-F_Research_Data_Factors"
            if model == "3"
            else "F-F_Research_Data_5_Factors_2x3"
        )
    else:
        raise ValueError(f"Frecuencia '{frequency}' no soportada.")

    try:
        raw = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
    except Exception as e:
        raise ConnectionError(f"No se pudieron descargar los factores: {e}") from e

    if raw is None or raw.empty:
        raise ValueError(f"Sin datos de factores entre {start} y {end}.")

    ff = raw / 100
    # El dataset mensual de la librería de Kenneth French devuelve un
    # índice PeriodIndex; pandas >= 2.x ya no permite convertirlo con
    # pd.to_datetime() directamente (TypeError: "Passing PeriodDtype
    # data is invalid. Use `data.to_timestamp()` instead") -- hay que
    # detectarlo y usar to_timestamp(). El dataset diario ya trae un
    # índice convertible directamente con pd.to_datetime().
    if isinstance(ff.index, pd.PeriodIndex):
        ff.index = ff.index.to_timestamp()
    else:
        ff.index = pd.to_datetime(ff.index)
    return ff


def run_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "3",
    risk_free: str = "RF",
    cov_type: str = "HAC",
    maxlags: int | None = None,
) -> FamaFrenchResult:
    """Ejecuta la regresión de factores sobre los retornos en exceso.

    Modelo: R_i - R_f = alpha + sum(beta_k * Factor_k) + epsilon

    Args:
        returns: Serie de retornos del activo (MISMA frecuencia que factors).
        factors: DataFrame de factores (salida de load_factors).
        model: '3' o '5'.
        risk_free: Nombre de la columna de tasa libre de riesgo.
        cov_type: 'HAC', 'HC3', 'nonrobust'.
        maxlags: Nº de lags para HAC. None → regla automática.

    Returns:
        FamaFrenchResult con p-values robustos.
    """
    if model not in FACTOR_COLS:
        raise ValueError(f"Modelo '{model}' no soportado.")

    factor_cols = FACTOR_COLS[model]

    df = pd.concat([returns.rename("ret"), factors], axis=1).dropna()
    if df.empty:
        raise ValueError("No hay datos superpuestos entre retornos y factores.")

    if len(df) < 20:
        raise ValueError(
            f"Solo {len(df)} observaciones superpuestas. "
            "¿Estás usando factores diarios con retornos diarios?"
        )

    df["excess"] = df["ret"] - df[risk_free]

    X = sm.add_constant(df[factor_cols])
    y = df["excess"]

    n = len(df)
    if cov_type == "HAC":
        if maxlags is None:
            maxlags = int(4 * (n / 100) ** (2 / 9))
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov_type == "nonrobust":
        fit = sm.OLS(y, X).fit()
    else:
        fit = sm.OLS(y, X).fit(cov_type=cov_type)

    return FamaFrenchResult(
        alpha=float(fit.params["const"]),
        alpha_pvalue=float(fit.pvalues["const"]),
        alpha_tstat=float(fit.tvalues["const"]),
        betas=fit.params[factor_cols],
        betas_pvalues=fit.pvalues[factor_cols],
        betas_tstats=fit.tvalues[factor_cols],
        r_squared=float(fit.rsquared),
        adj_r_squared=float(fit.rsquared_adj),
        n_obs=int(fit.nobs),
        cov_type=cov_type,
        maxlags=maxlags,
        summary=fit.summary().as_text(),
    )
