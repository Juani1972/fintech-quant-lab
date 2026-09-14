"""Regresiones Fama-French con descarga de factores de Kenneth French."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm


@dataclass
class FamaFrenchResult:
    """Resultado de la regresión Fama-French."""
    alpha: float
    alpha_pvalue: float
    betas: pd.Series
    betas_pvalues: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    summary: str


FACTOR_COLS = {
    "3": ["Mkt-RF", "SMB", "HML"],
    "5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
}


def load_factors(start: date, end: date, model: str = "3") -> pd.DataFrame:
    """Descarga los factores Fama-French del data library de Kenneth French.

    Args:
        start: Fecha de inicio.
        end: Fecha de fin.
        model: '3' o '5' factores.

    Returns:
        DataFrame con factores en formato decimal (no porcentaje), indexado por fecha.

    Raises:
        ValueError: Si el modelo no es '3' ni '5'.
        ConnectionError: Si falla la descarga.
    """
    if model not in FACTOR_COLS:
        raise ValueError(f"Modelo '{model}' no soportado. Usa '3' o '5'.")

    dataset = (
        "F-F_Research_Data_Factors"
        if model == "3"
        else "F-F_Research_Data_5_Factors_2x3"
    )

    try:
        raw = web.DataReader(dataset, "famafrench", start=start, end=end)[0]
    except Exception as e:
        raise ConnectionError(f"No se pudieron descargar los factores: {e}") from e

    ff = raw / 100  # convertir de % a decimal
    ff.index = pd.to_datetime(ff.index)
    return ff


def run_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "3",
    risk_free: str = "RF",
) -> FamaFrenchResult:
    """Ejecuta la regresión de factores sobre los retornos en exceso.

    Modelo: R_i - R_f = alpha + sum(beta_k * Factor_k) + epsilon

    Args:
        returns: Serie de retornos del activo.
        factors: DataFrame de factores (salida de load_factors).
        model: '3' o '5'.
        risk_free: Nombre de la columna de tasa libre de riesgo.

    Returns:
        FamaFrenchResult con alpha, betas y métricas.
    """
    if model not in FACTOR_COLS:
        raise ValueError(f"Modelo '{model}' no soportado.")

    factor_cols = FACTOR_COLS[model]

    df = pd.concat([returns.rename("ret"), factors], axis=1).dropna()
    if df.empty:
        raise ValueError("No hay datos superpuestos entre retornos y factores.")

    df["excess"] = df["ret"] - df[risk_free]

    X = sm.add_constant(df[factor_cols])
    y = df["excess"]
    fit = sm.OLS(y, X).fit()

    return FamaFrenchResult(
        alpha=fit.params["const"],
        alpha_pvalue=fit.pvalues["const"],
        betas=fit.params[factor_cols],
        betas_pvalues=fit.pvalues[factor_cols],
        r_squared=fit.rsquared,
        adj_r_squared=fit.rsquared_adj,
        n_obs=int(fit.nobs),
        summary=fit.summary().as_text(),
    )
