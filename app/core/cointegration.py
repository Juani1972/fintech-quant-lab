"""Análisis de cointegración y pairs trading."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass
class CointegrationResult:
    """Resultado del test de cointegración y spread."""
    pvalue: float
    alpha: float
    beta: float
    spread: pd.Series
    adf_spread_pvalue: float
    adf_spread_stat: float
    adf_spread_crit: dict[str, float]
    adf_pvalue_1: float
    adf_pvalue_2: float
    is_cointegrated: bool


def adf_test(series: pd.Series) -> float:
    """Retorna el p-valor del test ADF."""
    result = adfuller(series.dropna())
    return float(result[1])


def adf_full(series: pd.Series) -> tuple[float, float, dict[str, float]]:
    """Retorna (estadístico, p-valor, valores críticos) del test ADF."""
    result = adfuller(series.dropna())
    stat = float(result[0])
    pval = float(result[1])
    crit = {str(k): float(v) for k, v in result[4].items()}
    return stat, pval, crit


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    trend: str = "c",
    significance: float = 0.05,
    fit_until: pd.Timestamp | None = None,
) -> CointegrationResult:
    """Ejecuta el test de Engle-Granger y construye el spread.

    Modelo: y_t = alpha + beta * x_t + u_t
    Spread: u_t = y_t - alpha - beta * x_t

    Aviso de look-ahead: si no se especifica `fit_until`, alpha y beta
    se estiman por OLS sobre TODA la muestra, y ese mismo spread se usa
    después para generar señales de trading en cualquier punto de la
    serie — incluido su tramo inicial. Es una forma de ajuste in-sample:
    los coeficientes "conocen" datos futuros respecto a las señales que
    se generarían al principio. Esto es habitual en la literatura de
    pairs trading pedagógica, pero para una validación honesta (o para
    encadenar esto con walk-forward) pasa `fit_until` con la fecha de
    corte del tramo de entrenamiento: alpha, beta y el propio test de
    cointegración se calculan solo hasta esa fecha, y el spread se
    construye sobre toda la serie con esos coeficientes ya fijos.

    Args:
        y, x: Series de precios (o log-precios) a testear.
        trend: Término determinista para el test de cointegración.
        significance: Umbral de p-valor para `is_cointegrated`.
        fit_until: Si se da, alpha/beta y el test de cointegración solo
            usan datos con índice <= `fit_until` (evita look-ahead).

    Raises:
        ValueError: Si tras aplicar `fit_until` quedan menos de 20
            observaciones para estimar alpha/beta.
    """
    df = pd.concat([y, x], axis=1).dropna()
    y_, x_ = df.iloc[:, 0], df.iloc[:, 1]

    if fit_until is not None:
        fit_mask = df.index <= fit_until
        y_fit, x_fit = y_[fit_mask], x_[fit_mask]
        if len(y_fit) < 20:
            raise ValueError(
                f"Solo hay {len(y_fit)} observaciones hasta fit_until={fit_until}; "
                "se necesitan al menos 20 para estimar alpha/beta con fiabilidad."
            )
    else:
        y_fit, x_fit = y_, x_

    _, pvalue, _ = coint(y_fit, x_fit, trend=trend)

    fit = sm.OLS(y_fit, sm.add_constant(x_fit)).fit()
    alpha = float(fit.params.iloc[0])
    beta = float(fit.params.iloc[1])

    # El spread se construye sobre TODA la serie con alpha/beta ya fijos
    # (estimados solo hasta `fit_until`, si se especificó).
    spread = y_ - alpha - beta * x_

    adf_stat, adf_pval, adf_crit = adf_full(spread)

    return CointegrationResult(
        pvalue=float(pvalue),
        alpha=alpha,
        beta=beta,
        spread=spread,
        adf_spread_pvalue=adf_pval,
        adf_spread_stat=adf_stat,
        adf_spread_crit=adf_crit,
        adf_pvalue_1=adf_test(y_fit),
        adf_pvalue_2=adf_test(x_fit),
        is_cointegrated=pvalue < significance,
    )


def rolling_zscore(
    spread: pd.Series,
    window: int = 60,
    shift: int = 1,
) -> pd.Series:
    """Calcula el z-score rodante del spread."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()

    if shift > 0:
        mean = mean.shift(shift)
        std = std.shift(shift)

    return (spread - mean) / std


def half_life(spread: pd.Series) -> float:
    """Calcula la vida media de reversión a la media (en días)."""
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    df = pd.concat([spread_lag, spread_diff], axis=1).dropna()
    df.columns = ["lag", "diff"]

    beta = float(sm.OLS(df["diff"], sm.add_constant(df["lag"])).fit().params.iloc[1])
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)


def generate_signals(
    zscore: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
) -> pd.Series:
    """Genera señales de trading a partir del z-score.

    Returns:
        Serie con valores: 1 (long spread), -1 (short spread), 0 (neutral).
    """
    signals = pd.Series(0, index=zscore.index, dtype=int)
    position = 0
    for i, z in enumerate(zscore):
        if np.isnan(z):
            signals.iloc[i] = position
            continue
        if position == 0:
            if z > entry:
                position = -1
            elif z < -entry:
                position = 1
        elif abs(z) < exit_:
            position = 0
        signals.iloc[i] = position
    return signals
