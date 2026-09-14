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
    adf_pvalue_1: float
    adf_pvalue_2: float
    is_cointegrated: bool


def adf_test(series: pd.Series) -> float:
    """Retorna el p-valor del test ADF."""
    return adfuller(series.dropna())[1]


def engle_granger(y: pd.Series, x: pd.Series, trend: str = "c") -> CointegrationResult:
    """Ejecuta el test de Engle-Granger y construye el spread.

    Modelo: y_t = alpha + beta * x_t + u_t
    Spread: u_t = y_t - alpha - beta * x_t

    Args:
        y: Serie dependiente (precio).
        x: Serie independiente (precio).
        trend: 'c' constante, 'ct' constante+tendencia, 'n' ninguna.

    Returns:
        CointegrationResult con alpha, beta y spread correctamente centrado.
    """
    df = pd.concat([y, x], axis=1).dropna()
    y_, x_ = df.iloc[:, 0], df.iloc[:, 1]

    score, pvalue, _ = coint(y_, x_, trend=trend)

    # Regresión con constante para obtener alpha y beta
    fit = sm.OLS(y_, sm.add_constant(x_)).fit()
    alpha = fit.params.iloc[0]
    beta = fit.params.iloc[1]

    # Spread correcto: incluye alpha
    spread = y_ - alpha - beta * x_

    return CointegrationResult(
        pvalue=pvalue,
        alpha=alpha,
        beta=beta,
        spread=spread,
        adf_pvalue_1=adf_test(y_),
        adf_pvalue_2=adf_test(x_),
        is_cointegrated=pvalue < 0.05,
    )


def rolling_zscore(
    spread: pd.Series,
    window: int = 60,
    shift: int = 1,
) -> pd.Series:
    """Calcula el z-score rodante del spread.

    Args:
        spread: Serie del spread.
        window: Ventana de cálculo.
        shift: Si > 0, desplaza la media y std para usar solo información
            disponible ANTES de la observación actual (evita look-ahead
            en la generación de señales).
    """
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

    beta = sm.OLS(df["diff"], sm.add_constant(df["lag"])).fit().params.iloc[1]
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


def generate_signals(zscore: pd.Series, entry: float = 2.0, exit_: float = 0.5) -> pd.Series:
    """Genera señales de trading a partir del z-score.

    Returns:
        Serie con valores: 1 (long spread), -1 (short spread), 0 (neutral).
    """
    signals = pd.Series(0, index=zscore.index)
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
