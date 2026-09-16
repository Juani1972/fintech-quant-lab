"""Filtro de Kalman para hedge ratio dinámico en pairs trading.

El hedge ratio (beta) de una cointegración se puede estimar de tres formas:
    1. Estático: OLS sobre toda la muestra (lo que ya hace `cointegration.py`).
    2. Rolling: OLS en ventana móvil (rápido pero con saltos).
    3. Kalman: filtra el beta como un estado latente que evoluciona
       suavemente en el tiempo.

El Kalman es superior porque:
    - No requiere elegir un tamaño de ventana.
    - Se adapta a cambios estructurales sin saltos.
    - Da incertidumbre sobre el beta en cada punto.

Implementación del filtro de Kalman lineal estándar:

    Estado:        beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q)
    Observación:   y_t = alpha_t + beta_t * x_t + v_t,   v_t ~ N(0, R)

Referencias:
    Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their
    Rationale. Wiley.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    """Resultado del filtro de Kalman aplicado a un par de precios."""
    beta: pd.Series                    # hedge ratio dinámico
    alpha: pd.Series                   # intercepto dinámico
    spread: pd.Series                  # y - alpha - beta * x
    residuals: pd.Series               # residuos del filtro
    beta_std: pd.Series                # desviación estándar del beta
    log_likelihood: float


def kalman_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    r_var: float = 1e-3,
    initial_beta: float | None = None,
    initial_alpha: float | None = None,
) -> KalmanResult:
    """Estima el hedge ratio dinámico con filtro de Kalman.

    Args:
        y: Serie dependiente (precio).
        x: Serie independiente (precio).
        delta: Parámetro de transición (Q = delta / (1 - delta) * I).
            Valores más altos permiten que beta cambie más rápido entre
            observaciones, en teoría. En la práctica, con `y`/`x` en
            niveles de precio (no log-precios) y magnitudes grandes
            (p. ej. ~100+), el término x_t² en la ganancia de Kalman
            hace que el efecto de `delta` se sature a partir de cierto
            umbral -- y puede incluso invertirse para valores muy
            pequeños. La relación "mayor delta = beta más reactivo" NO
            es monótona de forma fiable en ese régimen. Si necesitas
            que `delta` se comporte de forma predecible, considera
            pasar log-precios (`np.log(precios)`) en vez de precios en
            niveles, que es la práctica estándar en la literatura de
            pairs trading precisamente para evitar esta dependencia de
            escala.
        r_var: Varianza del ruido de observación (R).
        initial_beta: Beta inicial. Si None, se usa OLS.
        initial_alpha: Alpha inicial. Si None, se usa OLS.

    Returns:
        KalmanResult con beta(t), alpha(t) y spread dinámico.

    Raises:
        ValueError: Si los datos son insuficientes o inválidos.
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("Se necesitan al menos 30 observaciones.")

    # Estimación inicial por OLS
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    coef, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    alpha0, beta0 = coef[0], coef[1]

    if initial_beta is not None:
        beta0 = initial_beta
    if initial_alpha is not None:
        alpha0 = initial_alpha

    # Estado: [alpha_t, beta_t]
    theta = np.array([alpha0, beta0], dtype=float)

    # Covarianza del estado
    P = np.eye(2) * 1.0

    # Ruido de transición
    Q = (delta / (1 - delta)) * np.eye(2)

    # Ruido de observación
    R = r_var

    betas = np.zeros(len(df))
    alphas = np.zeros(len(df))
    betas_std = np.zeros(len(df))
    residuals = np.zeros(len(df))
    log_lik = 0.0

    y_vals = df["y"].values
    x_vals = df["x"].values

    for t in range(len(df)):
        # --- Predicción ---
        P_pred = P + Q

        # --- Observación ---
        H = np.array([1.0, x_vals[t]])
        y_hat = H @ theta
        residual = y_vals[t] - y_hat
        S = H @ P_pred @ H.T + R

        # --- Log-likelihood ---
        log_lik += -0.5 * (np.log(2 * np.pi * S) + residual ** 2 / S)

        # --- Ganancia de Kalman ---
        K = (P_pred @ H.T) / S

        # --- Actualización ---
        theta = theta + K * residual
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        alphas[t] = theta[0]
        betas[t] = theta[1]
        betas_std[t] = np.sqrt(P[1, 1])
        residuals[t] = residual

    spread = df["y"] - alphas - betas * df["x"]

    return KalmanResult(
        beta=pd.Series(betas, index=df.index, name="beta"),
        alpha=pd.Series(alphas, index=df.index, name="alpha"),
        spread=spread.rename("spread"),
        residuals=pd.Series(residuals, index=df.index, name="residual"),
        beta_std=pd.Series(betas_std, index=df.index, name="beta_std"),
        log_likelihood=float(log_lik),
    )


def compare_static_dynamic(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    r_var: float = 1e-3,
) -> pd.DataFrame:
    """Compara beta estático (OLS) vs beta dinámico (Kalman).

    Returns:
        DataFrame con el beta OLS, el beta Kalman y la diferencia.
    """
    from app.core.cointegration import engle_granger

    static = engle_granger(y, x)
    dynamic = kalman_hedge_ratio(y, x, delta=delta, r_var=r_var)

    df = pd.DataFrame({
        "beta_ols": static.beta,
        "beta_kalman": dynamic.beta,
    })
    df["diferencia"] = df["beta_kalman"] - df["beta_ols"]
    return df


def rolling_ols_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Hedge ratio por OLS rodante (para comparación con Kalman)."""
    if window < 10:
        raise ValueError("window debe ser >= 10.")

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    betas = np.full(len(df), np.nan)

    y_vals = df["y"].values
    x_vals = df["x"].values

    for i in range(window, len(df)):
        ys = y_vals[i - window:i]
        xs = x_vals[i - window:i]
        X = np.column_stack([np.ones(window), xs])
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        betas[i] = coef[1]

    return pd.Series(betas, index=df.index, name="beta_rolling")
