"""Métricas de riesgo financiero.

Todas las funciones asumen retornos en frecuencia diaria por defecto
(`periods_per_year=252`). Ajusta el parámetro si usas otra frecuencia.

Convenciones:
    - VaR y ES se devuelven como números positivos (= pérdida).
    - Sharpe y Sortino se anualizan.
    - La tasa libre de riesgo `rf` es ANUAL; se convierte internamente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
#  Validaciones internas
# ============================================================
def _validate_returns(returns: pd.Series) -> pd.Series:
    """Valida que la serie de retornos es usable."""
    if not isinstance(returns, pd.Series):
        raise ValueError("returns debe ser una pd.Series.")
    clean = returns.dropna()
    if len(clean) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones no nulas.")
    if not np.isfinite(clean).all():
        raise ValueError("returns contiene valores no finitos (inf o NaN).")
    return clean


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence debe estar en (0, 1). Recibido: {confidence}")


def _validate_periods(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser > 0.")


# ============================================================
#  VaR y Expected Shortfall
# ============================================================
def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR histórico (positivo = pérdida)."""
    clean = _validate_returns(returns)
    _validate_confidence(confidence)
    return float(-np.percentile(clean, (1 - confidence) * 100))


def value_at_risk_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR paramétrico asumiendo normalidad.

    VaR = -(mu + z_alpha * sigma)
    """
    from scipy.stats import norm

    clean = _validate_returns(returns)
    _validate_confidence(confidence)
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    z = norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall histórico (CVaR), positivo = pérdida."""
    clean = _validate_returns(returns)
    _validate_confidence(confidence)
    var = -value_at_risk(clean, confidence)
    tail = clean[clean <= var]
    if len(tail) == 0:
        return np.nan
    return float(-tail.mean())


def expected_shortfall_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall paramétrico asumiendo normalidad."""
    from scipy.stats import norm

    clean = _validate_returns(returns)
    _validate_confidence(confidence)
    mu = clean.mean()
    sigma = clean.std(ddof=1)
    z = norm.ppf(1 - confidence)
    es = -(mu - sigma * norm.pdf(z) / (1 - confidence))
    return float(es)


def value_at_risk_cornish_fisher(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR modificado de Cornish-Fisher (positivo = pérdida).

    `value_at_risk_parametric` asume que los retornos son normales, lo
    que subestima el riesgo de colas cuando la distribución real tiene
    asimetría negativa o exceso de curtosis (el caso típico de
    retornos financieros diarios). Cornish-Fisher ajusta el cuantil
    normal `z` usando la asimetría y curtosis EMPÍRICAS de la propia
    serie, sin necesitar asumir ninguna distribución paramétrica
    concreta para las colas.

    VaR_CF = -(mu + z_cf * sigma), donde z_cf es el cuantil normal `z`
    corregido por asimetría (S) y curtosis (K):

        z_cf = z + (z²-1)S/6 + (z³-3z)(K-3)/24 - (2z³-5z)S²/36

    Con S=0 y K=3 (normal), z_cf = z y esto coincide exactamente con
    `value_at_risk_parametric`.

    Nota: la expansión de Cornish-Fisher es una aproximación y puede
    dar resultados poco fiables con asimetría/curtosis muy extremas
    (z_cf deja de ser monótono en la cola). Es razonablemente fiable
    para la magnitud de desviación de la normalidad típica en retornos
    diarios de acciones/índices.
    """
    from scipy.stats import kurtosis, norm, skew

    clean = _validate_returns(returns)
    _validate_confidence(confidence)

    mu = clean.mean()
    sigma = clean.std(ddof=1)
    s = float(skew(clean))
    k = float(kurtosis(clean, fisher=False))  # curtosis "normal" (no excess); normal=3

    z = norm.ppf(1 - confidence)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * (k - 3) / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(-(mu + z_cf * sigma))


def value_at_risk_filtered_historical(
    returns: pd.Series,
    conditional_volatility: pd.Series,
    forecast_volatility: float,
    confidence: float = 0.95,
) -> float:
    """VaR de simulación histórica filtrada (Filtered Historical Simulation).

    El VaR histórico simple (`value_at_risk`) asume implícitamente que
    la volatilidad futura será igual al promedio de todo el histórico
    usado. Esto es poco realista con clustering de volatilidad (el
    fenómeno que precisamente modela GARCH): si el mercado está ahora
    mismo más tranquilo o más agitado que su media histórica, el VaR
    histórico simple estará sesgado.

    La FHS corrige esto: estandariza cada retorno histórico dividiendo
    por su propia volatilidad condicional (de un modelo GARCH ya
    ajustado — ver `app.core.garch.fit_garch`), calcula el percentil
    histórico de esos residuos estandarizados (que ya no deberían tener
    el clustering de volatilidad), y lo reescala por la volatilidad
    PRONOSTICADA para el periodo actual.

    Args:
        returns: Retornos históricos.
        conditional_volatility: Volatilidad condicional estimada por un
            modelo GARCH para cada retorno histórico (mismo índice /
            alineable con `returns`; p.ej.
            `GarchResult.conditional_volatility` de `app.core.garch`).
        forecast_volatility: Volatilidad pronosticada para el periodo a
            estimar (p.ej. el primer valor de
            `garch.forecast_volatility(...)`).
        confidence: Nivel de confianza.

    Returns:
        VaR (positivo = pérdida), en las mismas unidades que `returns`.

    Raises:
        ValueError: Si quedan menos de 20 observaciones alineadas, si
            `conditional_volatility` tiene valores <= 0, o si
            `forecast_volatility` <= 0.
    """
    _validate_confidence(confidence)
    if forecast_volatility <= 0:
        raise ValueError("forecast_volatility debe ser > 0.")

    df = pd.concat(
        [returns.rename("r"), conditional_volatility.rename("vol")], axis=1,
    ).dropna()
    if len(df) < 20:
        raise ValueError(
            f"Se necesitan al menos 20 observaciones alineadas (hay {len(df)})."
        )
    if (df["vol"] <= 0).any():
        raise ValueError("conditional_volatility debe ser > 0 en todas las observaciones.")

    standardized = df["r"] / df["vol"]
    q = float(np.percentile(standardized, (1 - confidence) * 100))
    return float(-q * forecast_volatility)


def rolling_var(
    returns: pd.Series,
    window: int = 250,
    confidence: float = 0.95,
) -> pd.Series:
    """VaR rodante."""
    _validate_returns(returns)
    _validate_confidence(confidence)
    if window < 2:
        raise ValueError("window debe ser >= 2.")
    return returns.rolling(window).apply(
        lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
    )


# ============================================================
#  Drawdown
# ============================================================
def max_drawdown(prices: pd.Series) -> float:
    """Máximo drawdown (negativo)."""
    if not isinstance(prices, pd.Series):
        raise ValueError("prices debe ser una pd.Series.")
    clean = prices.dropna()
    if len(clean) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones.")
    running_max = clean.cummax()
    drawdown = (clean - running_max) / running_max
    return float(drawdown.min())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Serie temporal de drawdowns."""
    running_max = prices.cummax()
    return (prices - running_max) / running_max


# ============================================================
#  Ratios ajustados por riesgo (anualizados)
# ============================================================
def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Ratio de Sharpe anualizado.

    Args:
        returns: Retornos en la frecuencia de `periods_per_year`.
        rf: Tasa libre de riesgo ANUAL (ej. 0.02 = 2%).
        periods_per_year: 252 (diario), 52 (semanal), 12 (mensual).

    Returns:
        Sharpe anualizado, o NaN si la volatilidad es cero.
    """
    clean = _validate_returns(returns)
    _validate_periods(periods_per_year)

    daily_rf = rf / periods_per_year
    excess = clean - daily_rf

    if excess.std(ddof=1) == 0:
        return np.nan

    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Ratio de Sortino anualizado.

    Usa solo la desviación de los retornos negativos (downside deviation).
    """
    clean = _validate_returns(returns)
    _validate_periods(periods_per_year)

    daily_rf = rf / periods_per_year
    excess = clean - daily_rf
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return np.nan

    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Ratio de Calmar: retorno anual / |max drawdown|."""
    clean = _validate_returns(returns)
    _validate_periods(periods_per_year)

    equity = (1 + clean).cumprod()
    years = len(clean) / periods_per_year
    if years <= 0:
        return np.nan

    annual_return = equity.iloc[-1] ** (1 / years) - 1
    mdd = max_drawdown(equity)
    if mdd == 0:
        return np.nan
    return float(annual_return / abs(mdd))
