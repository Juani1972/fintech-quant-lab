"""Métricas de riesgo financiero."""
from __future__ import annotations

import numpy as np
import pandas as pd


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """VaR histórico (positivo = pérdida)."""
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR)."""
    var = -value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    return -tail.mean() if len(tail) else np.nan


def rolling_var(returns: pd.Series, window: int = 250, confidence: float = 0.95) -> pd.Series:
    """VaR rodante."""
    return returns.rolling(window).apply(
        lambda x: -np.percentile(x, (1 - confidence) * 100), raw=True
    )


def max_drawdown(prices: pd.Series) -> float:
    """Máximo drawdown (negativo)."""
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown.min()


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Serie temporal de drawdowns."""
    running_max = prices.cummax()
    return (prices - running_max) / running_max


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """Ratio de Sharpe anualizado."""
    excess = returns - rf / periods
    return np.sqrt(periods) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = 252) -> float:
    """Ratio de Sortino anualizado."""
    excess = returns - rf / periods
    downside = excess[excess < 0].std()
    return np.sqrt(periods) * excess.mean() / downside if downside else np.nan
