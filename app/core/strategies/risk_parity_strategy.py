"""
app/core/strategies/risk_parity_strategy.py

Risk parity simplificado (inverse-vol): cada activo recibe un peso
inversamente proporcional a su volatilidad reciente, de forma que
todos contribuyan aproximadamente lo mismo al riesgo total de la
cartera. Es una aproximación analítica cerrada, útil como estrategia
"lista para usar"; para un risk parity exacto (con matriz de
covarianza completa) usar `app/core/portfolio.py::risk_parity_weights`.

Devuelve una señal cross-sectional: índice = ticker, valor = peso
(0..1, suman 1).
"""

from __future__ import annotations

import pandas as pd

from .base import StrategyError


class RiskParityStrategy:
    name = "risk_parity"

    def __init__(self, lookback: int = 60) -> None:
        if lookback < 2:
            raise StrategyError("lookback debe ser >= 2.")
        self.lookback = lookback

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty or prices.shape[1] < 2:
            raise StrategyError("RiskParityStrategy necesita al menos 2 activos.")

        returns = prices.pct_change().dropna(how="all")
        window = returns.iloc[-self.lookback:] if len(returns) > self.lookback else returns

        if window.empty:
            raise StrategyError("No hay retornos suficientes para calcular volatilidad.")

        vol = window.std()
        vol = vol[vol > 0].dropna()

        if vol.empty:
            raise StrategyError("Todas las volatilidades son cero o inválidas.")

        inv_vol = 1.0 / vol
        weights = (inv_vol / inv_vol.sum()).reindex(prices.columns).dropna()
        weights.name = "signal"
        return weights
