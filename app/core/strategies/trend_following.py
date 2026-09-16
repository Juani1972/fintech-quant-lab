"""
app/core/strategies/trend_following.py

Trend following clásico por cruce de medias móviles: largo cuando la
media rápida está por encima de la lenta, corto cuando está por
debajo, plano en el resto de casos. Funciona sobre uno o varios
activos a la vez.

Devuelve una señal de serie temporal multi-activo: índice =
MultiIndex (fecha, ticker), valor = señal en {-1, 0, 1}.
"""

from __future__ import annotations

import pandas as pd

from .base import StrategyError


class TrendFollowing:
    name = "trend_following"

    def __init__(self, fast_window: int = 20, slow_window: int = 100) -> None:
        if fast_window < 1 or slow_window < 1:
            raise StrategyError("fast_window y slow_window deben ser >= 1.")
        if fast_window >= slow_window:
            raise StrategyError("fast_window debe ser menor que slow_window.")

        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            raise StrategyError("prices está vacío.")
        if len(prices) < self.slow_window + 1:
            raise StrategyError(
                f"Se necesitan al menos {self.slow_window + 1} observaciones, hay {len(prices)}."
            )

        fast = prices.rolling(self.fast_window).mean()
        slow = prices.rolling(self.slow_window).mean()

        raw = (fast > slow).astype(int) - (fast < slow).astype(int)
        raw = raw.iloc[self.slow_window:]

        stacked = raw.stack()
        stacked.index = stacked.index.set_names(["date", "ticker"])
        stacked.name = "signal"
        return stacked
