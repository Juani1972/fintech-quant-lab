"""
app/core/strategies/volatility_targeting.py

Volatility targeting: ajusta el apalancamiento de un único activo
para que la volatilidad realizada de la posición se mantenga
constante en torno a un objetivo (target_vol), subiendo el tamaño de
posición cuando el mercado está tranquilo y bajándolo cuando está
volátil.

Devuelve una señal de serie temporal sobre un solo activo: índice =
fecha, valor = apalancamiento objetivo (>= 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyError


class VolatilityTargeting:
    name = "volatility_targeting"

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback: int = 20,
        max_leverage: float = 3.0,
        annualization_factor: int = 252,
    ) -> None:
        """
        Args:
            target_vol: volatilidad anualizada objetivo (p. ej. 0.15 = 15%).
            lookback: ventana en observaciones para estimar la volatilidad realizada.
            max_leverage: tope de apalancamiento para evitar posiciones extremas
                cuando la volatilidad realizada es muy baja.
            annualization_factor: nº de periodos por año (252 para datos diarios).
        """
        if target_vol <= 0:
            raise StrategyError("target_vol debe ser > 0.")
        if lookback < 2:
            raise StrategyError("lookback debe ser >= 2.")
        if max_leverage <= 0:
            raise StrategyError("max_leverage debe ser > 0.")

        self.target_vol = target_vol
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.annualization_factor = annualization_factor

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            raise StrategyError("prices está vacío.")
        if prices.shape[1] != 1:
            raise StrategyError(
                "VolatilityTargeting opera sobre un único activo; "
                f"prices tiene {prices.shape[1]} columnas."
            )

        series = prices.iloc[:, 0]
        returns = series.pct_change().dropna()

        if len(returns) < self.lookback:
            raise StrategyError(
                f"Se necesitan al menos {self.lookback} retornos, hay {len(returns)}."
            )

        realized_vol = returns.rolling(self.lookback).std() * np.sqrt(self.annualization_factor)
        realized_vol = realized_vol.dropna()

        if realized_vol.empty:
            raise StrategyError("No se pudo calcular la volatilidad realizada.")

        leverage = (self.target_vol / realized_vol).clip(lower=0.0, upper=self.max_leverage)
        leverage.name = "signal"
        return leverage
