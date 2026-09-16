"""
app/core/strategies/carry_trade.py

Carry trade: opera largo cuando el "carry" (diferencial de tipos,
dividend yield, roll yield de futuros, etc.) es positivo y corto
cuando es negativo, con una banda muerta opcional para evitar
señales de ruido cuando el carry está cerca de cero.

`prices` debe incluir una columna `carry_yield` con el carry ya
calculado externamente (p. ej. diferencial de tipos entre dos
divisas, o el yield de dividendos menos el coste de financiación).
Este módulo no calcula el carry en sí — eso depende de la fuente de
datos y del tipo de activo.

Devuelve una señal de serie temporal sobre un solo activo: índice =
fecha, valor = señal en {-1, 0, 1}.
"""

from __future__ import annotations

import pandas as pd

from .base import StrategyError


class CarryTrade:
    name = "carry_trade"

    def __init__(self, deadband: float = 0.0) -> None:
        """
        Args:
            deadband: banda muerta alrededor de cero (en las mismas
                unidades que `carry_yield`) dentro de la cual la señal
                se mantiene plana (0), para evitar operar por ruido.
        """
        if deadband < 0:
            raise StrategyError("deadband debe ser >= 0.")
        self.deadband = deadband

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if "carry_yield" not in prices.columns:
            raise StrategyError("CarryTrade necesita una columna 'carry_yield' en prices.")

        carry = prices["carry_yield"].dropna()
        if carry.empty:
            raise StrategyError("La columna 'carry_yield' no tiene datos válidos.")

        signal = pd.Series(0, index=carry.index, dtype=int)
        signal[carry > self.deadband] = 1
        signal[carry < -self.deadband] = -1
        signal.name = "signal"
        return signal
