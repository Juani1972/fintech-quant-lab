"""
app/core/strategies/base.py

Contrato común que debe cumplir cualquier estrategia del proyecto,
tanto las nuevas como las 3 ya existentes (Pairs, Momentum, MeanRev)
una vez migradas a este patrón.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


class StrategyError(Exception):
    """Error genérico al generar señales de una estrategia."""


@runtime_checkable
class Strategy(Protocol):
    """
    Contrato de una estrategia.

    `generate_signals` recibe precios y devuelve una pd.Series de
    señales. El significado del índice depende del tipo de estrategia:

        - Estrategias cross-sectional (comparan activos entre sí en un
          instante dado): índice = ticker, valor = peso/señal (-1..1).
        - Estrategias de serie temporal sobre un solo activo: índice =
          fecha, valor = señal o tamaño de posición.
        - Estrategias de serie temporal multi-activo: índice =
          MultiIndex (fecha, ticker), valor = señal.

    Cada implementación documenta explícitamente cuál de los tres usa.
    """

    name: str

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series: ...
