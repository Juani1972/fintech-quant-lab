"""
app/core/strategies/cross_sectional_momentum.py

Momentum cross-sectional: compra los activos con mejor retorno
pasado (lookback) y vende los de peor retorno, dentro del mismo
cesto de activos.

Devuelve una señal cross-sectional: índice = ticker, valor = peso
(-1..1), con suma de valores absolutos igual a 1 (cartera dólar-neutral),
salvo que se use `top_n`, en cuyo caso reparte peso igual entre los
`top_n` largos y `top_n` cortos.
"""

from __future__ import annotations

import pandas as pd

from .base import StrategyError


class CrossSectionalMomentum:
    name = "cross_sectional_momentum"

    def __init__(self, lookback: int = 252, top_n: int | None = None) -> None:
        """
        Args:
            lookback: nº de observaciones hacia atrás para medir el retorno.
            top_n: si se indica, va largo en los top_n mejores y corto en
                los top_n peores (peso igual dentro de cada grupo). Si es
                None, usa un peso continuo proporcional al z-score del
                retorno de cada activo.
        """
        if lookback < 2:
            raise StrategyError("lookback debe ser >= 2.")
        self.lookback = lookback
        self.top_n = top_n

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty or prices.shape[1] < 2:
            raise StrategyError("CrossSectionalMomentum necesita al menos 2 activos.")
        if len(prices) < self.lookback + 1:
            raise StrategyError(
                f"Se necesitan al menos {self.lookback + 1} observaciones, hay {len(prices)}."
            )

        window = prices.iloc[-(self.lookback + 1):]
        trailing_return = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()

        if trailing_return.empty:
            raise StrategyError("No hay retornos válidos para calcular momentum.")

        if self.top_n:
            ranked = trailing_return.sort_values(ascending=False)
            n = max(1, min(self.top_n, len(ranked) // 2))
            longs = ranked.index[:n]
            shorts = ranked.index[-n:]

            signal = pd.Series(0.0, index=trailing_return.index)
            signal.loc[longs] = 1.0 / n
            signal.loc[shorts] = -1.0 / n
            signal.name = "signal"
            return signal

        std = trailing_return.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=trailing_return.index, name="signal")

        z = (trailing_return - trailing_return.mean()) / std
        denom = z.abs().sum()

        if denom == 0:
            return pd.Series(0.0, index=trailing_return.index, name="signal")

        signal = z / denom
        signal.name = "signal"
        return signal
