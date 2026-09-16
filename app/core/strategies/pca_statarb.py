"""
app/core/strategies/pca_statarb.py

Stat-arb basado en componentes principales: construye el primer
componente principal (PC1) de los retornos de un cesto de activos,
calcula el spread residual respecto a ese factor común y opera en
reversión a la media cuando el spread se aleja demasiado (en
desviaciones estándar) de su media reciente.

Implementa el PCA a mano con `numpy.linalg.svd`, sin depender de
scikit-learn, para no añadir una dependencia nueva al proyecto solo
por esta estrategia.

Devuelve una señal de serie temporal sobre el spread agregado:
índice = fecha, valor = señal en {-1, 0, 1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyError


class PCAStatArb:
    name = "pca_statarb"

    def __init__(self, lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5) -> None:
        """
        Args:
            lookback: ventana de retornos usada para estimar el PC1 y el spread.
            entry_z: umbral (en desviaciones estándar) para abrir posición.
            exit_z: umbral por debajo del cual se considera que ya no hay
                señal clara (la posición se mantiene plana, no se calcula
                aquí gestión de salida con estado — eso vive en la capa de
                ejecución/backtest, que sí lleva estado entre barras).
        """
        if lookback < 5:
            raise StrategyError("lookback debe ser >= 5.")
        if entry_z <= exit_z:
            raise StrategyError("entry_z debe ser mayor que exit_z.")

        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        if prices.shape[1] < 2:
            raise StrategyError("PCAStatArb necesita al menos 2 activos.")
        if len(prices) < self.lookback:
            raise StrategyError(
                f"Se necesitan al menos {self.lookback} observaciones, hay {len(prices)}."
            )

        returns = prices.pct_change().dropna(how="all").fillna(0.0)
        window = returns.iloc[-self.lookback:]

        if window.shape[0] < 2:
            raise StrategyError("Ventana insuficiente tras eliminar NaN.")

        centered = window - window.mean()

        # PCA vía SVD: evita depender de scikit-learn solo para esto.
        _, _, vt = np.linalg.svd(centered.values, full_matrices=False)
        pc1 = vt[0]

        spread = pd.Series(centered.values @ pc1, index=window.index)
        std = spread.std(ddof=0)

        if std == 0 or pd.isna(std):
            raise StrategyError("El spread no tiene varianza; no se puede calcular z-score.")

        z = (spread - spread.mean()) / std

        signal = pd.Series(0, index=z.index, dtype=int)
        signal[z > self.entry_z] = -1  # spread muy por encima de su media -> revertir a la baja
        signal[z < -self.entry_z] = 1  # spread muy por debajo de su media -> revertir al alza
        signal.name = "signal"
        return signal
