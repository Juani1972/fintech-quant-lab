"""
app/core/strategies/__init__.py

Registro central de las estrategias del proyecto. Sigue el mismo
patrón factory que `app/core/providers.py::get_provider`, para que la
UI de Streamlit (o la futura API REST) puedan listar y construir
estrategias por nombre sin importar cada clase a mano.

Las 3 estrategias ya existentes (Pairs, Momentum, MeanRev) deben
migrarse a este mismo patrón (clase con atributo `name` y método
`generate_signals(prices) -> pd.Series`) y añadirse a `_STRATEGIES`
para quedar disponibles también vía `get_strategy`.

Uso típico:

    from app.core.strategies import get_strategy

    strat = get_strategy("trend_following", fast_window=10, slow_window=50)
    signals = strat.generate_signals(prices)
"""

from __future__ import annotations

from typing import cast

from .base import Strategy, StrategyError
from .carry_trade import CarryTrade
from .cross_sectional_momentum import CrossSectionalMomentum
from .pca_statarb import PCAStatArb
from .risk_parity_strategy import RiskParityStrategy
from .trend_following import TrendFollowing
from .volatility_targeting import VolatilityTargeting

_STRATEGIES: dict[str, type] = {
    "cross_sectional_momentum": CrossSectionalMomentum,
    "risk_parity": RiskParityStrategy,
    "volatility_targeting": VolatilityTargeting,
    "trend_following": TrendFollowing,
    "carry_trade": CarryTrade,
    "pca_statarb": PCAStatArb,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """
    Factory de estrategias.

    Args:
        name: uno de 'cross_sectional_momentum', 'risk_parity',
            'volatility_targeting', 'trend_following', 'carry_trade',
            'pca_statarb'.
        **kwargs: argumentos del constructor de la estrategia elegida.

    Returns:
        Una instancia de Strategy lista para usar.

    Raises:
        StrategyError: si `name` no es una estrategia conocida.
    """
    key = name.strip().lower()
    strategy_cls = _STRATEGIES.get(key)

    if strategy_cls is None:
        disponibles = ", ".join(sorted(_STRATEGIES))
        raise StrategyError(f"Estrategia desconocida '{name}'. Disponibles: {disponibles}")

    return cast(Strategy, strategy_cls(**kwargs))


def list_strategies() -> list[str]:
    """Devuelve los nombres de todas las estrategias registradas, ordenados."""
    return sorted(_STRATEGIES)


__all__ = [
    "Strategy",
    "StrategyError",
    "CrossSectionalMomentum",
    "RiskParityStrategy",
    "VolatilityTargeting",
    "TrendFollowing",
    "CarryTrade",
    "PCAStatArb",
    "get_strategy",
    "list_strategies",
]
