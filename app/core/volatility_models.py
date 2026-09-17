"""Interfaz común y registro para modelos de volatilidad.

Mismo patrón que `app.core.strategies` (una clase base abstracta +
un registro con `get_*`/`list_*`): añadir un modelo nuevo significa
crear una clase que implemente `BaseVolatilityModel` y darla de alta
en `_MODELS`, sin tocar el código de los modelos existentes ni de
quien los use a través del registro.

`GARCHVolatilityModel` es un envoltorio fino sobre
`app.core.garch.fit_garch`/`forecast_volatility` -- no reimplementa
GARCH, delega en ese código ya probado (ver tests/test_garch.py).
`EWMAVolatilityModel` es una implementación nueva y genuinamente
distinta (RiskMetrics), para que la interfaz demuestre ser extensible
de verdad y no solo tener un único miembro decorativo.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class VolatilityModelError(Exception):
    """Error genérico al construir, ajustar o pronosticar con un
    modelo de volatilidad."""


class BaseVolatilityModel(ABC):
    """Contrato común que debe cumplir cualquier modelo de
    volatilidad condicional para poder usarse de forma intercambiable
    a través del registro de este módulo."""

    name: str

    @abstractmethod
    def fit(self, returns: pd.Series) -> BaseVolatilityModel:
        """Ajusta el modelo a una serie de retornos.

        Returns:
            El propio modelo (permite encadenar
            `modelo.fit(returns).forecast(30)`).
        """
        ...

    @abstractmethod
    def conditional_volatility(self) -> pd.Series:
        """Volatilidad condicional estimada in-sample, mismo índice
        que los retornos pasados a `fit()`. Requiere haber llamado a
        `fit()` antes."""
        ...

    @abstractmethod
    def forecast(self, horizon: int = 30) -> pd.Series:
        """Pronóstico de volatilidad a `horizon` periodos vista.
        Requiere haber llamado a `fit()` antes."""
        ...


class GARCHVolatilityModel(BaseVolatilityModel):
    """Envoltorio de `app.core.garch.fit_garch` sobre la interfaz
    común -- delega toda la lógica en ese módulo, ya probado por
    separado; esta clase no reimplementa nada de GARCH."""

    name = "garch"

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        vol: str = "Garch",
        dist: str = "normal",
        rescale: bool = True,
    ) -> None:
        self.p = p
        self.q = q
        self.vol = vol
        self.dist = dist
        self.rescale = rescale
        self._result: Any = None

    def fit(self, returns: pd.Series) -> GARCHVolatilityModel:
        from app.core.garch import fit_garch

        self._result = fit_garch(
            returns, p=self.p, q=self.q, vol=self.vol,
            dist=self.dist, rescale=self.rescale,
        )
        return self

    def conditional_volatility(self) -> pd.Series:
        self._check_fitted()
        return self._result.conditional_volatility

    def forecast(self, horizon: int = 30) -> pd.Series:
        self._check_fitted()
        from app.core.garch import forecast_volatility

        return forecast_volatility(self._result, horizon=horizon)

    def _check_fitted(self) -> None:
        if self._result is None:
            raise VolatilityModelError(
                "Llama a fit() antes de conditional_volatility()/forecast()."
            )


class EWMAVolatilityModel(BaseVolatilityModel):
    """Volatilidad exponencialmente ponderada, al estilo RiskMetrics
    (J.P. Morgan, 1996): var_t = lam * var_{t-1} + (1-lam) * r_{t-1}^2.

    lam=0.94 es el valor clásico de RiskMetrics para datos diarios
    (equivale a una "memoria" de en torno a 1/(1-lam) ≈ 17 días). A
    diferencia de GARCH, no hay ningún parámetro que se ajuste por
    máxima verosimilitud -- lam se fija de antemano -- así que el
    ajuste es instantáneo, pero también menos flexible: no captura
    reversión a una varianza de largo plazo distinta de la ponderación
    exponencial de los retornos pasados.
    """

    name = "ewma"

    def __init__(self, lam: float = 0.94) -> None:
        if not 0.0 < lam < 1.0:
            raise VolatilityModelError("lam debe estar entre 0 y 1 (excluidos los extremos).")
        self.lam = lam
        self._cond_vol: pd.Series | None = None

    def fit(self, returns: pd.Series) -> EWMAVolatilityModel:
        if len(returns.dropna()) < 2:
            raise VolatilityModelError("Se necesitan al menos 2 observaciones.")

        r = returns.dropna()
        # var_t depende de r_{t-1} (shift), no de r_t -- si no, la
        # "volatilidad condicional en t" incluiría información del
        # propio día t, dejando de ser una estimación ex-ante.
        squared = r.pow(2)
        ewma_var = squared.shift(1).ewm(alpha=1 - self.lam, adjust=False).mean()
        ewma_var.iloc[0] = squared.iloc[0]  # arranque: var(0) = r(0)^2
        self._cond_vol = np.sqrt(ewma_var)
        self._cond_vol.name = "ewma_volatility"
        return self

    def conditional_volatility(self) -> pd.Series:
        self._check_fitted()
        assert self._cond_vol is not None
        return self._cond_vol

    def forecast(self, horizon: int = 30) -> pd.Series:
        """Pronóstico EWMA: constante e igual a la última volatilidad
        estimada, para todo el horizonte. Es una limitación conocida
        del método (a diferencia de GARCH, que converge hacia una
        varianza incondicional de largo plazo), no un error de esta
        implementación -- EWMA no modela reversión a la media de la
        volatilidad."""
        self._check_fitted()
        assert self._cond_vol is not None
        last_vol = float(self._cond_vol.iloc[-1])
        return pd.Series([last_vol] * horizon, index=range(1, horizon + 1), name="ewma_forecast")

    def _check_fitted(self) -> None:
        if self._cond_vol is None:
            raise VolatilityModelError(
                "Llama a fit() antes de conditional_volatility()/forecast()."
            )


_MODELS: dict[str, type[BaseVolatilityModel]] = {
    "garch": GARCHVolatilityModel,
    "ewma": EWMAVolatilityModel,
}


def get_volatility_model(name: str, **kwargs: Any) -> BaseVolatilityModel:
    """Factory: instancia un modelo de volatilidad por nombre.

    Args:
        name: uno de 'garch', 'ewma'.
        **kwargs: argumentos del constructor del modelo elegido
            (p.ej. `p`, `q`, `vol`, `dist` para 'garch'; `lam` para
            'ewma').

    Returns:
        Una instancia de BaseVolatilityModel lista para `fit()`.

    Raises:
        VolatilityModelError: si `name` no es un modelo conocido.
    """
    key = name.strip().lower()
    cls = _MODELS.get(key)
    if cls is None:
        disponibles = ", ".join(sorted(_MODELS))
        raise VolatilityModelError(f"Modelo desconocido '{name}'. Disponibles: {disponibles}")
    return cls(**kwargs)


def list_volatility_models() -> list[str]:
    """Nombres de los modelos de volatilidad registrados, ordenados alfabéticamente."""
    return sorted(_MODELS)
