"""
tests/test_volatility_models.py

Tests unitarios de app/core/volatility_models.py. Todos los datos
son sintéticos y deterministas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.garch import fit_garch, forecast_volatility
from app.core.volatility_models import (
    BaseVolatilityModel,
    EWMAVolatilityModel,
    GARCHVolatilityModel,
    VolatilityModelError,
    get_volatility_model,
    list_volatility_models,
)


@pytest.fixture
def returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0003, 0.015, 500))


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

def test_list_volatility_models():
    assert list_volatility_models() == ["ewma", "garch"]


def test_get_volatility_model_garch():
    m = get_volatility_model("garch", p=1, q=1)
    assert isinstance(m, GARCHVolatilityModel)
    assert isinstance(m, BaseVolatilityModel)
    assert m.p == 1 and m.q == 1


def test_get_volatility_model_ewma():
    m = get_volatility_model("ewma", lam=0.9)
    assert isinstance(m, EWMAVolatilityModel)
    assert isinstance(m, BaseVolatilityModel)
    assert m.lam == 0.9


def test_get_volatility_model_case_insensitive():
    m = get_volatility_model("EWMA")
    assert isinstance(m, EWMAVolatilityModel)


def test_get_volatility_model_unknown_raises():
    with pytest.raises(VolatilityModelError, match="Modelo desconocido"):
        get_volatility_model("no-existe")


def test_cannot_instantiate_base_class_directly():
    with pytest.raises(TypeError):
        BaseVolatilityModel()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# GARCHVolatilityModel -- debe delegar en fit_garch/forecast_volatility
# sin cambiar ningún resultado (es un envoltorio, no una reimplementación)
# ---------------------------------------------------------------------------

def test_garch_model_matches_direct_fit_garch_call(returns):
    direct_result = fit_garch(returns, p=1, q=1, vol="Garch", dist="normal")
    direct_vol = direct_result.conditional_volatility
    direct_forecast = forecast_volatility(direct_result, horizon=10)

    wrapped = GARCHVolatilityModel(p=1, q=1, vol="Garch", dist="normal").fit(returns)

    assert np.allclose(wrapped.conditional_volatility().values, direct_vol.values)
    assert np.allclose(wrapped.forecast(horizon=10).values, direct_forecast.values)


def test_garch_model_fit_returns_self_for_chaining(returns):
    model = GARCHVolatilityModel()
    assert model.fit(returns) is model


def test_garch_model_raises_if_not_fitted():
    model = GARCHVolatilityModel()
    with pytest.raises(VolatilityModelError, match="Llama a fit"):
        model.conditional_volatility()
    with pytest.raises(VolatilityModelError, match="Llama a fit"):
        model.forecast(10)


# ---------------------------------------------------------------------------
# EWMAVolatilityModel
# ---------------------------------------------------------------------------

def test_ewma_model_matches_manual_riskmetrics_recursion():
    """La fórmula de RiskMetrics es var_t = lam*var_{t-1} + (1-lam)*r_{t-1}^2
    -- se compara contra un cálculo manual con un bucle explícito, no
    solo contra la propia implementación (para detectar un posible
    error de signo/desfase en el uso de pandas.ewm())."""
    np.random.seed(7)
    r = pd.Series(np.random.normal(0, 0.01, 50))
    lam = 0.94

    manual_var = [r.iloc[0] ** 2]
    for t in range(1, len(r)):
        manual_var.append(lam * manual_var[-1] + (1 - lam) * r.iloc[t - 1] ** 2)
    manual_vol = pd.Series(np.sqrt(manual_var), index=r.index)

    model = EWMAVolatilityModel(lam=lam).fit(r)
    assert np.allclose(model.conditional_volatility().values, manual_vol.values)


def test_ewma_model_forecast_is_constant_at_last_volatility():
    """Limitación documentada de EWMA (a diferencia de GARCH): el
    pronóstico no converge a una varianza de largo plazo, se queda
    fijo en la última volatilidad estimada."""
    np.random.seed(11)
    r = pd.Series(np.random.normal(0, 0.01, 200))
    model = EWMAVolatilityModel(lam=0.94).fit(r)

    forecast = model.forecast(horizon=20)
    assert forecast.nunique() == 1
    assert forecast.iloc[0] == pytest.approx(float(model.conditional_volatility().iloc[-1]))


def test_ewma_model_rejects_invalid_lambda():
    with pytest.raises(VolatilityModelError, match="lam debe estar"):
        EWMAVolatilityModel(lam=1.5)
    with pytest.raises(VolatilityModelError, match="lam debe estar"):
        EWMAVolatilityModel(lam=0.0)


def test_ewma_model_rejects_too_few_observations():
    model = EWMAVolatilityModel()
    with pytest.raises(VolatilityModelError, match="al menos 2"):
        model.fit(pd.Series([0.01]))


def test_ewma_model_raises_if_not_fitted():
    model = EWMAVolatilityModel()
    with pytest.raises(VolatilityModelError, match="Llama a fit"):
        model.conditional_volatility()
    with pytest.raises(VolatilityModelError, match="Llama a fit"):
        model.forecast(10)


# ---------------------------------------------------------------------------
# Extensibilidad real: ambos modelos se usan de forma intercambiable
# a través de la interfaz común, sin que quien los use necesite saber
# de qué clase concreta son -- este es el objetivo del patrón.
# ---------------------------------------------------------------------------

def test_both_models_are_interchangeable_through_the_common_interface(returns):
    for name in list_volatility_models():
        model = get_volatility_model(name)
        result = model.fit(returns)
        assert result is model
        vol = model.conditional_volatility()
        fc = model.forecast(horizon=5)
        assert len(vol) == len(returns)
        assert len(fc) == 5
        assert (vol.dropna() >= 0).all()
        assert (fc >= 0).all()
