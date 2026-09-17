"""Tests para el módulo garch."""
import numpy as np
import pandas as pd
import pytest

from app.core.garch import (
    check_stationarity,
    fit_garch,
    forecast_volatility,
    is_stationary,
    plain_language_summary,
)


def test_fit_garch_runs():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    result = fit_garch(returns, p=1, q=1)
    assert result.aic is not None
    assert len(result.conditional_volatility) == 500


def test_fit_garch_converged_flag():
    """Con datos bien condicionados, el optimizador debe converger."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    result = fit_garch(returns, p=1, q=1)
    assert result.converged is True


def test_fit_garch_gjr_garch_does_not_raise():
    """Regresión: 'GJR-GARCH' no es un valor de `vol` propio de la
    librería `arch` (se traduce a vol='GARCH', o=1 internamente). Antes
    de corregirlo, esto lanzaba `ValueError: Unknown model type in vol`.
    """
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0, 0.01, 300))
    result = fit_garch(returns, vol="GJR-GARCH")
    assert result.model_type == "GJR-GARCH"
    assert len(result.conditional_volatility) == 300


@pytest.mark.parametrize("rescale", [True, False])
def test_conditional_volatility_matches_returns_scale(rescale):
    """`conditional_volatility` debe quedar en las mismas unidades que
    los retornos de entrada, independientemente de `rescale` (que solo
    afecta a la escala interna del optimizador, no a la salida).
    Regresión del bug de unidades: antes, con rescale=True (el valor
    por defecto), `conditional_volatility` quedaba inflada x100.
    """
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0, 0.015, 500))  # ~1.5% diario
    result = fit_garch(returns, rescale=rescale)

    # La volatilidad condicional media debe ser del mismo orden de
    # magnitud que la desviación estándar de los retornos de entrada
    # (no 100x mayor, que sería el bug de unidades).
    ratio = result.conditional_volatility.mean() / returns.std()
    assert 0.3 < ratio < 3.0

    expected_factor = 100.0 if rescale else 1.0
    assert result.rescale_factor == expected_factor


def test_forecast_volatility_same_scale_as_conditional_volatility():
    """`forecast_volatility` y `conditional_volatility` deben quedar en
    la misma escala — si no, dos gráficas de la misma página (volatilidad
    condicional vs pronóstico) mostrarían magnitudes inconsistentes
    entre sí (justo el bug que había antes de dividir por rescale_factor
    en ambos sitios).
    """
    np.random.seed(2)
    returns = pd.Series(np.random.normal(0, 0.015, 500))
    result = fit_garch(returns, rescale=True)

    fc = forecast_volatility(result, horizon=5)
    last_cond_vol = result.conditional_volatility.iloc[-1]

    # Incluso con clustering de volatilidad, el pronóstico a 5 días no
    # debería diferir en un orden de magnitud de la última volatilidad
    # condicional observada.
    ratio = fc.iloc[0] / last_cond_vol
    assert 0.2 < ratio < 5.0


def test_is_stationary_true():
    params = pd.Series({"alpha[1]": 0.1, "beta[1]": 0.8})
    assert is_stationary(params) is True


def test_is_stationary_false():
    params = pd.Series({"alpha[1]": 0.6, "beta[1]": 0.6})
    assert is_stationary(params) is False


def test_check_stationarity_gjr():
    # alpha + beta + gamma/2 = 0.1 + 0.7 + 0.15 = 0.95 < 1 → estacionario
    params = pd.Series({"alpha[1]": 0.1, "beta[1]": 0.7, "gamma[1]": 0.3})
    assert check_stationarity(params, "GJR-GARCH") is True


def test_check_stationarity_egarch():
    params = pd.Series({"beta[1]": 0.9})
    assert check_stationarity(params, "EGARCH") is True


def test_check_stationarity_unknown():
    params = pd.Series({"alpha[1]": 0.1})
    with pytest.raises(ValueError, match="no reconocido"):
        check_stationarity(params, "UNKNOWN")


def _fake_garch_result(
    converged=True, params=None, model_type="Garch",
    conditional_volatility=None,
):
    """GarchResult mínimo construido a mano (sin ajustar de verdad),
    para probar plain_language_summary() con cada rama de forma
    determinista en vez de depender de que un ajuste real produzca el
    caso concreto que se quiere probar."""
    from app.core.garch import GarchResult

    if params is None:
        params = pd.Series({"alpha[1]": 0.05, "beta[1]": 0.90})  # estacionario
    if conditional_volatility is None:
        conditional_volatility = pd.Series([0.01] * 100)

    return GarchResult(
        model_result=None,
        conditional_volatility=conditional_volatility,
        standardized_residuals=pd.Series([0.0] * 100),
        params=params,
        pvalues=pd.Series(dict.fromkeys(params.index, 0.01)),
        aic=100.0,
        bic=110.0,
        model_type=model_type,
        converged=converged,
        rescale_factor=1.0,
    )


def test_plain_language_summary_not_converged():
    result = _fake_garch_result(converged=False)
    text, variant = plain_language_summary(result, pd.Series([0.01] * 30))
    assert variant == "danger"
    assert "no convergió" in text


def test_plain_language_summary_not_stationary():
    non_stationary_params = pd.Series({"alpha[1]": 0.5, "beta[1]": 0.6})  # suma > 1
    result = _fake_garch_result(params=non_stationary_params)
    text, variant = plain_language_summary(result, pd.Series([0.01] * 30))
    assert variant == "danger"
    assert "no es estacionario" in text


def test_plain_language_summary_rising_volatility():
    cond_vol = pd.Series([0.01] * 100)  # actual y media histórica = 0.01
    result = _fake_garch_result(conditional_volatility=cond_vol)
    forecast = pd.Series([0.02] * 30)  # el pronóstico dobla la vol actual
    text, variant = plain_language_summary(result, forecast)
    assert variant == "warning"
    assert "subiendo" in text


def test_plain_language_summary_falling_volatility():
    cond_vol = pd.Series([0.01] * 100)
    result = _fake_garch_result(conditional_volatility=cond_vol)
    forecast = pd.Series([0.005] * 30)  # el pronóstico es la mitad de la vol actual
    text, variant = plain_language_summary(result, forecast)
    assert variant == "success"
    assert "bajando" in text


def test_plain_language_summary_stable_volatility():
    cond_vol = pd.Series([0.01] * 100)
    result = _fake_garch_result(conditional_volatility=cond_vol)
    forecast = pd.Series([0.0102] * 30)  # prácticamente igual
    text, variant = plain_language_summary(result, forecast)
    assert variant == "success"
    assert "no anticipa un cambio notable" in text
