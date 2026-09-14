"""Tests para cointegration."""
import numpy as np
import pandas as pd
import pytest

from app.core.cointegration import engle_granger, half_life


def test_engle_granger_cointegrated():
    np.random.seed(0)
    n = 500
    x = pd.Series(np.cumsum(np.random.normal(0, 1, n)), name="x")
    noise = pd.Series(np.random.normal(0, 0.5, n), name="noise")
    y = 2 * x + noise
    result = engle_granger(y, x)
    assert result.is_cointegrated


def test_half_life_known_value_mean_reverting():
    """Con un proceso AR(1)/Ornstein-Uhlenbeck simulado con phi
    conocido, half_life debe recuperar aproximadamente el valor
    teórico ln(2)/(1-phi). Antes solo se comprobaba `hl > 0 or
    isinf(hl)`, una aserción casi imposible de fallar sobre un random
    walk (que además ni siquiera es el caso de uso real de esta
    función — se usa sobre spreads que SÍ deberían revertir a la media).
    """
    np.random.seed(3)
    n = 5000
    phi = 0.95
    noise = np.random.normal(0, 1, n)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + noise[t]

    hl = half_life(pd.Series(spread))
    theoretical = np.log(2) / (1 - phi)
    assert hl == pytest.approx(theoretical, rel=0.15)


def test_half_life_large_for_non_mean_reverting_series():
    """Un random walk (no estacionario, no revierte a la media) no
    debe dar una half-life corta y finita que sugeriría (falsamente)
    reversión rápida.
    """
    np.random.seed(1)
    spread = pd.Series(np.random.normal(0, 1, 500)).cumsum()
    hl = half_life(spread)
    assert np.isinf(hl) or hl > 100
