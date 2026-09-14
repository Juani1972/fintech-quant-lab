"""Tests para cointegration."""
import numpy as np
import pandas as pd

from app.core.cointegration import engle_granger, half_life


def test_engle_granger_cointegrated():
    np.random.seed(0)
    n = 500
    x = pd.Series(np.cumsum(np.random.normal(0, 1, n)), name="x")
    noise = pd.Series(np.random.normal(0, 0.5, n), name="noise")
    y = 2 * x + noise
    result = engle_granger(y, x)
    assert result.is_cointegrated


def test_half_life_positive():
    np.random.seed(1)
    spread = pd.Series(np.random.normal(0, 1, 500)).cumsum()
    hl = half_life(spread)
    assert hl > 0 or np.isinf(hl)
