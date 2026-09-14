"""Tests para garch."""
import numpy as np
import pandas as pd

from app.core.garch import fit_garch, is_stationary


def test_fit_garch_runs():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 500))
    result = fit_garch(returns, p=1, q=1)
    assert result.aic is not None
    assert len(result.conditional_volatility) == 500


def test_is_stationary():
    import pandas as pd
    params = pd.Series({"alpha[1]": 0.1, "beta[1]": 0.8})
    assert is_stationary(params) is True
    params_bad = pd.Series({"alpha[1]": 0.6, "beta[1]": 0.6})
    assert is_stationary(params_bad) is False
