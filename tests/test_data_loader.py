"""Tests para data_loader."""
import numpy as np
import pandas as pd

from app.core.data_loader import compute_log_returns


def test_log_returns_basic():
    prices = pd.DataFrame({"A": [100, 110, 121]}, index=pd.date_range("2024-01-01", periods=3))
    returns = compute_log_returns(prices)
    assert len(returns) == 2
    assert np.isclose(returns["A"].iloc[0], np.log(1.1))
