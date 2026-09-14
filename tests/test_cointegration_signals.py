"""Tests para la generación de señales de pairs trading."""
import numpy as np
import pandas as pd

from app.core.cointegration import generate_signals


def test_signals_entry_and_exit():
    # Serie de z-score diseñada: entra long, sale, entra short, sale
    z = pd.Series([0, 0.5, -2.5, -1.0, -0.3, 2.5, 1.0, 0.2])
    signals = generate_signals(z, entry=2.0, exit_=0.5)

    # Posiciones esperadas
    assert signals.iloc[0] == 0     # neutral
    assert signals.iloc[2] == 1     # entra long (z < -2)
    assert signals.iloc[4] == 0     # sale (|z| < 0.5)
    assert signals.iloc[5] == -1    # entra short (z > 2)
    assert signals.iloc[7] == 0     # sale


def test_signals_hold_position_through_noise():
    z = pd.Series([0, -2.5, -1.5, -1.8, -0.3])
    signals = generate_signals(z, entry=2.0, exit_=0.5)
    # Mantiene long hasta que |z| < 0.5
    assert signals.iloc[1] == 1
    assert signals.iloc[2] == 1
    assert signals.iloc[3] == 1
    assert signals.iloc[4] == 0


def test_signals_handles_nans():
    z = pd.Series([np.nan, -2.5, np.nan, -0.3])
    signals = generate_signals(z, entry=2.0, exit_=0.5)
    assert len(signals) == len(z)
