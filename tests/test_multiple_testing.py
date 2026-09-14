"""Tests para correcciones por múltiples contrastes."""
import numpy as np
import pandas as pd
import pytest

from app.core.multiple_testing import (
    benjamini_hochberg,
    bonferroni,
    correct_pvalues,
)


def test_bonferroni_multiplies_by_n():
    pv = pd.Series([0.01, 0.02, 0.03, 0.04])
    result = bonferroni(pv, alpha=0.05)
    assert result.method == "bonferroni"
    assert result.n_tests == 4
    assert result.pvalues_adjusted.iloc[0] == pytest.approx(0.04)


def test_bonferroni_caps_at_one():
    pv = pd.Series([0.5, 0.6, 0.7])
    result = bonferroni(pv, alpha=0.05)
    assert (result.pvalues_adjusted <= 1.0).all()
    assert result.n_rejected == 0


def test_bh_rejects_clear_signal():
    """Con p-values extremadamente pequeños, BH debe rechazar."""
    pv = pd.Series([1e-8, 1e-7, 1e-6, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    result = benjamini_hochberg(pv, alpha=0.05)
    assert result.n_rejected >= 3


def test_bh_no_rejects_when_all_large():
    pv = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
    result = benjamini_hochberg(pv, alpha=0.05)
    assert result.n_rejected == 0


def test_bh_monotonic():
    pv = pd.Series([0.001, 0.01, 0.02, 0.03, 0.04])
    result = benjamini_hochberg(pv, alpha=0.05)
    sorted_orig = result.pvalues_original.sort_values()
    sorted_adj = result.pvalues_adjusted.loc[sorted_orig.index]
    assert (sorted_adj.diff().dropna() >= -1e-9).all()


def test_bh_more_power_than_bonferroni():
    """BH siempre rechaza al menos tantas como Bonferroni."""
    np.random.seed(1)
    pv = pd.Series(np.random.uniform(0, 1, 50))
    bh = benjamini_hochberg(pv, alpha=0.05)
    bonf = bonferroni(pv, alpha=0.05)
    assert bh.n_rejected >= bonf.n_rejected


def test_rejects_invalid_pvalues():
    pv = pd.Series([0.5, 1.5])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        bonferroni(pv)


def test_rejects_empty():
    with pytest.raises(ValueError, match="al menos un p-value"):
        bonferroni([])


def test_dispatch_method():
    pv = pd.Series([0.01, 0.02, 0.03])
    r1 = correct_pvalues(pv, method="bonferroni")
    r2 = correct_pvalues(pv, method="bh")
    assert r1.method == "bonferroni"
    assert r2.method == "benjamini-hochberg"
    with pytest.raises(ValueError, match="no soportado"):
        correct_pvalues(pv, method="unknown")


def test_bonferroni_with_list():
    result = bonferroni([0.01, 0.02, 0.03])
    assert result.n_tests == 3
