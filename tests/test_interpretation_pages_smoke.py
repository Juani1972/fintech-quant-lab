"""Smoke tests end-to-end (con AppTest) de la nueva sección "📝
Interpretación" en las páginas que no tenían ninguna sección de
informe todavía (Cointegración, Fama-French, Riesgo, Walk-Forward,
Optimización, Robustez). GARCH y Backtest ya tenían cobertura de su
sección de informe en tests/test_app_smoke.py.

A diferencia de tests/test_interpretation.py (funciones puras con
dataclasses de mentira), esto ejercita el cableado real de cada
página: que las variables que se pasan a `interpret_*` y al prompt de
IA existan con el nombre correcto, que `render_interpretation_section`
reciba tipos válidos, etc. -- nada de esto lo detecta mypy si el
error está en el propio flujo de Streamlit (variable no definida en
una rama, KeyError con datos reales...).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).parent.parent / "app"


def _synthetic_prices(tickers: list[str], n_days: int = 800, seed: int = 3) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-01", periods=n_days)
    rng = np.random.default_rng(seed)
    data = {}
    for i, t in enumerate(tickers):
        data[t] = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))) + i
    return pd.DataFrame(data, index=idx)


def test_cointegracion_page_shows_interpretation():
    prices = _synthetic_prices(["AAA", "BBB"])
    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "2_🔗_Cointegración.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA, BBB"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar análisis" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))


def test_fama_french_page_shows_interpretation():
    prices = _synthetic_prices(["AAA"])
    idx = prices.index
    rng = np.random.default_rng(5)
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.01, len(idx)),
            "SMB": rng.normal(0, 0.005, len(idx)),
            "HML": rng.normal(0, 0.005, len(idx)),
            "RF": np.full(len(idx), 0.00005),
        },
        index=idx,
    )
    with (
        patch("app.core.data_loader.load_prices", return_value=prices),
        patch("app.core.fama_french.load_factors", return_value=factors),
    ):
        at = AppTest.from_file(str(APP_DIR / "pages" / "3_📊_Fama_French.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar regresión" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))


def test_riesgo_page_shows_interpretation():
    prices = _synthetic_prices(["AAA"])
    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "4_⚠️_Riesgo.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Calcular riesgo" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))


def test_walkforward_page_shows_interpretation():
    prices = _synthetic_prices(["AAA"])
    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "6_🔬_WalkForward.py"), default_timeout=30)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["wf_strategy"] = "Momentum"
        at.run()

        run_button = next(b for b in at.sidebar.button if "Ejecutar walk-forward" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))


def test_optimizacion_page_shows_interpretation():
    prices = _synthetic_prices(["AAA"])
    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "7_🎯_Optimización.py"), default_timeout=60)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["opt_strategy"] = "Momentum"
        at.session_state["opt_windows"] = [20, 45]
        at.run()

        run_button = next(b for b in at.sidebar.button if "Optimizar" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))


def test_robustez_page_shows_interpretation():
    prices = _synthetic_prices(["AAA"])
    with patch("app.core.data_loader.load_prices", return_value=prices):
        at = AppTest.from_file(str(APP_DIR / "pages" / "8_🛡️_Robustez.py"), default_timeout=60)
        at.session_state["global_tickers"] = "AAA"
        at.session_state["rob_strategy"] = "Momentum"
        at.session_state["rob_n_sims"] = 200
        at.run()

        run_button = next(b for b in at.sidebar.button if "Analizar robustez" in (b.label or ""))
        at = run_button.click().run()

    assert not at.exception
    assert any("Interpretación" in (h.value or "") for h in list(at.header) + list(at.subheader) + list(at.markdown))
