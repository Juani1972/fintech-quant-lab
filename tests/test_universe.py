"""Tests para el módulo de universos predefinidos."""
import pytest

from app.core.universe import (
    UNIVERSES,
    get_universe,
    list_universes,
    universe_to_string,
)


def test_list_universes_not_empty():
    names = list_universes()
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)


def test_list_universes_matches_dict():
    assert list_universes() == list(UNIVERSES.keys())


def test_get_universe_known():
    tickers = get_universe("Magnificent 7")
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert len(tickers) == 7


def test_get_universe_ibex35():
    tickers = get_universe("IBEX 35 (España)")
    assert "SAN.MC" in tickers
    assert "ITX.MC" in tickers
    assert len(tickers) == 10


def test_international_universes_use_yahoo_suffixes():
    """Los universos de mercados no estadounidenses deben llevar el
    sufijo de bolsa de Yahoo Finance (.MC, .DE, .PA, .L, .T) en TODOS
    sus tickers -- si falta en alguno, yfinance lo buscaría como ticker
    de EE.UU. y probablemente fallaría o traería la empresa equivocada.
    """
    international = {
        "IBEX 35 (España)": ".MC",
        "DAX 40 (Alemania)": (".DE", ".PA"),  # Airbus cotiza en París
        "CAC 40 (Francia)": ".PA",
        "FTSE 100 (Reino Unido)": ".L",
        "Nikkei 225 (Japón)": ".T",
    }
    for name, suffix in international.items():
        for t in get_universe(name):
            assert t.endswith(suffix), f"{t} en '{name}' no lleva el sufijo {suffix}"


def test_get_universe_unknown_returns_empty():
    assert get_universe("NoExisteEsteUniverso") == []


def test_get_universe_returns_copy():
    """Modificar la lista devuelta no debe afectar al dict original."""
    tickers = get_universe("Magnificent 7")
    tickers.append("XXX")
    assert "XXX" not in get_universe("Magnificent 7")


def test_universe_to_string():
    s = universe_to_string("Magnificent 7")
    assert isinstance(s, str)
    assert "AAPL" in s
    assert ", " in s  # separados por coma + espacio


def test_universe_to_string_unknown():
    assert universe_to_string("NoExisteEsteUniverso") == ""


@pytest.mark.parametrize("name", list_universes())
def test_every_universe_has_at_least_two_tickers(name: str):
    """Cada universo debe tener al menos 2 tickers (para pairs trading)."""
    tickers = get_universe(name)
    assert len(tickers) >= 2, f"Universo '{name}' tiene menos de 2 tickers"


@pytest.mark.parametrize("name", list_universes())
def test_every_universe_tickers_uppercase(name: str):
    """Todos los tickers deben estar en mayúsculas y sin espacios."""
    for t in get_universe(name):
        assert t == t.strip().upper(), f"Ticker '{t}' de '{name}' no está normalizado"


def test_no_duplicate_tickers_within_universe():
    """No debe haber tickers repetidos dentro de un universo."""
    for name, tickers in UNIVERSES.items():
        assert len(tickers) == len(set(tickers)), f"Duplicados en '{name}'"
