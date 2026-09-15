"""Universos predefinidos de tickers.

Presets útiles para no tener que escribir tickers a mano. El usuario
selecciona un universo y el campo de tickers se rellena automáticamente;
después puede editarlo libremente.
"""
from __future__ import annotations

from typing import Final


UNIVERSES: Final[dict[str, list[str]]] = {
    "Pares clásicos (cointegración)": [
        "KO", "PEP", "XOM", "CVX", "V", "MA", "GLD", "SLV", "GOLD", "NEM",
    ],
    "Magnificent 7": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    ],
    "ETFs populares": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "EEM", "EFA", "TLT", "GLD", "SLV",
    ],
    "Sectores (SPDR)": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
    ],
    "Bancos USA": [
        "JPM", "BAC", "WFC", "C", "GS", "MS",
    ],
    "Energía": [
        "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    ],
    "Semiconductores": [
        "NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "MU",
    ],
    "Salud y farmacéuticas": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN",
    ],
    "Consumo básico": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "MCD",
    ],
}


def list_universes() -> list[str]:
    """Devuelve los nombres de todos los universos disponibles."""
    return list(UNIVERSES.keys())


def get_universe(name: str) -> list[str]:
    """Devuelve la lista de tickers de un universo.

    Args:
        name: Nombre exacto del universo.

    Returns:
        Lista de tickers. Lista vacía si el nombre no existe.
    """
    return list(UNIVERSES.get(name, []))


def universe_to_string(name: str) -> str:
    """Devuelve los tickers de un universo como string separado por comas.

    Útil para rellenar `st.session_state["global_tickers"]`.
    """
    return ", ".join(get_universe(name))
