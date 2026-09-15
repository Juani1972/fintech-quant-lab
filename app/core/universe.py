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
    # --- Mercados internacionales (tickers de Yahoo Finance con el
    # sufijo de cada bolsa -- ver docs/MERCADOS.md para más detalle) ---
    "IBEX 35 (España)": [
        "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "CABK.MC",
        "FER.MC", "AENA.MC", "ELE.MC", "ACS.MC", "REP.MC",
    ],
    "DAX 40 (Alemania)": [
        "SIE.DE", "ALV.DE", "SAP.DE", "ENR.DE", "AIR.PA",
        "DTE.DE", "IFX.DE", "MUV2.DE", "DBK.DE", "DHL.DE",
    ],
    "CAC 40 (Francia)": [
        "MC.PA", "OR.PA", "SAN.PA", "TTE.PA", "SU.PA",
        "AIR.PA", "BNP.PA", "SAF.PA", "EL.PA", "AI.PA",
    ],
    "FTSE 100 (Reino Unido)": [
        "HSBA.L", "AZN.L", "SHEL.L", "ULVR.L", "RR.L",
        "BATS.L", "GSK.L", "BP.L", "RIO.L", "BARC.L",
    ],
    "Nikkei 225 (Japón)": [
        "7203.T", "6758.T", "9983.T", "9984.T", "8035.T",
        "6857.T", "6954.T", "4063.T", "9433.T", "6762.T",
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
