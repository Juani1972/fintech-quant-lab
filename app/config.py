"""Configuración global de la aplicación."""
from dataclasses import dataclass
from datetime import date

APP_TITLE = "📈 Fintech Quant Lab"
APP_ICON = "📈"
LAYOUT = "wide"

DEFAULT_TICKERS = ["KO", "PEP"]
DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date.today()

CACHE_TTL = 3600  # segundos
