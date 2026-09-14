"""Configuración global de la aplicación."""
from datetime import date, timedelta

APP_TITLE = "Fintech Quant Lab"
APP_ICON = "📈"
LAYOUT = "wide"

DEFAULT_TICKERS = ["KO", "PEP"]
DEFAULT_END = date.today()
DEFAULT_START = date.today() - timedelta(days=3 * 365)  # últimos 3 años

CACHE_TTL = 3600  # segundos
