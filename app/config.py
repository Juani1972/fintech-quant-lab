"""Configuración global de la aplicación."""
from datetime import date, timedelta

APP_TITLE = "Fintech Quant Lab"
APP_ICON = "📈"
LAYOUT = "wide"

DEFAULT_TICKERS = ["KO", "PEP"]
DEFAULT_END = date.today()
DEFAULT_START = date.today() - timedelta(days=3 * 365)

CACHE_TTL = 3600  # segundos — precios de mercado (cambian intradía)
CACHE_TTL_FACTORS = 86400  # segundos — factores Fama-French (se publican a diario)

# ============================================================
#  Tema
# ============================================================
THEME = {
    "primary": "#2563eb",
    "primary_dark": "#1d4ed8",
    "success": "#16a34a",
    "warning": "#f59e0b",
    "danger": "#dc2626",
    "muted": "#64748b",
    "text": "#1e293b",
    "border": "#e2e8f0",
}

# ============================================================
#  Registro de páginas (para la portada)
# ============================================================
PAGES = [
    {
        "icon": "📈",
        "name": "GARCH",
        "description": "Modelado de volatilidad condicional con diagnósticos de residuos "
                       "(Ljung-Box, ARCH-LM, Jarque-Bera) y pronóstico.",
    },
    {
        "icon": "🔗",
        "name": "Cointegración",
        "description": "Engle-Granger, ADF del spread, half-life y z-score causal. "
                       "Corrección por múltiples tests (Bonferroni, BH).",
    },
    {
        "icon": "📊",
        "name": "Fama-French",
        "description": "Regresión de 3 y 5 factores con errores estándar HAC (Newey-West). "
                       "Interpretación del alpha.",
    },
    {
        "icon": "⚠️",
        "name": "Riesgo",
        "description": "VaR histórico y paramétrico, Expected Shortfall, drawdown, "
                       "Sharpe, Sortino, Calmar.",
    },
    {
        "icon": "🧪",
        "name": "Backtest",
        "description": "Motor vectorizado con anti-look-ahead, comisión, slippage, "
                       "benchmark y métricas completas.",
    },
    {
        "icon": "🔬",
        "name": "Walk-Forward",
        "description": "Validación out-of-sample por ventanas sucesivas. "
                       "Detección de overfitting.",
    },
    {
        "icon": "🎯",
        "name": "Optimización",
        "description": "Grid search evaluado con walk-forward. Heatmap de sensibilidad "
                       "de parámetros.",
    },
    {
        "icon": "🛡️",
        "name": "Robustez",
        "description": "Monte Carlo, block bootstrap, sensibilidad de parámetros y "
                       "robustness score 0-100.",
    },
]
