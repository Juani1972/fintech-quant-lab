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
        "icon": "💹",
        "name": "Rentabilidad",
        "description": "Serie histórica de precios y rentabilidad de una o "
                       "varias empresas en el periodo elegido -- al instante, "
                       "sin configurar ningún modelo.",
    },
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
    {
        "icon": "📚",
        "name": "Histórico",
        "description": "Backtests guardados en SQLite, con informe de investigación "
                       "exportable a Markdown.",
    },
    {
        "icon": "📊",
        "name": "Comparar",
        "description": "Compara varias corridas guardadas del histórico lado a lado, "
                       "con el mejor/peor por métrica resaltado.",
    },
    {
        "icon": "📉",
        "name": "Regímenes",
        "description": "Detección de regímenes de volatilidad con un modelo oculto de "
                       "Markov (HMM).",
    },
    {
        "icon": "🎛️",
        "name": "Kalman",
        "description": "Hedge ratio dinámico con filtro de Kalman, para pairs trading "
                       "con beta cambiante en el tiempo.",
    },
    {
        "icon": "🔔",
        "name": "Alertas",
        "description": "Notificaciones multicanal (consola, archivo, email, Telegram, "
                       "Slack) cuando una métrica cruza un umbral.",
    },
    {
        "icon": "💼",
        "name": "Portfolio",
        "description": "Pesos de cartera óptimos: Hierarchical Risk Parity, Markowitz "
                       "y Equal Risk Contribution.",
    },
    {
        "icon": "🧭",
        "name": "Estrategias",
        "description": "Explora las señales de 6 estrategias -- Trend Following, "
                       "Volatility Targeting, PCA StatArb, Risk Parity, "
                       "Cross-Sectional Momentum y Carry Trade.",
    },
    {
        "icon": "📟",
        "name": "Papertrading",
        "description": "Envía órdenes en papel (sin capital real) contra la cuenta "
                       "de simulación de Alpaca.",
    },
    {
        "icon": "🧪",
        "name": "Experimentos",
        "description": "Congela config, datos y resultados de un backtest para "
                       "reproducirlo exactamente más adelante, con el commit "
                       "de git asociado.",
    },
    {
        "icon": "🔑",
        "name": "Licencia",
        "description": "Verificación de licencia con caché offline y huella de "
                       "máquina -- pendiente de un servidor de licencias real.",
    },
]
