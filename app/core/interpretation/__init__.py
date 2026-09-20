"""Interpretación determinista (sin IA) de los resultados de cada
módulo cuantitativo.

Cada función de este paquete traduce el resultado de un módulo de
`app.core` a una lista de puntos en español llano, con reglas fijas
basadas en los propios cálculos -- sin llamar a ningún proveedor de
IA. Sirve como informe "gratis e instantáneo" por sí solo, y como
contexto de hechos para la ampliación opcional con IA (ver
`app.report_ui.render_interpretation_section`).
"""
from app.core.interpretation.backtest import interpret_backtest
from app.core.interpretation.cointegration import interpret_cointegration
from app.core.interpretation.fama_french import interpret_fama_french
from app.core.interpretation.garch import interpret_garch
from app.core.interpretation.optimization import interpret_optimization
from app.core.interpretation.risk import interpret_risk
from app.core.interpretation.robustness import interpret_robustness
from app.core.interpretation.walkforward import interpret_walkforward

__all__ = [
    "interpret_backtest",
    "interpret_cointegration",
    "interpret_fama_french",
    "interpret_garch",
    "interpret_optimization",
    "interpret_risk",
    "interpret_robustness",
    "interpret_walkforward",
]
