"""Interpretación determinista (sin IA) de un análisis de robustez,
en lenguaje llano -- no vuelca los componentes técnicos del score
(claves como "degradacion_is_oos") sino que explica en una frase qué
comprueba cada uno."""
from __future__ import annotations

from app.core.robustness import MonteCarloResult, RobustnessReport, SensitivityResult


def interpret_robustness(
    report: RobustnessReport,
    mc: MonteCarloResult,
    sens: SensitivityResult,
) -> list[str]:
    """Interpreta el score de robustez agregado, la simulación de
    escenarios y la sensibilidad a un parámetro, en lenguaje llano."""
    bullets: list[str] = [
        f"Puntuación general de fiabilidad: **{report.final_score:.0f} "
        f"sobre 100** -- {report.interpretation}",
        "Esa puntuación combina varias comprobaciones: si el resultado "
        "se mantiene al pasar a datos nuevos, si sigue siendo bueno en "
        "escenarios simulados al azar, si depende demasiado de un "
        "parámetro concreto, y si hay suficientes operaciones para "
        "confiar en las cifras.",
    ]

    prob_positive = float((mc.distribution > 0).mean())
    if prob_positive < 0.5:
        bullets.append(
            f"Se simularon **{mc.n_simulations} escenarios** posibles "
            "barajando los resultados históricos, y en solo el "
            f"**{prob_positive:.0%}** de ellos la estrategia habría "
            "ganado dinero ajustado a riesgo -- podría perder en más de "
            "la mitad de los futuros plausibles."
        )
    elif prob_positive < 0.8:
        bullets.append(
            f"Se simularon **{mc.n_simulations} escenarios** posibles "
            f"barajando los resultados históricos, y en el "
            f"**{prob_positive:.0%}** de ellos la estrategia habría "
            "ganado dinero ajustado a riesgo -- mayoría favorable, pero "
            "con un margen de escenarios negativos no despreciable."
        )
    else:
        bullets.append(
            f"Se simularon **{mc.n_simulations} escenarios** posibles "
            f"barajando los resultados históricos, y en el "
            f"**{prob_positive:.0%}** de ellos la estrategia habría "
            "ganado dinero ajustado a riesgo -- un resultado consistente "
            "a través de los distintos escenarios."
        )

    if sens.stability_score >= 0.7:
        bullets.append(
            f"El resultado **no depende demasiado** del valor exacto "
            f"elegido para '{sens.param_name}' -- buena señal, no "
            "parece un ajuste frágil hecho a medida de estos datos."
        )
    elif sens.stability_score >= 0.4:
        bullets.append(
            f"El resultado depende **moderadamente** del valor exacto "
            f"elegido para '{sens.param_name}' -- revisa la gráfica en "
            "busca de un pico aislado frente a una zona amplia igual de "
            "buena."
        )
    else:
        bullets.append(
            f"El resultado **depende mucho** del valor exacto elegido "
            f"para '{sens.param_name}' -- señal de que podría estar "
            "ajustado en exceso a estos datos concretos, y funcionar "
            "peor con un valor ligeramente distinto."
        )

    return bullets
