"""Interpretación determinista (sin IA) de métricas de riesgo, en
lenguaje llano -- evita nombrar "VaR", "Expected Shortfall", "Sharpe"
etc. como tales y se centra en qué significan en la práctica.

Recibe un diccionario de métricas ya calculadas por la página de
Riesgo (no reimplementa los cálculos, ver `app.core.risk`) para no
acoplar este módulo a una única combinación de métricas -- claves
ausentes simplemente se omiten de la interpretación.
"""
from __future__ import annotations


def interpret_risk(metrics: dict[str, float | None], confidence: float) -> list[str]:
    """Interpreta las métricas de riesgo calculadas en la página.

    Args:
        metrics: dict con claves opcionales entre
            `var_hist, var_param, var_cf, var_fhs, es_hist, es_param,
            sharpe, sortino, calmar, max_dd` -- las que falten se omiten.
        confidence: nivel de confianza usado para el VaR/ES (p.ej. 0.95).

    Returns:
        Lista de puntos en español llano.
    """
    bullets: list[str] = []
    tail_prob = 1 - confidence

    var_hist = metrics.get("var_hist")
    if var_hist is not None:
        bullets.append(
            f"Basándonos en lo que ha pasado históricamente, hay un "
            f"{tail_prob:.0%} de probabilidad de perder más de un "
            f"**{var_hist:.1%}** en un solo día."
        )

    var_param = metrics.get("var_param")
    if var_hist is not None and var_param is not None and var_param > 1e-9:
        diff = abs(var_hist - var_param) / var_param
        if diff > 0.25:
            bullets.append(
                "Dos formas distintas de estimar ese riesgo dan "
                "resultados bastante distintos -- señal de que este "
                "activo tiene movimientos más extremos de lo que una "
                "campana de Gauss ('distribución normal') predeciría."
            )

    var_cf = metrics.get("var_cf")
    if var_cf is not None and var_param is not None and var_param > 1e-9 and var_cf > var_param * 1.1:
        bullets.append(
            "Teniendo en cuenta los movimientos extremos reales de este "
            f"activo (no solo el promedio), la pérdida probable en un "
            f"mal día sube a **{var_cf:.1%}** -- una estimación más "
            "realista que la anterior."
        )

    es_hist = metrics.get("es_hist")
    if es_hist is not None and var_hist is not None and var_hist > 1e-9:
        bullets.append(
            f"Y si ese mal día llega a ocurrir, la pérdida media en esos "
            f"peores casos ronda el **{es_hist:.1%}** -- claramente peor "
            "que el umbral anterior, porque solo tiene en cuenta los "
            "escenarios más duros."
        )

    var_fhs = metrics.get("var_fhs")
    if var_fhs is not None and var_hist is not None and var_hist > 1e-9:
        if var_fhs > var_hist * 1.15:
            bullets.append(
                "El mercado está **más agitado de lo habitual** ahora "
                "mismo, así que el riesgo real hoy es mayor que lo que "
                "sugiere el promedio de todo el histórico."
            )
        elif var_fhs < var_hist * 0.85:
            bullets.append(
                "El mercado está **más tranquilo de lo habitual** ahora "
                "mismo, así que el riesgo real hoy es menor que lo que "
                "sugiere el promedio de todo el histórico."
            )

    sharpe = metrics.get("sharpe")
    if sharpe is not None:
        if sharpe > 2:
            bullets.append(
                "La relación entre lo que gana y el riesgo que asume es "
                "**muy buena**: gana bastante en proporción a lo que se "
                "arriesga."
            )
        elif sharpe > 1:
            bullets.append(
                "La relación entre lo que gana y el riesgo que asume es "
                "**razonable**."
            )
        elif sharpe > 0:
            bullets.append(
                "La relación entre lo que gana y el riesgo que asume es "
                "**floja**: apenas compensa el riesgo asumido."
            )
        else:
            bullets.append(
                "En este periodo, el activo **ha perdido valor** una vez "
                "descontado el riesgo asumido."
            )

    sortino = metrics.get("sortino")
    if sortino is not None and sharpe is not None and sortino > sharpe * 1.3:
        bullets.append(
            "La mayor parte de sus altibajos vienen de **subidas**, no "
            "de caídas -- una señal favorable que la relación anterior, "
            "por sí sola, no distingue."
        )

    calmar = metrics.get("calmar")
    max_dd = metrics.get("max_dd")
    if calmar is not None and max_dd is not None:
        verdict = "asumible" if calmar > 1 else "considerable en relación a lo que gana"
        bullets.append(
            f"En su peor momento, este activo llegó a caer un "
            f"**{abs(max_dd):.1%}** desde su punto más alto -- comparado "
            f"con lo que gana al año, esa caída es **{verdict}**."
        )

    return bullets
