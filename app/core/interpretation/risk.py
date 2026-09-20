"""Interpretación determinista (sin IA) de métricas de riesgo.

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

    var_hist = metrics.get("var_hist")
    var_param = metrics.get("var_param")
    if var_hist is not None and var_param is not None and var_param > 1e-9:
        diff = abs(var_hist - var_param) / var_param
        if diff > 0.25:
            bullets.append(
                f"VaR histórico ({var_hist:.2%}) y paramétrico "
                f"({var_param:.2%}) difieren en más de un 25% -- indicio "
                "de que los retornos no son normales (colas pesadas o "
                "asimetría); prioriza el VaR de Cornish-Fisher o el "
                "filtrado por GARCH sobre el paramétrico simple."
            )
        else:
            bullets.append(
                f"VaR histórico ({var_hist:.2%}) y paramétrico "
                f"({var_param:.2%}) son similares -- la normalidad es una "
                "aproximación razonable para esta serie."
            )

    var_cf = metrics.get("var_cf")
    if var_cf is not None and var_param is not None and var_param > 1e-9 and var_cf > var_param * 1.1:
        bullets.append(
            f"El VaR de Cornish-Fisher ({var_cf:.2%}), que corrige por "
            f"asimetría y curtosis reales, es **mayor** que el paramétrico "
            f"simple ({var_param:.2%}) -- el riesgo de cola es peor de lo "
            "que sugiere una distribución normal."
        )

    es_hist = metrics.get("es_hist")
    if es_hist is not None and var_hist is not None and var_hist > 1e-9:
        ratio = es_hist / var_hist
        bullets.append(
            f"Expected Shortfall histórico ({es_hist:.2%}) es "
            f"{ratio:.2f}x el VaR histórico -- la pérdida media **una vez "
            f"superado** el VaR {confidence:.0%} es notablemente peor que "
            "el propio umbral del VaR."
        )

    var_fhs = metrics.get("var_fhs")
    if var_fhs is not None and var_hist is not None and var_hist > 1e-9:
        if var_fhs > var_hist * 1.15:
            bullets.append(
                f"El VaR filtrado por GARCH ({var_fhs:.2%}) es mayor que "
                f"el histórico simple ({var_hist:.2%}) -- la volatilidad "
                "pronosticada para hoy es más alta que el promedio "
                "histórico usado por el VaR simple."
            )
        elif var_fhs < var_hist * 0.85:
            bullets.append(
                f"El VaR filtrado por GARCH ({var_fhs:.2%}) es menor que "
                f"el histórico simple ({var_hist:.2%}) -- la volatilidad "
                "pronosticada para hoy es más baja que el promedio "
                "histórico usado por el VaR simple."
            )

    sharpe = metrics.get("sharpe")
    if sharpe is not None:
        if sharpe > 2:
            bullets.append(f"Sharpe {sharpe:.2f}: **muy bueno** en términos absolutos.")
        elif sharpe > 1:
            bullets.append(f"Sharpe {sharpe:.2f}: **razonable**.")
        elif sharpe > 0:
            bullets.append(f"Sharpe {sharpe:.2f}: **mediocre**, apenas compensa el riesgo asumido.")
        else:
            bullets.append(f"Sharpe {sharpe:.2f}: **negativo**, el activo pierde valor ajustado a riesgo.")

    sortino = metrics.get("sortino")
    if sortino is not None and sharpe is not None and sortino > sharpe * 1.3:
        bullets.append(
            f"Sortino ({sortino:.2f}) notablemente mayor que Sharpe "
            f"({sharpe:.2f}) -- la mayor parte de la volatilidad viene de "
            "subidas, no de caídas."
        )

    calmar = metrics.get("calmar")
    max_dd = metrics.get("max_dd")
    if calmar is not None and max_dd is not None:
        bullets.append(
            f"Calmar {calmar:.2f} con drawdown máximo de {abs(max_dd):.2%} "
            "-- relaciona el retorno anual con la peor caída sufrida; "
            "valores > 1 son razonables, > 3 son notables."
        )

    return bullets
