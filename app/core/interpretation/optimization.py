"""Interpretación determinista (sin IA) de un grid search con
walk-forward, en lenguaje llano -- evita nombrar "Deflated Sharpe
Ratio" como tal y explica el problema de "probar muchas
combinaciones" en términos de suerte vs. ventaja real."""
from __future__ import annotations

from typing import Any

from app.core.optimization import WalkForwardOptimizationResult


def interpret_optimization(
    result: WalkForwardOptimizationResult,
    dsr: dict[str, Any] | None,
) -> list[str]:
    """Interpreta un `WalkForwardOptimizationResult`: nº de
    configuraciones probadas, si la ganadora se mantiene en datos
    nuevos y, si se calculó, qué tan fiable es frente al azar.

    Args:
        result: resultado de `grid_search_walkforward()`.
        dsr: salida de `deflated_sharpe_ratio()`, o None si no se calculó.
    """
    bullets: list[str] = []

    n_combos = len(result.grid)
    bullets.append(
        f"Se probaron **{n_combos} configuraciones distintas** de la "
        "estrategia, y esta fue la que mejor funcionó en los datos "
        f"nuevos (no vistos durante el ajuste): `{result.best_params}`."
    )

    best_oos = result.best_oos_metrics.get(result.objective)
    best_is = result.best_is_metrics.get(result.objective)
    if best_oos is not None and best_is is not None and abs(best_is) > 1e-9:
        gap = (best_is - best_oos) / abs(best_is)
        if gap > 0.5:
            bullets.append(
                f"Esa configuración ganadora funciona **mucho peor** "
                f"({gap:.0%} de diferencia) en los datos nuevos que en "
                "los datos con los que se escogió -- señal de que podría "
                "estar sobreajustada al periodo de entrenamiento, no de "
                "que sea realmente la mejor opción."
            )
        else:
            bullets.append(
                f"Esa configuración ganadora funciona de forma "
                f"**razonablemente parecida** ({gap:.0%} de diferencia) "
                "en los datos nuevos que en los datos con los que se "
                "escogió."
            )

    if dsr is not None:
        dsr_val = dsr.get("dsr")
        if dsr_val is not None:
            if dsr_val < 0.95:
                bullets.append(
                    "Cuantas más configuraciones se prueban, más fácil "
                    "es que la 'ganadora' lo sea solo por azar, no "
                    "porque tenga una ventaja real. Teniendo en cuenta "
                    f"cuántas se probaron aquí, hay solo un "
                    f"**{dsr_val:.0%}** de probabilidad de que el "
                    "resultado ganador sea genuino y no un golpe de "
                    "suerte -- trátalo con cautela."
                )
            else:
                bullets.append(
                    "A pesar de haber probado varias configuraciones, "
                    f"hay un **{dsr_val:.0%}** de probabilidad de que el "
                    "resultado ganador sea genuino y no solo un golpe de "
                    "suerte -- una señal favorable."
                )

    oos_n_trades = result.best_oos_metrics.get("n_trades")
    if oos_n_trades is not None and oos_n_trades < 10:
        bullets.append(
            f"Esa configuración ganadora solo generó **{int(oos_n_trades)} "
            "operaciones** en los datos nuevos -- demasiado pocas para "
            "confiar en el resultado por sí solo."
        )

    return bullets
