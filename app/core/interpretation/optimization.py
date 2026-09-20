"""Interpretación determinista (sin IA) de un grid search con walk-forward."""
from __future__ import annotations

from typing import Any

from app.core.optimization import WalkForwardOptimizationResult


def interpret_optimization(
    result: WalkForwardOptimizationResult,
    dsr: dict[str, Any] | None,
) -> list[str]:
    """Interpreta un `WalkForwardOptimizationResult`: nº de
    combinaciones, brecha IS/OOS de la ganadora y, si se calculó, el
    Deflated Sharpe Ratio.

    Args:
        result: resultado de `grid_search_walkforward()`.
        dsr: salida de `deflated_sharpe_ratio()`, o None si no se calculó
            (solo tiene sentido con `objective='sharpe'`).
    """
    bullets: list[str] = []

    n_combos = len(result.grid)
    bullets.append(
        f"**{n_combos} combinaciones** evaluadas por su rendimiento "
        f"out-of-sample; ganadora: `{result.best_params}`."
    )

    best_oos = result.best_oos_metrics.get(result.objective)
    best_is = result.best_is_metrics.get(result.objective)
    if best_oos is not None and best_is is not None and abs(best_is) > 1e-9:
        gap = (best_is - best_oos) / abs(best_is)
        if gap > 0.5:
            bullets.append(
                f"El **{result.objective}** OOS ({best_oos:.2f}) es muy "
                f"inferior al IS ({best_is:.2f}, brecha {gap:.0%}) -- la "
                "combinación ganadora podría estar sobreajustada al "
                "periodo de entrenamiento."
            )
        else:
            bullets.append(
                f"El **{result.objective}** OOS ({best_oos:.2f}) se "
                f"mantiene razonablemente cerca del IS ({best_is:.2f})."
            )

    if dsr is not None:
        dsr_val = dsr.get("dsr")
        n_trials = dsr.get("n_trials")
        if dsr_val is not None:
            if dsr_val < 0.95:
                bullets.append(
                    f"Deflated Sharpe Ratio = **{dsr_val:.1%}** (por debajo "
                    f"del umbral habitual de 95%, sobre {n_trials} "
                    "combinaciones probadas) -- el Sharpe ganador podría "
                    "deberse en buena parte al azar de haber probado "
                    "muchas combinaciones."
                )
            else:
                bullets.append(
                    f"Deflated Sharpe Ratio = **{dsr_val:.1%}** (sobre "
                    f"{n_trials} combinaciones) -- por encima del umbral "
                    "habitual de 95%, el resultado probablemente no es "
                    "solo el máximo esperable por azar."
                )

    oos_n_trades = result.best_oos_metrics.get("n_trades")
    if oos_n_trades is not None and oos_n_trades < 10:
        bullets.append(
            f"La combinación ganadora solo generó **{int(oos_n_trades)} "
            "operaciones OOS** -- demasiado pocas para confiar en sus "
            "métricas por sí solas."
        )

    return bullets
