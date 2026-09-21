"""Interpretación determinista (sin IA) de un backtest, métrica a
métrica, en lenguaje llano.

Complementa a `app.core.backtest.plain_language_summary` (una única
frase de conclusión) con el desglose completo.
"""
from __future__ import annotations

from app.core.backtest import BacktestResult


def interpret_backtest(result: BacktestResult) -> list[str]:
    """Interpreta cada métrica de un `BacktestResult` por separado,
    sin jerga técnica."""
    m = result.metrics
    bullets: list[str] = []

    n_trades = int(m.get("n_trades", 0))
    if n_trades < 10:
        bullets.append(
            f"Solo se hicieron **{n_trades} operaciones** en todo el "
            "periodo -- muy pocas para sacar ninguna conclusión fiable; "
            "el resultado puede deberse al azar más que a que la "
            "estrategia funcione de verdad."
        )
    elif n_trades < 30:
        bullets.append(
            f"Se hicieron **{n_trades} operaciones** -- una cantidad "
            "moderada; las conclusiones son razonablemente fiables, pero "
            "conviene contrastarlas probando otro periodo de fechas."
        )
    else:
        bullets.append(
            f"Se hicieron **{n_trades} operaciones** -- suficientes para "
            "sacar conclusiones con cierta confianza."
        )

    sharpe = m.get("sharpe")
    if sharpe is not None and sharpe == sharpe:  # descarta NaN
        if sharpe < 0:
            bullets.append(
                f"La estrategia **ha perdido dinero** (ratio de "
                f"{sharpe:.2f}) una vez descontado el riesgo asumido."
            )
        elif sharpe > 3:
            bullets.append(
                f"El resultado parece **demasiado bueno para ser cierto** "
                f"(ratio de {sharpe:.2f}) -- antes de confiar en él, "
                "revisa la página de Optimización: con pocas operaciones, "
                "un resultado así suele deberse a que la estrategia se ha "
                "ajustado demasiado a estos datos concretos, no a una "
                "ventaja real."
            )
        elif sharpe < 1:
            bullets.append(
                f"La relación entre lo que gana y el riesgo que asume es "
                f"**floja** (ratio de {sharpe:.2f}) -- no destaca frente "
                "a algo tan simple como comprar y mantener."
            )
        else:
            bullets.append(
                "La relación entre lo que gana y el riesgo que asume es "
                f"**razonable** (ratio de {sharpe:.2f})."
            )

    win_rate = m.get("win_rate")
    profit_factor = m.get("profit_factor")
    if win_rate is not None and profit_factor is not None:
        if win_rate < 0.4 and profit_factor > 1.5:
            bullets.append(
                f"**Gana pocas veces** ({win_rate:.0%} de las "
                "operaciones), pero cuando acierta gana bastante más de "
                "lo que pierde cuando falla -- normal en estrategias que "
                "siguen tendencias, no un problema por sí mismo."
            )
        elif win_rate > 0.6 and profit_factor < 1.2:
            bullets.append(
                f"**Gana muchas veces** ({win_rate:.0%} de las "
                "operaciones), pero las ganancias son pequeñas y las "
                "pérdidas ocasionales se las comen casi enteras -- vigila "
                "el tamaño de esas pérdidas."
            )

    max_dd = m.get("max_drawdown")
    total_return = m.get("total_return")
    if max_dd is not None and total_return is not None:
        bullets.append(
            f"En todo el periodo ganó un **{total_return:.1%}**, pero en "
            f"su peor momento llegó a caer un **{abs(max_dd):.1%}** desde "
            "su punto más alto -- es el 'susto' máximo que habría que "
            "aguantar para conseguir ese resultado."
        )

    exposure = m.get("exposure")
    if exposure is not None:
        if exposure < 0.2:
            bullets.append(
                f"Solo estuvo invertido el **{exposure:.0%}** del "
                "tiempo -- la mayor parte del periodo el dinero estuvo "
                "fuera del mercado, sin arriesgar ni ganar."
            )
        elif exposure > 0.9:
            bullets.append(
                f"Estuvo invertido casi todo el tiempo "
                f"({exposure:.0%}) -- se comporta de forma parecida a "
                "comprar y mantener sin hacer nada."
            )

    return bullets
