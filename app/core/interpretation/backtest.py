"""Interpretación determinista (sin IA) de un backtest, métrica a
métrica.

Complementa a `app.core.backtest.plain_language_summary` (una única
frase de conclusión) con el desglose completo.
"""
from __future__ import annotations

from app.core.backtest import BacktestResult


def interpret_backtest(result: BacktestResult) -> list[str]:
    """Interpreta cada métrica de un `BacktestResult` por separado."""
    m = result.metrics
    bullets: list[str] = []

    n_trades = int(m.get("n_trades", 0))
    if n_trades < 10:
        bullets.append(
            f"Solo **{n_trades} operaciones** -- cualquier métrica aquí "
            "puede deberse al azar más que a una ventaja real."
        )
    elif n_trades < 30:
        bullets.append(
            f"**{n_trades} operaciones**: número moderado -- las "
            "conclusiones son más fiables que con pocas decenas, pero "
            "conviene contrastarlas con walk-forward."
        )
    else:
        bullets.append(f"**{n_trades} operaciones**: muestra razonable para sacar conclusiones.")

    sharpe = m.get("sharpe")
    if sharpe is not None and sharpe == sharpe:  # descarta NaN
        if sharpe < 0:
            bullets.append(f"Sharpe {sharpe:.2f}: la estrategia **pierde dinero ajustado a riesgo**.")
        elif sharpe > 3:
            bullets.append(
                f"Sharpe {sharpe:.2f}: **sospechosamente alto** -- revisa "
                "el Deflated Sharpe Ratio en Optimización antes de confiar en él."
            )
        elif sharpe < 1:
            bullets.append(f"Sharpe {sharpe:.2f}: mediocre en relación al riesgo asumido.")
        else:
            bullets.append(f"Sharpe {sharpe:.2f}: razonable.")

    win_rate = m.get("win_rate")
    profit_factor = m.get("profit_factor")
    if win_rate is not None and profit_factor is not None:
        if win_rate < 0.4 and profit_factor > 1.5:
            bullets.append(
                f"Win rate bajo ({win_rate:.1%}) pero profit factor alto "
                f"({profit_factor:.2f}) -- estrategia de pocas ganadoras "
                "grandes, típica de tendencia/momentum. Es normal, no un "
                "problema por sí mismo."
            )
        elif win_rate > 0.6 and profit_factor < 1.2:
            bullets.append(
                f"Win rate alto ({win_rate:.1%}) pero profit factor bajo "
                f"({profit_factor:.2f}) -- muchas ganadoras pequeñas que "
                "apenas compensan pérdidas grandes ocasionales; vigila el "
                "tamaño de las pérdidas."
            )

    max_dd = m.get("max_drawdown")
    total_return = m.get("total_return")
    if max_dd is not None and total_return is not None and max_dd != 0:
        ratio = total_return / abs(max_dd)
        bullets.append(
            f"Retorno total {total_return:.1%} frente a una caída máxima "
            f"de {abs(max_dd):.1%} (ratio {ratio:.2f}x)."
        )

    exposure = m.get("exposure")
    if exposure is not None:
        if exposure < 0.2:
            bullets.append(
                f"Exposición al mercado muy baja ({exposure:.1%}) -- la "
                "mayor parte del tiempo está fuera del mercado."
            )
        elif exposure > 0.9:
            bullets.append(
                f"Exposición al mercado casi permanente ({exposure:.1%}) "
                "-- se comporta de forma parecida a comprar y mantener."
            )

    return bullets
