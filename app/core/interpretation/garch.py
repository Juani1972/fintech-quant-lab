"""Interpretación determinista (sin IA) de un ajuste GARCH, en
lenguaje llano -- pensada para alguien sin formación estadística, no
para un lector que ya conozca los nombres de los tests.

Complementa a `app.core.garch.plain_language_summary` (una única
frase de conclusión) con un punto por cada diagnóstico por separado.
"""
from __future__ import annotations

import pandas as pd

from app.core.garch import GarchResult, check_stationarity


def interpret_garch(
    result: GarchResult,
    diagnostics: dict[str, float] | None,
    forecast: pd.Series,
) -> list[str]:
    """Interpreta cada aspecto de un `GarchResult`, sin jerga técnica.

    Args:
        result: resultado de `fit_garch()`.
        diagnostics: salida de `residual_diagnostics()`, o None si no
            se pudo calcular.
        forecast: salida de `forecast_volatility()`.

    Returns:
        Lista de puntos en español llano, uno por aspecto del ajuste.
    """
    bullets: list[str] = []

    # Convergencia + estacionariedad: en datos reales de bolsa casi
    # siempre salen bien las dos -- son un requisito técnico del
    # ajuste, no algo que distinga a una empresa de otra. Se juntan en
    # una sola línea de "salud del ajuste" cuando todo va bien, y se
    # detallan por separado solo cuando algo falla (que es cuando de
    # verdad aporta información).
    fit_ok = result.converged
    stationary = check_stationarity(result.params, result.model_type)
    if fit_ok and stationary:
        bullets.append(
            "El ajuste técnico es sólido: el modelo terminó de calcularse "
            "correctamente y la volatilidad estimada tiende a "
            "estabilizarse con el tiempo en vez de dispararse sin límite."
        )
    else:
        if not fit_ok:
            bullets.append(
                "El modelo **no terminó de ajustarse bien** -- trata estos "
                "resultados con cautela; prueba a cambiar la configuración "
                "en la barra lateral (otro tipo de modelo, u otro orden p/q)."
            )
        if not stationary:
            bullets.append(
                "La volatilidad estimada **podría no estabilizarse** con "
                "el tiempo -- el pronóstico a más largo plazo es menos "
                "fiable en este caso."
            )

    cur_vol = float(result.conditional_volatility.iloc[-1])
    mean_vol = float(result.conditional_volatility.mean())
    if cur_vol > mean_vol * 1.2:
        bullets.append(
            "El mercado está **más agitado de lo habitual** ahora mismo: "
            f"la volatilidad actual ({cur_vol:.4f}) está por encima de su "
            f"media histórica ({mean_vol:.4f})."
        )
    elif cur_vol < mean_vol * 0.8:
        bullets.append(
            "El mercado está **más tranquilo de lo habitual** ahora "
            f"mismo: la volatilidad actual ({cur_vol:.4f}) está por "
            f"debajo de su media histórica ({mean_vol:.4f})."
        )
    else:
        bullets.append(
            f"La volatilidad actual ({cur_vol:.4f}) está cerca de su "
            f"media histórica ({mean_vol:.4f}) -- nada fuera de lo normal."
        )

    if len(forecast):
        fc_vol = float(forecast.iloc[-1])
        if fc_vol > cur_vol * 1.1:
            bullets.append(
                f"De cara a los próximos {len(forecast)} días, el modelo "
                "anticipa que la volatilidad **subirá** respecto a hoy "
                f"(hasta {fc_vol:.4f})."
            )
        elif fc_vol < cur_vol * 0.9:
            bullets.append(
                f"De cara a los próximos {len(forecast)} días, el modelo "
                "anticipa que la volatilidad **bajará** respecto a hoy "
                f"(hasta {fc_vol:.4f})."
            )
        else:
            bullets.append(
                f"De cara a los próximos {len(forecast)} días, el modelo "
                "anticipa que la volatilidad se **mantendrá estable** "
                "respecto a hoy."
            )

    if diagnostics is not None:
        lb2 = diagnostics.get("ljung_box_squared_pvalue")
        arch_lm = diagnostics.get("arch_lm_pvalue")
        # El caso "sin patrones sin explicar" es el esperable para
        # cualquier GARCH razonablemente bien especificado -- no
        # distingue una empresa de otra, así que no se detalla; solo se
        # avisa cuando SÍ queda algo sin explicar, que es el caso
        # informativo.
        if lb2 is not None and arch_lm is not None and not (lb2 > 0.05 and arch_lm > 0.05):
            bullets.append(
                "Todavía **quedan patrones sin explicar** en los "
                "datos -- prueba otra configuración (otro tipo de "
                "modelo u orden) en la barra lateral para ver si "
                "mejora el ajuste."
            )
        jb = diagnostics.get("jarque_bera_pvalue")
        # Igual que arriba: las colas pesadas (jb < 0.05) son la norma
        # en retornos diarios de bolsa, casi para cualquier empresa --
        # lo informativo es el caso contrario (residuos que sí
        # parecen normales), así que solo ese se destaca.
        if jb is not None and jb >= 0.05:
            bullets.append(
                "Los movimientos más extremos de esta serie no se salen "
                "mucho de lo que el modelo asume por defecto -- menos "
                "habitual que en la mayoría de acciones, donde suele "
                "haber más sustos de los que una distribución normal "
                "predeciría."
            )

    bullets.append(
        "Si pruebas otras configuraciones (otro tipo de modelo u orden "
        "p/q) y quieres saber cuál encaja mejor con los datos, compara "
        "los valores AIC y BIC en el 'Resumen completo del modelo' más "
        "abajo -- cuanto más bajos, mejor; solo sirven para comparar "
        "entre configuraciones de esta misma serie, no como una nota "
        "absoluta de calidad."
    )

    return bullets
