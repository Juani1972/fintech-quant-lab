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

    if result.converged:
        bullets.append(
            "El modelo terminó de ajustarse correctamente, así que estos "
            "resultados son fiables."
        )
    else:
        bullets.append(
            "El modelo **no terminó de ajustarse bien** -- trata estos "
            "resultados con cautela; prueba a cambiar la configuración en "
            "la barra lateral (otro tipo de modelo, u otro orden p/q)."
        )

    if check_stationarity(result.params, result.model_type):
        bullets.append(
            "La volatilidad estimada **tiende a estabilizarse** con el "
            "tiempo en vez de dispararse sin límite -- un comportamiento "
            "razonable para un activo financiero."
        )
    else:
        bullets.append(
            "La volatilidad estimada **podría no estabilizarse** con el "
            "tiempo -- el pronóstico a más largo plazo es menos fiable "
            "en este caso."
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
        if lb2 is not None and arch_lm is not None:
            if lb2 > 0.05 and arch_lm > 0.05:
                bullets.append(
                    "El modelo **capturó bien** los periodos de calma y "
                    "de nerviosismo del mercado -- no se le ha escapado "
                    "ningún patrón evidente en los datos."
                )
            else:
                bullets.append(
                    "Todavía **quedan patrones sin explicar** en los "
                    "datos -- prueba otra configuración (otro tipo de "
                    "modelo u orden) en la barra lateral para ver si "
                    "mejora el ajuste."
                )
        jb = diagnostics.get("jarque_bera_pvalue")
        if jb is not None and jb < 0.05:
            bullets.append(
                "Los movimientos más extremos (subidas o caídas fuera de "
                "lo común) no siguen la forma que el modelo asume por "
                "defecto -- si notas que el pronóstico falla en momentos "
                "de mucha turbulencia, prueba a cambiar la "
                "'Distribución' a 't' o 'skewt' en la barra lateral."
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
