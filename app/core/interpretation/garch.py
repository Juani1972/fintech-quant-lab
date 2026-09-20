"""Interpretación determinista (sin IA) de un ajuste GARCH.

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
    """Interpreta cada métrica de un `GarchResult` por separado.

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
        bullets.append("El optimizador **convergió**: los parámetros estimados son fiables.")
    else:
        bullets.append(
            "El optimizador **no convergió** -- trata los parámetros con "
            "cautela, prueba otro orden (p, q) o distribución."
        )

    if check_stationarity(result.params, result.model_type):
        bullets.append(
            "El modelo es **estacionario**: la volatilidad revierte a un "
            "nivel de largo plazo en vez de explotar o desvanecerse."
        )
    else:
        bullets.append(
            "El modelo **no es estacionario** según sus parámetros -- el "
            "pronóstico de volatilidad puede no ser fiable a largo plazo."
        )

    cur_vol = float(result.conditional_volatility.iloc[-1])
    mean_vol = float(result.conditional_volatility.mean())
    if cur_vol > mean_vol * 1.2:
        bullets.append(
            f"La volatilidad actual ({cur_vol:.4f}) está **por encima** de "
            f"su media histórica ({mean_vol:.4f}) -- el mercado está más "
            "agitado de lo habitual."
        )
    elif cur_vol < mean_vol * 0.8:
        bullets.append(
            f"La volatilidad actual ({cur_vol:.4f}) está **por debajo** de "
            f"su media histórica ({mean_vol:.4f}) -- el mercado está más "
            "tranquilo de lo habitual."
        )
    else:
        bullets.append(
            f"La volatilidad actual ({cur_vol:.4f}) está cerca de su media "
            f"histórica ({mean_vol:.4f})."
        )

    if len(forecast):
        fc_vol = float(forecast.iloc[-1])
        if fc_vol > cur_vol * 1.1:
            bullets.append(
                f"El pronóstico a {len(forecast)} días ({fc_vol:.4f}) "
                "anticipa que la volatilidad **subirá** respecto a hoy."
            )
        elif fc_vol < cur_vol * 0.9:
            bullets.append(
                f"El pronóstico a {len(forecast)} días ({fc_vol:.4f}) "
                "anticipa que la volatilidad **bajará** respecto a hoy."
            )
        else:
            bullets.append(
                f"El pronóstico a {len(forecast)} días ({fc_vol:.4f}) se "
                "mantiene estable respecto a hoy."
            )

    if diagnostics is not None:
        lb2 = diagnostics.get("ljung_box_squared_pvalue")
        arch_lm = diagnostics.get("arch_lm_pvalue")
        if lb2 is not None and arch_lm is not None:
            if lb2 > 0.05 and arch_lm > 0.05:
                bullets.append(
                    f"Ljung-Box² (p={lb2:.4f}) y ARCH-LM (p={arch_lm:.4f}) "
                    "no detectan heterocedasticidad residual: el modelo "
                    "capturó bien el clustering de volatilidad."
                )
            else:
                bullets.append(
                    f"Ljung-Box² (p={lb2:.4f}) o ARCH-LM (p={arch_lm:.4f}) "
                    "por debajo de 0.05: queda estructura sin explicar en "
                    "los residuos -- considera otro orden o modelo."
                )
        jb = diagnostics.get("jarque_bera_pvalue")
        if jb is not None:
            if jb < 0.05:
                bullets.append(
                    f"Jarque-Bera rechaza normalidad de los residuos "
                    f"(p={jb:.4f}) -- si usaste distribución 'normal', "
                    "prueba 't' o 'skewt' para colas más realistas."
                )
            else:
                bullets.append(
                    f"Jarque-Bera no rechaza normalidad de los residuos (p={jb:.4f})."
                )

    bullets.append(
        f"AIC={result.aic:.2f}, BIC={result.bic:.2f} -- solo son "
        "comparables entre modelos ajustados **sobre la misma serie**: "
        "úsalos para elegir entre GARCH/EGARCH/GJR-GARCH u órdenes "
        "distintos, no como una medida absoluta de calidad."
    )

    return bullets
