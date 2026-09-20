"""Interpretación determinista (sin IA) de una regresión Fama-French,
en lenguaje llano -- evita nombrar el alpha/t-stat/p-valor como tales
y traduce cada beta a qué tipo de empresa se parece el activo."""
from __future__ import annotations

from app.core.fama_french import FamaFrenchResult


def _factor_sentence(factor: str, beta: float) -> str | None:
    """Traduce un factor con su beta a una frase en español llano, o
    None si el factor no se reconoce (se omite en vez de mostrar
    jerga sin explicar)."""
    if factor == "Mkt-RF":
        if beta > 1:
            return f"Se mueve **más que el mercado en general** (subidas y bajadas amplificadas ~{beta:.1f}x)."
        return f"Se mueve **menos que el mercado en general** (subidas y bajadas atenuadas ~{beta:.1f}x)."
    if factor == "SMB":
        tilt = "empresas pequeñas" if beta > 0 else "empresas grandes"
        return f"Se comporta más como una empresa **{tilt}**."
    if factor == "HML":
        tilt = "baratas respecto a sus fundamentales ('valor')" if beta > 0 else "de crecimiento"
        return f"Se comporta más como una empresa de acciones **{tilt}**."
    if factor == "RMW":
        tilt = "rentables" if beta > 0 else "poco rentables"
        return f"Se comporta más como una empresa **{tilt}**."
    if factor == "CMA":
        tilt = "conservadora en sus inversiones" if beta > 0 else "agresiva en sus inversiones"
        return f"Se comporta más como una empresa **{tilt}**."
    return None


def interpret_fama_french(result: FamaFrenchResult) -> list[str]:
    """Interpreta el alpha, la bondad de ajuste y cada beta de factor
    de un `FamaFrenchResult`, en lenguaje llano."""
    bullets: list[str] = []

    annualized_alpha = result.alpha * 252
    if result.alpha_pvalue < 0.05:
        sign = "positivo" if result.alpha > 0 else "negativo"
        bullets.append(
            f"Este activo genera un rendimiento **{sign} y no "
            f"explicado** por el mercado ni por el resto de factores "
            f"considerados -- en torno a un {abs(annualized_alpha):.2%} "
            "al año que no viene de ninguna de esas causas habituales."
        )
    else:
        bullets.append(
            "El rendimiento de este activo **se explica bien** por el "
            "mercado y el resto de factores -- no hay indicios de que "
            "tenga algo especial propio."
        )

    if result.r_squared >= 0.7:
        fit_desc = "en gran parte"
    elif result.r_squared >= 0.3:
        fit_desc = "en parte"
    else:
        fit_desc = "solo una pequeña parte"
    bullets.append(
        f"Los movimientos de este activo se explican **{fit_desc}** por "
        f"el mercado y el resto de factores ({result.r_squared:.0%} de "
        "sus variaciones)."
    )

    significant_factors = [
        f for f in result.betas.index
        if float(result.betas_pvalues[f]) < 0.05
    ]
    if not significant_factors:
        bullets.append(
            "Ninguno de los factores analizados muestra una relación "
            "suficientemente clara con este activo en este periodo."
        )
    else:
        for factor in significant_factors:
            sentence = _factor_sentence(str(factor), float(result.betas[factor]))
            if sentence:
                bullets.append(sentence)

    return bullets
