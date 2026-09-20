"""Interpretación determinista (sin IA) de una regresión Fama-French."""
from __future__ import annotations

from app.core.fama_french import FamaFrenchResult

_FACTOR_MEANING = {
    "Mkt-RF": "sensibilidad al mercado (beta de mercado)",
    "SMB": "sesgo a empresas pequeñas (positivo) o grandes (negativo)",
    "HML": "sesgo a valor (positivo) o crecimiento (negativo)",
    "RMW": "sesgo a empresas rentables (positivo) o poco rentables (negativo)",
    "CMA": "sesgo a inversión conservadora (positivo) o agresiva (negativo)",
}


def interpret_fama_french(result: FamaFrenchResult) -> list[str]:
    """Interpreta cada componente de un `FamaFrenchResult`: alpha, bondad
    de ajuste y cada beta de factor por separado."""
    bullets: list[str] = []

    annualized_alpha = result.alpha * 252
    if result.alpha_pvalue < 0.05:
        sign = "positivo" if result.alpha > 0 else "negativo"
        bullets.append(
            f"Alpha **estadísticamente significativo** y {sign} "
            f"(t={result.alpha_tstat:.2f}, p={result.alpha_pvalue:.4f}) -- "
            f"~{annualized_alpha:.2%} anualizado de retorno **no explicado** "
            "por los factores."
        )
    else:
        bullets.append(
            f"Alpha **no significativo** (t={result.alpha_tstat:.2f}, "
            f"p={result.alpha_pvalue:.4f}) -- el retorno del activo se "
            "explica razonablemente bien por los factores, sin evidencia "
            "de exceso de retorno propio."
        )

    if result.r_squared >= 0.7:
        fit_desc = "alto"
    elif result.r_squared >= 0.3:
        fit_desc = "moderado"
    else:
        fit_desc = "bajo"
    bullets.append(
        f"R² = {result.r_squared:.3f} (ajustado {result.adj_r_squared:.3f}): "
        f"ajuste **{fit_desc}** -- los factores explican "
        f"{result.r_squared:.0%} de la varianza de los retornos en exceso."
    )

    for factor in result.betas.index:
        beta = float(result.betas[factor])
        pval = float(result.betas_pvalues[factor])
        meaning = _FACTOR_MEANING.get(str(factor), "")
        sig = "significativo" if pval < 0.05 else "no significativo"
        bullets.append(
            f"**{factor}**: beta={beta:.3f} ({sig}, p={pval:.4f})"
            + (f" -- {meaning}." if meaning else ".")
        )

    bullets.append(
        f"Errores estándar: **{result.cov_type}**"
        + (f" (maxlags={result.maxlags})" if result.maxlags else "")
        + f", N={result.n_obs} observaciones."
    )

    return bullets
