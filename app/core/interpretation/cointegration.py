"""Interpretación determinista (sin IA) de un test de cointegración."""
from __future__ import annotations

from app.core.cointegration import CointegrationResult


def interpret_cointegration(result: CointegrationResult, half_life_value: float) -> list[str]:
    """Interpreta cada componente de un `CointegrationResult`.

    Args:
        result: resultado de `engle_granger()`.
        half_life_value: resultado de `half_life()` sobre `result.spread`.

    Returns:
        Lista de puntos en español llano, uno por aspecto del test.
    """
    bullets: list[str] = []

    if result.is_cointegrated:
        bullets.append(
            f"El test de Engle-Granger encuentra **cointegración** "
            f"(p-valor {result.pvalue:.4f} < 0.05): existe una combinación "
            "lineal estable de ambas series."
        )
    else:
        bullets.append(
            f"El test de Engle-Granger **no encuentra cointegración** "
            f"(p-valor {result.pvalue:.4f} >= 0.05) -- no hay base "
            "estadística para asumir reversión a la media."
        )

    if result.adf_spread_pvalue < 0.05:
        bullets.append(
            f"El ADF sobre el spread confirma que es **estacionario** "
            f"(p={result.adf_spread_pvalue:.4f}, estadístico "
            f"{result.adf_spread_stat:.3f})."
        )
    else:
        bullets.append(
            f"El ADF sobre el spread **no rechaza** la existencia de raíz "
            f"unitaria (p={result.adf_spread_pvalue:.4f}) -- inconsistente "
            "con un spread que revierte a la media."
        )

    bullets.append(
        f"Ratio de cobertura (hedge ratio): beta={result.beta:.4f}, "
        f"alpha={result.alpha:.4f} -- el spread se construye como "
        "`y - alpha - beta * x`."
    )

    if half_life_value == float("inf"):
        bullets.append(
            "La vida media de reversión es **indefinida** -- el spread no "
            "revierte de forma medible en el histórico usado."
        )
    elif half_life_value > 60:
        bullets.append(
            f"Vida media de reversión: **{half_life_value:.0f} días** -- "
            "lenta; el capital quedaría inmovilizado mucho tiempo por "
            "operación."
        )
    elif half_life_value <= 15:
        bullets.append(
            f"Vida media de reversión: **{half_life_value:.0f} días** -- "
            "rápida, favorable para pairs trading."
        )
    else:
        bullets.append(
            f"Vida media de reversión: **{half_life_value:.0f} días** -- moderada."
        )

    n_individually_stationary = sum(
        1 for p in (result.adf_pvalue_1, result.adf_pvalue_2) if p < 0.05
    )
    if n_individually_stationary > 0:
        bullets.append(
            f"{n_individually_stationary} de las 2 series individuales ya "
            "son estacionarias por sí solas (ADF p < 0.05) -- si ambas lo "
            "fueran, cointegrarlas no aportaría nada nuevo frente a operar "
            "cada serie por separado."
        )
    else:
        bullets.append(
            "Ninguna de las 2 series es estacionaria por separado (ambas "
            "ADF p >= 0.05) -- el patrón habitual para que la "
            "cointegración tenga sentido."
        )

    return bullets
