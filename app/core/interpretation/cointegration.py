"""Interpretación determinista (sin IA) de un test de cointegración,
en lenguaje llano -- evita nombrar los tests estadísticos (Engle-
Granger, ADF) y sus p-valores, y se centra en qué significan para
alguien sin formación estadística.
"""
from __future__ import annotations

from app.core.cointegration import CointegrationResult


def interpret_cointegration(
    result: CointegrationResult,
    half_life_value: float,
    ticker_1: str | None = None,
    ticker_2: str | None = None,
) -> list[str]:
    """Interpreta cada componente de un `CointegrationResult`.

    Args:
        result: resultado de `engle_granger()`.
        half_life_value: resultado de `half_life()` sobre `result.spread`.
        ticker_1, ticker_2: símbolos de los dos activos, para poder
            explicar la relación entre ellos con nombres concretos.
            Si se omiten, se usan términos genéricos.

    Returns:
        Lista de puntos en español llano, uno por aspecto del test.
    """
    t1 = ticker_1 or "el primer activo"
    t2 = ticker_2 or "el segundo activo"
    bullets: list[str] = []

    if result.is_cointegrated:
        bullets.append(
            f"El análisis confirma que **{t1} y {t2} se mueven juntos** "
            "a largo plazo: cuando la diferencia entre ambos se aleja de "
            "lo habitual, tiende a converger de nuevo con el tiempo."
        )
    else:
        bullets.append(
            f"El análisis **no confirma** que {t1} y {t2} se muevan "
            "juntos a largo plazo -- no hay base para asumir que, si se "
            "separan, vayan a converger de nuevo."
        )

    if result.adf_spread_pvalue < 0.05:
        bullets.append(
            "La diferencia entre ambos activos **se comporta de forma "
            "estable**, sin alejarse indefinidamente de su nivel "
            "habitual."
        )
    else:
        bullets.append(
            "La diferencia entre ambos activos **no se comporta de "
            "forma estable** -- podría alejarse de su nivel habitual sin "
            "volver, algo que no encaja con la idea de operar este par."
        )

    bullets.append(
        f"Por cada 1€ que sube {t1}, {t2} tiende a moverse "
        f"{abs(result.beta):.2f}€ en la misma dirección -- es la "
        "relación que mantiene estable la diferencia entre ambos."
    )

    if half_life_value == float("inf"):
        bullets.append(
            "No se puede estimar cuánto tardaría esa diferencia en "
            "volver a su nivel habitual -- en la práctica, no hay una "
            "señal clara de cuándo entrar o salir de la operación."
        )
    elif half_life_value > 60:
        bullets.append(
            "Cuando la diferencia entre ambos activos se aleja de lo "
            f"habitual, tarda de media **{half_life_value:.0f} días** en "
            "volver -- bastante lento; el dinero quedaría inmovilizado "
            "mucho tiempo en cada operación."
        )
    elif half_life_value <= 15:
        bullets.append(
            "Cuando la diferencia entre ambos activos se aleja de lo "
            f"habitual, tarda de media solo **{half_life_value:.0f} "
            "días** en volver -- una velocidad favorable para este tipo "
            "de estrategia."
        )
    else:
        bullets.append(
            "Cuando la diferencia entre ambos activos se aleja de lo "
            f"habitual, tarda de media **{half_life_value:.0f} días** en "
            "volver -- una velocidad moderada."
        )

    n_individually_stable = sum(
        1 for p in (result.adf_pvalue_1, result.adf_pvalue_2) if p < 0.05
    )
    if n_individually_stable == 2:
        bullets.append(
            f"Aviso: {t1} y {t2}, cada uno por separado, ya tienden a "
            "mantenerse cerca de un nivel estable -- en ese caso, "
            "combinarlos no aporta una ventaja especial frente a "
            "analizarlos por separado."
        )

    return bullets
