"""Interpretación determinista (sin IA) de un walk-forward analysis,
en lenguaje llano -- evita hablar de "IS/OOS" o "Sharpe" y se centra
en la idea de "probarlo en datos que no había visto"."""
from __future__ import annotations

from app.core.walkforward import WalkForwardResult


def interpret_walkforward(result: WalkForwardResult) -> list[str]:
    """Interpreta un `WalkForwardResult`: nº de pruebas, si el
    resultado se mantiene en datos nuevos, resultado final y margen
    de seguridad entre entrenamiento y prueba."""
    bullets: list[str] = []
    is_m = result.is_metrics_agg
    oos_m = result.oos_metrics_agg
    params = result.params

    n_windows = params.get("n_windows", 0)
    n_skipped = params.get("n_windows_skipped", 0)
    bullets.append(
        f"La estrategia se puso a prueba **{n_windows} veces**, cada vez "
        "entrenándola con un tramo de datos y evaluándola después en "
        "datos que no había visto"
        + (f" ({n_skipped} intentos se descartaron por errores)" if n_skipped else "")
        + "."
    )

    is_sharpe = is_m.get("sharpe")
    oos_sharpe = oos_m.get("sharpe")
    if (
        is_sharpe is not None and oos_sharpe is not None
        and is_sharpe == is_sharpe and oos_sharpe == oos_sharpe  # descarta NaN
        and abs(is_sharpe) > 1e-9
    ):
        degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe)
        if degradation > 0.5:
            bullets.append(
                f"El resultado **empeora mucho** ({degradation:.0%}) al "
                "pasar de los datos de entrenamiento a los datos nuevos "
                "-- fuerte señal de que la estrategia se ha ajustado "
                "demasiado al pasado y podría no funcionar igual de bien "
                "en el futuro."
            )
        elif degradation > 0.25:
            bullets.append(
                f"El resultado **empeora de forma moderada** "
                f"({degradation:.0%}) al pasar a datos nuevos -- normal "
                "hasta cierto punto, pero conviene vigilarlo."
            )
        else:
            bullets.append(
                "El resultado se mantiene **razonablemente parecido** "
                "entre los datos de entrenamiento y los datos nuevos -- "
                "buena señal de que la estrategia no depende de haber "
                "'memorizado' el pasado."
            )

    oos_return = oos_m.get("total_return")
    oos_dd = oos_m.get("max_drawdown")
    if oos_return is not None and oos_dd is not None:
        bullets.append(
            "En los datos nuevos (los que de verdad importan), la "
            f"estrategia ganó un **{oos_return:.1%}** en total, con una "
            f"caída máxima del **{abs(oos_dd):.1%}** desde su punto más "
            "alto."
        )

    embargo = params.get("embargo", 0)
    if embargo == 0:
        bullets.append(
            "No se dejó ningún margen entre el tramo de entrenamiento y "
            "el de prueba -- si la estrategia usa indicadores con "
            "memoria larga, podría estar aprovechando algo de "
            "información que en la práctica no tendría disponible."
        )
    else:
        bullets.append(
            f"Se dejó un margen de **{embargo} días** entre el tramo de "
            "entrenamiento y el de prueba, para evitar que la estrategia "
            "se asome a información futura."
        )

    return bullets
