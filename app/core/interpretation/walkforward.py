"""Interpretación determinista (sin IA) de un walk-forward analysis."""
from __future__ import annotations

from app.core.walkforward import WalkForwardResult


def interpret_walkforward(result: WalkForwardResult) -> list[str]:
    """Interpreta un `WalkForwardResult`: nº de ventanas, degradación
    IS/OOS del Sharpe, resultado out-of-sample y configuración de embargo."""
    bullets: list[str] = []
    is_m = result.is_metrics_agg
    oos_m = result.oos_metrics_agg
    params = result.params

    n_windows = params.get("n_windows", 0)
    n_skipped = params.get("n_windows_skipped", 0)
    bullets.append(
        f"**{n_windows} ventanas** completadas"
        + (f", {n_skipped} descartadas por errores o señales vacías" if n_skipped else "")
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
                f"**Degradación severa** del Sharpe de IS a OOS "
                f"({degradation:.0%}, de {is_sharpe:.2f} a {oos_sharpe:.2f}) "
                "-- fuerte sospecha de overfitting a los datos de entrenamiento."
            )
        elif degradation > 0.25:
            bullets.append(
                f"Degradación **moderada** del Sharpe de IS a OOS "
                f"({degradation:.0%}, de {is_sharpe:.2f} a {oos_sharpe:.2f})."
            )
        else:
            bullets.append(
                f"Degradación **aceptable** del Sharpe de IS a OOS "
                f"({degradation:.0%}, de {is_sharpe:.2f} a {oos_sharpe:.2f}) "
                "-- buena señal de que la estrategia generaliza."
            )

    oos_return = oos_m.get("total_return")
    oos_dd = oos_m.get("max_drawdown")
    if oos_return is not None and oos_dd is not None:
        bullets.append(
            f"Out-of-sample: retorno total {oos_return:.1%}, caída máxima "
            f"{abs(oos_dd):.1%} -- esta es la lectura que más importa, ya "
            "que refleja datos no vistos durante el ajuste."
        )

    embargo = params.get("embargo", 0)
    if embargo == 0:
        bullets.append(
            "Sin **embargo** entre entrenamiento y test (0 barras) -- "
            "riesgo de fuga de información en la frontera si la estrategia "
            "usa indicadores con memoria larga."
        )
    else:
        bullets.append(
            f"Embargo de **{embargo} barras** entre entrenamiento y test, "
            "para reducir la fuga de información."
        )

    return bullets
