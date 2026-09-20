"""Interpretación determinista (sin IA) de un análisis de robustez."""
from __future__ import annotations

from app.core.robustness import MonteCarloResult, RobustnessReport, SensitivityResult


def interpret_robustness(
    report: RobustnessReport,
    mc: MonteCarloResult,
    sens: SensitivityResult,
) -> list[str]:
    """Interpreta el score de robustez agregado, la simulación Monte
    Carlo y el análisis de sensibilidad de un parámetro."""
    bullets: list[str] = [
        f"Score de robustez: **{report.final_score:.1f}/100** -- {report.interpretation}"
    ]

    for name, points in report.components.items():
        bullets.append(f"Componente `{name}`: {points:.1f} puntos.")

    prob_positive = float((mc.distribution > 0).mean())
    if prob_positive < 0.5:
        bullets.append(
            f"Monte Carlo: solo el **{prob_positive:.0%}** de las "
            f"simulaciones ({mc.n_simulations}) da Sharpe positivo -- la "
            "estrategia podría perder dinero en más de la mitad de los "
            "escenarios plausibles."
        )
    elif prob_positive < 0.8:
        bullets.append(
            f"Monte Carlo: el **{prob_positive:.0%}** de las simulaciones "
            f"({mc.n_simulations}) da Sharpe positivo -- mayoría "
            "favorable, pero con un margen de escenarios negativos no "
            "despreciable."
        )
    else:
        bullets.append(
            f"Monte Carlo: el **{prob_positive:.0%}** de las simulaciones "
            f"({mc.n_simulations}) da Sharpe positivo -- resultado "
            "consistente a través de los escenarios simulados."
        )

    if sens.stability_score >= 0.7:
        bullets.append(
            f"Sensibilidad de `{sens.param_name}`: score de estabilidad "
            f"**{sens.stability_score:.2f}** -- el resultado no depende de "
            "forma frágil del valor exacto de este parámetro."
        )
    elif sens.stability_score >= 0.4:
        bullets.append(
            f"Sensibilidad de `{sens.param_name}`: score de estabilidad "
            f"**{sens.stability_score:.2f}** -- estabilidad moderada; "
            "revisa la gráfica en busca de un pico aislado frente a una "
            "meseta."
        )
    else:
        bullets.append(
            f"Sensibilidad de `{sens.param_name}`: score de estabilidad "
            f"**{sens.stability_score:.2f}** -- baja estabilidad, el "
            "resultado depende mucho del valor exacto elegido, señal de "
            "posible overfitting de ese parámetro."
        )

    return bullets
