"""Detección de regímenes de mercado con Hidden Markov Models (HMM).

Un HMM gaussiano asume que los retornos provienen de una mezcla de
distribuciones normales (los "regímenes"), cada una con su propia media
y volatilidad. El modelo aprende la secuencia de regímenes y las
probabilidades de transición entre ellos.

Aplicaciones:
    - Identificar periodos de alta vs baja volatilidad.
    - Bull vs bear market.
    - Filtrar señales de trading según el régimen activo.

Referencia:
    Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
    Nonstationary Time Series and the Business Cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# ============================================================
#  Modelo
# ============================================================
@dataclass
class RegimeResult:
    """Resultado del ajuste de un HMM."""
    n_states: int
    states: pd.Series                    # secuencia de estados (0..n-1)
    state_probs: pd.DataFrame            # probabilidades posteriores por estado
    means: np.ndarray                    # media de retornos por estado
    variances: np.ndarray                # varianza por estado
    transition_matrix: np.ndarray        # P(i → j)
    log_likelihood: float
    aic: float
    bic: float
    model: GaussianHMM = field(repr=False)

    def state_volatilities(self, periods_per_year: int = 252) -> pd.Series:
        """Volatilidad anualizada por estado."""
        return pd.Series(
            np.sqrt(self.variances * periods_per_year),
            index=[f"Régimen {i}" for i in range(self.n_states)],
            name="volatilidad_anual",
        )

    def state_means_annualized(self, periods_per_year: int = 252) -> pd.Series:
        """Media anualizada por estado."""
        return pd.Series(
            self.means * periods_per_year,
            index=[f"Régimen {i}" for i in range(self.n_states)],
            name="retorno_anual",
        )


# ============================================================
#  Fit
# ============================================================
def fit_hmm(
    returns: pd.Series,
    n_states: int = 2,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int | None = 42,
) -> RegimeResult:
    """Ajusta un Gaussian HMM a una serie de retornos.

    Args:
        returns: Serie de retornos (diarios idealmente).
        n_states: Número de regímenes (2 = bull/bear o alta/baja vol).
        covariance_type: 'diag', 'full', 'tied' o 'spherical'.
        n_iter: Iteraciones máximas del algoritmo EM.
        random_state: Semilla para reproducibilidad.

    Returns:
        RegimeResult con estados, parámetros y métricas.

    Raises:
        ValueError: Si los datos son insuficientes o inválidos.
    """
    if n_states < 2:
        raise ValueError("n_states debe ser >= 2.")
    clean = returns.dropna()
    if len(clean) < n_states * 20:
        raise ValueError(
            f"Datos insuficientes: {len(clean)} observaciones para "
            f"{n_states} estados. Se recomiendan al menos {n_states * 20}."
        )

    X = clean.values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    states_arr = model.predict(X)
    probs_arr = model.predict_proba(X)

    # Ordenar los estados por volatilidad ascendente (Régimen 0 = baja vol)
    order = np.argsort(model.covars_.flatten())
    remap = {old: new for new, old in enumerate(order)}

    states_sorted = np.array([remap[s] for s in states_arr])
    probs_sorted = probs_arr[:, order]
    means_sorted = model.means_.flatten()[order]
    variances_sorted = model.covars_.flatten()[order]
    trans_sorted = model.transmat_[np.ix_(order, order)]

    n_params = (
        n_states * 2                            # medias + varianzas
        + n_states * (n_states - 1)             # transiciones libres
        + n_states - 1                          # pesos iniciales libres
    )
    aic = -2 * model.score(X) + 2 * n_params
    bic = -2 * model.score(X) + n_params * np.log(len(clean))

    return RegimeResult(
        n_states=n_states,
        states=pd.Series(states_sorted, index=clean.index, name="state"),
        state_probs=pd.DataFrame(
            probs_sorted,
            index=clean.index,
            columns=[f"P(Régimen {i})" for i in range(n_states)],
        ),
        means=means_sorted,
        variances=variances_sorted,
        transition_matrix=trans_sorted,
        log_likelihood=float(model.score(X)),
        aic=float(aic),
        bic=float(bic),
        model=model,
    )


# ============================================================
#  Análisis por régimen
# ============================================================
def regime_summary(
    returns: pd.Series,
    result: RegimeResult,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estadísticas descriptivas por régimen.

    Returns:
        DataFrame con una fila por régimen: nº obs, % del tiempo,
        media anualizada, volatilidad anualizada, Sharpe y duración media.
    """
    df = pd.DataFrame({
        "return": returns.reindex(result.states.index),
        "state": result.states,
    }).dropna()

    rows = []
    for s in range(result.n_states):
        sub = df[df["state"] == s]["return"]
        if len(sub) == 0:
            rows.append({
                "Régimen": f"Régimen {s}",
                "Nº obs": 0,
                "% tiempo": 0.0,
                "Retorno anual": np.nan,
                "Volatilidad anual": np.nan,
                "Sharpe": np.nan,
                "Duración media (días)": np.nan,
            })
            continue

        mean_ann = float(sub.mean() * periods_per_year)
        vol_ann = float(sub.std() * np.sqrt(periods_per_year))
        sharpe = mean_ann / vol_ann if vol_ann > 0 else np.nan

        # Duración media: contar runs consecutivos
        states_series = df["state"]
        durations = []
        run = 1
        for i in range(1, len(states_series)):
            if states_series.iloc[i] == states_series.iloc[i - 1]:
                run += 1
            else:
                if states_series.iloc[i - 1] == s:
                    durations.append(run)
                run = 1
        if len(states_series) > 0 and states_series.iloc[-1] == s:
            durations.append(run)

        rows.append({
            "Régimen": f"Régimen {s}",
            "Nº obs": len(sub),
            "% tiempo": len(sub) / len(df),
            "Retorno anual": mean_ann,
            "Volatilidad anual": vol_ann,
            "Sharpe": sharpe,
            "Duración media (días)": float(np.mean(durations)) if durations else np.nan,
        })

    return pd.DataFrame(rows).set_index("Régimen")


def current_regime(result: RegimeResult) -> int:
    """Devuelve el régimen más reciente."""
    return int(result.states.iloc[-1])


def regime_stats_table(result: RegimeResult) -> pd.DataFrame:
    """Tabla con la matriz de transición y los parámetros por estado."""
    trans = pd.DataFrame(
        result.transition_matrix,
        index=[f"Desde R{i}" for i in range(result.n_states)],
        columns=[f"A R{j}" for j in range(result.n_states)],
    )
    return trans
