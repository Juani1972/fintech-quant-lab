"""
app/core/portfolio.py

Construcción de carteras: desde el clásico Markowitz hasta Hierarchical
Risk Parity (HRP, López de Prado 2016). Todo implementado con álgebra
lineal analítica o algoritmos iterativos simples — sin `cvxpy` ni
`scikit-learn` — usando solo `numpy`, `pandas` y `scipy.cluster.hierarchy`
(ya presente en el stack del proyecto).

Uso típico:

    from app.core.portfolio import hrp_weights, rebalance_schedule

    weights = hrp_weights(returns)
    schedule = rebalance_schedule(weights_over_time, method="calendar", frequency="M")
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


class PortfolioError(Exception):
    """Error genérico al construir una cartera."""


# ---------------------------------------------------------------------------
# Markowitz (media-varianza), forma cerrada sin restricciones de desigualdad
# ---------------------------------------------------------------------------

def markowitz_weights(
    returns: pd.DataFrame,
    target_return: float | None = None,
    risk_free: float = 0.0,
) -> pd.Series:
    """
    Pesos de media-varianza en forma cerrada (sin restricción de no
    negatividad, por lo que puede haber posiciones cortas).

    - Si `target_return` es None: cartera tangente (máximo ratio de
      Sharpe) usando el exceso de retorno sobre `risk_free`.
    - Si `target_return` se indica: cartera de varianza mínima que
      alcanza exactamente ese retorno esperado (solución de Lagrange
      de dos fondos).

    Args:
        returns: DataFrame de retornos periódicos, una columna por activo.
        target_return: retorno objetivo (misma frecuencia que `returns`).
        risk_free: tipo libre de riesgo (misma frecuencia que `returns`),
            usado solo cuando `target_return` es None.

    Returns:
        pd.Series de pesos indexada por activo, que suma 1.0.

    Raises:
        PortfolioError: si `returns` está vacío, tiene menos de 2 activos,
            o la matriz de covarianza es singular.
    """
    if returns.empty or returns.shape[1] < 2:
        raise PortfolioError("markowitz_weights necesita al menos 2 activos.")

    mu = returns.mean()
    cov = returns.cov()

    try:
        inv_cov = np.linalg.inv(cov.values)
    except np.linalg.LinAlgError as exc:
        raise PortfolioError("La matriz de covarianza es singular (activos colineales).") from exc

    ones = np.ones(len(mu))

    if target_return is None:
        excess = mu.values - risk_free
        raw = inv_cov @ excess
        denom = ones @ raw
        if denom == 0:
            raise PortfolioError("No se pudo normalizar la cartera tangente (denominador 0).")
        weights = raw / denom
    else:
        a = ones @ inv_cov @ ones
        b = ones @ inv_cov @ mu.values
        c = mu.values @ inv_cov @ mu.values
        denom = a * c - b * b
        if denom == 0:
            raise PortfolioError("Sistema de Markowitz degenerado; no se puede resolver.")
        lam = (c - b * target_return) / denom
        gam = (a * target_return - b) / denom
        weights = lam * (inv_cov @ ones) + gam * (inv_cov @ mu.values)

    return pd.Series(weights, index=returns.columns, name="weight")


# ---------------------------------------------------------------------------
# Risk parity (Equal Risk Contribution), algoritmo iterativo
# ---------------------------------------------------------------------------

def risk_parity_weights(
    returns: pd.DataFrame,
    budget: pd.Series | None = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> pd.Series:
    """
    Pesos de Equal Risk Contribution (ERC): cada activo contribuye al
    riesgo total de la cartera en proporción a su `budget` (por defecto,
    a partes iguales). Se resuelve por descenso cíclico de coordenadas,
    sin depender de un solver de optimización externo.

    Args:
        returns: DataFrame de retornos periódicos, una columna por activo.
        budget: presupuesto de riesgo objetivo por activo (debe sumar 1).
            Si es None, se usa reparto igualitario.
        max_iter: iteraciones máximas del descenso cíclico.
        tol: tolerancia de convergencia sobre el cambio de pesos.

    Returns:
        pd.Series de pesos (todos > 0) indexada por activo, que suma 1.0.

    Raises:
        PortfolioError: si `returns` está vacío, tiene menos de 2 activos,
            o `budget` no coincide con las columnas de `returns`.
    """
    if returns.empty or returns.shape[1] < 2:
        raise PortfolioError("risk_parity_weights necesita al menos 2 activos.")

    cov = returns.cov().values
    n = cov.shape[0]

    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        if set(budget.index) != set(returns.columns):
            raise PortfolioError("budget debe tener exactamente los mismos activos que returns.")
        b = budget.reindex(returns.columns).values
        if not np.isclose(b.sum(), 1.0, atol=1e-6):
            raise PortfolioError("budget debe sumar 1.0.")

    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        w_prev = w.copy()
        port_var = w @ cov @ w
        marginal = cov @ w

        for i in range(n):
            # Resuelve w_i tal que su contribución de riesgo == b_i * riesgo total,
            # manteniendo el resto de pesos fijos (descenso cíclico de coordenadas).
            others = marginal[i] - cov[i, i] * w[i]
            a = cov[i, i]
            c = -b[i] * port_var if port_var > 0 else -b[i] * 1e-12
            # a*w_i^2 + others*w_i + c = 0  (root positivo)
            discriminant = others**2 - 4 * a * c
            discriminant = max(discriminant, 0.0)
            w[i] = (-others + np.sqrt(discriminant)) / (2 * a) if a > 0 else w[i]
            w[i] = max(w[i], 1e-8)
            port_var = w @ cov @ w
            marginal = cov @ w

        w = w / w.sum()

        if np.abs(w - w_prev).max() < tol:
            break

    return pd.Series(w, index=returns.columns, name="weight")


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity (López de Prado, 2016)
# ---------------------------------------------------------------------------

def _get_quasi_diag(link: np.ndarray) -> list[int]:
    """Ordena los activos según el dendrograma (quasi-diagonalización)."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])

    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return cast(list[int], sort_ix.tolist())


def _get_cluster_var(cov: pd.DataFrame, items: list) -> float:
    """Varianza de un clúster usando pesos de varianza inversa dentro del clúster."""
    sub_cov = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub_cov.values)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov.values @ ivp)


def _get_rec_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Bisección recursiva: reparte el peso entre subclústeres según su varianza relativa."""
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while clusters:
        clusters = [
            c[start:end]
            for c in clusters
            for start, end in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]

        for i in range(0, len(clusters), 2):
            c0 = clusters[i]
            c1 = clusters[i + 1]

            var0 = _get_cluster_var(cov, c0)
            var1 = _get_cluster_var(cov, c1)
            alpha = 1.0 - var0 / (var0 + var1)

            weights[c0] *= alpha
            weights[c1] *= 1.0 - alpha

    return weights


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity (López de Prado, 2016).

    A diferencia de Markowitz, no requiere invertir la matriz de
    covarianza (más estable con muchos activos correlacionados) y no
    requiere estimar retornos esperados, solo la estructura de
    correlación/covarianza.

    Pasos: (1) clustering jerárquico sobre la matriz de distancias de
    correlación, (2) quasi-diagonalización según el dendrograma, (3)
    bisección recursiva repartiendo el peso según la varianza relativa
    de cada subclúster.

    Args:
        returns: DataFrame de retornos periódicos, una columna por activo.

    Returns:
        pd.Series de pesos (todos > 0) indexada por activo, que suma 1.0.

    Raises:
        PortfolioError: si `returns` tiene menos de 2 activos o
            `scipy` no está instalado.
    """
    if returns.empty or returns.shape[1] < 2:
        raise PortfolioError("hrp_weights necesita al menos 2 activos.")

    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError as exc:  # pragma: no cover
        raise PortfolioError("scipy no está instalado. Añádelo a requirements.txt.") from exc

    corr = returns.corr()
    cov = returns.cov()

    dist_arr = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, None))
    np.fill_diagonal(dist_arr, 0.0)
    dist = pd.DataFrame(dist_arr, index=corr.index, columns=corr.columns)

    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="single")

    sort_ix_int = _get_quasi_diag(link)
    sort_ix = corr.index[sort_ix_int].tolist()

    weights = _get_rec_bisection(cov, sort_ix)
    weights = weights.reindex(returns.columns)
    weights.name = "weight"
    return weights


# ---------------------------------------------------------------------------
# Calendario de rebalanceo
# ---------------------------------------------------------------------------

def rebalance_schedule(
    weights: pd.DataFrame,
    method: str = "calendar",
    frequency: str = "M",
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calendario de rebalanceo sobre una serie temporal de pesos objetivo.

    Args:
        weights: DataFrame indexado por fecha, una columna por activo,
            con el peso objetivo (o el peso que tendría la cartera sin
            rebalancear, para el método 'threshold') en cada fecha.
        method: 'calendar' (rebalancea a fechas fijas según `frequency`)
            o 'threshold' (rebalancea cuando algún peso se desvía más
            de `threshold` desde el último rebalanceo).
        frequency: frecuencia pandas ('M', 'Q', 'W'...) para el método 'calendar'.
        threshold: desviación absoluta máxima tolerada antes de rebalancear,
            para el método 'threshold'.

    Returns:
        DataFrame igual al índice de `weights`, con una columna booleana
        'rebalance' marcando en qué fechas toca rebalancear.

    Raises:
        PortfolioError: si `weights` está vacío o `method` no es válido.
    """
    if weights.empty:
        raise PortfolioError("weights está vacío.")

    if method == "calendar":
        period = weights.index.to_series().dt.to_period(frequency)
        is_new_period = period.ne(period.shift(1))
        is_new_period.iloc[0] = True
        result = pd.DataFrame({"rebalance": is_new_period.values}, index=weights.index)
        return result

    if method == "threshold":
        flags = [True]
        last_rebalanced = weights.iloc[0]

        for i in range(1, len(weights)):
            current = weights.iloc[i]
            drift = (current - last_rebalanced).abs().max()
            if drift > threshold:
                flags.append(True)
                last_rebalanced = current
            else:
                flags.append(False)

        return pd.DataFrame({"rebalance": flags}, index=weights.index)

    raise PortfolioError(f"method desconocido '{method}'. Usa 'calendar' o 'threshold'.")
