"""Motor de walk-forward analysis.

Divide los datos en ventanas sucesivas de entrenamiento (in-sample)
y prueba (out-of-sample). Para cada ventana, entrena el modelo y
evalúa la estrategia solo en el periodo OOS.

Esto es la herramienta estándar para detectar overfitting: si la
performance OOS se degrada mucho respecto a IS, la estrategia no
generaliza.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.core.backtest import BacktestMode, run_backtest


SignalGenerator = Callable[[pd.Series, pd.Series], pd.Series]


@dataclass
class WalkForwardWindow:
    """Resultado de una ventana walk-forward."""
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]


@dataclass
class WalkForwardResult:
    """Resultado completo del walk-forward."""
    windows: list[WalkForwardWindow]
    is_metrics_agg: dict[str, float]
    oos_metrics_agg: dict[str, float]
    oos_equity_concat: pd.Series
    params: dict


def walk_forward_analysis(
    prices: pd.Series,
    signal_generator: SignalGenerator,
    train_size: int = 504,
    test_size: int = 126,
    step: int | None = None,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    mode: BacktestMode = "percent",
) -> WalkForwardResult:
    """Ejecuta walk-forward analysis.

    Args:
        prices: Serie de precios o spread.
        signal_generator: Función `(precios_train, precios_full) -> señales_full`.
        train_size: Nº de barras de la ventana de entrenamiento.
        test_size: Nº de barras de la ventana de test.
        step: Desplazamiento entre ventanas. Si None, usa test_size.
        initial_capital: Capital para cada backtest OOS.
        commission: Comisión por operación.
        slippage: Slippage por operación.
        mode: 'percent' para precios, 'absolute' para spreads.

    Returns:
        WalkForwardResult con las métricas IS/OOS agregadas.

    Raises:
        ValueError: Si los datos son insuficientes.
    """
    if step is None:
        step = test_size

    n = len(prices)
    if n < train_size + test_size:
        raise ValueError(
            f"Datos insuficientes: {n} barras. "
            f"Se necesitan al menos {train_size + test_size}."
        )

    windows: list[WalkForwardWindow] = []
    oos_equity_parts: list[pd.Series] = []

    start = 0
    while start + train_size + test_size <= n:
        train_slice = prices.iloc[start : start + train_size]
        test_slice = prices.iloc[start + train_size : start + train_size + test_size]
        visible = prices.iloc[: start + train_size + test_size]

        try:
            signals_full = signal_generator(train_slice, visible)
        except Exception:
            start += step
            continue

        if signals_full is None or len(signals_full) == 0:
            start += step
            continue

        signals_is = signals_full.reindex(train_slice.index).fillna(0)
        res_is = run_backtest(
            train_slice, signals_is,
            initial_capital=initial_capital,
            commission=commission, slippage=slippage,
            mode=mode,
        )

        signals_oos = signals_full.reindex(test_slice.index).fillna(0)
        res_oos = run_backtest(
            test_slice, signals_oos,
            initial_capital=initial_capital,
            commission=commission, slippage=slippage,
            mode=mode,
        )

        windows.append(WalkForwardWindow(
            train_start=train_slice.index[0],
            train_end=train_slice.index[-1],
            test_start=test_slice.index[0],
            test_end=test_slice.index[-1],
            is_metrics=res_is.metrics,
            oos_metrics=res_oos.metrics,
        ))

        oos_equity_parts.append(res_oos.equity_curve)
        start += step

    if not windows:
        raise ValueError("No se pudo completar ninguna ventana walk-forward.")

    oos_equity_concat = _compound_equity(oos_equity_parts, initial_capital)
    is_agg = _aggregate_metrics([w.is_metrics for w in windows])
    oos_agg = _aggregate_metrics([w.oos_metrics for w in windows])

    return WalkForwardResult(
        windows=windows,
        is_metrics_agg=is_agg,
        oos_metrics_agg=oos_agg,
        oos_equity_concat=oos_equity_concat,
        params={
            "train_size": train_size,
            "test_size": test_size,
            "step": step,
            "n_windows": len(windows),
            "initial_capital": initial_capital,
            "commission": commission,
            "slippage": slippage,
            "mode": mode,
        },
    )


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Promedia las métricas entre ventanas."""
    if not metrics_list:
        return {}
    keys = set().union(*[m.keys() for m in metrics_list])
    result: dict[str, float] = {}
    for k in keys:
        vals = [m.get(k, np.nan) for m in metrics_list]
        vals_clean = [v for v in vals if v is not None and np.isfinite(v)]
        result[k] = float(np.mean(vals_clean)) if vals_clean else np.nan
    return result


def _compound_equity(
    equity_parts: list[pd.Series],
    initial_capital: float,
) -> pd.Series:
    """Concatena curvas OOS aplicando composición de capital."""
    if not equity_parts:
        return pd.Series(dtype=float)

    compounded = []
    capital = initial_capital

    for eq in equity_parts:
        if len(eq) == 0:
            continue
        factor = capital / eq.iloc[0]
        eq_scaled = eq * factor
        compounded.append(eq_scaled)
        capital = eq_scaled.iloc[-1]

    result = pd.concat(compounded)
    result = result[~result.index.duplicated(keep="first")].sort_index()
    return result


def signal_from_pairs_trading(
    window: int = 60,
    entry: float = 2.0,
    exit_: float = 0.5,
    shift: int = 1,
) -> SignalGenerator:
    """Generador de señales para pairs trading (asume que `prices` es el spread)."""
    from app.core.cointegration import generate_signals, rolling_zscore

    def generator(train_prices: pd.Series, full_prices: pd.Series) -> pd.Series:
        z = rolling_zscore(full_prices, window=window, shift=shift)
        return generate_signals(z, entry=entry, exit_=exit_)

    return generator


def signal_from_momentum(window: int = 60) -> SignalGenerator:
    """Generador de señales de momentum simple."""

    def generator(train_prices: pd.Series, full_prices: pd.Series) -> pd.Series:
        ret = full_prices.pct_change(window)
        signals = pd.Series(0, index=full_prices.index, dtype=int)
        signals[ret > 0] = 1
        signals[ret < 0] = -1
        return signals

    return generator


def signal_from_mean_reversion(
    window: int = 30,
    entry: float = 1.5,
    exit_: float = 0.5,
) -> SignalGenerator:
    """Generador de señales de reversión a la media."""

    def generator(train_prices: pd.Series, full_prices: pd.Series) -> pd.Series:
        mean = full_prices.rolling(window).mean().shift(1)
        std = full_prices.rolling(window).std().shift(1)
        z = (full_prices - mean) / std

        signals = pd.Series(0, index=full_prices.index, dtype=int)
        position = 0
        for i, zi in enumerate(z):
            if np.isnan(zi):
                signals.iloc[i] = position
                continue
            if position == 0:
                if zi > entry:
                    position = -1
                elif zi < -entry:
                    position = 1
            elif abs(zi) < exit_:
                position = 0
            signals.iloc[i] = position
        return signals

    return generator
