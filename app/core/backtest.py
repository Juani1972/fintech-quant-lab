"""Motor de backtesting vectorizado con anti-look-ahead por diseño.

Modos:
    - 'percent':  los precios son estrictamente positivos. Retornos = pct_change().
    - 'absolute': los precios pueden cruzar cero (típico de un spread de
      cointegración). P&L de la posición = diff() del nivel dividido por
      el capital inicial.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

BacktestMode = Literal["percent", "absolute"]


@dataclass
class BacktestResult:
    """Resultado completo de un backtest."""
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, float | str] = field(default_factory=dict)

    def summary(self) -> str:
        """Devuelve un resumen legible de las métricas."""
        lines = ["=" * 50, "  BACKTEST RESULT", "=" * 50]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                if "return" in k.lower() or "drawdown" in k.lower() or "rate" in k.lower():
                    lines.append(f"  {k:<22} {v:>10.2%}")
                else:
                    lines.append(f"  {k:<22} {v:>10.4f}")
            else:
                lines.append(f"  {k:<22} {v:>10}")
        lines.append("=" * 50)
        return "\n".join(lines)


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    mode: BacktestMode = "percent",
) -> BacktestResult:
    """Ejecuta un backtest vectorizado sobre una serie de precios o spread.

    Args:
        prices: Serie de precios (>0) o spread (cualquier valor, con mode='absolute').
        signals: Serie de señales objetivo en t (-1, 0, 1). Se ejecutan en t+1.
        initial_capital: Capital inicial.
        commission: Comisión por operación.
        slippage: Deslizamiento por operación.
        periods_per_year: Días de trading al año.
        risk_free_rate: Tasa libre de riesgo anualizada.
        mode: 'percent' (default) o 'absolute'.
            - 'percent': precios > 0, retornos = pct_change().
            - 'absolute': precios pueden cruzar cero. P&L de posición =
              diff() del nivel / capital inicial. Adecuado para spreads.

    Returns:
        BacktestResult.

    Raises:
        ValueError: Si los datos son inválidos o las series no están alineadas.
    """
    # --- Validaciones ---
    if not isinstance(prices, pd.Series) or not isinstance(signals, pd.Series):
        raise ValueError("prices y signals deben ser pd.Series.")
    if len(prices) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones.")
    if mode not in ("percent", "absolute"):
        raise ValueError("mode debe ser 'percent' o 'absolute'.")
    if mode == "percent" and (prices.dropna() <= 0).any():
        raise ValueError(
            "En modo 'percent' los precios deben ser estrictamente positivos. "
            "Si operas con un spread que cruza cero, usa mode='absolute'."
        )
    if not signals.dropna().isin([-1, 0, 1]).all():
        raise ValueError("signals solo puede contener -1, 0 o 1.")
    if commission < 0 or slippage < 0:
        raise ValueError("commission y slippage deben ser >= 0.")

    # --- Alinear series ---
    df = pd.concat([prices.rename("price"), signals.rename("signal")], axis=1).dropna()
    if len(df) < 2:
        raise ValueError("Tras alinear, quedan menos de 2 observaciones.")

    prices_ = df["price"]
    signals_ = df["signal"]

    # --- Anti-look-ahead ---
    positions = signals_.shift(1).fillna(0)

    # --- Retornos brutos según modo ---
    if mode == "percent":
        asset_returns = prices_.pct_change().fillna(0)
        gross_returns = positions * asset_returns
    else:  # absolute
        asset_changes = prices_.diff().fillna(0)
        gross_returns = positions * asset_changes / initial_capital

    # --- Costes ---
    position_changes = positions.diff().abs().fillna(positions.abs())
    total_cost_per_trade = commission + slippage
    costs = position_changes * total_cost_per_trade

    # --- Retornos netos ---
    net_returns = gross_returns - costs

    # --- Curva de capital ---
    equity = initial_capital * (1 + net_returns).cumprod()

    # --- Trades (emparejar entradas con salidas) ---
    trades = _extract_trades(prices_, positions, equity, initial_capital, mode=mode)

    # --- Métricas ---
    metrics = _compute_metrics(
        net_returns=net_returns,
        equity=equity,
        positions=positions,
        trades=trades,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        initial_capital=initial_capital,
    )

    params: dict[str, float | str] = {
        "initial_capital": initial_capital,
        "commission": commission,
        "slippage": slippage,
        "periods_per_year": periods_per_year,
        "risk_free_rate": risk_free_rate,
        "mode": mode,
    }

    return BacktestResult(
        equity_curve=equity,
        returns=net_returns,
        positions=positions,
        trades=trades,
        metrics=metrics,
        params=params,
    )


def _extract_trades(
    prices: pd.Series,
    positions: pd.Series,
    equity: pd.Series,
    initial_capital: float,
    mode: BacktestMode = "percent",
) -> pd.DataFrame:
    """Empareja entradas y salidas para construir la tabla de operaciones.

    Una operación empieza cuando la posición pasa de 0 a ±1 y termina
    cuando vuelve a 0 o cambia de signo.

    `equity` debe ser la curva de capital ya calculada por el llamante
    (``initial_capital * (1 + net_returns).cumprod()``): se reutiliza en
    vez de recalcularse en cada iteración, lo que evita un coste O(n²)
    en backtests largos (walk-forward, grid search).
    """
    empty = pd.DataFrame(
        columns=["entry_date", "exit_date", "direction", "entry_price",
                 "exit_price", "pnl_pct", "pnl_abs", "bars_held"]
    )
    if positions.abs().sum() == 0:
        return empty

    def _pnl_pct(exit_price: float, entry_p: float, pos: int) -> float:
        if mode == "percent":
            return (exit_price - entry_p) / entry_p * pos
        # absolute: P&L relativo al capital inicial (adecuado para spreads
        # que pueden cruzar cero, donde un % sobre entry_p no tiene sentido)
        return (exit_price - entry_p) * pos / initial_capital

    price_vals = prices.to_numpy()
    equity_vals = equity.to_numpy()
    pos_vals = positions.to_numpy()
    dates = positions.index

    trades = []
    current_pos = 0
    entry_idx = None
    entry_price: float | None = None
    entry_equity = None

    for i in range(len(positions)):
        pos = pos_vals[i]
        price = price_vals[i]
        equity_i = equity_vals[i]

        if pos != current_pos:
            if current_pos != 0 and entry_idx is not None:
                assert entry_price is not None, "entry_price se fija junto con entry_idx"
                pnl_abs = equity_i - entry_equity
                pnl_pct = _pnl_pct(price, entry_price, current_pos)
                trades.append({
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[i],
                    "direction": "long" if current_pos > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "pnl_abs": pnl_abs,
                    "bars_held": i - entry_idx,
                })

            if pos != 0:
                entry_idx = i
                entry_price = price
                entry_equity = equity_i

            current_pos = pos

    if current_pos != 0 and entry_idx is not None:
        assert entry_price is not None, "entry_price se fija junto con entry_idx"
        last_idx = len(positions) - 1
        last_price = price_vals[last_idx]
        last_equity = equity_vals[last_idx]
        pnl_abs = last_equity - entry_equity
        pnl_pct = _pnl_pct(last_price, entry_price, current_pos)
        trades.append({
            "entry_date": dates[entry_idx],
            "exit_date": dates[last_idx],
            "direction": "long" if current_pos > 0 else "short",
            "entry_price": entry_price,
            "exit_price": last_price,
            "pnl_pct": pnl_pct,
            "pnl_abs": pnl_abs,
            "bars_held": last_idx - entry_idx,
        })

    return pd.DataFrame(trades)


def _compute_metrics(
    net_returns: pd.Series,
    equity: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float,
    initial_capital: float,
) -> dict[str, float]:
    """Calcula todas las métricas de rendimiento y riesgo."""
    metrics: dict[str, float] = {}

    n_periods = len(net_returns)
    years = n_periods / periods_per_year

    total_return = equity.iloc[-1] / initial_capital - 1
    metrics["total_return"] = total_return
    metrics["annual_return"] = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    daily_rf = risk_free_rate / periods_per_year
    excess = net_returns - daily_rf
    vol_annual = net_returns.std() * np.sqrt(periods_per_year)
    metrics["annual_volatility"] = vol_annual

    if net_returns.std() > 0:
        metrics["sharpe"] = excess.mean() / net_returns.std() * np.sqrt(periods_per_year)
    else:
        metrics["sharpe"] = np.nan

    downside = net_returns[net_returns < daily_rf]
    if len(downside) > 0 and downside.std() > 0:
        metrics["sortino"] = excess.mean() / downside.std() * np.sqrt(periods_per_year)
    else:
        metrics["sortino"] = np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    metrics["max_drawdown"] = drawdown.min()

    if metrics["max_drawdown"] != 0:
        metrics["calmar"] = metrics["annual_return"] / abs(metrics["max_drawdown"])
    else:
        metrics["calmar"] = np.nan

    metrics["n_trades"] = len(trades)
    if len(trades) > 0:
        wins = trades[trades["pnl_abs"] > 0]
        losses = trades[trades["pnl_abs"] < 0]
        metrics["win_rate"] = len(wins) / len(trades)
        gross_profit = wins["pnl_abs"].sum() if len(wins) else 0.0
        gross_loss = abs(losses["pnl_abs"].sum()) if len(losses) else 0.0
        metrics["profit_factor"] = (
            gross_profit / gross_loss if gross_loss > 0 else np.inf
        )
        metrics["avg_trade_pnl"] = trades["pnl_abs"].mean()
        metrics["avg_bars_held"] = trades["bars_held"].mean()
    else:
        metrics["win_rate"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_trade_pnl"] = 0.0
        metrics["avg_bars_held"] = 0.0

    metrics["exposure"] = (positions != 0).mean()
    metrics["turnover"] = positions.diff().abs().sum() / years if years > 0 else np.nan

    return metrics


def buy_and_hold(prices: pd.Series, initial_capital: float = 100_000.0) -> pd.Series:
    """Curva de capital de una estrategia buy & hold."""
    returns = prices.pct_change().fillna(0)
    return initial_capital * (1 + returns).cumprod()


def compare_to_benchmark(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> pd.DataFrame:
    """Compara la curva de capital de la estrategia con un benchmark."""
    df = pd.concat(
        [strategy_equity.rename("strategy"), benchmark_equity.rename("benchmark")],
        axis=1,
    ).dropna()
    df["strategy_norm"] = df["strategy"] / df["strategy"].iloc[0]
    df["benchmark_norm"] = df["benchmark"] / df["benchmark"].iloc[0]
    return df


def benchmark_metrics(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Métricas de la estrategia relativas a un benchmark.

    `compare_to_benchmark` solo normaliza las dos curvas para dibujarlas
    juntas; esta función añade las métricas que de verdad responden a
    "¿esta estrategia bate al benchmark, y cómo de arriesgado es ese
    exceso de retorno?":

    - beta: sensibilidad de los retornos de la estrategia a los del
      benchmark (regresión OLS simple retorno_estrategia ~ retorno_benchmark).
    - jensen_alpha: exceso de retorno anualizado de la estrategia sobre
      lo que predeciría el CAPM dado su beta.
    - tracking_error: desviación estándar anualizada de la diferencia
      de retornos (estrategia - benchmark).
    - information_ratio: exceso de retorno anualizado sobre el
      benchmark dividido por el tracking error.

    Raises:
        ValueError: Si tras alinear ambas series quedan menos de 3
            observaciones, o si el benchmark no tiene varianza.
    """
    df = pd.concat(
        [strategy_equity.rename("strategy"), benchmark_equity.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(df) < 3:
        raise ValueError("Se necesitan al menos 3 observaciones alineadas.")

    r_s = df["strategy"].pct_change().dropna()
    r_b = df["benchmark"].pct_change().dropna()
    r_s, r_b = r_s.align(r_b, join="inner")

    var_b = r_b.var(ddof=1)
    if var_b == 0 or not np.isfinite(var_b):
        raise ValueError("El benchmark no tiene varianza; beta no está definido.")

    beta = float(np.cov(r_s, r_b, ddof=1)[0, 1] / var_b)

    rf_period = risk_free_rate / periods_per_year
    alpha_period = float((r_s.mean() - rf_period) - beta * (r_b.mean() - rf_period))
    jensen_alpha = alpha_period * periods_per_year

    excess = r_s - r_b
    tracking_error = float(excess.std(ddof=1) * np.sqrt(periods_per_year))
    information_ratio = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))
        if excess.std(ddof=1) > 0 else float("nan")
    )

    return {
        "beta": beta,
        "jensen_alpha": jensen_alpha,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
    }
