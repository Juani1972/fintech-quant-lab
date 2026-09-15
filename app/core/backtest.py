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
    params: dict[str, float] = field(default_factory=dict)

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

    # --- Trades ---
    trades = _extract_trades(
        prices_, positions, net_returns, initial_capital, mode=mode,
    )

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

    params = {
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
    net_returns: pd.Series,
    initial_capital: float,
    mode: BacktestMode = "percent",
) -> pd.DataFrame:
    """Empareja entradas y salidas para construir la tabla de operaciones."""
    empty = pd.DataFrame(
        columns=["entry_date", "exit_date", "direction", "entry_price",
                 "exit_price", "pnl_pct", "pnl_abs", "bars_held"]
    )
    if positions.abs().sum() == 0:
        return empty

    trades = []
    current_pos = 0
    entry_date = None
    entry_price = None
    entry_equity = None

    def _pnl_pct(price: float, entry_p: float, pos: int) -> float:
        if mode == "percent":
            return (price - entry_p) / entry_p * pos
        # absolute: P&L relativo al capital inicial
        return (price - entry_p) * pos / initial_capital

    for i, (date, pos) in enumerate(positions.items()):
        price = prices.loc[date]
        equity = initial_capital * (1 + net_returns.iloc[: i + 1]).prod()

        if pos != current_pos:
            # Cerrar posición anterior
            if current_pos != 0 and entry_date is not None:
                pnl_abs = equity - entry_equity
                pnl_pct = _pnl_pct(price, entry_price, current_pos)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "long" if current_pos > 0 else "short",
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "pnl_abs": pnl_abs,
                    "bars_held": i - positions.index.get_loc(entry_date),
                })

            # Abrir nueva posición
            if pos != 0:
                entry_date = date
                entry_price = price
                entry_equity = equity

            current_pos = pos

    # Cerrar posición abierta al final
    if current_pos != 0 and entry_date is not None:
        last_date = positions.index[-1]
        last_price = prices.loc[last_date]
        last_equity = initial_capital * (1 + net_returns).prod()
        pnl_abs = last_equity - entry_equity
        pnl_pct = _pnl_pct(last_price, entry_price, current_pos)
        trades.append({
            "entry_date": entry_date,
            "exit_date": last_date,
            "direction": "long" if current_pos > 0 else "short",
            "entry_price": entry_price,
            "exit_price": last_price,
            "pnl_pct": pnl_pct,
            "pnl_abs": pnl_abs,
            "bars_held": len(positions) - positions.index.get_loc(entry_date) - 1,
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
