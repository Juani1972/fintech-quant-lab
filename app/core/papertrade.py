"""
app/core/papertrade.py

Trading en papel (paper trading) en vivo contra Alpaca, sin arriesgar
capital real. Usa la API REST de Alpaca directamente vía `requests`
(en vez del SDK `alpaca-py`) para no añadir una dependencia pesada
solo para esto — el proyecto ya depende de `requests` para
AlphaVantageProvider y license.py.

Uso típico:

    from app.core.papertrade import PaperAccount, StrategyRunner

    account = PaperAccount(api_key="...", api_secret="...")
    account.submit_order(symbol="AAPL", qty=10, side="buy")

    runner = StrategyRunner(
        strategy=mi_estrategia,
        symbols=["AAPL", "MSFT"],
        capital=10_000,
        account=account,
    )
    runner.run_once()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


class PaperTradeError(Exception):
    """Error genérico al operar contra la cuenta de paper trading."""


@dataclass
class PaperAccount:
    """Cliente ligero de la API REST de paper trading de Alpaca."""

    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_secret:
            raise PaperTradeError("api_key y api_secret son obligatorios.")
        self.base_url = self.base_url.rstrip("/")

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise PaperTradeError("requests no está instalado. Añádelo a requirements.txt.") from exc

        url = f"{self.base_url}{path}"
        try:
            resp = requests.request(
                method, url, headers=self._headers(), timeout=self.timeout, **kwargs
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise PaperTradeError(f"Fallo al llamar a Alpaca ({method} {path}): {exc}") from exc

        if not resp.content:
            return None

        try:
            return resp.json()
        except ValueError as exc:
            raise PaperTradeError(f"Respuesta de Alpaca no es JSON válido: {exc}") from exc

    def get_account(self) -> dict:
        """Devuelve el estado de la cuenta (equity, cash, buying_power...)."""
        return self._request("GET", "/v2/account")

    def get_positions(self) -> list[dict]:
        """Devuelve la lista de posiciones abiertas."""
        result = self._request("GET", "/v2/positions")
        return result or []

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        order_type: str = "market",
        limit_price: float | None = None,
        time_in_force: str = "day",
    ) -> dict:
        """
        Envía una orden.

        Args:
            symbol: ticker a operar.
            qty: cantidad de acciones/contratos.
            side: 'buy' o 'sell'.
            order_type: 'market', 'limit', 'stop', 'stop_limit'.
            limit_price: precio límite, obligatorio si order_type es 'limit' o 'stop_limit'.
            time_in_force: 'day', 'gtc', 'ioc', 'fok'...

        Returns:
            La orden creada, tal como la devuelve Alpaca.

        Raises:
            PaperTradeError: si los parámetros son inválidos o la llamada falla.
        """
        if qty <= 0:
            raise PaperTradeError("qty debe ser > 0.")
        if side not in ("buy", "sell"):
            raise PaperTradeError("side debe ser 'buy' o 'sell'.")
        if order_type in ("limit", "stop_limit") and limit_price is None:
            raise PaperTradeError(f"order_type='{order_type}' requiere limit_price.")

        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)

        return self._request("POST", "/v2/orders", json=payload)

    def cancel_all(self) -> int:
        """Cancela todas las órdenes abiertas. Devuelve cuántas se cancelaron."""
        open_orders = self._request("GET", "/v2/orders", params={"status": "open"}) or []
        self._request("DELETE", "/v2/orders")
        return len(open_orders)

    def close_all_positions(self) -> int:
        """Cierra todas las posiciones abiertas. Devuelve cuántas se cerraron."""
        positions = self.get_positions()
        self._request("DELETE", "/v2/positions")
        return len(positions)


@dataclass
class StrategyRunner:
    """
    Ejecuta una estrategia en paper trading con datos en vivo.

    `strategy` es un callable que recibe una `pd.Series` de precios
    recientes de un símbolo y devuelve una señal entera: 1 (largo),
    -1 (corto) o 0 (plano). `StrategyRunner` traduce esa señal en
    órdenes a través de `account`, comparando contra la posición
    actual para no duplicar órdenes en cada poll.
    """

    strategy: Callable[[Any], int]
    symbols: list[str]
    capital: float
    account: PaperAccount
    price_fetcher: Callable[[str], Any] | None = None
    poll_interval: int = 60
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise PaperTradeError("symbols no puede estar vacío.")
        if self.capital <= 0:
            raise PaperTradeError("capital debe ser > 0.")

    def _target_qty(self, signal: int, price: float) -> float:
        """Tamaño de posición: reparte el capital a partes iguales entre símbolos."""
        if price <= 0:
            raise PaperTradeError("price debe ser > 0 para calcular el tamaño de posición.")
        allocation = self.capital / len(self.symbols)
        return round((allocation / price) * abs(signal), 4)

    def run_once(self) -> list[dict]:
        """
        Ejecuta una pasada: para cada símbolo, calcula la señal, la
        compara contra la posición actual y envía la orden necesaria
        para alinear la posición con la señal.

        Returns:
            Lista de órdenes enviadas en esta pasada (puede estar vacía
            si ningún símbolo necesitaba ajuste).
        """
        if self.price_fetcher is None:
            raise PaperTradeError("price_fetcher no está configurado.")

        positions = {p["symbol"]: float(p["qty"]) for p in self.account.get_positions()}
        orders = []

        for symbol in self.symbols:
            prices = self.price_fetcher(symbol)
            if prices is None or len(prices) == 0:
                continue

            signal = self.strategy(prices)
            last_price = float(prices.iloc[-1])

            target_qty = self._target_qty(signal, last_price)
            target_signed = target_qty if signal >= 0 else -target_qty

            current_qty = positions.get(symbol, 0.0)
            delta = target_signed - current_qty

            if abs(delta) < 1e-6:
                continue

            side = "buy" if delta > 0 else "sell"
            order = self.account.submit_order(symbol=symbol, qty=abs(delta), side=side)
            orders.append(order)

        return orders

    def run_forever(self) -> None:
        """
        Bucle de ejecución continua: llama a `run_once` cada
        `poll_interval` segundos hasta que se llame a `stop()`.
        """
        self._running = True
        while self._running:
            self.run_once()
            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Detiene un `run_forever` en curso en la próxima iteración."""
        self._running = False
