"""
tests/test_papertrade.py

Tests unitarios de app/core/papertrade.py. `requests` está mockeado
en PaperAccount; StrategyRunner se testea con un PaperAccount de
mentira (MagicMock) para no depender de la red en ningún caso.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.core.papertrade import PaperAccount, PaperTradeError, StrategyRunner


def _fake_response(json_data=None, content=b"{}"):
    resp = MagicMock()
    resp.json.return_value = json_data if json_data is not None else {}
    resp.content = content
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# PaperAccount
# ---------------------------------------------------------------------------

class TestPaperAccountInit:
    def test_requires_api_key_and_secret(self):
        with pytest.raises(PaperTradeError, match="obligatorios"):
            PaperAccount(api_key="", api_secret="secret")

    def test_strips_trailing_slash_from_base_url(self):
        account = PaperAccount(api_key="k", api_secret="s", base_url="https://example.com/")
        assert account.base_url == "https://example.com"


class TestGetAccount:
    def test_returns_account_payload(self):
        payload = {"equity": "10000.00", "cash": "5000.00"}
        with patch("requests.request", return_value=_fake_response(payload)) as mocked:
            account = PaperAccount(api_key="k", api_secret="s")
            result = account.get_account()

        assert result == payload
        assert mocked.call_args.kwargs["headers"]["APCA-API-KEY-ID"] == "k"

    def test_network_error_raises(self):
        import requests

        with patch("requests.request", side_effect=requests.ConnectionError("sin red")):
            account = PaperAccount(api_key="k", api_secret="s")
            with pytest.raises(PaperTradeError, match="Fallo al llamar a Alpaca"):
                account.get_account()


class TestGetPositions:
    def test_returns_list(self):
        payload = [{"symbol": "AAPL", "qty": "10"}]
        with patch("requests.request", return_value=_fake_response(payload)):
            account = PaperAccount(api_key="k", api_secret="s")
            result = account.get_positions()

        assert result == payload

    def test_empty_response_returns_empty_list(self):
        with patch("requests.request", return_value=_fake_response(None, content=b"")):
            account = PaperAccount(api_key="k", api_secret="s")
            result = account.get_positions()

        assert result == []


class TestSubmitOrder:
    def test_valid_market_order(self):
        payload = {"id": "abc123", "symbol": "AAPL", "qty": "10", "side": "buy"}
        with patch("requests.request", return_value=_fake_response(payload)) as mocked:
            account = PaperAccount(api_key="k", api_secret="s")
            result = account.submit_order(symbol="AAPL", qty=10, side="buy")

        assert result == payload
        sent_json = mocked.call_args.kwargs["json"]
        assert sent_json["symbol"] == "AAPL"
        assert sent_json["side"] == "buy"

    def test_negative_qty_raises(self):
        account = PaperAccount(api_key="k", api_secret="s")
        with pytest.raises(PaperTradeError, match="qty debe ser > 0"):
            account.submit_order(symbol="AAPL", qty=-5, side="buy")

    def test_invalid_side_raises(self):
        account = PaperAccount(api_key="k", api_secret="s")
        with pytest.raises(PaperTradeError, match="side debe ser"):
            account.submit_order(symbol="AAPL", qty=5, side="hold")

    def test_limit_order_without_price_raises(self):
        account = PaperAccount(api_key="k", api_secret="s")
        with pytest.raises(PaperTradeError, match="limit_price"):
            account.submit_order(symbol="AAPL", qty=5, side="buy", order_type="limit")

    def test_limit_order_with_price_succeeds(self):
        payload = {"id": "abc123"}
        with patch("requests.request", return_value=_fake_response(payload)) as mocked:
            account = PaperAccount(api_key="k", api_secret="s")
            account.submit_order(symbol="AAPL", qty=5, side="buy", order_type="limit", limit_price=150.0)

        sent_json = mocked.call_args.kwargs["json"]
        assert sent_json["limit_price"] == "150.0"


class TestCancelAllAndClosePositions:
    def test_cancel_all_returns_count(self):
        open_orders = [{"id": "1"}, {"id": "2"}]
        responses = [_fake_response(open_orders), _fake_response(None, content=b"")]

        with patch("requests.request", side_effect=responses):
            account = PaperAccount(api_key="k", api_secret="s")
            count = account.cancel_all()

        assert count == 2

    def test_close_all_positions_returns_count(self):
        positions = [{"symbol": "AAPL"}]
        responses = [_fake_response(positions), _fake_response(None, content=b"")]

        with patch("requests.request", side_effect=responses):
            account = PaperAccount(api_key="k", api_secret="s")
            count = account.close_all_positions()

        assert count == 1


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------

def _always_long(prices: pd.Series) -> int:
    return 1


def _always_flat(prices: pd.Series) -> int:
    return 0


class TestStrategyRunnerInit:
    def test_empty_symbols_raises(self):
        account = MagicMock()
        with pytest.raises(PaperTradeError, match="symbols"):
            StrategyRunner(strategy=_always_long, symbols=[], capital=10_000, account=account)

    def test_non_positive_capital_raises(self):
        account = MagicMock()
        with pytest.raises(PaperTradeError, match="capital"):
            StrategyRunner(strategy=_always_long, symbols=["AAPL"], capital=0, account=account)


class TestRunOnce:
    def test_opens_position_when_flat_and_signal_is_long(self):
        account = MagicMock()
        account.get_positions.return_value = []
        account.submit_order.return_value = {"id": "order-1"}

        prices = pd.Series([100.0, 101.0, 102.0])
        runner = StrategyRunner(
            strategy=_always_long,
            symbols=["AAPL"],
            capital=10_000,
            account=account,
            price_fetcher=lambda symbol: prices,
        )

        orders = runner.run_once()

        assert len(orders) == 1
        _, kwargs = account.submit_order.call_args
        assert kwargs["side"] == "buy"
        assert kwargs["symbol"] == "AAPL"

    def test_no_order_when_already_aligned(self):
        account = MagicMock()
        # 10_000 / 1 símbolo / 100 precio = 100 acciones ya en cartera.
        account.get_positions.return_value = [{"symbol": "AAPL", "qty": "100.0"}]

        prices = pd.Series([100.0] * 5)
        runner = StrategyRunner(
            strategy=_always_long,
            symbols=["AAPL"],
            capital=10_000,
            account=account,
            price_fetcher=lambda symbol: prices,
        )

        orders = runner.run_once()

        assert orders == []
        account.submit_order.assert_not_called()

    def test_closes_position_when_signal_turns_flat(self):
        account = MagicMock()
        account.get_positions.return_value = [{"symbol": "AAPL", "qty": "50.0"}]
        account.submit_order.return_value = {"id": "order-2"}

        prices = pd.Series([100.0] * 5)
        runner = StrategyRunner(
            strategy=_always_flat,
            symbols=["AAPL"],
            capital=10_000,
            account=account,
            price_fetcher=lambda symbol: prices,
        )

        orders = runner.run_once()

        assert len(orders) == 1
        _, kwargs = account.submit_order.call_args
        assert kwargs["side"] == "sell"

    def test_missing_price_fetcher_raises(self):
        account = MagicMock()
        runner = StrategyRunner(strategy=_always_long, symbols=["AAPL"], capital=10_000, account=account)

        with pytest.raises(PaperTradeError, match="price_fetcher"):
            runner.run_once()

    def test_skips_symbol_with_no_price_data(self):
        account = MagicMock()
        account.get_positions.return_value = []

        runner = StrategyRunner(
            strategy=_always_long,
            symbols=["AAPL"],
            capital=10_000,
            account=account,
            price_fetcher=lambda symbol: pd.Series(dtype=float),
        )

        orders = runner.run_once()
        assert orders == []


class TestStop:
    def test_stop_flips_running_flag(self):
        account = MagicMock()
        runner = StrategyRunner(strategy=_always_long, symbols=["AAPL"], capital=10_000, account=account)

        runner._running = True
        runner.stop()

        assert runner._running is False
