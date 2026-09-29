"""AlpacaBroker tests — against FakeAlpaca, an in-memory stand-in for the API."""

import pytest

from hedge_fund.brokers.alpaca import AlpacaBroker, AlpacaError
from hedge_fund.brokers.fake_alpaca import FakeAlpaca
from hedge_fund.brokers.models import Order


def _broker(fake, **kwargs):
    return AlpacaBroker("key", "secret", session=fake, sleep=lambda s: None, **kwargs)


def test_sends_the_key_headers_to_the_paper_endpoint():
    fake = FakeAlpaca()
    broker = _broker(fake)
    assert fake.headers == {"APCA-API-KEY-ID": "key", "APCA-API-SECRET-KEY": "secret"}
    assert broker._base == "https://paper-api.alpaca.markets"


def test_missing_keys_fail_with_the_env_var_names(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaError, match="APCA_API_KEY_ID"):
        AlpacaBroker.from_env(session=FakeAlpaca())


def test_reads_signed_positions_and_cash():
    fake = FakeAlpaca(cash=5_000.0, positions={"AAPL": 10, "MSFT": -3})
    broker = _broker(fake)
    assert {t: p.shares for t, p in broker.positions().items()} == {"AAPL": 10, "MSFT": -3}
    assert broker.cash() == 5_000.0


def test_market_date_is_the_exchange_date_when_open():
    assert _broker(FakeAlpaca(date="2026-09-29")).market_date() == "2026-09-29"


def test_market_date_refuses_a_closed_market_or_a_blocked_account():
    with pytest.raises(AlpacaError, match="market is closed"):
        _broker(FakeAlpaca(is_open=False)).market_date()
    with pytest.raises(AlpacaError, match="cannot trade"):
        _broker(FakeAlpaca(status="ACCOUNT_CLOSED")).market_date()


def test_buy_fills_at_the_venue_price_with_signed_slippage():
    fake = FakeAlpaca(prices={"AAPL": 199.0})
    fill = _broker(fake).place_order(Order(ticker="AAPL", side="buy", quantity=10, price=200.0))
    assert (fill.quantity, fill.price, fill.commission) == (10, 199.0, 0.0)
    assert fill.slippage == pytest.approx(-10.0)  # beat the reference close by $1 a share
    assert fake.positions == {"AAPL": 10}
    assert fake.cash == pytest.approx(100_000.0 - 1_990.0)
    assert fake.submitted[0] | {"client_order_id": None} == {
        "symbol": "AAPL", "qty": "10", "side": "buy", "type": "market",
        "time_in_force": "day", "client_order_id": None}


def test_a_flip_through_zero_goes_out_as_close_then_open():
    fake = FakeAlpaca(positions={"AAPL": 10}, prices={"AAPL": 100.0})
    fill = _broker(fake).place_order(Order(ticker="AAPL", side="sell", quantity=15, price=100.0))
    assert [o["qty"] for o in fake.submitted] == ["10", "5"]
    assert fill.quantity == 15
    assert fake.positions == {"AAPL": -5}


def test_a_rejected_order_raises_with_the_venues_reason():
    fake = FakeAlpaca(prices={"XYZ": 1.0}, rejected={"XYZ"})
    with pytest.raises(AlpacaError, match="not tradable"):
        _broker(fake).place_order(Order(ticker="XYZ", side="buy", quantity=1, price=1.0))


def test_an_order_that_never_fills_is_canceled_and_raises():
    fake = FakeAlpaca(prices={"AAPL": 100.0}, stuck={"AAPL"})
    with pytest.raises(AlpacaError, match="did not fill"):
        _broker(fake, fill_timeout=2.0).place_order(
            Order(ticker="AAPL", side="buy", quantity=1, price=100.0))
    assert fake.canceled == ["o1"]
    assert fake.positions == {}


def test_credit_is_a_no_op_since_real_dividends_land_on_their_own():
    fake = FakeAlpaca(cash=1_000.0)
    broker = _broker(fake)
    broker.credit(50.0, "dividend accrual")
    assert broker.cash() == 1_000.0
    assert AlpacaBroker.books_dividends is True
