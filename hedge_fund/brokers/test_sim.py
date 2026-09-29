"""SimBroker tests — deterministic fills and bookkeeping."""

import pytest

from hedge_fund.brokers.models import Order
from hedge_fund.brokers.sim import SimBroker


def test_buy_updates_cash_and_position():
    broker = SimBroker(cash=10_000.0)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert broker.cash() == pytest.approx(9_000.0)
    assert broker.positions()["AAPL"].shares == 10
    assert fill.quantity == 10
    assert fill.price == 100.0


def test_sell_updates_cash_and_position():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=4, price=110.0))
    assert broker.positions()["AAPL"].shares == 6
    assert broker.cash() == pytest.approx(-1_000.0 + 440.0)


def test_position_removed_at_zero():
    broker = SimBroker(cash=1_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=5, price=100.0))
    assert broker.positions() == {}


def test_sell_past_zero_creates_short():
    broker = SimBroker(cash=0.0)
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=3, price=100.0))
    assert broker.positions()["AAPL"].shares == -3
    assert broker.cash() == pytest.approx(300.0)


def test_nonpositive_price_raises():
    broker = SimBroker(cash=1_000.0)
    with pytest.raises(ValueError):
        broker.place_order(Order(ticker="AAPL", side="buy", quantity=1, price=0.0))


def test_positions_returns_a_copy():
    broker = SimBroker(cash=1_000.0)
    broker.place_order(Order(ticker="AAPL", side="buy", quantity=5, price=100.0))
    broker.positions().clear()
    assert broker.positions()["AAPL"].shares == 5


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def test_default_broker_is_frictionless():
    fill = SimBroker(cash=10_000.0).place_order(
        Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    assert fill.price == 100.0
    assert fill.commission == 0.0 and fill.slippage == 0.0 and fill.cost == 0.0


def test_buy_pays_slippage_and_commission():
    broker = SimBroker(cash=10_000.0, commission_bps=10.0, slippage_bps=20.0)
    fill = broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    # 20 bps against the buyer: fill at 100.20; commission 10 bps on 1,002.
    assert fill.price == pytest.approx(100.20)
    assert fill.slippage == pytest.approx(2.0)
    assert fill.commission == pytest.approx(1.002)
    assert fill.cost == pytest.approx(3.002)
    assert broker.cash() == pytest.approx(10_000.0 - 1_002.0 - 1.002)
    assert broker.positions()["AAPL"].shares == 10


def test_sell_receives_less_and_pays_commission():
    broker = SimBroker(cash=0.0, commission_bps=10.0, slippage_bps=20.0)
    fill = broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    assert fill.price == pytest.approx(99.80)
    assert fill.slippage == pytest.approx(2.0)
    assert fill.commission == pytest.approx(0.998)
    assert broker.cash() == pytest.approx(998.0 - 0.998)
    assert broker.positions()["AAPL"].shares == -10


def test_round_trip_loses_exactly_the_costs():
    broker = SimBroker(cash=10_000.0, commission_bps=5.0, slippage_bps=5.0)
    buy = broker.place_order(Order(ticker="AAPL", side="buy", quantity=10, price=100.0))
    sell = broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    assert broker.positions() == {}
    assert broker.cash() == pytest.approx(10_000.0 - buy.cost - sell.cost)


def test_negative_costs_rejected():
    with pytest.raises(ValueError):
        SimBroker(cash=1.0, slippage_bps=-1.0)


def test_opens_on_a_carried_book():
    broker = SimBroker(cash=500.0, positions={"AAPL": 10, "MSFT": -3, "NVDA": 0})
    assert {t: p.shares for t, p in broker.positions().items()} == {"AAPL": 10, "MSFT": -3}
    broker.place_order(Order(ticker="AAPL", side="sell", quantity=10, price=100.0))
    assert "AAPL" not in broker.positions()
    assert broker.cash() == pytest.approx(1_500.0)
