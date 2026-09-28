"""build_orders tests — pure diffing math."""

from hedge_fund.brokers.models import Position
from hedge_fund.pipeline.execution import build_orders


def _positions(**shares):
    return {t: Position(ticker=t, shares=s) for t, s in shares.items()}


def test_floor_sizing_never_overshoots():
    orders = build_orders({"AAPL": 0.25}, {}, {"AAPL": 300.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].quantity == 8  # 2500 / 300 = 8.33 -> 8


def test_delta_against_existing_position():
    orders = build_orders(
        {"AAPL": 0.25}, _positions(AAPL=5), {"AAPL": 250.0}, equity=10_000.0,
    )
    assert len(orders) == 1
    assert orders[0].side == "buy"
    assert orders[0].quantity == 5  # target 10, held 5


def test_held_name_missing_from_targets_is_closed():
    orders = build_orders({}, _positions(AAPL=7), {"AAPL": 100.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == 7


def test_subshare_delta_emits_nothing():
    orders = build_orders({"AAPL": 0.005}, {}, {"AAPL": 100.0}, equity=10_000.0)
    assert orders == []  # 50 dollars / 100 = 0.5 shares -> floor 0


def test_sells_before_buys_alphabetical():
    orders = build_orders(
        {"AAPL": 0.2, "MSFT": 0.0, "NVDA": 0.2, "AMZN": 0.0},
        _positions(MSFT=10, NVDA=1, AMZN=5),
        {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0, "AMZN": 100.0},
        equity=10_000.0,
    )
    assert [(o.ticker, o.side) for o in orders] == [
        ("AMZN", "sell"), ("MSFT", "sell"),
        ("AAPL", "buy"), ("NVDA", "buy"),
    ]


def test_short_target_sells_past_zero():
    orders = build_orders({"AAPL": -0.2}, {}, {"AAPL": 100.0}, equity=10_000.0)
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == 20


# ---------------------------------------------------------------------------
# No-trade band
# ---------------------------------------------------------------------------

def test_band_skips_small_rebalance():
    # Held 100, target 125: a 250-dollar top-up on 10k equity is 2.5%, under a 5% band.
    orders = build_orders(
        {"AAPL": 0.125}, _positions(AAPL=100), {"AAPL": 10.0}, equity=10_000.0,
        min_trade_pct=0.05,
    )
    assert orders == []


def test_band_lets_large_rebalance_through():
    orders = build_orders(
        {"AAPL": 0.20}, _positions(AAPL=100), {"AAPL": 10.0}, equity=10_000.0,
        min_trade_pct=0.05,
    )
    assert len(orders) == 1 and orders[0].quantity == 100


def test_band_never_blocks_an_exit():
    # Two dust positions: one closed by an explicit zero target, one by absence.
    orders = build_orders(
        {"AAPL": 0.0}, _positions(AAPL=1, MSFT=1), {"AAPL": 10.0, "MSFT": 10.0},
        equity=10_000.0, min_trade_pct=0.05,
    )
    assert [(o.ticker, o.side, o.quantity) for o in orders] == [
        ("AAPL", "sell", 1), ("MSFT", "sell", 1),
    ]


def test_band_does_not_open_a_position_below_it():
    orders = build_orders({"AAPL": 0.01}, {}, {"AAPL": 10.0}, equity=10_000.0,
                          min_trade_pct=0.05)
    assert orders == []


def test_band_applies_to_shorts_too():
    # Short 100, target short 125 -> skipped; target flat -> covers in full.
    assert build_orders({"AAPL": -0.125}, _positions(AAPL=-100), {"AAPL": 10.0},
                        equity=10_000.0, min_trade_pct=0.05) == []
    cover = build_orders({}, _positions(AAPL=-100), {"AAPL": 10.0},
                         equity=10_000.0, min_trade_pct=0.05)
    assert [(o.side, o.quantity) for o in cover] == [("buy", 100)]


def test_zero_band_is_the_old_behavior():
    orders = build_orders({"AAPL": 0.125}, _positions(AAPL=100), {"AAPL": 10.0},
                          equity=10_000.0, min_trade_pct=0.0)
    assert len(orders) == 1 and orders[0].quantity == 25


def test_negative_band_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_orders({}, {}, {}, equity=1.0, min_trade_pct=-0.1)
