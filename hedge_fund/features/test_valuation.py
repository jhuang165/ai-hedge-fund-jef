"""Valuation-at-a-price tests — pure arithmetic over one metrics row."""

import pytest

from hedge_fund.data.models import FinancialMetrics
from hedge_fund.features.valuation import (
    dividend_per_share,
    implied_price,
    quantize_price,
    reprice,
)


def _row(**kw):
    base = dict(ticker="T", report_period="2024-12-31", period="ttm",
                earnings_per_share=5.0, book_value_per_share=20.0,
                free_cash_flow_per_share=4.0,
                price_to_earnings_ratio=20.0, price_to_book_ratio=5.0,
                price_to_sales_ratio=3.0, free_cash_flow_yield=0.04,
                market_cap=1_000e6, enterprise_value=1_200e6,
                enterprise_value_to_ebitda_ratio=12.0,
                enterprise_value_to_revenue_ratio=4.0, peg_ratio=2.0,
                return_on_equity=0.25, payout_ratio=0.4)
    base.update(kw)
    return FinancialMetrics(**base)


def test_implied_price_from_pe_then_pb():
    assert implied_price(_row()) == pytest.approx(100.0)           # 20 x 5
    assert implied_price(_row(price_to_earnings_ratio=None)) == pytest.approx(100.0)  # 5 x 20
    assert implied_price(_row(price_to_earnings_ratio=None, price_to_book_ratio=None)) is None
    assert implied_price(_row(earnings_per_share=-5.0, price_to_book_ratio=None)) is None


def test_reprice_restrikes_every_price_based_field():
    r = reprice(_row(), 150.0)  # struck at 100, now 150
    assert r.price_to_earnings_ratio == pytest.approx(30.0)
    assert r.price_to_book_ratio == pytest.approx(7.5)
    assert r.free_cash_flow_yield == pytest.approx(4.0 / 150)
    assert r.market_cap == pytest.approx(1_500e6)
    assert r.enterprise_value == pytest.approx(1_700e6)             # +500m of equity value
    assert r.enterprise_value_to_ebitda_ratio == pytest.approx(12.0 * 1_700 / 1_200)
    assert r.enterprise_value_to_revenue_ratio == pytest.approx(4.0 * 1_700 / 1_200)
    assert r.price_to_sales_ratio == pytest.approx(4.5)
    assert r.peg_ratio == pytest.approx(3.0)
    # Non-valuation facts untouched; the original row untouched.
    assert r.return_on_equity == 0.25 and r.earnings_per_share == 5.0
    assert _row().price_to_earnings_ratio == 20.0


def test_reprice_at_the_struck_price_is_identity():
    r = reprice(_row(), 100.0)
    for f in ("price_to_earnings_ratio", "price_to_book_ratio", "market_cap",
              "enterprise_value", "peg_ratio", "free_cash_flow_yield"):
        assert getattr(r, f) == pytest.approx(getattr(_row(), f))


def test_reprice_without_implied_price_keeps_cap_as_filed():
    row = _row(price_to_earnings_ratio=None, price_to_book_ratio=None)
    r = reprice(row, 150.0)
    assert r.price_to_earnings_ratio == pytest.approx(30.0)   # exact from EPS
    assert r.market_cap == row.market_cap                     # cannot be moved honestly
    assert r.enterprise_value == row.enterprise_value


def test_reprice_negative_eps_gives_negative_pe_and_no_pb_for_negative_book():
    r = reprice(_row(earnings_per_share=-2.0, book_value_per_share=-1.0), 50.0)
    assert r.price_to_earnings_ratio == pytest.approx(-25.0)
    assert r.price_to_book_ratio == 5.0                       # left as filed


def test_reprice_rejects_nonpositive_price():
    with pytest.raises(ValueError):
        reprice(_row(), 0.0)


def test_dividend_per_share():
    assert dividend_per_share(_row()) == pytest.approx(2.0)   # 0.4 x 5
    assert dividend_per_share(_row(payout_ratio=None)) is None
    assert dividend_per_share(_row(earnings_per_share=-1.0)) is None
    assert dividend_per_share(_row(payout_ratio=9.0)) is None
    assert dividend_per_share(_row(payout_ratio=0.0)) == 0.0


def test_quantize_price_snaps_to_ten_percent_rungs():
    prices = [p / 4 for p in range(200, 2000)]                 # 50 .. 500 in quarters
    snapped = [quantize_price(p, 0.10) for p in prices]
    # Never more than half a rung (5%) from the true price.
    assert all(abs(q / p - 1) <= 0.05 + 1e-9 for p, q in zip(prices, snapped))
    # Consecutive rungs are 10% apart.
    rungs = sorted(set(snapped))
    assert all(b / a == pytest.approx(1.10, rel=1e-6) for a, b in zip(rungs, rungs[1:]))
    # Prices more than a rung apart never share one.
    assert quantize_price(100.0, 0.10) != quantize_price(111.0, 0.10)
    # Disabled: the price itself.
    assert quantize_price(103.0, 0.0) == 103.0
