"""Valuation at a price — the filed numbers, marked to today.

The provider's financial-metrics rows are snapshots at filing time: the
market cap, P/E, P/B and every other price-based multiple on the latest
row were struck at a price from weeks or months ago. A value investor
judging cheapness in June on a March price is not judging today's
opportunity. The per-share facts on the row (EPS, book value per share,
free cash flow per share) are what was filed and do not move with the
tape, so the multiples can be re-struck exactly at any price:

    P/E = price / EPS      P/B = price / BVPS      FCF yield = FCF/sh / price

Market cap needs a share count the row does not carry. It is implied from
the provider's own numbers — market_cap / (P/E x EPS), the price the row
was struck at — and the enterprise-value multiples move by the same
change in market cap. Where that implied price cannot be formed, those
fields are left as filed.

Dividends come from the same row: payout_ratio x EPS is the trailing
twelve-month dividend per share, which the backtest accrues on the book
because the provider's prices are unadjusted and it has no dividends feed.
"""

from __future__ import annotations

import math

from hedge_fund.data.models import FinancialMetrics

# A payout ratio above this is a data artifact, not a dividend policy.
_MAX_PAYOUT = 5.0


def implied_price(m: FinancialMetrics) -> float | None:
    """The price the row's multiples were struck at, from the row itself."""
    if m.price_to_earnings_ratio and m.earnings_per_share:
        price = m.price_to_earnings_ratio * m.earnings_per_share
        if price > 0:
            return price
    if m.price_to_book_ratio and m.book_value_per_share:
        price = m.price_to_book_ratio * m.book_value_per_share
        if price > 0:
            return price
    return None


def reprice(m: FinancialMetrics, price: float) -> FinancialMetrics:
    """A copy of *m* with its price-based valuation struck at *price*.

    Per-share multiples are exact. Market cap, enterprise value and the
    EV multiples scale by the implied share count; if no implied price
    exists they stay as filed. Everything non-valuation is untouched.
    """
    if price <= 0:
        raise ValueError(f"cannot reprice at {price}")
    update: dict[str, float | None] = {}

    eps = m.earnings_per_share
    if eps:
        update["price_to_earnings_ratio"] = price / eps
    bvps = m.book_value_per_share
    if bvps and bvps > 0:
        update["price_to_book_ratio"] = price / bvps
    fcfps = m.free_cash_flow_per_share
    if fcfps is not None:
        update["free_cash_flow_yield"] = fcfps / price

    struck = implied_price(m)
    if struck is not None:
        ratio = price / struck
        if m.market_cap is not None:
            new_cap = m.market_cap * ratio
            update["market_cap"] = new_cap
            if m.enterprise_value is not None:
                new_ev = m.enterprise_value + (new_cap - m.market_cap)
                update["enterprise_value"] = new_ev
                if m.enterprise_value:
                    ev_ratio = new_ev / m.enterprise_value
                    if m.enterprise_value_to_ebitda_ratio is not None:
                        update["enterprise_value_to_ebitda_ratio"] = (
                            m.enterprise_value_to_ebitda_ratio * ev_ratio)
                    if m.enterprise_value_to_revenue_ratio is not None:
                        update["enterprise_value_to_revenue_ratio"] = (
                            m.enterprise_value_to_revenue_ratio * ev_ratio)
        if m.price_to_sales_ratio is not None:
            update["price_to_sales_ratio"] = m.price_to_sales_ratio * ratio
        if (m.peg_ratio is not None and m.price_to_earnings_ratio
                and "price_to_earnings_ratio" in update):
            update["peg_ratio"] = m.peg_ratio * (
                update["price_to_earnings_ratio"] / m.price_to_earnings_ratio)

    return m.model_copy(update=update)


def dividend_per_share(m: FinancialMetrics) -> float | None:
    """Trailing-twelve-month dividend per share: payout_ratio x EPS.

    None when it cannot be formed. A loss-maker's payout ratio is
    meaningless and a payout above _MAX_PAYOUT is a data artifact; both
    yield None rather than a number.
    """
    if m.payout_ratio is None or m.earnings_per_share is None:
        return None
    if m.earnings_per_share <= 0 or not (0 <= m.payout_ratio <= _MAX_PAYOUT):
        return None
    return m.payout_ratio * m.earnings_per_share


def quantize_price(price: float, step: float) -> float:
    """Snap *price* to a geometric grid of *step* (0.10 = 10% rungs).

    Two prices on the same rung produce the same number, so a prompt built
    from it stays identical — and cached — until the price has moved
    enough to matter. step <= 0 returns the price unchanged.
    """
    if step <= 0 or price <= 0:
        return price
    rung = math.log(1 + step)
    return round(math.exp(round(math.log(price) / rung) * rung), 4)
