"""FundamentalsSnapshot tests — mocked data client, no network."""

import pytest
from datetime import date

from hedge_fund.data.models import CompanyFacts, FinancialMetrics
from hedge_fund.features.snapshot import InsufficientData, build_snapshot


class MockDataClient:
    """Returns canned metrics; records what it was asked for."""

    def __init__(self, metrics=None, facts=None, close=None):
        self._metrics = metrics or []
        self._facts = facts
        self._close = close
        self.metrics_calls = []

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        from hedge_fund.data.models import Price
        if self._close is None:
            return []
        return [Price(open=self._close, close=self._close, high=self._close,
                      low=self._close, volume=1000, time=f"{end_date}T00:00:00Z")]

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        self.metrics_calls.append(
            {"ticker": ticker, "end_date": end_date, "period": period, "limit": limit}
        )
        return self._metrics

    def get_company_facts(self, ticker):
        return self._facts


def _metric(report_period, **kwargs):
    defaults = {
        "ticker": "TEST",
        "period": "ttm",
        "filing_date": report_period,  # simplification for tests
        "return_on_equity": 0.20,
        "net_margin": 0.25,
        "gross_margin": 0.40,
        "book_value_per_share": 10.0,
        "debt_to_equity": 0.5,
        "market_cap": 1e9,
    }
    defaults.update(kwargs)
    return FinancialMetrics(report_period=report_period, **defaults)


def _history(n=8):
    """n periods, newest first, quarter-spaced."""
    quarters = ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
                "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"]
    return [_metric(q) for q in quarters[:n]]


def test_as_of_passes_through_to_data_client():
    client = MockDataClient(metrics=_history())
    build_snapshot("TEST", "2025-01-15", client)
    call = client.metrics_calls[0]
    assert call["end_date"] == "2025-01-15"
    assert call["ticker"] == "TEST"


def test_insufficient_data_raises():
    client = MockDataClient(metrics=_history(3))  # below MIN_PERIODS
    with pytest.raises(InsufficientData):
        build_snapshot("TEST", "2025-01-15", client)


def test_aggregates():
    metrics = _history(4)
    # oldest gross margin 0.30, newest 0.40 -> trend +0.10
    metrics[-1] = _metric("2024-03-31", gross_margin=0.30)
    # BVPS oldest 8.0 -> newest 10.0 over the span of the report periods
    metrics[-1].book_value_per_share = 8.0
    span_years = (date.fromisoformat(metrics[0].report_period)
                  - date.fromisoformat(metrics[-1].report_period)).days / 365.25
    client = MockDataClient(metrics=metrics)

    snap = build_snapshot("TEST", "2025-01-15", client)

    assert snap.roe_avg == pytest.approx(0.20)
    assert snap.gross_margin_trend == pytest.approx(0.10)
    assert snap.debt_to_equity_latest == pytest.approx(0.5)
    assert snap.market_cap_latest == pytest.approx(1e9)
    assert snap.bvps_cagr == pytest.approx((10.0 / 8.0) ** (1 / span_years) - 1, abs=1e-4)


def test_market_cap_comes_from_pit_metrics_not_facts():
    """company_facts market cap is latest-only (lookahead); the snapshot must
    use the most recent FILED metrics row instead."""
    facts = CompanyFacts(ticker="TEST", sector="Tech")
    client = MockDataClient(metrics=_history(), facts=facts)

    snap = build_snapshot("TEST", "2020-06-30", client)

    assert snap.market_cap_latest == pytest.approx(1e9)  # from metrics row
    assert snap.sector == "Tech"  # facts used only for slow-moving attributes


def test_content_hash_stable_and_sensitive():
    client_a = MockDataClient(metrics=_history())
    client_b = MockDataClient(metrics=_history())
    snap_a = build_snapshot("TEST", "2025-01-15", client_a)
    snap_b = build_snapshot("TEST", "2025-01-15", client_b)
    assert snap_a.content_hash == snap_b.content_hash  # same data -> same key

    changed = _history()
    changed[0] = _metric("2024-12-31", return_on_equity=0.35)
    snap_c = build_snapshot("TEST", "2025-01-15", MockDataClient(metrics=changed))
    assert snap_c.content_hash != snap_a.content_hash  # new filing -> new key


def test_same_data_different_as_of_same_render_and_hash():
    """Between filings the snapshot is unchanged — the hash and the rendered
    prompt must be identical on any as-of date, or the LLM cache never hits."""
    snap_jan = build_snapshot("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    snap_feb = build_snapshot("TEST", "2025-02-15", MockDataClient(metrics=_history()))

    assert snap_jan.as_of != snap_feb.as_of  # the field itself still differs
    assert snap_jan.content_hash == snap_feb.content_hash
    assert snap_jan.render() == snap_feb.render()


def test_render_contains_the_facts():
    snap = build_snapshot("TEST", "2025-01-15", MockDataClient(metrics=_history()))
    text = snap.render()
    assert "2025-01-15" not in text  # as_of must never leak into the prompt
    assert "2024-12-31" in text
    assert "publicly filed" in text


# ---------------------------------------------------------------------------
# Valuation at the current price
# ---------------------------------------------------------------------------

def _priced_history(n=8):
    """Latest row struck at 100 (P/E 20 x EPS 5); older rows as filed."""
    rows = _history(n)
    rows[0] = _metric(rows[0].report_period, earnings_per_share=5.0,
                      price_to_earnings_ratio=20.0, book_value_per_share=20.0,
                      price_to_book_ratio=5.0, free_cash_flow_per_share=4.0,
                      market_cap=1_000e6)
    return rows


def test_no_price_leaves_the_filed_valuation():
    snap = build_snapshot("TEST", "2025-01-15", MockDataClient(_priced_history()))
    assert snap.price is None
    assert snap.market_cap_latest == 1_000e6
    assert "Market cap (latest filed)" in snap.render()


def test_price_restrikes_the_summary_valuation_only():
    snap = build_snapshot("TEST", "2025-01-15", MockDataClient(_priced_history(), close=150.0))
    assert snap.price == pytest.approx(150.0, rel=0.05)             # on a 10% rung near 150
    assert snap.market_cap_latest == pytest.approx(1_000e6 * snap.price / 100, rel=1e-6)
    assert snap.pe_at_price == pytest.approx(snap.price / 5.0)
    assert snap.pb_at_price == pytest.approx(snap.price / 20.0)
    assert snap.fcf_yield_at_price == pytest.approx(4.0 / snap.price)
    # History rows stay as struck at filing — the row still says P/E 20.
    assert snap.periods[0].price_to_earnings_ratio == 20.0
    text = snap.render()
    assert "Price: ~" in text and "Market cap at that price" in text
    assert "as struck at each filing" in text


def test_prices_on_the_same_rung_share_a_prompt_and_hash():
    a = build_snapshot("TEST", "2025-01-15", MockDataClient(_priced_history(), close=150.0))
    b = build_snapshot("TEST", "2025-01-22", MockDataClient(_priced_history(), close=153.0))
    c = build_snapshot("TEST", "2025-01-29", MockDataClient(_priced_history(), close=175.0))
    assert a.content_hash == b.content_hash and a.render() == b.render()
    assert c.content_hash != a.content_hash


def test_price_step_zero_uses_the_exact_close():
    snap = build_snapshot("TEST", "2025-01-15", MockDataClient(_priced_history(), close=153.0),
                          price_step=0.0)
    assert snap.price == 153.0
