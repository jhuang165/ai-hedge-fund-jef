"""Scorecard tests — grading saved calls against a canned tape."""

import pytest

from hedge_fund.data.models import Price
from hedge_fund.research.models import ResearchReport
from hedge_fund.research.scorecard import grade


class Tape:
    """Closes per ticker per date; the last close on or before a date wins."""

    def __init__(self, series):
        self._series = series

    def get_prices(self, ticker, start_date, end_date, **kw):
        days = self._series.get(ticker, {})
        return [Price(open=c, close=c, high=c, low=c, volume=1, time=f"{d}T00:00:00Z")
                for d, c in sorted(days.items()) if start_date <= d <= end_date]


def _report(ticker, as_of, signal, confidence=70, action=None):
    action = action or {"bullish": "buy", "neutral": "watch", "bearish": "avoid"}[signal]
    return ResearchReport(ticker=ticker, as_of=as_of, model="m", signal=signal,
                          confidence=confidence, action=action, action_rationale="r",
                          thesis="t", prompt_key="k")


# 2025-01-02 -> 2025-04-02 (90 days). SPY +10%. AAPL +20% (beats), XYZ -5% (lags),
# MSFT +10% (matches: zero excess).
TAPE = Tape({
    "SPY":  {"2025-01-02": 100.0, "2025-04-02": 110.0, "2025-07-01": 120.0},
    "AAPL": {"2025-01-02": 100.0, "2025-04-02": 120.0},
    "XYZ":  {"2025-01-02": 50.0, "2025-04-02": 47.5},
    "MSFT": {"2025-01-02": 200.0, "2025-04-02": 220.0},
    "NEW":  {"2025-04-01": 10.0},
})


def test_graded_calls_are_hits_or_misses_on_excess_return():
    reports = [
        _report("AAPL", "2025-01-02", "bullish", 80),   # +20% vs +10%: hit
        _report("XYZ", "2025-01-02", "bearish", 90),    # -5% vs +10%: hit
        _report("MSFT", "2025-01-02", "bullish", 60),   # matches the market: miss
    ]
    card = grade(reports, TAPE, horizon_days=90, benchmark="SPY", as_of="2025-06-01")

    assert (card.n_reports, card.n_graded, card.n_pending, card.n_skipped) == (3, 3, 0, 0)
    by = {c.ticker: c for c in card.calls}
    assert by["AAPL"].excess_pct == pytest.approx(0.10) and by["AAPL"].hit is True
    assert by["XYZ"].excess_pct == pytest.approx(-0.15) and by["XYZ"].hit is True
    assert by["MSFT"].excess_pct == pytest.approx(0.0) and by["MSFT"].hit is False
    assert by["AAPL"].matured_on == "2025-04-02"
    assert card.hit_rate == pytest.approx(2 / 3, abs=1e-4)
    # Bull avg excess (0.10 + 0.0) / 2 = 0.05; bear -0.15 -> spread 0.20.
    assert card.bull_bear_spread_pct == pytest.approx(0.20)
    # Scores 80, -90, 60 vs excess .10, -.15, 0: perfectly rank-ordered.
    assert card.rank_ic == pytest.approx(1.0)
    actions = {g.key: g for g in card.by_action}
    assert actions["buy"].n == 2 and actions["buy"].hit_rate == pytest.approx(0.5)
    assert actions["avoid"].avg_excess_pct == pytest.approx(-0.15)


def test_neutral_calls_count_but_are_not_hits_or_misses():
    card = grade([_report("MSFT", "2025-01-02", "neutral", 50)], TAPE,
                 horizon_days=90, as_of="2025-06-01")
    (call,) = card.calls
    assert call.hit is None and call.direction == 0
    assert card.hit_rate is None
    assert card.bull_bear_spread_pct is None
    assert card.by_signal[0].hit_rate is None


def test_immature_calls_are_pending_not_wrong():
    card = grade([_report("AAPL", "2025-01-02", "bullish")], TAPE,
                 horizon_days=90, as_of="2025-03-01")
    assert card.n_graded == 0 and card.n_pending == 1
    assert card.pending[0].matures_on == "2025-04-02"


def test_missing_prices_are_skipped_with_a_reason():
    reports = [_report("NEW", "2025-01-02", "bullish"),      # no close on the report date
               _report("AAPL", "2025-01-02", "bullish")]
    card = grade(reports, Tape({**TAPE._series, "SPY": {}}), horizon_days=90, as_of="2025-06-01")
    reasons = {s.ticker: s.reason for s in card.skipped}
    assert "no close for NEW on 2025-01-02" in reasons["NEW"]
    assert "no SPY close" in reasons["AAPL"]
    assert card.n_graded == 0


def test_same_ticker_same_day_is_one_call_last_wins():
    reports = [_report("AAPL", "2025-01-02", "bearish"), _report("AAPL", "2025-01-02", "bullish")]
    card = grade(reports, TAPE, horizon_days=90, as_of="2025-06-01")
    assert card.n_reports == 1 and card.calls[0].signal == "bullish"


def test_rank_ic_needs_three_calls_and_variation():
    two = [_report("AAPL", "2025-01-02", "bullish"), _report("XYZ", "2025-01-02", "bearish")]
    assert grade(two, TAPE, horizon_days=90, as_of="2025-06-01").rank_ic is None
    same = [_report(t, "2025-01-02", "bullish", 70) for t in ("AAPL", "XYZ", "MSFT")]
    assert grade(same, TAPE, horizon_days=90, as_of="2025-06-01").rank_ic is None


def test_bad_horizon_rejected():
    with pytest.raises(ValueError):
        grade([], TAPE, horizon_days=0)
