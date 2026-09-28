"""backtest_fund tests — fake data client + fake analysts, real broker + pipeline."""

import pytest

from hedge_fund.backtesting.fund import backtest_fund, rebalance_grid
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.models import Signal


# ---------------------------------------------------------------------------
# Fakes (date-aware variants of the run_cycle test fakes)
# ---------------------------------------------------------------------------

class FakeDataClient:
    """Canned closes per ticker per date: {ticker: {date: close}}."""

    def __init__(self, series):
        self._series = series

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return getattr(self, "metrics", {}).get(ticker, [])[:limit]

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        days = self._series.get(ticker, {})
        return [
            Price(open=close, close=close, high=close, low=close,
                  volume=1000, time=f"{day}T00:00:00Z")
            for day, close in sorted(days.items())
            if start_date <= day <= end_date
        ]


class FakeAnalyst:
    """Fixed conviction per ticker, on every date."""

    def __init__(self, name, views=None):
        self._name = name
        self._views = views or {}

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=self._views.get(ticker, 0.0))


def _spec(**overrides):
    base = dict(
        name="test-fund",
        strategies=[{"name": "solo", "models": [{"name": "a"}]}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        # Frictionless and bandless so the arithmetic below stays hand-computable;
        # the cost model has its own test.
        execution={"min_trade_pct": 0.0, "commission_bps": 0.0, "slippage_bps": 0.0},
        capital=100_000.0,
        rebalance="weekly",
    )
    return FundSpec(**{**base, **overrides})


# Three trading weeks (Mon–Fri). Weekly grid = each Friday.
WEEKDAYS = [
    "2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07",
    "2024-06-10", "2024-06-11", "2024-06-12", "2024-06-13", "2024-06-14",
    "2024-06-17", "2024-06-18", "2024-06-19", "2024-06-20", "2024-06-21",
]
FRIDAYS = ["2024-06-07", "2024-06-14", "2024-06-21"]

# Closes chosen so 100k always targets exactly 500 AAPL shares — the fund
# buys once and then correctly has nothing to trade.
SERIES = {
    "SPY": {day: close for day, close in
            zip(FRIDAYS, [100.0, 102.0, 101.0])},
    "AAPL": {day: close for day, close in
             zip(FRIDAYS, [200.0, 210.0, 190.0])},
}


def _run(series=SERIES, spec=None):
    spec = spec or _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    return backtest_fund(fund, "2024-06-03", "2024-06-21",
                         FakeDataClient(series), ["AAPL"])


# ---------------------------------------------------------------------------
# rebalance_grid
# ---------------------------------------------------------------------------

def test_grid_daily_is_identity():
    assert rebalance_grid(WEEKDAYS, "daily") == WEEKDAYS


def test_grid_weekly_takes_last_trading_day_of_each_iso_week():
    # A short holiday week (no Friday) still contributes its last day.
    days = ["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02", "2024-07-05"]
    assert rebalance_grid(days, "weekly") == ["2024-06-28", "2024-07-05"]
    assert rebalance_grid(WEEKDAYS, "weekly") == FRIDAYS


def test_grid_monthly_splits_where_weekly_does_not():
    # Dec 30 2024 – Jan 3 2025 is ONE ISO week but TWO calendar months.
    days = ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    assert rebalance_grid(days, "weekly") == ["2025-01-03"]
    assert rebalance_grid(days, "monthly") == ["2024-12-31", "2025-01-03"]


def test_grid_unknown_cadence_raises():
    with pytest.raises(ValueError, match="cadence"):
        rebalance_grid(WEEKDAYS, "hourly")


# ---------------------------------------------------------------------------
# backtest_fund
# ---------------------------------------------------------------------------

def test_happy_path_hand_computed():
    result = _run()

    assert result.dates == FRIDAYS
    assert len(result.records) == 3
    # Week 1: buy 500 @ 200 (full conviction, 100% cap). Weeks 2-3: the
    # closes are chosen so the target stays exactly 500 shares — no churn.
    assert result.records[0].positions == {"AAPL": 500}
    assert result.nav == [100_000.0, 105_000.0, 95_000.0]
    assert result.metrics.n_orders == 1
    # Benchmark scaled to starting capital off its first grid close.
    assert result.benchmark_nav == [100_000.0, 102_000.0, 101_000.0]

    m = result.metrics
    assert m.total_return_pct == pytest.approx(-0.05)
    assert m.benchmark_return_pct == pytest.approx(0.01)
    assert m.excess_return_pct == pytest.approx(-0.06)
    # Peak 105k -> trough 95k.
    assert m.max_drawdown_pct == pytest.approx(10_000 / 105_000, abs=1e-6)
    assert m.n_cycles == 3
    assert m.total_costs == 0.0 and m.costs_pct == 0.0


def test_costs_come_out_of_nav_and_the_band_stops_churn():
    """The mandate's defaults (5 bps slippage, 0.5% band) applied end to end."""
    result = _run(spec=_spec(execution={"slippage_bps": 5.0, "commission_bps": 0.0,
                                        "min_trade_pct": 0.005}))

    # Week 1: 500 AAPL bought at 200 x (1 + 5 bps) = 200.10 -> 50 dollars of
    # slippage, and NAV marks at the 200 close, so it is 50 dollars short.
    first = result.records[0]
    assert first.positions == {"AAPL": 500}
    assert first.fills[0].price == pytest.approx(200.10)
    assert first.costs == pytest.approx(50.0)
    assert result.nav[0] == pytest.approx(99_950.0)

    # Week 2: equity 104,950 sizes to 499 shares, a 1-share sell worth 210
    # dollars. Under the 0.5% band (~525 dollars) it is not worth trading.
    assert result.metrics.n_orders == 1
    assert result.records[1].positions == {"AAPL": 500}
    assert result.records[1].costs == 0.0

    # The frictionless run is the same curve shifted up by the costs paid.
    frictionless = _run()
    assert [a + 50.0 for a in result.nav] == pytest.approx(frictionless.nav)
    assert result.metrics.total_costs == pytest.approx(50.0)
    assert result.metrics.costs_pct == pytest.approx(0.0005)


def test_band_disabled_churns_the_dust_share():
    """Same costs, no band: week 2 sells the 1-share drift and pays for it."""
    result = _run(spec=_spec(execution={"slippage_bps": 5.0, "commission_bps": 0.0,
                                        "min_trade_pct": 0.0}))
    assert result.metrics.n_orders >= 2
    assert result.records[1].positions == {"AAPL": 499}
    assert result.records[1].costs > 0.0


def test_positions_carry_across_cycles_not_restart():
    result = _run()
    # Same book all three weeks; only the marks moved.
    assert [r.positions for r in result.records] == [{"AAPL": 500}] * 3
    assert result.records[1].orders == []
    assert result.records[2].orders == []


def test_deterministic_json_round_trip():
    first, second = _run(), _run()
    assert first.model_dump_json() == second.model_dump_json()
    from hedge_fund.backtesting.fund import FundBacktestResult
    assert FundBacktestResult.model_validate_json(first.model_dump_json()) == first


def test_on_cycle_fires_per_tick_in_order():
    seen = []
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    backtest_fund(fund, "2024-06-03", "2024-06-21", FakeDataClient(SERIES),
                  ["AAPL"],
                  on_cycle=lambda i, n, record: seen.append((i, n, record.as_of)))
    assert seen == [(0, 3, FRIDAYS[0]), (1, 3, FRIDAYS[1]), (2, 3, FRIDAYS[2])]


def test_universe_round_trips_onto_the_result():
    """The study's tickers are recorded — the mandate never held them."""
    result = _run()
    assert result.universe == ["AAPL"]
    assert all(r.universe == ["AAPL"] for r in result.records)


def test_missing_benchmark_raises():
    series = {"AAPL": SERIES["AAPL"]}  # no SPY bars at all
    with pytest.raises(ValueError, match="trading grid"):
        _run(series=series)


def test_grid_follows_mandate_cadence():
    spec = _spec(rebalance="monthly")
    result = _run(spec=spec)
    assert result.dates == ["2024-06-21"]  # one June rebalance
    assert result.rebalance == "monthly"


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def test_single_sleeve_attribution_hand_computed():
    result = _run()
    (solo,) = result.attribution.strategies

    assert solo.name == "solo" and solo.slice == 1.0
    # Sleeve is 100% AAPL: 200 -> 210 (+5%) -> 190 (-9.52%). First period 0.
    assert solo.returns == pytest.approx([0.0, 0.05, 190 / 210 - 1])
    assert solo.nav == pytest.approx([100_000.0, 105_000.0, 95_000.0])
    assert solo.total_return_pct == pytest.approx(-0.05)
    assert solo.max_drawdown_pct == pytest.approx(10_000 / 105_000)
    assert solo.sharpe_ratio == result.metrics.sharpe_ratio  # identical curve, identical math
    assert solo.contribution_pct == pytest.approx(0.05 + (190 / 210 - 1), abs=1e-6)
    # Frictionless, uncapped, whole shares: the sleeve explains everything.
    assert result.attribution.residual_pct == pytest.approx(0.0, abs=1e-6)


def test_two_sleeves_sum_to_the_netted_book():
    """Opposing sleeves on one name: contributions net to the fund's return."""
    spec = _spec(strategies=[
        {"name": "bull", "weight": 3.0, "models": [{"name": "a"}]},
        {"name": "bear", "weight": 1.0, "models": [{"name": "b"}]},
    ])
    fund = Fund(spec, models={
        "bull": [FakeAnalyst("a", views={"AAPL": 1.0})],
        "bear": [FakeAnalyst("b", views={"AAPL": -1.0})],
    })
    result = backtest_fund(fund, "2024-06-03", "2024-06-21", FakeDataClient(SERIES), ["AAPL"])

    bull, bear = result.attribution.strategies
    assert bull.returns == pytest.approx([0.0, 0.05, 190 / 210 - 1])
    assert bear.returns == pytest.approx([0.0, -0.05, -(190 / 210 - 1)])
    assert bull.contribution_pct == pytest.approx(0.75 * sum(bull.returns), abs=1e-6)
    assert bear.contribution_pct == pytest.approx(0.25 * sum(bear.returns), abs=1e-6)
    # Netted book is 50% long AAPL: 250 shares in week 1. In week 2 the
    # target is 244.04 shares and the fund floors to 244, so the sleeves
    # explain the fund's return up to that whole-share rounding.
    assert result.records[0].positions == {"AAPL": 250}
    contributions = bull.contribution_pct + bear.contribution_pct
    assert result.attribution.fund_return_pct == pytest.approx(contributions, abs=1e-4)
    assert result.attribution.residual_pct == pytest.approx(
        result.attribution.fund_return_pct - contributions, abs=1e-6)
    assert abs(result.attribution.residual_pct) < 1e-4


def test_costs_and_clamps_land_in_the_residual():
    # 25% position cap on a 100%-conviction sleeve, plus slippage.
    spec = _spec(risk={"max_position_pct": 0.25, "max_gross_exposure": 1.0},
                 execution={"slippage_bps": 5.0, "commission_bps": 0.0,
                            "min_trade_pct": 0.0})
    result = _run(spec=spec)
    (solo,) = result.attribution.strategies
    # The sleeve's paper record is the unclamped 100% AAPL book...
    assert solo.total_return_pct == pytest.approx(-0.05)
    # ...while the fund held a quarter of that and paid to trade it.
    fund_arith = result.attribution.fund_return_pct
    assert fund_arith > solo.contribution_pct  # cap muted the loss
    assert result.attribution.residual_pct == pytest.approx(fund_arith - solo.contribution_pct)
    assert result.attribution.residual_pct > 0


# ---------------------------------------------------------------------------
# Dated universe and the survivorship warning
# ---------------------------------------------------------------------------

def test_static_universe_carries_the_survivorship_warning():
    result = _run()
    assert result.universe_kind == "static"
    assert len(result.warnings) == 1
    assert "Survivorship bias" in result.warnings[0]
    assert "AAPL" in result.warnings[0] and result.start in result.warnings[0]


def test_dated_universe_changes_the_book_and_carries_no_warning():
    from hedge_fund.fund.universe import DatedUniverse

    series = {
        "SPY": SERIES["SPY"],
        "AAPL": SERIES["AAPL"],
        "MSFT": {day: 100.0 for day in FRIDAYS},
    }
    # Week 1: AAPL only. From week 2: MSFT replaces AAPL.
    universe = DatedUniverse(entries=[
        {"from": "2024-06-01", "tickers": ["AAPL"]},
        {"from": "2024-06-10", "tickers": ["MSFT"]},
    ])
    spec = _spec()
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})]})
    result = backtest_fund(fund, "2024-06-03", "2024-06-21", FakeDataClient(series), universe)

    assert result.universe_kind == "dated"
    assert result.warnings == []
    assert result.universe == ["AAPL", "MSFT"]
    assert [r.universe for r in result.records] == [["AAPL"], ["MSFT"], ["MSFT"]]
    # Dropping out of the universe closes the position at that cycle's mark,
    # and the newcomer is bought with the proceeds.
    assert result.records[0].positions == {"AAPL": 500}
    assert result.records[1].positions == {"MSFT": 1050}    # 500 x 210 = 105,000 / 100
    assert [(o.ticker, o.side) for o in result.records[1].orders] == [
        ("AAPL", "sell"), ("MSFT", "buy"),
    ]


def test_dated_universe_starting_after_the_backtest_raises():
    from hedge_fund.fund.universe import DatedUniverse

    universe = DatedUniverse(entries=[{"from": "2024-06-10", "tickers": ["AAPL"]}])
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    with pytest.raises(ValueError, match="no membership as of 2024-06-07"):
        backtest_fund(fund, "2024-06-03", "2024-06-21", FakeDataClient(SERIES), universe)


# ---------------------------------------------------------------------------
# Drawdown kill-switch, end to end
# ---------------------------------------------------------------------------

def test_kill_switch_flattens_and_stays_flat():
    # AAPL 200 -> 210 -> 190: week 3 opens 9.5% below the week-2 peak.
    spec = _spec(risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0,
                       "max_drawdown_pct": 0.09})
    result = _run(spec=spec)

    r1, r2, r3 = result.records
    assert (r1.peak_nav, r1.drawdown) == (100_000.0, 0.0)
    assert (r2.peak_nav, r2.drawdown) == (100_000.0, 0.0)      # equity 105k above the old peak
    assert r3.peak_nav == 105_000.0
    assert r3.drawdown == pytest.approx(10_000 / 105_000)
    assert [c.limit for c in r3.clamps] == ["max_drawdown_pct"]
    assert r3.final_weights == {"AAPL": 0.0}
    assert r3.positions == {} and r3.nav == pytest.approx(95_000.0)
    assert result.metrics.kill_switch_date == "2024-06-21"


def test_kill_switch_stays_flat_on_later_cycles():
    days = WEEKDAYS + ["2024-06-24", "2024-06-25", "2024-06-26", "2024-06-27", "2024-06-28"]
    fridays = FRIDAYS + ["2024-06-28"]
    series = {
        "SPY": {d: c for d, c in zip(fridays, [100.0, 102.0, 101.0, 110.0])},
        "AAPL": {d: c for d, c in zip(fridays, [200.0, 210.0, 190.0, 260.0])},
    }
    spec = _spec(risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0,
                       "max_drawdown_pct": 0.09})
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    result = backtest_fund(fund, "2024-06-03", "2024-06-28", FakeDataClient(series), ["AAPL"])

    # Killed in week 3; week 4's rally is missed because a flat book never
    # climbs back out of its drawdown.
    assert result.metrics.kill_switch_date == "2024-06-21"
    r4 = result.records[3]
    assert [c.limit for c in r4.clamps] == ["max_drawdown_pct"]
    assert r4.positions == {} and r4.nav == pytest.approx(95_000.0)


def test_no_kill_switch_means_no_drawdown_clamp():
    result = _run()
    assert result.metrics.kill_switch_date is None
    assert all(c.limit != "max_drawdown_pct" for r in result.records for c in r.clamps)
    assert result.records[2].drawdown == pytest.approx(10_000 / 105_000)  # still measured


# ---------------------------------------------------------------------------
# Dividends through the backtest
# ---------------------------------------------------------------------------

def test_backtest_accrues_dividends_and_benchmark_yield():
    from hedge_fund.data.models import FinancialMetrics

    data = FakeDataClient(SERIES)
    data.metrics = {"AAPL": [FinancialMetrics(
        ticker="AAPL", report_period="2024-03-31", period="ttm", filing_date="2024-05-01",
        earnings_per_share=4.0, payout_ratio=3.65 / 4.0)]}          # $3.65/yr
    spec = _spec(dividends={"accrue": True, "benchmark_yield": 0.0365})
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    result = backtest_fund(fund, "2024-06-03", "2024-06-21", data, ["AAPL"])

    # 500 shares x $3.65 x 7/365 = $35 per weekly gap, twice.
    assert [sum(r.dividends.values()) for r in result.records] == pytest.approx([0.0, 35.0, 35.0])
    assert result.metrics.total_dividends == pytest.approx(70.0)
    assert result.nav == pytest.approx([100_000.0, 105_035.0, 95_070.0])
    # Benchmark: price return x (1 + 3.65%)^(days/365).
    assert result.benchmark_nav[0] == pytest.approx(100_000.0)
    assert result.benchmark_nav[1] == pytest.approx(102_000.0 * 1.0365 ** (7 / 365))
    assert result.benchmark_nav[2] == pytest.approx(101_000.0 * 1.0365 ** (14 / 365))


def test_no_dividend_data_means_no_accrual():
    result = _run()
    assert result.metrics.total_dividends == 0.0
    assert all(r.dividends == {} for r in result.records)
