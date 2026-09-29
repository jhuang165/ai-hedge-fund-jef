"""run_cycle end-to-end tests — fake data client + fake analysts + real SimBroker."""

import pytest

from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.models import Price
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.models import Signal
from hedge_fund.pipeline.models import CycleRecord
from hedge_fund.pipeline.run_cycle import run_cycle


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeDataClient:
    """Canned closes per ticker; a ticker absent from `closes` has no bars."""

    def __init__(self, closes, metrics=None):
        self._closes = closes
        self._metrics = metrics or {}

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return self._metrics.get(ticker, [])[:limit]

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        close = self._closes.get(ticker)
        if close is None:
            return []
        return [Price(open=close, close=close, high=close, low=close,
                      volume=1000, time=f"{end_date}T00:00:00Z")]


class FakeAnalyst:
    """Fixed conviction per ticker; counts predict calls."""

    def __init__(self, name, views=None, abstain=False, error=None):
        self._name = name
        self._views = views or {}
        self._abstain = abstain
        self._error = error
        self.predict_calls = []

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        self.predict_calls.append(ticker)
        if self._error is not None:
            raise self._error
        metadata = {"abstained": True} if self._abstain else {}
        value = 0.0 if self._abstain else self._views.get(ticker, 0.0)
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=value, metadata=metadata)


def _spec(strategies=None, max_position_pct=0.25):
    if strategies is None:
        strategies = [{"name": "solo", "models": [{"name": "a"}]}]
    return FundSpec(
        name="test-fund",
        strategies=strategies,
        risk={"max_position_pct": max_position_pct, "max_gross_exposure": 1.0},
        capital=100_000.0,
    )


CLOSES = {"AAPL": 200.0, "MSFT": 400.0, "NVDA": 100.0}
# What to trade is a run-time argument, not a mandate field.
UNIVERSE = ["AAPL", "MSFT", "NVDA"]


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_full_cycle_record_is_consistent():
    spec = _spec(strategies=[
        {"name": "long", "models": [{"name": "a"}]},
        {"name": "short", "models": [{"name": "b"}]},
    ])
    fund = Fund(spec, models={
        "long": [FakeAnalyst("a", views={"AAPL": 1.0, "NVDA": 0.5})],
        "short": [FakeAnalyst("b", views={"MSFT": -1.0})],
    })
    broker = SimBroker(cash=100_000.0)

    record = run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES),
                       UNIVERSE)

    assert record.fund == "test-fund"
    assert record.equity_before == pytest.approx(100_000.0)
    assert len(record.strategies) == 2
    assert all(len(sr.signals) == 3 for sr in record.strategies)  # 3 tickers x 1 analyst
    # Weights respect the hard caps
    for w in record.final_weights.values():
        assert abs(w) <= 0.25 + 1e-12
    # Fills mirror orders one-to-one, and the books balance
    assert len(record.fills) == len(record.orders) > 0
    assert record.nav == pytest.approx(
        record.cash + sum(s * record.marks[t] for t, s in record.positions.items())
    )
    # The short strategy's bearish view -> short position
    assert record.positions["MSFT"] < 0


def test_netting_math_two_strategies_unequal_slices():
    """Two sleeves, overlapping ticker, 3:1 slices — hand-computed netting."""
    spec = _spec(strategies=[
        {"name": "s1", "weight": 3.0, "models": [{"name": "a"}]},
        {"name": "s2", "weight": 1.0, "models": [{"name": "b"}]},
    ], max_position_pct=1.0)
    fund = Fund(spec, models={
        # s1 sleeve: AAPL 0.5, MSFT 0.5 (equal convictions, gross 1.0)
        "s1": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
        # s2 sleeve: MSFT -1.0 (only conviction takes full gross)
        "s2": [FakeAnalyst("b", views={"MSFT": -1.0})],
    })

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), UNIVERSE)

    s1, s2 = record.strategies
    assert s1.slice == pytest.approx(0.75)
    assert s2.slice == pytest.approx(0.25)
    assert s1.weights == {"AAPL": pytest.approx(0.5), "MSFT": pytest.approx(0.5),
                          "NVDA": 0.0}
    assert s2.weights["MSFT"] == pytest.approx(-1.0)
    # Netted: AAPL = .75*.5 = .375 ; MSFT = .75*.5 + .25*(-1) = .125
    assert record.target_weights["AAPL"] == pytest.approx(0.375)
    assert record.target_weights["MSFT"] == pytest.approx(0.125)


def test_slices_normalize():
    """weights 2/2 must mean exactly the same as 1/1."""
    def run(w1, w2):
        spec = _spec(strategies=[
            {"name": "s1", "weight": w1, "models": [{"name": "a"}]},
            {"name": "s2", "weight": w2, "models": [{"name": "b"}]},
        ])
        fund = Fund(spec, models={
            "s1": [FakeAnalyst("a", views={"AAPL": 1.0})],
            "s2": [FakeAnalyst("b", views={"NVDA": -0.5})],
        })
        return run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                         FakeDataClient(CLOSES), UNIVERSE)

    assert run(2.0, 2.0).target_weights == run(1.0, 1.0).target_weights


def test_deterministic_and_json_round_trips():
    def make():
        spec = _spec()
        fund = Fund(spec, models={
            "solo": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": -0.5})],
        })
        return run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                         FakeDataClient(CLOSES), UNIVERSE)

    first, second = make(), make()
    assert first.model_dump_json() == second.model_dump_json()
    assert CycleRecord.model_validate_json(first.model_dump_json()) == first


def test_second_cycle_rebalances_not_restarts():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(max_position_pct=1.0), models={"solo": [analyst]})
    broker = SimBroker(cash=100_000.0)
    data = FakeDataClient({"AAPL": 200.0})

    first = run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    second = run_cycle(fund, "2024-06-04", broker, data, ["AAPL"])

    assert first.positions["AAPL"] == 500  # 100k at 200
    assert second.orders == []  # already at target; nothing to trade
    assert second.positions["AAPL"] == 500


# ---------------------------------------------------------------------------
# Abstain / flat behavior
# ---------------------------------------------------------------------------

def test_all_abstain_closes_the_book_to_flat():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(max_position_pct=1.0), models={"solo": [analyst]})
    broker = SimBroker(cash=100_000.0)
    data = FakeDataClient({"AAPL": 200.0})
    run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    assert broker.positions()["AAPL"].shares == 500

    analyst._abstain = True
    record = run_cycle(fund, "2024-06-04", broker, data, ["AAPL"])

    assert record.positions == {}  # book closed to flat
    assert record.nav == pytest.approx(100_000.0)  # flat closes at same price


# ---------------------------------------------------------------------------
# Pricing edge cases
# ---------------------------------------------------------------------------

def test_unpriced_unowned_ticker_skipped_and_analysts_never_called():
    analyst = FakeAnalyst("a", views={"AAPL": 1.0})
    fund = Fund(_spec(), models={"solo": [analyst]})
    closes = dict(CLOSES)
    del closes["NVDA"]

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(closes), UNIVERSE)

    assert [s.ticker for s in record.skipped] == ["NVDA"]
    assert "NVDA" not in analyst.predict_calls
    assert "NVDA" not in record.final_weights


def test_unpriced_held_ticker_raises():
    broker = SimBroker(cash=100_000.0)
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    run_cycle(fund, "2024-06-03", broker, FakeDataClient(CLOSES), UNIVERSE)
    assert broker.positions()  # something is held

    closes = {t: c for t, c in CLOSES.items() if t not in broker.positions()}
    with pytest.raises(ValueError, match="cannot value the book"):
        run_cycle(fund, "2024-06-04", broker, FakeDataClient(closes), UNIVERSE)


def test_universe_is_a_run_time_argument():
    """The same fund, pointed at different names, trades different names —
    and the record says what it was asked to trade."""
    fund = Fund(_spec(max_position_pct=1.0), models={
        "solo": [FakeAnalyst("a", views={"AAPL": 1.0, "MSFT": 1.0})],
    })

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), ["aapl", "AAPL"])

    assert record.universe == ["AAPL"]  # upper-cased and de-duped
    assert "MSFT" not in record.final_weights


def test_empty_universe_raises():
    fund = Fund(_spec(), models={"solo": [FakeAnalyst("a")]})
    with pytest.raises(ValueError, match="universe is empty"):
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                  FakeDataClient(CLOSES), [])


def test_analyst_error_propagates():
    """Fail loud: an infrastructure failure must not become a quiet no-trade."""
    fund = Fund(_spec(), models={
        "solo": [FakeAnalyst("a", error=ConnectionError("API down"))],
    })
    with pytest.raises(ConnectionError):
        run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                  FakeDataClient(CLOSES), UNIVERSE)


# ---------------------------------------------------------------------------
# Risk-scaled sizing
# ---------------------------------------------------------------------------

class HistoryDataClient(FakeDataClient):
    """A year of daily bars per ticker with a chosen daily amplitude, so the
    price snapshot can compute a realized vol; the last close is `closes`."""

    def __init__(self, closes, amplitude):
        super().__init__(closes)
        self._amp = amplitude
        self.price_calls = 0

    def get_prices(self, ticker, start_date, end_date, **kwargs):
        from datetime import date, timedelta
        self.price_calls += 1
        close = self._closes.get(ticker)
        if close is None:
            return []
        end = date.fromisoformat(end_date)
        start = date.fromisoformat(start_date)
        bars, day, i = [], end, 0
        while day >= start:
            if day.weekday() < 5:
                # Alternate up and down by the amplitude: vol scales with it.
                px = close * (1 + self._amp[ticker] * (1 if i % 2 else -1))
                bars.append(Price(open=px, close=px, high=px, low=px, volume=1000,
                                  time=f"{day.isoformat()}T00:00:00Z"))
                i += 1
            day -= timedelta(days=1)
        bars[0] = Price(open=close, close=close, high=close, low=close, volume=1000,
                        time=f"{end_date}T00:00:00Z")
        return sorted(bars, key=lambda b: b.time)


def test_vol_scaled_sizing_tilts_toward_the_calmer_name():
    spec = _spec(max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0, "NVDA": 1.0})]})
    data = HistoryDataClient({"AAPL": 200.0, "NVDA": 100.0}, {"AAPL": 0.01, "NVDA": 0.04})

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0), data, ["AAPL", "NVDA"])

    assert set(record.vols) == {"AAPL", "NVDA"}
    assert record.vols["NVDA"] > record.vols["AAPL"]
    # Equal views: the calmer name gets more dollars, and the ratio of the
    # weights is the inverse ratio of the vols.
    w = record.target_weights
    assert w["AAPL"] > w["NVDA"] > 0
    assert w["AAPL"] / w["NVDA"] == pytest.approx(record.vols["NVDA"] / record.vols["AAPL"])


def test_vol_scaling_off_means_no_vols_fetched():
    spec = _spec(strategies=[{"name": "solo", "models": [{"name": "a"}],
                              "blend": {"vol_scaled": False}}], max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0, "NVDA": 1.0})]})
    data = HistoryDataClient({"AAPL": 200.0, "NVDA": 100.0}, {"AAPL": 0.01, "NVDA": 0.04})

    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0), data, ["AAPL", "NVDA"])

    assert record.vols == {}
    assert data.price_calls == 2  # marks only, no snapshot fetch
    assert record.target_weights["AAPL"] == pytest.approx(record.target_weights["NVDA"])


def test_too_little_history_sizes_as_typical_name():
    """The default fake serves one bar: no vol, plain conviction weights."""
    spec = _spec(max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0, "NVDA": 1.0})]})
    record = run_cycle(fund, "2024-06-03", SimBroker(cash=100_000.0),
                       FakeDataClient(CLOSES), ["AAPL", "NVDA"])
    assert record.vols == {}
    assert record.target_weights["AAPL"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Dividend accrual
# ---------------------------------------------------------------------------

def _payer(dps_ttm, eps=4.0):
    from hedge_fund.data.models import FinancialMetrics
    return [FinancialMetrics(ticker="X", report_period="2024-03-31", period="ttm",
                             filing_date="2024-05-01", earnings_per_share=eps,
                             payout_ratio=dps_ttm / eps)]


def test_dividends_accrue_on_the_held_book_between_cycles():
    spec = _spec(max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    data = FakeDataClient(CLOSES, metrics={"AAPL": _payer(dps_ttm=3.65)})
    broker = SimBroker(cash=100_000.0)
    first = run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    assert first.dividends == {}                       # nothing to accrue from
    assert first.positions == {"AAPL": 500}

    second = run_cycle(fund, "2024-06-13", broker, data, ["AAPL"], prev_as_of="2024-06-03")
    # 500 shares x $3.65/yr x 10/365 days = $50.
    assert second.dividends == {"AAPL": pytest.approx(50.0)}
    assert second.cash_before == pytest.approx(first.cash + 50.0)
    assert second.equity_before == pytest.approx(first.nav + 50.0)


def test_a_broker_that_books_real_dividends_gets_no_estimate():
    from hedge_fund.brokers.alpaca import AlpacaBroker
    from hedge_fund.brokers.fake_alpaca import FakeAlpaca

    fake = FakeAlpaca(positions={"AAPL": 500}, cash=0.0, prices=CLOSES)
    broker = AlpacaBroker("k", "s", session=fake, sleep=lambda s: None)
    fund = Fund(_spec(max_position_pct=1.0),
                models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    data = FakeDataClient(CLOSES, metrics={"AAPL": _payer(dps_ttm=3.65)})
    record = run_cycle(fund, "2024-06-13", broker, data, ["AAPL"], prev_as_of="2024-06-03")
    assert record.dividends == {}
    assert record.cash_before == 0.0


def test_short_owes_the_dividend_and_nonpayers_accrue_nothing():
    spec = _spec(max_position_pct=1.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": -1.0, "MSFT": 1.0})]})
    data = FakeDataClient(CLOSES, metrics={"AAPL": _payer(dps_ttm=3.65),
                                           "MSFT": _payer(dps_ttm=0.0)})
    broker = SimBroker(cash=100_000.0)
    first = run_cycle(fund, "2024-06-03", broker, data, ["AAPL", "MSFT"])
    assert first.positions["AAPL"] < 0 < first.positions["MSFT"]
    second = run_cycle(fund, "2024-06-13", broker, data, ["AAPL", "MSFT"], prev_as_of="2024-06-03")
    assert second.dividends["AAPL"] == pytest.approx(first.positions["AAPL"] * 3.65 * 10 / 365)
    assert second.dividends["AAPL"] < 0
    assert "MSFT" not in second.dividends


def test_accrual_can_be_switched_off():
    spec = FundSpec(name="t", strategies=[{"name": "solo", "models": [{"name": "a"}]}],
                    risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
                    dividends={"accrue": False}, capital=100_000.0)
    fund = Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})
    data = FakeDataClient(CLOSES, metrics={"AAPL": _payer(dps_ttm=3.65)})
    broker = SimBroker(cash=100_000.0)
    run_cycle(fund, "2024-06-03", broker, data, ["AAPL"])
    second = run_cycle(fund, "2024-06-13", broker, data, ["AAPL"], prev_as_of="2024-06-03")
    assert second.dividends == {}
