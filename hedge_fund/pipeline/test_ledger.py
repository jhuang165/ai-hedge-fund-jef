"""Ledger tests — the book carried between runs through receipts on disk."""

import json

import pytest

from hedge_fund.fund.spec import Fund
from hedge_fund.pipeline.ledger import LedgerError, carried_book, run_carried, run_receipts
from hedge_fund.pipeline.test_run_cycle import CLOSES, UNIVERSE, FakeAnalyst, FakeDataClient, _spec


def _fund(views):
    return Fund(_spec(), models={"solo": [FakeAnalyst("a", views=views)]})


VIEWS = {"AAPL": 1.0, "MSFT": 0.5}


def test_first_run_opens_on_capital_and_saves_a_receipt(tmp_path):
    ran = run_carried(_fund(VIEWS), "2024-06-03", FakeDataClient(CLOSES), UNIVERSE,
                      root=tmp_path)

    assert ran.carry.as_of is None
    assert ran.record.prev_as_of is None
    assert ran.record.equity_before == pytest.approx(100_000.0)
    assert ran.record.positions  # it traded
    assert ran.path.parent == tmp_path
    assert json.loads(ran.path.read_text())["nav"] == pytest.approx(ran.record.nav)
    assert "first run" in ran.carry.describe()


def test_second_run_carries_the_book(tmp_path):
    first = run_carried(_fund(VIEWS), "2024-06-03", FakeDataClient(CLOSES), UNIVERSE,
                        root=tmp_path).record
    # Prices move between runs: the carried book is marked at the new closes.
    moved = {**CLOSES, "AAPL": 220.0}
    ran = run_carried(_fund(VIEWS), "2024-06-10", FakeDataClient(moved), UNIVERSE,
                      root=tmp_path)
    second = ran.record

    assert ran.carry.as_of == "2024-06-03"
    assert ran.carry.positions == first.positions
    assert second.prev_as_of == "2024-06-03"
    assert second.cash_before == pytest.approx(first.cash)
    expected = first.cash + sum(s * moved[t] for t, s in first.positions.items())
    assert second.equity_before == pytest.approx(expected)
    assert second.equity_before != pytest.approx(100_000.0)  # not reset to capital
    assert "2024-06-03 run" in ran.carry.describe()


def test_rerun_with_unchanged_views_and_prices_does_not_churn(tmp_path):
    fund, data = _fund(VIEWS), FakeDataClient(CLOSES)
    run_carried(fund, "2024-06-03", data, UNIVERSE, root=tmp_path)
    again = run_carried(fund, "2024-06-03", data, UNIVERSE, root=tmp_path).record
    assert again.orders == []
    assert len(run_receipts("test-fund", tmp_path)) == 2


def test_a_dropped_ticker_is_closed_out(tmp_path):
    run_carried(_fund(VIEWS), "2024-06-03", FakeDataClient(CLOSES), UNIVERSE,
                root=tmp_path)
    record = run_carried(_fund(VIEWS), "2024-06-10", FakeDataClient(CLOSES), ["AAPL"],
                         root=tmp_path).record
    assert "MSFT" not in record.positions
    assert any(o.ticker == "MSFT" and o.side == "sell" for o in record.orders)


def test_running_before_the_last_receipt_is_refused(tmp_path):
    run_carried(_fund(VIEWS), "2024-06-10", FakeDataClient(CLOSES), UNIVERSE,
                root=tmp_path)
    with pytest.raises(LedgerError, match="2024-06-10"):
        run_carried(_fund(VIEWS), "2024-06-03", FakeDataClient(CLOSES), UNIVERSE,
                    root=tmp_path)
    assert len(run_receipts("test-fund", tmp_path)) == 1  # nothing written


def _receipt(root, name, stamp, **fields):
    (root / f"{name}-run-{stamp}.json").write_text(json.dumps(
        {"fund": name, "positions": {}, **fields}))


def test_chain_is_ordered_by_as_of_and_peak_spans_it(tmp_path):
    # Written out of order on purpose: the stamp says when a run executed,
    # the chain follows the as-of date.
    _receipt(tmp_path, "test-fund", "2024-07-01-000000", as_of="2024-06-10",
             equity_before=120_000.0, nav=90_000.0, cash=90_000.0)
    _receipt(tmp_path, "test-fund", "2024-06-01-000000", as_of="2024-06-03",
             equity_before=100_000.0, nav=120_000.0, cash=500.0,
             positions={"AAPL": 10})

    carry = carried_book(_spec(), "2024-06-17", tmp_path)
    assert carry.as_of == "2024-06-10"
    assert carry.cash == 90_000.0
    assert carry.positions == {}
    assert carry.peak_nav == 120_000.0


def test_only_this_funds_run_receipts_count(tmp_path):
    # "test-fund-run-…" also matches the glob of a fund named "test-fund"
    # when the other fund is "test-fund-run"; the name inside decides.
    _receipt(tmp_path, "test-fund-run", "x", as_of="2024-06-10",
             equity_before=1.0, nav=1.0, cash=1.0)
    (tmp_path / "test-fund-backtest-x.json").write_text("{}")
    (tmp_path / "test-fund-run-broken.json").write_text("not json")

    carry = carried_book(_spec(), "2024-06-03", tmp_path)
    assert carry.as_of is None
    assert carry.cash == 100_000.0


# ---------------------------------------------------------------------------
# Paper: the Alpaca account holds the book
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from hedge_fund.brokers.alpaca import AlpacaBroker, AlpacaError  # noqa: E402
from hedge_fund.brokers.fake_alpaca import FakeAlpaca  # noqa: E402
from hedge_fund.pipeline import ledger  # noqa: E402

TODAY = datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def _paper_fund(views=VIEWS):
    spec = _spec()
    spec = spec.model_copy(update={
        "execution": spec.execution.model_copy(update={"broker": "alpaca-paper"})})
    return Fund(spec, models={"solo": [FakeAnalyst("a", views=views)]})


@pytest.fixture
def account(monkeypatch):
    fake = FakeAlpaca(date=TODAY, prices={t: p * 1.001 for t, p in CLOSES.items()})
    monkeypatch.setattr(ledger.AlpacaBroker, "from_env", classmethod(
        lambda cls: AlpacaBroker("k", "s", session=fake, sleep=lambda s: None)))
    return fake


def test_paper_run_trades_the_account_and_saves_its_book(tmp_path, account):
    ran = run_carried(_paper_fund(), TODAY, FakeDataClient(CLOSES), UNIVERSE, root=tmp_path)
    record = ran.record

    assert account.positions and record.positions == account.positions
    assert record.cash == pytest.approx(account.cash)
    assert record.peak_nav is None and record.drawdown is None  # no mark to fall from yet
    assert record.warnings == []
    assert json.loads(ran.path.read_text())["spec"]["execution"]["broker"] == "alpaca-paper"
    assert "Alpaca paper account" in ran.carry.describe()


def test_paper_run_reads_the_account_and_flags_drift(tmp_path, account):
    run_carried(_paper_fund(), TODAY, FakeDataClient(CLOSES), UNIVERSE, root=tmp_path)
    account.positions["AAPL"] -= 7  # someone traded in the dashboard
    record = run_carried(_paper_fund(), TODAY, FakeDataClient(CLOSES), UNIVERSE,
                         root=tmp_path).record

    assert record.prev_as_of == TODAY
    assert len(record.warnings) == 1 and record.warnings[0].startswith("AAPL:")


def test_first_paper_run_needs_a_flat_account(tmp_path, account):
    account.positions["TSLA"] = 5
    with pytest.raises(LedgerError, match="TSLA"):
        run_carried(_paper_fund(), TODAY, FakeDataClient(CLOSES), UNIVERSE, root=tmp_path)
    assert account.submitted == [] and run_receipts("test-fund", tmp_path) == []


def test_adopting_an_account_hands_its_positions_to_the_fund(tmp_path, account):
    account.positions["TSLA"] = 5
    account.prices["TSLA"] = 250.0
    record = run_carried(_paper_fund(), TODAY, FakeDataClient({**CLOSES, "TSLA": 250.0}),
                         UNIVERSE, root=tmp_path, adopt_account=True).record
    assert record.warnings == ["adopted the account's TSLA position (5 shares) into the fund's book"]
    assert "TSLA" not in record.positions  # not in the universe: the fund closed it


def test_paper_runs_trade_today_only(tmp_path, account):
    with pytest.raises(LedgerError, match="as of today"):
        run_carried(_paper_fund(), "2024-06-03", FakeDataClient(CLOSES), UNIVERSE,
                    root=tmp_path)


def test_a_closed_market_stops_the_run_before_any_analyst_is_asked(tmp_path, account):
    account.is_open = False
    fund = _paper_fund()
    with pytest.raises(AlpacaError, match="closed"):
        run_carried(fund, TODAY, FakeDataClient(CLOSES), UNIVERSE, root=tmp_path)
    assert fund.strategies[0][1][0].predict_calls == []


def test_each_broker_keeps_its_own_chain(tmp_path, account):
    # A simulated history — even one dated later — neither blocks nor seeds
    # the paper chain.
    _receipt(tmp_path, "test-fund", "x", as_of="2099-01-01", equity_before=1.0,
             nav=1.0, cash=1.0, positions={"AAPL": 99})
    record = run_carried(_paper_fund(), TODAY, FakeDataClient(CLOSES), UNIVERSE,
                         root=tmp_path).record
    assert record.prev_as_of is None and record.warnings == []
    assert carried_book(_spec(), "2099-01-02", tmp_path).positions == {"AAPL": 99}
