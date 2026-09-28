"""Hold-out and sweep tests — fake analysts, canned tape, real backtests."""

import pytest

from hedge_fund.backtesting.test_fund import FakeAnalyst, FakeDataClient
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.validation import holdout, parse_sweep, set_path, sweep

# Six Fridays. First half AAPL rallies, second half it falls.
FRIDAYS = ["2024-06-07", "2024-06-14", "2024-06-21", "2024-06-28", "2024-07-05", "2024-07-12"]
SERIES = {
    "SPY": {d: 100.0 for d in FRIDAYS},
    "AAPL": dict(zip(FRIDAYS, [100.0, 110.0, 121.0, 121.0, 110.0, 100.0])),
}


def _spec(**overrides):
    base = dict(
        name="t", strategies=[{"name": "solo", "models": [{"name": "a"}]}],
        risk={"max_position_pct": 1.0, "max_gross_exposure": 1.0},
        execution={"min_trade_pct": 0.0, "commission_bps": 0.0, "slippage_bps": 0.0},
        capital=100_000.0, rebalance="weekly",
    )
    return FundSpec(**{**base, **overrides})


def _fund(spec):
    return Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": 1.0})]})


def test_holdout_splits_the_window_and_reports_both_halves():
    result = holdout(_fund(_spec()), "2024-06-03", "2024-06-24", "2024-07-12",
                     FakeDataClient(SERIES), ["AAPL"])
    assert (result.train_start, result.train_end) == ("2024-06-07", "2024-06-21")
    assert (result.test_start, result.test_end) == ("2024-06-28", "2024-07-12")
    # Fresh start each half: +21% in train, then 121 -> 100 in test.
    assert result.train.total_return_pct == pytest.approx(0.21)
    # Whole shares: 826 of 121 leaves dust in cash, so slightly less than 100/121 - 1.
    assert result.test.total_return_pct == pytest.approx(100 / 121 - 1, abs=1e-3)
    assert result.train.sharpe_ratio > 0 > result.test.sharpe_ratio
    assert result.sharpe_decay == pytest.approx(
        result.train.sharpe_ratio - result.test.sharpe_ratio, abs=1e-4)
    assert not result.holds
    assert "does not survive" in result.verdict


def test_holdout_holds_when_both_halves_work():
    steady = {"SPY": SERIES["SPY"],
              "AAPL": dict(zip(FRIDAYS, [100.0, 105.0, 110.0, 115.0, 120.0, 125.0]))}
    result = holdout(_fund(_spec()), "2024-06-03", "2024-06-24", "2024-07-12",
                     FakeDataClient(steady), ["AAPL"])
    assert result.holds and result.verdict == "holds out of sample"


def test_holdout_rejects_bad_split():
    with pytest.raises(ValueError):
        holdout(_fund(_spec()), "2024-06-03", "2024-08-01", "2024-07-12",
                FakeDataClient(SERIES), ["AAPL"])


def test_sweep_runs_every_combination_and_judges_the_surface():
    result = sweep(
        _spec(), {"risk.max_position_pct": [0.25, 1.0], "execution.slippage_bps": [0, 50]},
        "2024-06-03", "2024-06-21", FakeDataClient(SERIES), ["AAPL"],
        build_fund=_fund,
    )
    assert result.paths == ["risk.max_position_pct", "execution.slippage_bps"]
    assert [p.overrides for p in result.points] == [
        {"risk.max_position_pct": 0.25, "execution.slippage_bps": 0},
        {"risk.max_position_pct": 0.25, "execution.slippage_bps": 50},
        {"risk.max_position_pct": 1.0, "execution.slippage_bps": 0},
        {"risk.max_position_pct": 1.0, "execution.slippage_bps": 50},
    ]
    full = result.points[2].metrics
    assert full.total_return_pct == pytest.approx(0.21)               # uncapped, free
    assert result.points[3].metrics.total_return_pct < 0.21          # slippage bites
    assert 0.04 < result.points[0].metrics.total_return_pct < 0.06    # a quarter of the book
    assert result.points[0].test is None
    assert result.sharpe_min <= result.sharpe_median <= result.sharpe_max
    assert result.share_positive == 1.0
    assert not result.fragile and result.verdict.startswith("robust")


def test_sweep_with_split_also_scores_the_test_half():
    result = sweep(_spec(), {"risk.max_position_pct": [1.0]},
                   "2024-06-03", "2024-07-12", FakeDataClient(SERIES), ["AAPL"],
                   split="2024-06-24", build_fund=_fund)
    (point,) = result.points
    assert point.test is not None
    assert point.test.total_return_pct == pytest.approx(100 / 121 - 1, abs=1e-3)


def test_sweep_flags_a_sign_change_as_fragile():
    # Long at 100% wins in this window; short (gross target as a short view
    # needs a bearish analyst) — instead flip the sign via a bearish sleeve.
    def build(spec):
        view = spec.strategies[0].models[0].params.get("view", 1.0)
        return Fund(spec, models={"solo": [FakeAnalyst("a", views={"AAPL": view})]})

    result = sweep(_spec(), {"strategies.0.models.0.params.view": [1.0, -1.0]},
                   "2024-06-03", "2024-06-21", FakeDataClient(SERIES), ["AAPL"],
                   build_fund=build)
    assert result.fragile and "changes sign" in result.verdict
    assert result.share_positive == 0.5


def test_sweep_needs_overrides():
    with pytest.raises(ValueError):
        sweep(_spec(), {}, "2024-06-03", "2024-06-21", FakeDataClient(SERIES), ["AAPL"])


def test_set_path_walks_dicts_and_lists():
    data = _spec().model_dump()
    set_path(data, "strategies.0.models.0.params.scale", 2.0)
    set_path(data, "risk.max_position_pct", 0.5)
    assert data["strategies"][0]["models"][0]["params"] == {"scale": 2.0}
    assert data["risk"]["max_position_pct"] == 0.5
    with pytest.raises(ValueError, match="not a list index"):
        set_path(data, "strategies.first.weight", 1)
    with pytest.raises(ValueError, match="out of range"):
        set_path(data, "strategies.5.weight", 1)
    with pytest.raises(ValueError, match="cannot descend"):
        set_path(data, "name.oops", 1)


def test_parse_sweep_reads_yaml_scalars():
    parsed = parse_sweep(["a.b=0.5,1,2", "c=true,x"])
    assert parsed == {"a.b": [0.5, 1, 2], "c": [True, "x"]}
    assert isinstance(parsed["a.b"][0], float) and isinstance(parsed["a.b"][1], int)
    with pytest.raises(ValueError):
        parse_sweep(["nope"])
    with pytest.raises(ValueError):
        parse_sweep(["a="])
