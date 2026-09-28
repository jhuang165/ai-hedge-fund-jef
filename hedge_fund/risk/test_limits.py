"""apply_limits tests — pure math."""

import pytest

from hedge_fund.risk.limits import RiskLimits, apply_limits

LIMITS = RiskLimits(max_position_pct=0.25, max_gross_exposure=1.0)


def test_position_clamp_records_event():
    result = apply_limits({"AAPL": 0.6, "MSFT": 0.2}, LIMITS)
    assert result.weights["AAPL"] == pytest.approx(0.25)
    assert result.weights["MSFT"] == pytest.approx(0.2)  # untouched
    assert len(result.clamps) == 1
    clamp = result.clamps[0]
    assert clamp.limit == "max_position_pct"
    assert clamp.ticker == "AAPL"
    assert clamp.before == pytest.approx(0.6)
    assert clamp.after == pytest.approx(0.25)


def test_gross_clamp_scales_all_and_records_one_event():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=1.0)
    result = apply_limits({"AAPL": 0.8, "MSFT": 0.8}, limits)
    assert result.weights["AAPL"] == pytest.approx(0.5)
    assert result.weights["MSFT"] == pytest.approx(0.5)
    assert len(result.clamps) == 1
    assert result.clamps[0].limit == "max_gross_exposure"
    assert result.clamps[0].ticker is None
    assert result.clamps[0].before == pytest.approx(1.6)


def test_position_then_gross_never_reviolates():
    weights = {t: 0.5 for t in ["A", "B", "C", "D", "E", "F"]}  # gross 3.0
    result = apply_limits(weights, LIMITS)
    # Position cap first (0.5 -> 0.25 each, gross 1.5), then gross scale to 1.0.
    for w in result.weights.values():
        assert abs(w) <= LIMITS.max_position_pct + 1e-12
    gross = sum(abs(w) for w in result.weights.values())
    assert gross == pytest.approx(1.0)
    kinds = [c.limit for c in result.clamps]
    assert kinds.count("max_position_pct") == 6
    assert kinds.count("max_gross_exposure") == 1


def test_within_limits_passes_through_untouched():
    weights = {"AAPL": 0.2, "MSFT": -0.1}
    result = apply_limits(weights, LIMITS)
    assert result.weights == weights
    assert result.clamps == []


def test_shorts_clamped_by_absolute_value():
    result = apply_limits({"AAPL": -0.6}, LIMITS)
    assert result.weights["AAPL"] == pytest.approx(-0.25)


def test_clamped_exposure_not_redistributed():
    """Risk only shrinks; freed exposure stays as cash."""
    result = apply_limits({"AAPL": 0.9, "MSFT": 0.05}, LIMITS)
    assert result.weights["AAPL"] == pytest.approx(0.25)
    assert result.weights["MSFT"] == pytest.approx(0.05)  # NOT topped up


# ---------------------------------------------------------------------------
# Net exposure cap
# ---------------------------------------------------------------------------

def test_net_cap_shrinks_the_long_side_only():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=2.0, max_net_exposure=0.2)
    result = apply_limits({"A": 0.5, "B": 0.3, "C": -0.2}, limits)  # net +0.6
    # Longs scaled by (0.2 + 0.2) / 0.8 = 0.5; the short is untouched.
    assert result.weights == {"A": pytest.approx(0.25), "B": pytest.approx(0.15),
                              "C": pytest.approx(-0.2)}
    assert sum(result.weights.values()) == pytest.approx(0.2)
    (clamp,) = result.clamps
    assert clamp.limit == "max_net_exposure"
    assert clamp.before == pytest.approx(0.6) and clamp.after == pytest.approx(0.2)


def test_net_cap_shrinks_the_short_side_when_net_short():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=2.0, max_net_exposure=0.1)
    result = apply_limits({"A": 0.2, "B": -0.6}, limits)  # net -0.4
    assert result.weights["A"] == pytest.approx(0.2)
    assert result.weights["B"] == pytest.approx(-0.3)   # (0.1 + 0.2) / 0.6 = 0.5
    assert sum(result.weights.values()) == pytest.approx(-0.1)
    assert result.clamps[0].after == pytest.approx(-0.1)


def test_net_cap_zero_forces_dollar_neutral():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=2.0, max_net_exposure=0.0)
    result = apply_limits({"A": 0.6, "B": -0.2}, limits)
    assert sum(result.weights.values()) == pytest.approx(0.0, abs=1e-12)
    assert result.weights == {"A": pytest.approx(0.2), "B": pytest.approx(-0.2)}


def test_net_cap_within_limit_is_silent():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=2.0, max_net_exposure=0.5)
    result = apply_limits({"A": 0.4, "B": -0.1}, limits)
    assert result.clamps == []


def test_net_then_gross_keeps_both():
    limits = RiskLimits(max_position_pct=1.0, max_gross_exposure=1.0, max_net_exposure=0.2)
    result = apply_limits({"A": 1.0, "B": -0.6}, limits)  # net 0.4, gross 1.6
    net = sum(result.weights.values())
    gross = sum(abs(w) for w in result.weights.values())
    assert gross == pytest.approx(1.0)
    assert abs(net) <= 0.2 + 1e-12
    assert [c.limit for c in result.clamps] == ["max_net_exposure", "max_gross_exposure"]


# ---------------------------------------------------------------------------
# Drawdown kill-switch
# ---------------------------------------------------------------------------

KILL = RiskLimits(max_position_pct=0.25, max_gross_exposure=1.0, max_drawdown_pct=0.2)


def test_kill_switch_flattens_everything():
    result = apply_limits({"A": 0.6, "B": -0.2}, KILL, drawdown=0.25)
    assert result.weights == {"A": 0.0, "B": 0.0}
    assert result.killed
    (clamp,) = result.clamps
    assert clamp.limit == "max_drawdown_pct"
    assert clamp.before == pytest.approx(0.8) and clamp.after == 0.0


def test_kill_switch_trips_exactly_at_the_limit():
    assert apply_limits({"A": 0.1}, KILL, drawdown=0.2).killed
    assert not apply_limits({"A": 0.1}, KILL, drawdown=0.1999).killed


def test_kill_switch_needs_a_drawdown_to_judge():
    assert not apply_limits({"A": 0.6}, KILL, drawdown=None).killed
    assert not apply_limits({"A": 0.6}, LIMITS, drawdown=0.9).killed  # no limit set


def test_killed_result_does_not_report_other_clamps():
    result = apply_limits({"A": 0.9}, KILL, drawdown=0.5)  # would also breach position cap
    assert [c.limit for c in result.clamps] == ["max_drawdown_pct"]
