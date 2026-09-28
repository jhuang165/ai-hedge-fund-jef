"""blend_signals tests — pure math, hand-built signals."""

import pytest

from hedge_fund.models import Signal
from hedge_fund.portfolio.construction import blend_signals


def _sig(model, ticker, value, abstained=False):
    metadata = {"abstained": True} if abstained else {}
    return Signal(model_name=model, ticker=ticker, date="2024-06-03",
                  value=value, metadata=metadata)


def test_weighted_mean_with_unequal_weights():
    signals = [_sig("a", "AAPL", 1.0), _sig("b", "AAPL", 0.0)]
    result = blend_signals(signals, {"a": 3.0, "b": 1.0}, gross_target=1.0)
    assert result.convictions["AAPL"] == pytest.approx(0.75)  # (3*1 + 1*0) / 4


def test_abstain_excluded_from_denominator():
    """bullish + abstain must blend to fully bullish, not half."""
    signals = [_sig("a", "AAPL", 1.0), _sig("b", "AAPL", 0.0, abstained=True)]
    result = blend_signals(signals, {"a": 1.0, "b": 1.0}, gross_target=1.0)
    assert result.convictions["AAPL"] == pytest.approx(1.0)


def test_non_abstained_zero_dilutes():
    """A real neutral vote (e.g. PEAD outside its window) is a vote."""
    signals = [_sig("a", "AAPL", 1.0), _sig("b", "AAPL", 0.0)]
    result = blend_signals(signals, {"a": 1.0, "b": 1.0}, gross_target=1.0)
    assert result.convictions["AAPL"] == pytest.approx(0.5)


def test_weights_sum_to_gross_target():
    signals = [
        _sig("a", "AAPL", 0.8),
        _sig("a", "MSFT", -0.4),
        _sig("a", "NVDA", 0.2),
    ]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0)
    gross = sum(abs(w) for w in result.weights.values())
    assert gross == pytest.approx(1.0)
    assert result.weights["MSFT"] < 0  # bearish view -> negative weight


def test_market_neutral_sleeve_sums_to_zero():
    signals = [
        _sig("a", "AAPL", 1.0),
        _sig("a", "MSFT", 0.2),
        _sig("a", "NVDA", -0.6),
    ]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0, market_neutral=True)
    assert sum(result.weights.values()) == pytest.approx(0.0)  # dollar-neutral
    assert sum(abs(w) for w in result.weights.values()) == pytest.approx(1.0)
    # Ranking survives demeaning: best-liked long, least-liked short
    assert result.weights["AAPL"] > 0 > result.weights["NVDA"]
    # Raw convictions are reported un-demeaned (the audit trail keeps the views)
    assert result.convictions["MSFT"] == pytest.approx(0.2)


def test_market_neutral_uniform_views_go_flat():
    """If the desk likes everything equally, there is no relative view."""
    signals = [_sig("a", t, 0.8) for t in ("AAPL", "MSFT", "NVDA")]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0, market_neutral=True)
    assert all(w == 0.0 for w in result.weights.values())


def test_all_abstain_yields_flat_book():
    signals = [
        _sig("a", "AAPL", 0.0, abstained=True),
        _sig("b", "AAPL", 0.0, abstained=True),
    ]
    result = blend_signals(signals, {"a": 1.0, "b": 1.0}, gross_target=1.0)
    assert result.convictions == {"AAPL": 0.0}
    assert result.weights == {"AAPL": 0.0}


# ---------------------------------------------------------------------------
# Risk-scaled sizing
# ---------------------------------------------------------------------------

def test_vol_scaling_gives_equal_views_equal_risk():
    """Same conviction, 4x the vol -> a quarter of the dollars."""
    signals = [_sig("a", "AAPL", 1.0), _sig("a", "TSLA", 1.0)]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0,
                           vols={"AAPL": 0.20, "TSLA": 0.80})
    assert result.weights["AAPL"] == pytest.approx(0.8)
    assert result.weights["TSLA"] == pytest.approx(0.2)
    # Convictions are reported unscaled — the view is the view.
    assert result.convictions == {"AAPL": 1.0, "TSLA": 1.0}


def test_no_vols_is_plain_conviction_weighting():
    signals = [_sig("a", "AAPL", 1.0), _sig("a", "TSLA", 1.0)]
    for vols in (None, {}):
        result = blend_signals(signals, {"a": 1.0}, gross_target=1.0, vols=vols)
        assert result.weights == {"AAPL": pytest.approx(0.5), "TSLA": pytest.approx(0.5)}


def test_missing_vol_is_sized_as_typical():
    """No history for NVDA: it gets the median vol, not a free ride."""
    signals = [_sig("a", "AAPL", 1.0), _sig("a", "TSLA", 1.0), _sig("a", "NVDA", 1.0)]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0,
                           vols={"AAPL": 0.20, "TSLA": 0.80})
    # median vol 0.5: factors AAPL 2.5, TSLA 0.625, NVDA 1.0 -> normalized
    total = 2.5 + 0.625 + 1.0
    assert result.weights["AAPL"] == pytest.approx(2.5 / total)
    assert result.weights["TSLA"] == pytest.approx(0.625 / total)
    assert result.weights["NVDA"] == pytest.approx(1.0 / total)


def test_vol_floor_stops_a_sleepy_name_taking_the_book():
    signals = [_sig("a", "AAPL", 1.0), _sig("a", "XYZ", 1.0)]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0,
                           vols={"AAPL": 0.25, "XYZ": 0.0001})
    # XYZ floored to 5%: factor ratio 5:1, not 2500:1.
    assert result.weights["XYZ"] == pytest.approx(5 / 6)
    assert result.weights["AAPL"] == pytest.approx(1 / 6)


def test_market_neutral_after_vol_scaling_is_still_dollar_neutral():
    signals = [_sig("a", "AAPL", 1.0), _sig("a", "TSLA", 0.5), _sig("a", "MSFT", -1.0)]
    result = blend_signals(signals, {"a": 1.0}, gross_target=1.0, market_neutral=True,
                           vols={"AAPL": 0.20, "TSLA": 0.80, "MSFT": 0.30})
    assert sum(result.weights.values()) == pytest.approx(0.0, abs=1e-12)
    assert sum(abs(w) for w in result.weights.values()) == pytest.approx(1.0)
