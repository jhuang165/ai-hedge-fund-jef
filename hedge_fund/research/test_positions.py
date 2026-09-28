"""Position grammar and portfolio-file tests."""

import pytest

from hedge_fund.research.models import Position
from hedge_fund.research.positions import (
    ResearchTarget,
    load_portfolio,
    merge_targets,
    parse_target,
)


def test_bare_ticker():
    t = parse_target("aapl")
    assert t == ResearchTarget(ticker="AAPL")
    assert t.position is None


def test_ticker_with_position():
    t = parse_target("AAPL:100@150.25")
    assert t.ticker == "AAPL"
    assert t.position == Position(ticker="AAPL", shares=100.0, cost_basis=150.25)
    assert t.position.side == "long"


def test_short_and_fractional():
    t = parse_target("tsla:-12.5@240")
    assert t.position.shares == -12.5
    assert t.position.side == "short"


def test_share_classes_and_dots():
    assert parse_target("BRK.B:1@400").ticker == "BRK.B"


@pytest.mark.parametrize("bad", ["", "AAPL:", "AAPL:100", "AAPL@150", "AAPL:abc@1", "AAPL:1@-5", "123"])
def test_bad_grammar_raises(bad):
    with pytest.raises(ValueError, match="TICKER:SHARES@COST"):
        parse_target(bad)


def test_zero_shares_is_not_a_position():
    with pytest.raises(ValueError):
        parse_target("AAPL:0@10")


def test_load_portfolio(tmp_path):
    f = tmp_path / "pf.yaml"
    f.write_text(
        "positions:\n"
        "  - ticker: aapl\n    shares: 100\n    cost_basis: 150.25\n"
        "  - ticker: MSFT\n    shares: -20\n    cost_basis: 410\n"
    )
    targets = load_portfolio(f)
    assert [t.ticker for t in targets] == ["AAPL", "MSFT"]
    assert targets[1].position.side == "short"


def test_load_portfolio_bare_list(tmp_path):
    f = tmp_path / "pf.yaml"
    f.write_text("- {ticker: NVDA, shares: 5, cost_basis: 100}\n")
    assert load_portfolio(f)[0].ticker == "NVDA"


def test_load_portfolio_rejects_unknown_keys(tmp_path):
    f = tmp_path / "pf.yaml"
    f.write_text("positions:\n  - {ticker: NVDA, shares: 5, cost: 100}\n")
    with pytest.raises(ValueError):
        load_portfolio(f)


def test_empty_portfolio(tmp_path):
    f = tmp_path / "pf.yaml"
    f.write_text("")
    assert load_portfolio(f) == []


def test_merge_keeps_order_and_lets_positions_override():
    from_file = [parse_target("AAPL:10@100"), parse_target("MSFT:5@300")]
    given = [parse_target("NVDA"), parse_target("AAPL:20@120"), parse_target("MSFT")]
    merged = merge_targets(from_file, given)
    assert [t.ticker for t in merged] == ["AAPL", "MSFT", "NVDA"]
    assert merged[0].position.shares == 20       # positional position overrides the file
    assert merged[1].position.shares == 5        # bare ticker does not erase a file position
    assert merged[2].position is None
