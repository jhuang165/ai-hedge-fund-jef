"""Report store tests — one file per report, corrupt files skipped."""

from hedge_fund.research.models import ResearchReport
from hedge_fund.research.store import load_reports, save_report


def _report(ticker, as_of="2025-06-30"):
    return ResearchReport(ticker=ticker, as_of=as_of, model="m", signal="bullish",
                          confidence=70, action="buy", action_rationale="r",
                          thesis="t", prompt_key="k")


def test_round_trip_and_ordering(tmp_path):
    save_report(_report("MSFT", "2025-06-30"), tmp_path)
    save_report(_report("AAPL", "2025-01-02"), tmp_path)
    save_report(_report("AAPL", "2025-06-30"), tmp_path)
    loaded = load_reports(tmp_path)
    assert [(r.as_of, r.ticker) for r in loaded] == [
        ("2025-01-02", "AAPL"), ("2025-06-30", "AAPL"), ("2025-06-30", "MSFT")]
    assert loaded[0].action == "buy"


def test_corrupt_file_is_skipped(tmp_path):
    save_report(_report("AAPL"), tmp_path)
    (tmp_path / "2025-06-30_BAD_000000.json").write_text("{not json")
    assert [r.ticker for r in load_reports(tmp_path)] == ["AAPL"]


def test_missing_directory_is_empty(tmp_path):
    assert load_reports(tmp_path / "nope") == []
