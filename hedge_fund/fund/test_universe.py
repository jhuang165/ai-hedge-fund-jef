"""DatedUniverse tests — point-in-time membership and the survivorship warning."""

import pytest

from hedge_fund.fund.universe import (
    DatedUniverse,
    load_universe,
    normalize_tickers,
    survivorship_warning,
)

ENTRIES = [
    {"from": "2020-01-01", "tickers": ["aapl", "GE", "XOM"]},
    {"from": "2022-06-01", "tickers": ["AAPL", "XOM", "NVDA"]},
]


def test_membership_as_of_each_date():
    u = DatedUniverse(entries=ENTRIES)
    assert u.as_of("2020-01-01") == ["AAPL", "GE", "XOM"]
    assert u.as_of("2022-05-31") == ["AAPL", "GE", "XOM"]
    assert u.as_of("2022-06-01") == ["AAPL", "XOM", "NVDA"]
    assert u.as_of("2030-01-01") == ["AAPL", "XOM", "NVDA"]


def test_before_first_entry_raises():
    u = DatedUniverse(entries=ENTRIES)
    with pytest.raises(ValueError, match="no membership as of 2019-12-31"):
        u.as_of("2019-12-31")


def test_entries_sorted_and_union_in_first_appearance_order():
    u = DatedUniverse(entries=list(reversed(ENTRIES)))
    assert u.start == "2020-01-01"
    assert u.tickers == ["AAPL", "GE", "XOM", "NVDA"]


def test_duplicate_dates_and_typos_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        DatedUniverse(entries=[ENTRIES[0], {**ENTRIES[1], "from": "2020-01-01"}])
    with pytest.raises(ValueError):
        DatedUniverse(entries=[{"from": "2020-01-01", "tikcers": ["AAPL"]}])
    with pytest.raises(ValueError):
        DatedUniverse(entries=[{"from": "2020-01-01", "tickers": []}])


def test_load_universe_accepts_list_or_mapping(tmp_path):
    as_list = tmp_path / "list.yaml"
    as_list.write_text("- from: 2020-01-01\n  tickers: [AAPL, GE]\n- from: 2021-01-01\n  tickers: [AAPL]\n")
    as_map = tmp_path / "map.yaml"
    as_map.write_text("entries:\n  - from: 2020-01-01\n    tickers: [AAPL, GE]\n")
    assert load_universe(as_list).as_of("2021-06-01") == ["AAPL"]
    assert load_universe(as_map).tickers == ["AAPL", "GE"]


def test_from_accepts_date_objects_and_rejects_bad_strings():
    from datetime import date
    u = DatedUniverse(entries=[{"from": date(2020, 1, 1), "tickers": ["AAPL"]}])
    assert u.start == "2020-01-01"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        DatedUniverse(entries=[{"from": "01/01/2020", "tickers": ["AAPL"]}])


def test_normalize_tickers():
    assert normalize_tickers([" aapl", "AAPL", "msft "]) == ["AAPL", "MSFT"]
    with pytest.raises(ValueError, match="empty"):
        normalize_tickers(["", " "])


def test_survivorship_warning_only_for_static_lists():
    assert survivorship_warning(DatedUniverse(entries=ENTRIES), "2020-01-03") is None
    w = survivorship_warning(["AAPL", "MSFT"], "2020-01-03")
    assert "Survivorship" in w and "AAPL, MSFT" in w and "2020-01-03" in w
    many = survivorship_warning([f"T{i}" for i in range(9)], "2020-01-03")
    assert "T5, …" in many and "T6" not in many
