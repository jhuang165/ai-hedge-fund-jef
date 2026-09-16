"""diagnose() tests — fake LLM, fake data client, fake search client. No network."""

import json

import pytest

from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import FinancialMetrics
from hedge_fund.llm import PromptCache
from hedge_fund.llm.client import LLMParseError
from hedge_fund.research.diagnose import DEFAULT_QUERIES, diagnose
from hedge_fund.research.models import ResearchReport
from hedge_fund.research.search import SearchClientError, SearchResult

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    """Canned-response LLM; counts calls; can raise instead."""

    model = "fake-model"

    def __init__(self, response="", error=None):
        self._response = response
        self._error = error
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


class MockDataClient:
    def __init__(self, metrics=None, error=None):
        self._metrics = metrics or []
        self._error = error

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        if self._error is not None:
            raise self._error
        return self._metrics

    def get_company_facts(self, ticker):
        return None


class FakeSearchClient:
    """Canned results keyed by the exact query string; a query not present
    in results_by_query and not in errors_by_query returns []."""

    def __init__(self, results_by_query=None, errors_by_query=None):
        self._results = results_by_query or {}
        self._errors = errors_by_query or {}
        self.calls: list[str] = []

    def search(self, query, max_results=5):
        self.calls.append(query)
        if query in self._errors:
            raise self._errors[query]
        return self._results.get(query, [])


def _history(n=8):
    quarters = ["2024-12-31", "2024-09-30", "2024-06-30", "2024-03-31",
                "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"]
    return [
        FinancialMetrics(
            ticker="TEST", report_period=q, period="ttm", filing_date=q,
            return_on_equity=0.2, gross_margin=0.4, book_value_per_share=10.0,
            market_cap=1e9,
        )
        for q in quarters[:n]
    ]


def _result(query, url="https://a.example/1"):
    return SearchResult(query=query, title="headline", url=url, content="body text")


def _queries(ticker):
    return [q.format(ticker=ticker) for q in DEFAULT_QUERIES]


BULLISH = json.dumps({
    "signal": "bullish", "confidence": 80, "thesis": "Strong quarter.\nMore detail.",
    "catalysts": [{"text": "beat estimates", "source_indices": [0]}],
    "risks": [],
})


def _search_client_all_queries(ticker):
    return FakeSearchClient({q: [_result(q)] for q in _queries(ticker)})


def _diagnose(tmp_path, llm, data=None, search=None, **kwargs):
    return diagnose(
        "TEST", "2025-01-15",
        data if data is not None else MockDataClient(metrics=_history()),
        search if search is not None else _search_client_all_queries("TEST"),
        llm,
        cache=PromptCache(tmp_path / "llm"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_builds_report(tmp_path):
    report = _diagnose(tmp_path, FakeLLM(BULLISH))

    assert isinstance(report, ResearchReport)
    assert report.ticker == "TEST"
    assert report.signal == "bullish"
    assert report.confidence == 80
    assert report.catalysts[0].source_indices == [0]
    assert report.snapshot_hash is not None
    assert len(report.queries) == len(DEFAULT_QUERIES)
    assert report.warnings == []


def test_source_dedupe_by_url(tmp_path):
    same_url = "https://dup.example/1"
    search = FakeSearchClient({q: [_result(q, same_url)] for q in _queries("TEST")})

    report = _diagnose(tmp_path, FakeLLM(BULLISH), search=search)

    assert len(report.sources) == 1


# ---------------------------------------------------------------------------
# Fundamentals degrade gracefully; infra failures propagate
# ---------------------------------------------------------------------------

def test_insufficient_fundamentals_degrades_to_search_only(tmp_path):
    report = _diagnose(tmp_path, FakeLLM(BULLISH), data=MockDataClient(metrics=_history(2)))

    assert report.snapshot_hash is None
    assert any("fundamentals unavailable" in w for w in report.warnings)


def test_fd_client_error_propagates(tmp_path):
    data = MockDataClient(error=FDClientError("API down", status_code=500))
    with pytest.raises(FDClientError):
        _diagnose(tmp_path, FakeLLM(BULLISH), data=data)


# ---------------------------------------------------------------------------
# Search failure contract
# ---------------------------------------------------------------------------

def test_one_query_failure_is_tolerated(tmp_path):
    queries = _queries("TEST")
    failing, ok = queries[0], queries[1:]
    search = FakeSearchClient(
        results_by_query={q: [_result(q)] for q in ok},
        errors_by_query={failing: SearchClientError("boom")},
    )

    report = _diagnose(tmp_path, FakeLLM(BULLISH), search=search)

    assert len(report.queries) == len(ok)
    assert any("query failed" in w for w in report.warnings)


def test_all_queries_failing_raises(tmp_path):
    queries = _queries("TEST")
    search = FakeSearchClient(errors_by_query={q: SearchClientError("boom") for q in queries})

    with pytest.raises(SearchClientError):
        _diagnose(tmp_path, FakeLLM(BULLISH), search=search)


# ---------------------------------------------------------------------------
# LLM failure contract — propagates, unlike LLMAgent's abstain
# ---------------------------------------------------------------------------

def test_llm_call_failure_propagates(tmp_path):
    with pytest.raises(TimeoutError):
        _diagnose(tmp_path, FakeLLM(error=TimeoutError("llm timed out")))


def test_malformed_json_raises_and_persists_audit_record(tmp_path):
    with pytest.raises(LLMParseError):
        _diagnose(tmp_path, FakeLLM("not json at all"))

    records = list((tmp_path / "llm").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert "parse_error" in record
    assert record["response"] == "not json at all"


def test_invalid_signal_raises(tmp_path):
    bad = json.dumps({"signal": "very bullish", "confidence": 80, "thesis": "x",
                       "catalysts": [], "risks": []})
    with pytest.raises(ValueError):
        _diagnose(tmp_path, FakeLLM(bad))


def test_out_of_range_source_index_raises(tmp_path):
    bad = json.dumps({
        "signal": "bullish", "confidence": 80, "thesis": "x",
        "catalysts": [{"text": "c", "source_indices": [99]}], "risks": [],
    })
    with pytest.raises(ValueError):
        _diagnose(tmp_path, FakeLLM(bad))


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_cache_hit_skips_llm_call(tmp_path):
    llm = FakeLLM(BULLISH)
    cache = PromptCache(tmp_path / "llm")
    data = MockDataClient(metrics=_history())
    search = _search_client_all_queries("TEST")

    first = diagnose("TEST", "2025-01-15", data, search, llm, cache=cache)
    second = diagnose("TEST", "2025-01-15", data, search, llm, cache=cache)

    assert llm.calls == 1
    assert first.signal == second.signal
    assert first.prompt_key == second.prompt_key
