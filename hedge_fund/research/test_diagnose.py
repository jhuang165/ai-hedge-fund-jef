"""diagnose() tests — fake LLM, fake data client, fake search client. No network."""

import json

import pytest

from hedge_fund.data.client import FDClientError
from hedge_fund.data.models import FinancialMetrics
from hedge_fund.features.technicals import YEAR
from hedge_fund.features.test_technicals import bars, linear
from hedge_fund.llm import PromptCache
from hedge_fund.llm.client import LLMParseError
from hedge_fund.models import Signal
from hedge_fund.research.diagnose import DEFAULT_QUERIES, diagnose
from hedge_fund.research.models import Position, ResearchReport
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
    """Fundamentals, a year of rising bars, no insider trades, no earnings."""

    def __init__(self, metrics=None, error=None, prices=None):
        self._metrics = metrics or []
        self._error = error
        self._prices = bars(linear(YEAR + 40), end="2025-01-15") if prices is None else prices

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        if self._error is not None:
            raise self._error
        return self._metrics

    def get_company_facts(self, ticker):
        return None

    def get_prices(self, ticker, start_date, end_date, **kw):
        return [p for p in self._prices if start_date <= p.time[:10] <= end_date]

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        return []

    def get_earnings_history(self, ticker, limit=12):
        return []


class FakeModel:
    """A stand-in quant model with a fixed view."""

    def __init__(self, name="fake", value=0.5, abstain=False):
        self._name, self._value, self._abstain = name, value, abstain
        self.calls = []

    @property
    def name(self):
        return self._name

    def predict(self, ticker, date, data_client):
        self.calls.append((ticker, date))
        meta = {"abstained": True, "abstain_reason": "no data"} if self._abstain else {}
        return Signal(model_name=self._name, ticker=ticker, date=date,
                      value=0.0 if self._abstain else self._value,
                      reasoning="fixed view", metadata=meta)


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


def _response(**overrides):
    data = {
        "signal": "bullish", "confidence": 80, "action": "buy",
        "action_rationale": "cheap and improving",
        "thesis": "Strong quarter.\nMore detail.",
        "catalysts": [{"text": "beat estimates", "source_indices": [0]}],
        "risks": [],
    }
    data.update(overrides)
    return json.dumps(data)


BULLISH = _response()


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
    assert report.action == "buy"
    assert report.action_rationale == "cheap and improving"
    assert report.snapshot_hash is not None
    assert report.technicals is not None
    assert report.position is None
    assert len(report.queries) == len(DEFAULT_QUERIES)
    assert report.warnings == []
    # Every registered quant model was consulted.
    assert {s.model_name for s in report.desk} >= {"momentum", "mean-reversion", "insider-flow", "quality-value", "pead"}


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
    with pytest.raises(ValueError):
        _diagnose(tmp_path, FakeLLM(_response(signal="very bullish")))


def test_out_of_range_source_index_raises(tmp_path):
    bad = _response(catalysts=[{"text": "c", "source_indices": [99]}])
    with pytest.raises(ValueError):
        _diagnose(tmp_path, FakeLLM(bad))


# ---------------------------------------------------------------------------
# Actions and positions
# ---------------------------------------------------------------------------

def test_held_action_on_a_not_held_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not-held"):
        _diagnose(tmp_path, FakeLLM(_response(action="trim")))


def test_flat_action_on_a_held_name_is_rejected(tmp_path):
    position = Position(ticker="TEST", shares=100, cost_basis=90.0)
    with pytest.raises(ValueError, match="held"):
        _diagnose(tmp_path, FakeLLM(_response(action="buy")), position=position)


def test_missing_action_is_rejected(tmp_path):
    raw = json.loads(_response())
    del raw["action"]
    with pytest.raises(ValueError, match="invalid action"):
        _diagnose(tmp_path, FakeLLM(json.dumps(raw)))


def test_position_is_marked_and_prompted(tmp_path):
    llm = FakeLLM(_response(action="hold"))
    position = Position(ticker="TEST", shares=100, cost_basis=90.0)
    data = MockDataClient(metrics=_history())

    report = _diagnose(tmp_path, llm, data=data, position=position)

    assert report.action == "hold"
    assert report.position is not None
    assert report.position.side == "long"
    assert report.position.last_close == report.technicals.last_close
    assert report.position.unrealized_pnl_pct == pytest.approx(
        (report.technicals.last_close - 90.0) / 90.0)
    # The prompt tells the model what is held and which actions apply.
    record = json.loads(next((tmp_path / "llm").glob("*.json")).read_text())
    assert "long 100 shares" in record["user"]
    assert "add, hold, trim, exit" in record["user"]


def test_short_position_pnl_is_signed_by_side(tmp_path):
    position = Position(ticker="TEST", shares=-50, cost_basis=200.0)
    report = _diagnose(tmp_path, FakeLLM(_response(action="exit")), position=position)
    close = report.technicals.last_close
    assert report.position.side == "short"
    assert report.position.market_value == pytest.approx(-50 * close)
    assert report.position.unrealized_pnl == pytest.approx(-50 * (close - 200.0))


def test_position_without_price_history_is_unmarked(tmp_path):
    position = Position(ticker="TEST", shares=10, cost_basis=5.0)
    data = MockDataClient(metrics=_history(), prices=[])
    report = _diagnose(tmp_path, FakeLLM(_response(action="hold")), data=data, position=position)
    assert report.technicals is None
    assert report.position.last_close is None
    assert report.position.unrealized_pnl_pct is None
    assert any("price history unavailable" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Desk readout
# ---------------------------------------------------------------------------

def test_desk_readout_goes_into_the_prompt(tmp_path):
    bull = FakeModel("bull", 0.7)
    silent = FakeModel("silent", abstain=True)
    report = _diagnose(tmp_path, FakeLLM(BULLISH), desk=[bull, silent])

    assert [s.model_name for s in report.desk] == ["bull", "silent"]
    assert bull.calls == [("TEST", "2025-01-15")]
    record = json.loads(next((tmp_path / "llm").glob("*.json")).read_text())
    assert "bull: +0.70 — fixed view" in record["user"]
    assert "silent: abstained — no data" in record["user"]
    assert "Price action for TEST" in record["user"]
    assert len(record["desk"]) == 2


def test_empty_desk_is_allowed(tmp_path):
    report = _diagnose(tmp_path, FakeLLM(BULLISH), desk=[])
    assert report.desk == []


def test_reports_rank_by_signed_confidence(tmp_path):
    bull = _diagnose(tmp_path, FakeLLM(_response(signal="bullish", confidence=60)))
    bear = _diagnose(tmp_path, FakeLLM(_response(signal="bearish", confidence=90, action="avoid")))
    flat = _diagnose(tmp_path, FakeLLM(_response(signal="neutral", confidence=90, action="watch")))
    assert sorted([bull, bear, flat], key=lambda r: r.score, reverse=True) == [bull, flat, bear]


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
