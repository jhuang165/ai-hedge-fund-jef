"""TavilyClient contract tests — mocked HTTP, no API key required.

Mirrors hedge_fund/data/test_client_contract.py's pattern: fail-loud on
infrastructure failures, retry-then-raise on exhausted 429s, and a search
that finds nothing is a normal success (empty list), not a failure.
"""

import pytest
import requests

from hedge_fund.research.search import SearchClientError, TavilyClient


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


@pytest.fixture
def client():
    c = TavilyClient(api_key="test-key")
    yield c
    c.close()


def _stub(client, responses):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    client._session.request = fake_request
    return calls


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------

def test_search_returns_results(client):
    _stub(client, [_FakeResponse(200, {"results": [
        {"title": "Apple beats Q3", "url": "https://a.example/1",
         "content": "Apple reported...", "score": 0.9, "published_date": "2026-09-10"},
    ]})])

    results = client.search("AAPL stock news")

    assert len(results) == 1
    r = results[0]
    assert r.query == "AAPL stock news"
    assert r.title == "Apple beats Q3"
    assert r.url == "https://a.example/1"
    assert r.score == 0.9
    assert r.published_date == "2026-09-10"


def test_empty_results_returns_empty_list(client):
    """Zero hits is a normal successful search outcome, not a failure."""
    _stub(client, [_FakeResponse(200, {"results": []})])
    assert client.search("ZZZZ stock news") == []


# ---------------------------------------------------------------------------
# Fail-loud contract
# ---------------------------------------------------------------------------

def test_http_401_raises(client):
    _stub(client, [_FakeResponse(401, text="bad key")])
    with pytest.raises(SearchClientError) as exc_info:
        client.search("AAPL stock news")
    assert exc_info.value.status_code == 401


def test_http_500_raises(client):
    _stub(client, [_FakeResponse(500, text="internal error")])
    with pytest.raises(SearchClientError) as exc_info:
        client.search("AAPL stock news")
    assert exc_info.value.status_code == 500


def test_network_error_raises(client):
    _stub(client, [requests.ConnectionError("boom")])
    with pytest.raises(SearchClientError):
        client.search("AAPL stock news")


def test_429_retries_then_raises_when_exhausted(client, monkeypatch):
    monkeypatch.setattr("hedge_fund.research.search.time.sleep", lambda s: None)
    _stub(client, [_FakeResponse(429)] * (len(TavilyClient._RETRY_DELAYS) + 1))
    with pytest.raises(SearchClientError) as exc_info:
        client.search("AAPL stock news")
    assert exc_info.value.status_code == 429


def test_429_then_success_recovers(client, monkeypatch):
    monkeypatch.setattr("hedge_fund.research.search.time.sleep", lambda s: None)
    _stub(client, [
        _FakeResponse(429),
        _FakeResponse(200, {"results": [
            {"title": "t", "url": "https://a.example/1", "content": "c"},
        ]}),
    ])
    results = client.search("AAPL stock news")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------

def test_search_sends_expected_body(client):
    calls = _stub(client, [_FakeResponse(200, {"results": []})])
    client.search("AAPL stock news", max_results=3)

    call = calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.tavily.com/search"
    assert call["json"] == {
        "query": "AAPL stock news",
        "max_results": 3,
        "search_depth": "basic",
        "topic": "finance",
        "time_range": "month",
    }


def test_auth_header_uses_bearer_token():
    client = TavilyClient(api_key="my-key")
    assert client._session.headers["Authorization"] == "Bearer my-key"
    client.close()
