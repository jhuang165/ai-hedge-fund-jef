import json
from datetime import datetime, timezone

import pytest

from hedge_fund.reporter.brief import (
    Brief,
    repair_json_text,
    CodexClient,
    LLMError,
    build_prompt,
    parse_brief_json,
    produce_brief,
    render_previous,
)
from hedge_fund.reporter.evidence import Evidence, Headline, Quote

NOW = datetime(2026, 10, 9, 15, 30, tzinfo=timezone.utc)


def evidence(ticker="NVDA") -> Evidence:
    return Evidence(
        ticker=ticker, as_of="2026-10-09", gathered_at="2026-10-09T15:29:00+00:00",
        quote=Quote(price=185.5, previous_close=180.0, change_pct=185.5 / 180 - 1, as_of="2026-10-09T15:29:00+00:00"),
        headlines=[Headline(title="Nvidia rallies", url="https://n.example/1", published="2026-10-09T13:00+00:00",
                            source="Reuters", feed="google")],
        warnings=["fundamentals unavailable: thin"],
    )


GOOD = {
    "signal": "Bullish", "confidence": 71, "action": "buy",
    "headline": "Rallies on d-Matrix stake",
    "whats_new": ["Stake reported (source 0)"],
    "summary": "Para one.\n\nPara two.",
    "catalysts": [{"text": "Stake", "source_indices": [0]}, {"text": "bad idx", "source_indices": [5, "x"]}, "plain string"],
    "risks": [],
    "watch_next": ["Earnings Nov 19"],
    "sources": [{"title": "Info", "url": "https://n.example/1", "published": "2026-10-09"}, "https://n.example/2", {"bad": 1}],
    "nothing_new": False,
}


def test_parse_brief_json_normalizes():
    data = parse_brief_json("Here you go:\n```json\n" + json.dumps(GOOD) + "\n```")
    assert data["signal"] == "bullish" and data["confidence"] == 71.0 and data["action"] == "buy"
    assert [s.url for s in data["sources"]] == ["https://n.example/1", "https://n.example/2"]
    assert data["catalysts"][1].source_indices == []          # out-of-range dropped, not fatal
    assert data["catalysts"][2].text == "plain string"
    assert data["summary"] == "Para one.\n\nPara two."


def test_parse_brief_json_fills_action_by_policy():
    d = parse_brief_json(json.dumps({"signal": "bullish", "confidence": 55}))
    assert d["action"] == "watch"
    d = parse_brief_json(json.dumps({"signal": "bullish", "confidence": 60}))
    assert d["action"] == "buy"
    d = parse_brief_json(json.dumps({"signal": "bearish", "confidence": 10}))
    assert d["action"] == "avoid"


@pytest.mark.parametrize("bad", [
    "no json here",
    json.dumps({"signal": "maybe", "confidence": 50}),
    json.dumps({"signal": "bullish", "confidence": 101}),
    json.dumps({"signal": "bullish", "confidence": "high"}),
    json.dumps({"signal": "bullish", "confidence": 70, "action": "add"}),
    "[1, 2]",
])
def test_parse_brief_json_rejects(bad):
    with pytest.raises(ValueError):
        parse_brief_json(bad)


def test_build_prompt_contains_evidence_and_previous():
    prev = {"generated_at": "2026-10-09T14:30:00+00:00", "signal": "neutral", "confidence": 50.0, "action": "watch",
            "headline": "Quiet", "quote": {"price": 180.0, "as_of": "t0"}, "whats_new": ["a"], "watch_next": ["b"]}
    system, user = build_prompt(evidence(), prev, NOW)
    assert "web search tool" in system
    assert "Ticker: NVDA" in user and "2026-10-09T15:30" in user
    assert "185.50" in user and "[H0]" in user and "Nvidia rallies" in user
    assert "Previous brief (written 2026-10-09T14:30:00+00:00): neutral 50, action watch." in user
    assert "Price then: 180.00" in user and "You said to watch: b" in user
    assert "Data warnings: fundamentals unavailable: thin" in user
    assert "first brief on the name" in render_previous(None)


class FakeClient(CodexClient):
    def __init__(self, answer):
        super().__init__("http://unused")
        self.answer = answer
        self.calls = []

    def complete(self, system, user, model):
        self.calls.append((system, user, model))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_produce_brief_attaches_evidence_and_previous():
    client = FakeClient(json.dumps(GOOD))
    prev = Brief(ticker="NVDA", generated_at="2026-10-09T14:30:00+00:00", model="m", signal="neutral",
                 confidence=50, action="watch", headline="Quiet", quote=Quote())
    b = produce_brief(evidence(), prev, client, "gpt-6-luna", now=NOW)
    assert b.model == "gpt-6-luna" and b.generated_at == "2026-10-09T15:30:00+00:00"
    assert b.signal == "bullish" and b.action == "buy" and b.score == 71.0
    assert b.quote.price == 185.5 and b.headlines[0].title == "Nvidia rallies"
    assert b.prev_signal == "neutral" and b.prev_confidence == 50 and b.prev_generated_at == "2026-10-09T14:30:00+00:00"
    assert b.warnings == ["fundamentals unavailable: thin"]
    assert client.calls[0][2] == "gpt-6-luna"
    # Round-trips through JSON (what the store keeps).
    assert Brief(**json.loads(b.model_dump_json())).headline == "Rallies on d-Matrix stake"


def test_produce_brief_propagates_bad_answers():
    with pytest.raises(ValueError):
        produce_brief(evidence(), None, FakeClient("garbage"), "m", now=NOW)
    with pytest.raises(LLMError):
        produce_brief(evidence(), None, FakeClient(LLMError("dead")), "m", now=NOW)


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.posts.append((url, headers, json.loads(data), timeout))
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        return _Resp(200, {"data": [{"id": "gpt-6-luna"}]})


def test_codex_client_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("hedge_fund.reporter.brief.time.sleep", lambda s: None)
    session = _Session([_Resp(503, text="busy"), _Resp(200, {"choices": [{"message": {"content": "{}"}}]})])
    c = CodexClient("http://w/v1/", "k", timeout_s=5, reasoning_effort="high", session=session)
    assert c.complete("s", "u", "gpt-6-luna") == "{}"
    url, headers, body, timeout = session.posts[0]
    assert url == "http://w/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    assert body["model"] == "gpt-6-luna" and body["x_codex"] == {"reasoning_effort": "high"} and body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert timeout == 5
    assert c.models() == ["gpt-6-luna"] and c.healthy()[0] is True


def test_codex_client_gives_up(monkeypatch):
    monkeypatch.setattr("hedge_fund.reporter.brief.time.sleep", lambda s: None)
    session = _Session([_Resp(500), _Resp(500), _Resp(500)])
    c = CodexClient("http://w/v1", session=session, retries=2)
    with pytest.raises(LLMError):
        c.complete("s", "u", "m")
    assert len(session.posts) == 3


def test_parse_brief_json_repairs_trailing_commas_from_a_line_filter():
    # Pretty-printed JSON whose last array item was deleted in transit.
    damaged = """{
  "signal": "neutral",
  "confidence": 57,
  "whats_new": [
    "Board change announced (source 0)",
  ],
  "sources": [{"title": "t", "url": "https://x.example/1", "published": null},],
}"""
    d = parse_brief_json(damaged)
    assert d["signal"] == "neutral" and d["whats_new"] == ["Board change announced (source 0)"]
    assert d["sources"][0].url == "https://x.example/1"
    assert repair_json_text('{"a": [1, 2,], "b": {"c": 1,},}') == '{"a": [1, 2], "b": {"c": 1}}'


def test_system_prompt_demands_one_line_json():
    from hedge_fund.reporter.brief import SYSTEM_PROMPT
    assert "ONE line" in SYSTEM_PROMPT and "minified" in SYSTEM_PROMPT
