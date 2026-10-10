"""Dossier tests — the research pipeline with the reasoner outside the
process. No LLM, no search client, no network."""

import json

import pytest

from hedge_fund.research.dossier import (
    CONVICTION_THRESHOLD,
    SYSTEM_PROMPT,
    Dossier,
    assemble_report,
    build_dossier,
    derive_action,
)
from hedge_fund.research.models import Position, PositionMark, ResearchReport, SearchResult
from hedge_fund.research.test_diagnose import FakeModel, MockDataClient, _history

AS_OF = "2025-01-15"


def _sources(n=2):
    return [SearchResult(query="T stock news", title=f"s{i}", url=f"https://x/{i}", content="…")
            for i in range(n)]


def _answer(**over):
    base = {
        "signal": "bullish", "confidence": 72,
        "action_rationale": "Momentum and quality agree. A guidance cut would change my mind.",
        "thesis": "A thesis.",
        "catalysts": [{"text": "c", "source_indices": [0]}],
        "risks": [{"text": "r", "source_indices": [1]}],
    }
    return {**base, **over}


# ---------------------------------------------------------------------------
# build_dossier
# ---------------------------------------------------------------------------

def test_dossier_gathers_evidence_and_renders_the_same_prompt():
    client = MockDataClient(metrics=_history())
    model = FakeModel("mom", 0.6)
    d = build_dossier("T", AS_OF, client, desk=[model])

    assert d.snapshot is not None and d.technicals is not None
    assert [s.model_name for s in d.desk] == ["mom"] and model.calls == [("T", AS_OF)]
    assert not d.held and d.allowed_actions == ("buy", "watch", "avoid")
    assert d.queries == ["T stock news", "T earnings report",
                         "T analyst rating price target", "T stock risks controversy lawsuit"]
    prompt = d.render()
    assert "mom: +0.60" in prompt and "Sources: none returned." in prompt
    assert "Allowed actions: buy, watch, avoid" in prompt
    with_sources = d.render(_sources(1))
    assert "[0] (undated) s0" in with_sources


def test_dossier_marks_a_position_and_switches_vocabulary():
    client = MockDataClient(metrics=_history())
    d = build_dossier("T", AS_OF, client, position=Position(ticker="T", shares=10, cost_basis=50),
                      desk=[])
    assert d.held and d.position.last_close == d.technicals.last_close
    assert d.allowed_actions == ("add", "hold", "trim", "exit")
    assert "Allowed actions: add, hold, trim, exit" in d.render()


def test_dossier_degrades_to_warnings_when_a_name_is_too_new():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=[]), desk=[])
    assert d.snapshot is None
    assert d.warnings and d.warnings[0].startswith("fundamentals unavailable")


def test_dossier_round_trips_through_json():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=_history()), desk=[FakeModel()],
                      position=Position(ticker="T", shares=-5, cost_basis=20))
    back = Dossier(**json.loads(json.dumps(d.model_dump(mode="json"))))
    assert back == d
    assert back.render() == d.render()


# ---------------------------------------------------------------------------
# derive_action policy
# ---------------------------------------------------------------------------

def _mark(side):
    shares = 10 if side == "long" else -10
    return PositionMark.mark(Position(ticker="T", shares=shares, cost_basis=10), 12.0)


@pytest.mark.parametrize("signal,conf,position,expected", [
    ("bullish", CONVICTION_THRESHOLD, None, "buy"),
    ("bullish", CONVICTION_THRESHOLD - 1, None, "watch"),
    ("neutral", 90, None, "watch"),
    ("bearish", 10, None, "avoid"),
    ("bullish", 80, "long", "add"),
    ("bullish", 40, "long", "hold"),
    ("neutral", 80, "long", "hold"),
    ("bearish", 40, "long", "trim"),
    ("bearish", 80, "long", "exit"),
    ("bearish", 80, "short", "add"),
    ("bullish", 40, "short", "trim"),
    ("bullish", 80, "short", "exit"),
])
def test_derive_action(signal, conf, position, expected):
    mark = _mark(position) if position else None
    assert derive_action(signal, conf, mark) == expected


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------

def test_assemble_builds_a_valid_report_and_fills_the_action_by_policy():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=_history()), desk=[FakeModel()])
    report = assemble_report(d, sources=_sources(), queries=d.queries, answer=_answer(),
                             model="claude-code")
    assert isinstance(report, ResearchReport)
    assert report.action == "buy" and report.signal == "bullish" and report.confidence == 72
    assert report.model == "claude-code" and len(report.sources) == 2
    assert report.desk == d.desk and report.snapshot_hash == d.snapshot.content_hash
    assert report.prompt_key  # a stable key over the same prompt the LLM path sends
    assert "catalysts" in SYSTEM_PROMPT


def test_assemble_keeps_an_explicit_action():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=_history()), desk=[])
    report = assemble_report(d, sources=_sources(), queries=[], answer=_answer(action="watch"),
                             model="m")
    assert report.action == "watch"


def test_assemble_rejects_the_wrong_vocabulary_and_bad_indices():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=_history()), desk=[])
    with pytest.raises(ValueError, match="invalid action"):
        assemble_report(d, sources=_sources(), queries=[], answer=_answer(action="hold"), model="m")
    with pytest.raises(ValueError, match="source index out of range"):
        assemble_report(d, sources=_sources(1), queries=[], answer=_answer(), model="m")
    with pytest.raises(ValueError, match="invalid signal"):
        assemble_report(d, sources=_sources(), queries=[], answer=_answer(signal="meh"), model="m")


def test_assemble_held_name_uses_held_vocabulary():
    d = build_dossier("T", AS_OF, MockDataClient(metrics=_history()), desk=[],
                      position=Position(ticker="T", shares=10, cost_basis=50))
    report = assemble_report(d, sources=_sources(), queries=[],
                             answer=_answer(signal="bearish", confidence=85), model="m")
    assert report.action == "exit" and report.position.side == "long"
