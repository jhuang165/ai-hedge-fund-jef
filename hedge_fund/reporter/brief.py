"""One brief: the prompt, the call, and the validated answer.

The brief is the hourly counterpart of the research desk's report. It is
shorter, it is written against the previous brief ("what changed"), and it
leans on the model's own live web search for news in the last few hours —
the RSS headlines in the evidence are a starting point the model is told
to verify and extend, not the universe of what it may cite.

The model is reached through any OpenAI-compatible chat endpoint. In
production that is circlemouth/Codex-Wrapper wrapping `codex exec` with
the user's ChatGPT sign-in; the wrapper returns plain text, so the answer
is parsed with the same `extract_json` the rest of the fund uses, and
validated the same way `diagnose()` validates a research answer.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from hedge_fund.llm import LLMParseError, extract_json
from hedge_fund.reporter.evidence import Evidence, Headline, Quote
from hedge_fund.research.dossier import SIGNALS, derive_action
from hedge_fund.research.models import FLAT_ACTIONS

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the hourly news-and-tape desk for a small systematic \
hedge fund. Every hour you write a short, cited brief on one stock. You are \
given: a live quote, a price-action snapshot built from daily bars, a \
fundamentals snapshot, the fund's quant models' readings (conviction in \
[-1, +1]), a list of recent headlines from RSS feeds, and the brief you \
wrote last time (if any).

You have a web search tool. USE IT: search for news about the company from \
the last few hours and today, including earnings, guidance, analyst actions, \
regulatory or legal news, product and customer news, macro or sector moves \
that explain the tape, and anything that contradicts the RSS headlines. Cite \
URLs you actually found; never invent a URL, a number, or a quote. Prefer \
primary sources and dated reporting. If nothing material happened, say so \
plainly — a quiet hour is a valid finding.

Write for a portfolio manager who read your last brief: lead with what is \
new since then. Reconcile the narrative with the numbers (quote, trend, the \
quant desk); where they disagree, say which you trust and why.

Then give a signal and ONE action for a name the fund does not hold: \
"buy" (confident bullish, confidence >= 60), "watch" (interesting, not yet, \
or neutral), or "avoid" (bearish). Explain what would change your mind.

Respond with ONLY a JSON object of this exact shape (no prose outside it):
{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <0-100>,
  "action": "buy" | "watch" | "avoid",
  "headline": "<one line, <= 120 characters: the hour in a sentence>",
  "whats_new": ["<bullet: a development since the last brief, with (source N) refs>", ...],
  "summary": "<2-3 short paragraphs: tape, news, how the desk's numbers fit, the call>",
  "catalysts": [{"text": "<claim>", "source_indices": [<int>, ...]}],
  "risks": [{"text": "<claim>", "source_indices": [<int>, ...]}],
  "watch_next": ["<what to watch in the next hours/days: dates, levels, events>", ...],
  "sources": [{"title": "<title>", "url": "<url>", "published": "<YYYY-MM-DD or ISO, or null>"}],
  "nothing_new": <true if no material development since the last brief>
}
`source_indices` index into YOUR `sources` list (0-based). Include every \
RSS headline you relied on in `sources` too, by its URL. Keep `sources` to \
the items you actually used, at most 12.

FORMAT RULE: print the whole JSON object on ONE line, minified (no \
pretty-printing, no line breaks between keys or array items; use \\n inside \
strings for paragraph breaks). The transport between you and the reader \
drops individual lines that look like log metadata, so a multi-line object \
can arrive broken."""


class Source(BaseModel):
    title: str
    url: str
    published: str | None = None


class Claim(BaseModel):
    text: str
    source_indices: list[int] = Field(default_factory=list)


class Brief(BaseModel):
    """A validated hourly brief, plus the evidence it was written from."""

    ticker: str
    generated_at: str
    model: str
    signal: Literal["bullish", "neutral", "bearish"]
    confidence: float
    action: Literal["buy", "watch", "avoid"]
    headline: str
    whats_new: list[str] = Field(default_factory=list)
    summary: str = ""
    catalysts: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    watch_next: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    nothing_new: bool = False

    # Evidence carried along for the page and for the next prompt.
    quote: Quote
    headlines: list[Headline] = Field(default_factory=list)
    technicals: dict[str, Any] | None = None
    desk: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # The brief before this one, for the "changed since" markers on the page.
    prev_signal: str | None = None
    prev_confidence: float | None = None
    prev_action: str | None = None
    prev_generated_at: str | None = None

    @property
    def score(self) -> float:
        sign = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}[self.signal]
        return sign * self.confidence


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def render_previous(previous: Brief | dict[str, Any] | None) -> str:
    if previous is None:
        return "Previous brief: none — this is the first brief on the name. Treat everything recent as new."
    p = previous if isinstance(previous, dict) else previous.model_dump()
    lines = [
        f"Previous brief (written {p.get('generated_at')}): {p.get('signal')} {p.get('confidence'):.0f}, "
        f"action {p.get('action')}.",
        f"  Headline then: {p.get('headline', '')}",
    ]
    q = p.get("quote") or {}
    if q.get("price") is not None:
        lines.append(f"  Price then: {q['price']:.2f} (quote as of {q.get('as_of', '?')})")
    if p.get("whats_new"):
        lines.append("  What was new then: " + " | ".join(str(x) for x in p["whats_new"][:5]))
    if p.get("watch_next"):
        lines.append("  You said to watch: " + " | ".join(str(x) for x in p["watch_next"][:5]))
    return "\n".join(lines)


def build_prompt(evidence: Evidence, previous: Brief | dict[str, Any] | None, now: datetime | None = None) -> tuple[str, str]:
    """(system, user) for one brief."""
    now = now or datetime.now(timezone.utc)
    user = "\n\n".join([
        f"Ticker: {evidence.ticker}. Current time: {now.isoformat(timespec='minutes')} "
        f"(UTC). Search the web for news on {evidence.ticker} from the last few hours and today before answering.",
        evidence.render(),
        render_previous(previous),
    ])
    return SYSTEM_PROMPT, user


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """The endpoint failed (after retries) or returned nothing usable."""


class CodexClient:
    """A minimal OpenAI-compatible chat client (requests, no SDK): the
    wrapper speaks /v1/chat/completions and nothing more is needed."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        *,
        timeout_s: int = 600,
        reasoning_effort: str = "",
        retries: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.retries = retries
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def models(self) -> list[str]:
        r = self._session.get(f"{self.base_url}/models", headers=self._headers(), timeout=20)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m]

    def healthy(self) -> tuple[bool, str]:
        try:
            ids = self.models()
            return True, f"{len(ids)} models"
        except Exception as exc:
            return False, str(exc)

    def complete(self, system: str, user: str, model: str) -> str:
        body: dict[str, Any] = {
            "model": model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.reasoning_effort:
            body["x_codex"] = {"reasoning_effort": self.reasoning_effort}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self._session.post(
                    f"{self.base_url}/chat/completions", headers=self._headers(),
                    data=json.dumps(body), timeout=self.timeout_s,
                )
                if r.status_code >= 500 or r.status_code == 429:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise LLMError("empty completion")
                return content
            except (requests.RequestException, LLMError, KeyError, ValueError) as exc:
                last = exc
                log.warning("completion attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.retries:
                    time.sleep(min(30, 5 * (attempt + 1)))
        raise LLMError(f"completion failed after {self.retries + 1} attempts: {last}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def repair_json_text(text: str) -> str:
    """Undo the damage a line-filtering transport does to pretty-printed
    JSON: a deleted last array item or last key leaves a trailing comma,
    which strict JSON rejects. Content that was deleted cannot come back;
    this only makes the remainder parse."""
    return _TRAILING_COMMA.sub(r"\1", text)


def parse_brief_json(text: str) -> dict[str, Any]:
    """The raw answer, validated and normalized. Raises ValueError."""
    try:
        data = extract_json(text)
    except LLMParseError:
        try:
            data = extract_json(repair_json_text(text))
        except LLMParseError as exc:
            raise ValueError(f"no JSON object in the answer: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("answer is not a JSON object")

    signal = str(data.get("signal", "")).strip().lower()
    if signal not in SIGNALS:
        raise ValueError(f"invalid signal {data.get('signal')!r}")
    try:
        confidence = float(data.get("confidence", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence is not a number: {data.get('confidence')!r}") from exc
    if not 0 <= confidence <= 100:
        raise ValueError(f"confidence out of range: {confidence}")

    action = str(data.get("action") or "").strip().lower()
    if not action:
        action = derive_action(signal, confidence, None)
    if action not in FLAT_ACTIONS:
        raise ValueError(f"invalid action {data.get('action')!r}; expected one of {FLAT_ACTIONS}")

    sources = []
    for s in _as_list(data.get("sources")):
        if isinstance(s, dict) and s.get("url"):
            sources.append(Source(title=str(s.get("title") or s["url"]), url=str(s["url"]),
                                  published=_opt_str(s.get("published"))))
        elif isinstance(s, str) and s.startswith("http"):
            sources.append(Source(title=s, url=s))
    n = len(sources)

    def claims(raw: object) -> list[Claim]:
        out = []
        for item in _as_list(raw):
            if isinstance(item, str):
                out.append(Claim(text=item))
            elif isinstance(item, dict) and item.get("text"):
                idx = [i for i in _as_list(item.get("source_indices")) if isinstance(i, int) and 0 <= i < n]
                out.append(Claim(text=str(item["text"]), source_indices=idx))
        return out

    return {
        "signal": signal,
        "confidence": confidence,
        "action": action,
        "headline": str(data.get("headline") or "").strip()[:200],
        "whats_new": [str(x) for x in _as_list(data.get("whats_new")) if str(x).strip()],
        "summary": str(data.get("summary") or "").strip(),
        "catalysts": claims(data.get("catalysts")),
        "risks": claims(data.get("risks")),
        "watch_next": [str(x) for x in _as_list(data.get("watch_next")) if str(x).strip()],
        "sources": sources,
        "nothing_new": bool(data.get("nothing_new", False)),
    }


def _as_list(v: object) -> list:
    return v if isinstance(v, list) else []


def _opt_str(v: object) -> str | None:
    return None if v is None or v == "" else str(v)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def produce_brief(
    evidence: Evidence,
    previous: Brief | dict[str, Any] | None,
    client: CodexClient,
    model: str,
    *,
    now: datetime | None = None,
) -> Brief:
    """Prompt, call, validate, and attach the evidence. ValueError on an
    answer that does not follow the contract; LLMError on a dead endpoint."""
    now = now or datetime.now(timezone.utc)
    system, user = build_prompt(evidence, previous, now)
    raw = client.complete(system, user, model)
    data = parse_brief_json(raw)
    p = previous if isinstance(previous, dict) or previous is None else previous.model_dump()
    return Brief(
        ticker=evidence.ticker,
        generated_at=now.isoformat(timespec="seconds"),
        model=model,
        quote=evidence.quote,
        headlines=evidence.headlines,
        technicals=evidence.technicals.model_dump() if evidence.technicals else None,
        desk=[s.model_dump() for s in evidence.desk],
        warnings=evidence.warnings,
        prev_signal=p.get("signal") if p else None,
        prev_confidence=p.get("confidence") if p else None,
        prev_action=p.get("action") if p else None,
        prev_generated_at=p.get("generated_at") if p else None,
        **data,
    )
