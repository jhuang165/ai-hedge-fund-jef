"""diagnose() — retrieve-then-generate stock research.

Runs a fixed set of web searches alongside the existing point-in-time
fundamentals snapshot, folds both into one prompt, and asks an LLM for a
cited bullish/neutral/bearish diagnosis. Mirrors the structure of
hedge_fund/signals/llm_agent.py's LLMAgent.predict(), but this is a one-off,
user-requested report — not a portfolio input — so failures propagate
instead of degrading to an abstained/neutral result. See
hedge_fund/research/models.py for the ResearchReport shape.
"""

from __future__ import annotations

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import FundamentalsSnapshot, InsufficientData, build_snapshot
from hedge_fund.llm import LLMClient, PromptCache, extract_json, prompt_key
from hedge_fund.research.models import Claim, ResearchReport, SearchResult
from hedge_fund.research.search import SearchClient, SearchClientError

# One query per angle a research analyst would check first: general news
# flow, the most recent print/guidance, sell-side sentiment, and the risk
# surface a plain news query under-indexes.
DEFAULT_QUERIES: tuple[str, ...] = (
    "{ticker} stock news",
    "{ticker} earnings report",
    "{ticker} analyst rating price target",
    "{ticker} stock risks controversy lawsuit",
)

_SIGNALS = {"bullish", "neutral", "bearish"}

_SYSTEM_PROMPT = """You are an equity research analyst producing a single, \
evidence-based diagnosis of a stock. You are given a fundamentals snapshot \
(if available) and a numbered list of web search results. Only claim what \
that evidence supports — do not invent facts. Every catalyst or risk you \
list must cite the indices of the sources that support it; use an empty \
list if a claim rests on fundamentals alone.

Respond with ONLY a JSON object of this exact shape:
{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <0-100>,
  "thesis": "<2-4 paragraph written diagnosis>",
  "catalysts": [{"text": "<claim>", "source_indices": [<int>, ...]}],
  "risks": [{"text": "<claim>", "source_indices": [<int>, ...]}]
}"""


def diagnose(
    ticker: str,
    as_of: str,
    data_client: DataClient,
    search_client: SearchClient,
    llm: LLMClient,
    *,
    cache: PromptCache | None = None,
    queries: tuple[str, ...] = DEFAULT_QUERIES,
    max_results_per_query: int = 5,
) -> ResearchReport:
    """Produce a cited diagnosis of *ticker* as of *as_of* (YYYY-MM-DD).

    Raises on: a fundamentals infrastructure failure, every search query
    failing, an LLM transport failure, or an unparseable/invalid LLM
    response. A one-off, user-requested report is worse for silently
    degrading into a fake "neutral" than for surfacing the real error.
    """
    cache = cache if cache is not None else PromptCache()

    snapshot, warnings = _build_snapshot_best_effort(ticker, as_of, data_client)
    sources, ran_queries, warnings = _run_searches(
        ticker, search_client, queries, max_results_per_query, warnings,
    )

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(snapshot, sources)
    key = prompt_key("research", llm.model, system, user)

    cached = cache.get(key)
    if cached is not None and "parsed" in cached:
        data = cached["parsed"]
    else:
        response = llm.complete(system, user)
        record = {
            "agent": "research",
            "model": llm.model,
            "ticker": ticker,
            "as_of": as_of,
            "snapshot_hash": snapshot.content_hash if snapshot else None,
            "queries": ran_queries,
            "sources": [s.model_dump() for s in sources],
            "system": system,
            "user": user,
            "response": response,
        }
        try:
            data = _parse(response, len(sources))
        except Exception as exc:
            cache.put(key, {**record, "parse_error": str(exc)})
            raise
        cache.put(key, {**record, "parsed": data})

    return ResearchReport(
        ticker=ticker,
        as_of=as_of,
        model=llm.model,
        signal=data["signal"],
        confidence=data["confidence"],
        thesis=data["thesis"],
        catalysts=[Claim(**c) for c in data["catalysts"]],
        risks=[Claim(**c) for c in data["risks"]],
        sources=sources,
        queries=ran_queries,
        snapshot_hash=snapshot.content_hash if snapshot else None,
        prompt_key=key,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_snapshot_best_effort(
    ticker: str,
    as_of: str,
    data_client: DataClient,
) -> tuple[FundamentalsSnapshot | None, list[str]]:
    """InsufficientData (e.g. a recent IPO) is a data fact, not a failure —
    proceed with search alone. Any other data-layer error (e.g.
    FDClientError) propagates: an infrastructure failure must never look
    like "this company just has no fundamentals".
    """
    try:
        return build_snapshot(ticker, as_of, data_client), []
    except InsufficientData as exc:
        return None, [f"fundamentals unavailable: {exc}"]


def _run_searches(
    ticker: str,
    search_client: SearchClient,
    queries: tuple[str, ...],
    max_results: int,
    warnings: list[str],
) -> tuple[list[SearchResult], list[str], list[str]]:
    """Run every templated query. One query failing is tolerated (skipped,
    warned); every query failing propagates — search is this feature's
    entire value-add, so a report built without any live context would be a
    stale fundamentals summary mislabeled as "research".
    """
    warnings = list(warnings)
    sources: list[SearchResult] = []
    ran_queries: list[str] = []
    last_error: SearchClientError | None = None

    for template in queries:
        query = template.format(ticker=ticker)
        try:
            sources.extend(search_client.search(query, max_results=max_results))
            ran_queries.append(query)
        except SearchClientError as exc:
            last_error = exc
            warnings.append(f"query failed: {query!r}: {exc}")

    if not ran_queries:
        assert last_error is not None
        raise last_error

    return _dedupe_by_url(sources), ran_queries, warnings


def _dedupe_by_url(sources: list[SearchResult]) -> list[SearchResult]:
    """Keep the first occurrence of each URL — preserves order so a claim's
    source_indices stay stable once assigned by the LLM.
    """
    seen: set[str] = set()
    deduped = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        deduped.append(s)
    return deduped


def _build_user_prompt(snapshot: FundamentalsSnapshot | None, sources: list[SearchResult]) -> str:
    fundamentals = snapshot.render() if snapshot else "No fundamentals data available."

    if sources:
        lines = ["Sources:"]
        for i, s in enumerate(sources):
            lines.append(f"[{i}] ({s.published_date or 'undated'}) {s.title}")
            lines.append(f"    url: {s.url}")
            lines.append(f"    {s.content}")
        source_block = "\n".join(lines)
    else:
        source_block = "Sources: none returned."

    return f"{fundamentals}\n\n{source_block}"


def _parse(response: str, n_sources: int) -> dict:
    """Extract + validate the diagnosis JSON. Raises on any violation."""
    data = extract_json(response)

    signal = str(data.get("signal", "")).lower()
    if signal not in _SIGNALS:
        raise ValueError(f"invalid signal {data.get('signal')!r}")

    confidence = float(data.get("confidence", 0))
    if not 0 <= confidence <= 100:
        raise ValueError(f"confidence out of range: {confidence}")

    thesis = str(data.get("thesis", ""))

    catalysts = _parse_claims(data.get("catalysts", []), n_sources)
    risks = _parse_claims(data.get("risks", []), n_sources)

    return {
        "signal": signal,
        "confidence": confidence,
        "thesis": thesis,
        "catalysts": catalysts,
        "risks": risks,
    }


def _parse_claims(raw: object, n_sources: int) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of claims, got {type(raw).__name__}")
    claims = []
    for item in raw:
        if not isinstance(item, dict) or "text" not in item:
            raise ValueError(f"invalid claim: {item!r}")
        indices = item.get("source_indices", []) or []
        for i in indices:
            if not isinstance(i, int) or not 0 <= i < n_sources:
                raise ValueError(f"source index out of range: {i!r}")
        claims.append({"text": str(item["text"]), "source_indices": list(indices)})
    return claims
