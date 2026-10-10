"""diagnose() — retrieve-then-generate stock research.

Runs a fixed set of web searches alongside the point-in-time fundamentals
snapshot, the price-action snapshot, and a "desk readout" — every quant
alpha model's view on the name as of the date — folds all of it into one
prompt, and asks an LLM for a cited bullish/neutral/bearish diagnosis with
a concrete action. Mirrors the structure of
hedge_fund/signals/llm_agent.py's LLMAgent.predict(), but this is a one-off,
user-requested report — not a portfolio input — so failures propagate
instead of degrading to an abstained/neutral result. See
hedge_fund/research/models.py for the ResearchReport shape.

The desk readout is what makes the report more than a news summary: the
LLM sees the same momentum, reversal, insider-flow, quality-value and
earnings-drift readings the fund's systematic pods trade on, and has to
reconcile the narrative with the numbers.

The evidence-gathering and the answer validation live in
hedge_fund/research/dossier.py, so a reasoner other than an API-called
LLM (Claude Code inside the checkout, say) can run the same pipeline with
itself in the middle. This module is the LLM-in-the-middle composition.
"""

from __future__ import annotations

from hedge_fund.data.protocol import DataClient
from hedge_fund.llm import LLMClient, PromptCache, prompt_key
from hedge_fund.research.dossier import (
    CONVICTION_THRESHOLD,
    DEFAULT_QUERIES,
    SIGNALS,
    SYSTEM_PROMPT,
    Dossier,
    assemble_report,
    build_dossier,
    build_user_prompt,
    default_desk,
    derive_action,
    parse_response,
    render_desk,
    render_position,
)
from hedge_fund.research.models import Position, ResearchReport, SearchResult
from hedge_fund.research.search import SearchClient, SearchClientError
from hedge_fund.signals.base import AlphaModel

# Names the rest of the package (and its tests) have always imported from here.
_SYSTEM_PROMPT = SYSTEM_PROMPT
_SIGNALS = SIGNALS
_build_user_prompt = build_user_prompt
_parse = parse_response

__all__ = [
    "CONVICTION_THRESHOLD",
    "DEFAULT_QUERIES",
    "Dossier",
    "assemble_report",
    "build_dossier",
    "default_desk",
    "derive_action",
    "diagnose",
    "render_desk",
    "render_position",
]


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
    position: Position | None = None,
    desk: list[AlphaModel] | None = None,
) -> ResearchReport:
    """Produce a cited diagnosis of *ticker* as of *as_of* (YYYY-MM-DD).

    *position* is what the user holds, if anything — it changes the action
    vocabulary (add/hold/trim/exit vs buy/watch/avoid). *desk* is the list
    of quant models to read out; None means every registered quant model.

    Raises on: a fundamentals or price infrastructure failure, every search
    query failing, an LLM transport failure, or an unparseable/invalid LLM
    response. A one-off, user-requested report is worse for silently
    degrading into a fake "neutral" than for surfacing the real error.
    """
    cache = cache if cache is not None else PromptCache()

    dossier = build_dossier(ticker, as_of, data_client, position=position, desk=desk)
    sources, ran_queries, search_warnings = _run_searches(
        ticker, search_client, queries, max_results_per_query,
    )

    system = SYSTEM_PROMPT
    user = dossier.render(sources)
    key = prompt_key("research", llm.model, system, user)

    cached = cache.get(key)
    if cached is not None and "parsed" in cached and "action" in cached["parsed"]:
        data = cached["parsed"]
    else:
        response = llm.complete(system, user)
        record = {
            "agent": "research",
            "model": llm.model,
            "ticker": ticker,
            "as_of": as_of,
            "snapshot_hash": dossier.snapshot.content_hash if dossier.snapshot else None,
            "technicals_hash": dossier.technicals.content_hash if dossier.technicals else None,
            "desk": [s.model_dump() for s in dossier.desk],
            "position": dossier.position.model_dump() if dossier.position else None,
            "queries": ran_queries,
            "sources": [s.model_dump() for s in sources],
            "system": system,
            "user": user,
            "response": response,
        }
        try:
            data = parse_response(response, len(sources), held=dossier.held)
        except Exception as exc:
            cache.put(key, {**record, "parse_error": str(exc)})
            raise
        cache.put(key, {**record, "parsed": data})

    return assemble_report(
        dossier, sources=sources, queries=ran_queries, answer=data,
        model=llm.model, warnings=search_warnings,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _run_searches(
    ticker: str,
    search_client: SearchClient,
    queries: tuple[str, ...],
    max_results: int,
) -> tuple[list[SearchResult], list[str], list[str]]:
    """Run every templated query. One query failing is tolerated (skipped,
    warned); every query failing propagates — search is this feature's
    entire value-add, so a report built without any live context would be a
    stale fundamentals summary mislabeled as "research".
    """
    warnings: list[str] = []
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

    return dedupe_by_url(sources), ran_queries, warnings


def dedupe_by_url(sources: list[SearchResult]) -> list[SearchResult]:
    """Keep the first occurrence of each URL — preserves order so a claim's
    source_indices stay stable once assigned by the reasoner.
    """
    seen: set[str] = set()
    deduped = []
    for s in sources:
        if s.url in seen:
            continue
        seen.add(s.url)
        deduped.append(s)
    return deduped


_dedupe_by_url = dedupe_by_url
