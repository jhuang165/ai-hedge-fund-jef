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
"""

from __future__ import annotations

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import FundamentalsSnapshot, InsufficientData, build_snapshot
from hedge_fund.features.technicals import PriceSnapshot, build_price_snapshot
from hedge_fund.llm import LLMClient, PromptCache, extract_json, prompt_key
from hedge_fund.models import Signal
from hedge_fund.research.models import (
    FLAT_ACTIONS,
    HELD_ACTIONS,
    Claim,
    Position,
    PositionMark,
    ResearchReport,
    SearchResult,
)
from hedge_fund.research.search import SearchClient, SearchClientError
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, QUANT_MODEL_NAMES
from hedge_fund.signals.base import AlphaModel

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
evidence-based diagnosis of a stock and a concrete recommendation. You are \
given, in order: a fundamentals snapshot (if available), a price-action \
snapshot (if available), a desk readout — the systematic models' views on \
the name, each a conviction in [-1, +1] with its arithmetic — the user's \
current position in the name (if any), and a numbered list of web search \
results.

Reconcile the narrative with the numbers. Where the search results and the \
desk disagree, say which you trust and why. Only claim what the evidence \
supports — do not invent facts. Every catalyst or risk you list must cite \
the indices of the sources that support it; use an empty list if a claim \
rests on fundamentals, price action, or the desk readout alone.

Then recommend ONE action:
- If the user holds the name: "add" (increase), "hold" (keep as is), \
"trim" (reduce), or "exit" (close). Judge the position as it stands — its \
size, its unrealized gain or loss, and whether the thesis that would justify \
holding it still stands. Never anchor on the cost basis: a loss is not a \
reason to hold, a gain is not a reason to sell.
- If the user does not hold it: "buy", "watch" (interesting, but not yet), \
or "avoid".
Explain the action in two or three sentences, including what would change \
your mind.

Respond with ONLY a JSON object of this exact shape:
{
  "signal": "bullish" | "neutral" | "bearish",
  "confidence": <0-100>,
  "action": "<one of the allowed actions>",
  "action_rationale": "<2-3 sentences>",
  "thesis": "<2-4 paragraph written diagnosis>",
  "catalysts": [{"text": "<claim>", "source_indices": [<int>, ...]}],
  "risks": [{"text": "<claim>", "source_indices": [<int>, ...]}]
}"""


def default_desk() -> list[AlphaModel]:
    """Every registered quant model, fresh instances — the systematic desk
    whose readings go into the prompt. Quant only: an LLM persona in the
    readout would be one model's opinion feeding another's."""
    return [ALPHA_MODEL_REGISTRY[name]() for name in QUANT_MODEL_NAMES]


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
    desk = default_desk() if desk is None else desk

    snapshot, warnings = _build_snapshot_best_effort(ticker, as_of, data_client)
    technicals, warnings = _build_technicals_best_effort(ticker, as_of, data_client, warnings)
    readout = [model.predict(ticker, as_of, data_client) for model in desk]
    mark = PositionMark.mark(position, technicals.last_close if technicals else None) if position else None
    sources, ran_queries, warnings = _run_searches(
        ticker, search_client, queries, max_results_per_query, warnings,
    )

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(snapshot, technicals, readout, mark, sources)
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
            "snapshot_hash": snapshot.content_hash if snapshot else None,
            "technicals_hash": technicals.content_hash if technicals else None,
            "desk": [s.model_dump() for s in readout],
            "position": mark.model_dump() if mark else None,
            "queries": ran_queries,
            "sources": [s.model_dump() for s in sources],
            "system": system,
            "user": user,
            "response": response,
        }
        try:
            data = _parse(response, len(sources), held=mark is not None)
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
        action=data["action"],
        action_rationale=data["action_rationale"],
        thesis=data["thesis"],
        catalysts=[Claim(**c) for c in data["catalysts"]],
        risks=[Claim(**c) for c in data["risks"]],
        sources=sources,
        queries=ran_queries,
        desk=readout,
        technicals=technicals,
        position=mark,
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


def _build_technicals_best_effort(
    ticker: str,
    as_of: str,
    data_client: DataClient,
    warnings: list[str],
) -> tuple[PriceSnapshot | None, list[str]]:
    """Same contract as the fundamentals: too little price history is a
    fact about the name (a fresh listing), any other error propagates."""
    try:
        return build_price_snapshot(ticker, as_of, data_client), list(warnings)
    except InsufficientData as exc:
        return None, [*warnings, f"price history unavailable: {exc}"]


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


def _build_user_prompt(
    snapshot: FundamentalsSnapshot | None,
    technicals: PriceSnapshot | None,
    readout: list[Signal],
    mark: PositionMark | None,
    sources: list[SearchResult],
) -> str:
    blocks = [
        snapshot.render() if snapshot else "No fundamentals data available.",
        technicals.render() if technicals else "No price history available.",
        render_desk(readout),
        render_position(mark),
    ]

    if sources:
        lines = ["Sources:"]
        for i, s in enumerate(sources):
            lines.append(f"[{i}] ({s.published_date or 'undated'}) {s.title}")
            lines.append(f"    url: {s.url}")
            lines.append(f"    {s.content}")
        blocks.append("\n".join(lines))
    else:
        blocks.append("Sources: none returned.")

    return "\n\n".join(blocks)


def render_desk(readout: list[Signal]) -> str:
    """The systematic desk's views, one line per model."""
    if not readout:
        return "Desk readout: no systematic models consulted."
    lines = ["Desk readout (systematic models, conviction in [-1, +1]; "
             "'abstained' means the model could not form a view):"]
    for s in readout:
        if s.metadata.get("abstained") is True:
            lines.append(f"  {s.model_name}: abstained — {s.metadata.get('abstain_reason', '')}")
        else:
            lines.append(f"  {s.model_name}: {s.value:+.2f} — {s.reasoning or 'no view'}")
    return "\n".join(lines)


def render_position(mark: PositionMark | None) -> str:
    if mark is None:
        return "Position: the user does not hold this name. Allowed actions: buy, watch, avoid."
    line = (
        f"Position: the user is {mark.side} {abs(mark.shares):g} shares "
        f"at an average cost of {mark.cost_basis:.2f}"
    )
    if mark.last_close is not None:
        line += (
            f"; at the last close of {mark.last_close:.2f} that is "
            f"{mark.market_value:,.0f} of market value with an unrealized "
            f"{mark.unrealized_pnl_pct:+.1%} ({mark.unrealized_pnl:+,.0f})"
        )
    return line + ". Allowed actions: add, hold, trim, exit."


def _parse(response: str, n_sources: int, held: bool) -> dict:
    """Extract + validate the diagnosis JSON. Raises on any violation."""
    data = extract_json(response)

    signal = str(data.get("signal", "")).lower()
    if signal not in _SIGNALS:
        raise ValueError(f"invalid signal {data.get('signal')!r}")

    confidence = float(data.get("confidence", 0))
    if not 0 <= confidence <= 100:
        raise ValueError(f"confidence out of range: {confidence}")

    allowed = HELD_ACTIONS if held else FLAT_ACTIONS
    action = str(data.get("action", "")).lower()
    if action not in allowed:
        raise ValueError(
            f"invalid action {data.get('action')!r} for a "
            f"{'held' if held else 'not-held'} name; expected one of {allowed}"
        )

    thesis = str(data.get("thesis", ""))
    catalysts = _parse_claims(data.get("catalysts", []), n_sources)
    risks = _parse_claims(data.get("risks", []), n_sources)

    return {
        "signal": signal,
        "confidence": confidence,
        "action": action,
        "action_rationale": str(data.get("action_rationale", "")),
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
