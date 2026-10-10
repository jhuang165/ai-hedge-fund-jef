"""The research pipeline, split at the LLM.

`diagnose()` (hedge_fund/research/diagnose.py) does three things in a row:
gather the evidence the fund already has on a name, search the web, and ask
an LLM for a diagnosis. This module is the first and last of those as
standalone pieces, so a reasoner that is not an API call — a person, or an
agent such as Claude Code running inside the checkout — can sit in the
middle:

    dossier = build_dossier("AAPL", "2026-10-06", data_client)   # no LLM, no search key
    ...the caller searches and reasons, producing the same JSON the LLM would...
    report = assemble_report(dossier, sources=..., queries=..., answer=..., model=...)

A `Dossier` is everything the prompt is built from: the point-in-time
fundamentals snapshot, the price-action snapshot, every quant model's
reading, and the user's marked position. It renders to the same prompt
text `diagnose()` sends, so the two paths see identical evidence.

`assemble_report()` validates the answer exactly as the LLM path does
(signal vocabulary, confidence range, the action vocabulary for a held or
not-held name, source indices in range). If the answer leaves `action`
out, `derive_action()` fills it from the signal and confidence by a fixed,
documented policy — the trade action is then the project's rule, not the
reasoner's opinion.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import FundamentalsSnapshot, InsufficientData, build_snapshot
from hedge_fund.features.technicals import PriceSnapshot, build_price_snapshot
from hedge_fund.llm import extract_json, prompt_key
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

SIGNALS = {"bullish", "neutral", "bearish"}

# Below this confidence a bullish or bearish signal is a lean, not a call:
# it earns "watch"/"hold"/"trim" rather than "buy"/"add"/"exit".
CONVICTION_THRESHOLD = 60.0

SYSTEM_PROMPT = """You are an equity research analyst producing a single, \
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


class Dossier(BaseModel):
    """Everything the fund knows about a name before anyone reasons about it."""

    ticker: str
    as_of: str
    snapshot: FundamentalsSnapshot | None = None   # None: too few filed periods
    technicals: PriceSnapshot | None = None        # None: too little price history
    desk: list[Signal] = Field(default_factory=list)
    position: PositionMark | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def held(self) -> bool:
        return self.position is not None

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return HELD_ACTIONS if self.held else FLAT_ACTIONS

    @property
    def queries(self) -> list[str]:
        return [q.format(ticker=self.ticker) for q in DEFAULT_QUERIES]

    def render(self, sources: list[SearchResult] | None = None) -> str:
        """The user prompt `diagnose()` sends: evidence blocks, then the
        numbered sources (or a placeholder when the caller has none yet)."""
        return build_user_prompt(self.snapshot, self.technicals, self.desk, self.position, sources or [])


def build_dossier(
    ticker: str,
    as_of: str,
    data_client: DataClient,
    *,
    position: Position | None = None,
    desk: list[AlphaModel] | None = None,
) -> Dossier:
    """Gather the evidence for *ticker* as of *as_of* (YYYY-MM-DD).

    InsufficientData from the fundamentals or price layers is a fact about
    the name (a recent IPO) and becomes a warning; any other data-layer
    error propagates, because an infrastructure failure must never look
    like "this company just has no fundamentals".
    """
    desk = default_desk() if desk is None else desk
    warnings: list[str] = []
    try:
        snapshot: FundamentalsSnapshot | None = build_snapshot(ticker, as_of, data_client)
    except InsufficientData as exc:
        snapshot = None
        warnings.append(f"fundamentals unavailable: {exc}")
    try:
        technicals: PriceSnapshot | None = build_price_snapshot(ticker, as_of, data_client)
    except InsufficientData as exc:
        technicals = None
        warnings.append(f"price history unavailable: {exc}")
    readout = [model.predict(ticker, as_of, data_client) for model in desk]
    mark = PositionMark.mark(position, technicals.last_close if technicals else None) if position else None
    return Dossier(ticker=ticker, as_of=as_of, snapshot=snapshot, technicals=technicals,
                   desk=readout, position=mark, warnings=warnings)


def derive_action(signal: str, confidence: float, position: PositionMark | None) -> str:
    """The project's standing policy for turning a research signal into a
    trade action, used when the reasoner supplies none.

    Not held: a confident bullish call is "buy"; a weak bullish or a neutral
    call is "watch"; bearish is "avoid".
    Held: a confident call in the position's direction is "add"; a weak one,
    or neutral, is "hold"; a confident call against the position is "exit";
    a weak call against it is "trim".
    """
    confident = confidence >= CONVICTION_THRESHOLD
    if position is None:
        if signal == "bullish":
            return "buy" if confident else "watch"
        return "watch" if signal == "neutral" else "avoid"
    with_position = {"long": "bullish", "short": "bearish"}[position.side]
    if signal == "neutral":
        return "hold"
    if signal == with_position:
        return "add" if confident else "hold"
    return "exit" if confident else "trim"


def assemble_report(
    dossier: Dossier,
    *,
    sources: list[SearchResult],
    queries: list[str],
    answer: dict,
    model: str,
    warnings: list[str] | None = None,
) -> ResearchReport:
    """Validate *answer* (the diagnosis JSON) against the dossier and build
    the report. Raises ValueError on any violation, exactly like the LLM
    path; a missing `action` is filled by `derive_action()`."""
    data = parse_answer(answer, len(sources), held=dossier.held, position=dossier.position)
    user = dossier.render(sources)
    return ResearchReport(
        ticker=dossier.ticker,
        as_of=dossier.as_of,
        model=model,
        signal=data["signal"],
        confidence=data["confidence"],
        action=data["action"],
        action_rationale=data["action_rationale"],
        thesis=data["thesis"],
        catalysts=[Claim(**c) for c in data["catalysts"]],
        risks=[Claim(**c) for c in data["risks"]],
        sources=sources,
        queries=queries,
        desk=dossier.desk,
        technicals=dossier.technicals,
        position=dossier.position,
        snapshot_hash=dossier.snapshot.content_hash if dossier.snapshot else None,
        prompt_key=prompt_key("research", model, SYSTEM_PROMPT, user),
        warnings=[*dossier.warnings, *(warnings or [])],
    )


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def build_user_prompt(
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


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------

def parse_response(response: str, n_sources: int, held: bool) -> dict:
    """Extract + validate the diagnosis JSON from raw LLM text."""
    return parse_answer(extract_json(response), n_sources, held=held, fill_action=False)


def parse_answer(
    data: dict,
    n_sources: int,
    *,
    held: bool,
    position: PositionMark | None = None,
    fill_action: bool = True,
) -> dict:
    """Validate a diagnosis dict. Raises ValueError on any violation. With
    *fill_action*, a missing action is derived by policy rather than
    rejected (the LLM path never fills: a model that forgot the action is
    a model that did not follow the prompt)."""
    signal = str(data.get("signal", "")).lower()
    if signal not in SIGNALS:
        raise ValueError(f"invalid signal {data.get('signal')!r}")

    confidence = float(data.get("confidence", 0))
    if not 0 <= confidence <= 100:
        raise ValueError(f"confidence out of range: {confidence}")

    allowed = HELD_ACTIONS if held else FLAT_ACTIONS
    action = str(data.get("action") or "").lower()
    if not action and fill_action:
        action = derive_action(signal, confidence, position)
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
