"""Pydantic models for the research command's output.

A `ResearchReport` is the read-only counterpart to `Signal` (see
hedge_fund/models.py): both speak the same `signal`/`confidence` vocabulary,
but a report is a one-off, user-facing artifact with cited sources, not a
portfolio input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """One search hit, kept verbatim so a human can verify a citation."""

    query: str  # which templated query produced it
    title: str
    url: str
    content: str
    score: float | None = None
    published_date: str | None = None


class Claim(BaseModel):
    """One catalyst or risk statement, tied back to the evidence for it."""

    text: str
    source_indices: list[int] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The complete output of `diagnose()` — a cited stock diagnosis."""

    ticker: str
    as_of: str
    model: str
    signal: Literal["bullish", "neutral", "bearish"]
    confidence: float  # 0-100, same scale as Signal.metadata["confidence"]
    thesis: str
    catalysts: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    sources: list[SearchResult] = Field(default_factory=list)  # index-addressable
    queries: list[str] = Field(default_factory=list)  # queries that actually succeeded
    snapshot_hash: str | None = None  # None when fundamentals were unavailable
    prompt_key: str
    warnings: list[str] = Field(default_factory=list)
