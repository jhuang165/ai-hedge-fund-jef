"""What goes into a brief before any model reasons about it.

Three ingredients, each fetched by a function the scheduler can swap for a
fake in tests:

- the research desk's `Dossier` (fundamentals snapshot, price action,
  the quant models' readings) through the fund's own data layer — the
  same evidence `aihf research` uses, so the hourly brief and the
  research report never disagree about the numbers;
- a live quote from Yahoo (`yfinance` fast_info): last price, previous
  close, day range, volume. The dossier's bars are daily closes; this is
  what the stock is doing *now*;
- headlines from the last two days: Google News RSS (broad, every outlet)
  and Yahoo Finance's per-ticker RSS (real article URLs). Both are free
  and keyless. They are context, not the model's only news source — the
  Codex CLI behind the wrapper has its own live web search and is told to
  use it.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

from pydantic import BaseModel, Field

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.technicals import PriceSnapshot
from hedge_fund.models import Signal
from hedge_fund.research.dossier import Dossier, build_dossier, render_desk

log = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; aihf-reporter/1.0)"


class Quote(BaseModel):
    price: float | None = None
    previous_close: float | None = None
    change_pct: float | None = None       # vs previous close, fraction
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    avg_volume_3m: float | None = None
    market_cap: float | None = None
    currency: str | None = None
    as_of: str = ""                       # UTC ISO, when we asked

    def render(self) -> str:
        if self.price is None:
            return "Live quote: unavailable."
        head = f"Live quote (as of {self.as_of}): {self.price:.2f}"
        if self.previous_close is not None:
            chg = f"{self.change_pct:+.2%}" if self.change_pct is not None else "n/a"
            head += f" ({chg} vs previous close {self.previous_close:.2f})"
        if self.day_low is not None and self.day_high is not None:
            head += f"; day range {self.day_low:.2f}-{self.day_high:.2f}"
        if self.volume is not None and self.avg_volume_3m:
            head += (f"; volume {self.volume:,.0f} vs 3m avg {self.avg_volume_3m:,.0f} "
                     f"({self.volume / self.avg_volume_3m:.0%} of normal)")
        return head + "."


class Headline(BaseModel):
    title: str
    url: str
    published: str | None = None          # ISO UTC when parseable
    source: str | None = None
    feed: str                              # google | yahoo


class Evidence(BaseModel):
    ticker: str
    as_of: str                             # the dossier's as-of date (YYYY-MM-DD)
    gathered_at: str                       # UTC ISO
    quote: Quote
    headlines: list[Headline] = Field(default_factory=list)
    technicals: PriceSnapshot | None = None
    desk: list[Signal] = Field(default_factory=list)
    fundamentals_text: str = ""
    warnings: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """The evidence block of the prompt."""
        blocks = [
            self.quote.render(),
            self.technicals.render() if self.technicals else "Price history: unavailable.",
            self.fundamentals_text or "Fundamentals: unavailable.",
            render_desk(self.desk),
            render_headlines(self.headlines),
        ]
        if self.warnings:
            blocks.append("Data warnings: " + "; ".join(self.warnings))
        return "\n\n".join(blocks)


def render_headlines(headlines: list[Headline]) -> str:
    if not headlines:
        return "Recent headlines (RSS): none found."
    lines = ["Recent headlines (RSS, newest first; verify anything you rely on):"]
    for i, h in enumerate(headlines):
        when = h.published or "undated"
        src = f" — {h.source}" if h.source else ""
        lines.append(f"[H{i}] ({when}) {h.title}{src}")
        lines.append(f"     {h.url}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

def fetch_quote(ticker: str) -> Quote:
    """A live quote from Yahoo. Never raises: a missing quote is a warning in
    the brief, not a reason to skip the name."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        import yfinance as yf

        fi = yf.Ticker(ticker).fast_info

        def get(key: str) -> float | None:
            return _float(fi[key]) if key in fi else None

        price, prev = get("lastPrice"), get("previousClose")
        return Quote(
            price=price, previous_close=prev,
            change_pct=(price / prev - 1.0) if price and prev else None,
            day_high=get("dayHigh"), day_low=get("dayLow"),
            volume=get("lastVolume"), avg_volume_3m=get("threeMonthAverageVolume"),
            market_cap=get("marketCap"),
            currency=str(fi["currency"]) if "currency" in fi else None,
            as_of=stamp,
        )
    except Exception as exc:  # network, parsing, a delisted name
        log.warning("quote for %s failed: %s", ticker, exc)
        return Quote(as_of=stamp)


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN -> None


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------

Fetcher = Callable[[str], bytes]


def _http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # fixed, known hosts
        return resp.read()


def google_news_url(ticker: str, company: str | None = None, days: int = 2) -> str:
    q = f"{ticker} stock" if not company else f"{company} OR {ticker} stock"
    q = f"{q} when:{days}d"
    return "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )


def yahoo_rss_url(ticker: str) -> str:
    return "https://feeds.finance.yahoo.com/rss/2.0/headline?" + urllib.parse.urlencode(
        {"s": ticker, "region": "US", "lang": "en-US"}
    )


def parse_rss(raw: bytes, feed: str) -> list[Headline]:
    """Items of an RSS 2.0 feed as Headlines; a malformed feed yields []."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out: list[Headline] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        if not title or not url:
            continue
        source = (item.findtext("source") or "").strip() or None
        if feed == "google" and " - " in title:
            # Google titles end in " - Publisher"; the <source> element names the same outlet.
            title, suffix = title.rsplit(" - ", 1)
            source = source or suffix.strip()
        published = _iso(item.findtext("pubDate"))
        out.append(Headline(title=title, url=url, published=published, source=source, feed=feed))
    return out


def _iso(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw.strip())
    except (TypeError, ValueError, IndexError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="minutes")


_NOISE = re.compile(r"\b(should you buy|is it too late|motley fool|zacks|here's why|stock-split)\b", re.I)


def fetch_headlines(
    ticker: str,
    *,
    company: str | None = None,
    lookback_hours: int = 48,
    limit: int = 14,
    fetcher: Fetcher = _http_get,
    now: datetime | None = None,
) -> list[Headline]:
    """Merge Google News and Yahoo headlines, newest first, deduplicated on
    title, limited to the lookback window. Feed failures are logged and
    skipped — one dead feed must not empty the brief."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=lookback_hours)
    items: list[Headline] = []
    for feed, url in (("google", google_news_url(ticker, company)), ("yahoo", yahoo_rss_url(ticker))):
        try:
            items.extend(parse_rss(fetcher(url), feed))
        except Exception as exc:
            log.warning("%s feed for %s failed: %s", feed, ticker, exc)

    def keep(h: Headline) -> bool:
        if h.published is None:
            return True  # undated is better than dropped; the model is told to verify
        return datetime.fromisoformat(h.published) >= cutoff

    seen: set[str] = set()
    fresh: list[Headline] = []
    for h in sorted((h for h in items if keep(h)), key=lambda h: h.published or "", reverse=True):
        key = re.sub(r"\W+", " ", h.title.lower()).strip()[:80]
        if key in seen:
            continue
        seen.add(key)
        fresh.append(h)
    # Listicles and promo pieces to the back, not out: they are still news flow.
    fresh.sort(key=lambda h: 1 if _NOISE.search(h.title) else 0)
    fresh = fresh[:limit]
    fresh.sort(key=lambda h: h.published or "", reverse=True)
    return fresh


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def gather_evidence(
    ticker: str,
    as_of: str,
    data_client: DataClient,
    *,
    quote_fn: Callable[[str], Quote] = fetch_quote,
    headlines_fn: Callable[[str], list[Headline]] | None = None,
    dossier_fn: Callable[[str, str, DataClient], Dossier] = build_dossier,
) -> Evidence:
    """Everything the brief prompt is built from. The dossier may fail on a
    thin name (that is a warning); the quote and headlines never raise."""
    warnings: list[str] = []
    technicals = None
    desk: list[Signal] = []
    fundamentals_text = ""
    try:
        dossier = dossier_fn(ticker, as_of, data_client)
        technicals = dossier.technicals
        desk = dossier.desk
        fundamentals_text = dossier.snapshot.render() if dossier.snapshot else ""
        warnings.extend(dossier.warnings)
    except Exception as exc:  # an infrastructure failure in the data layer
        log.warning("dossier for %s failed: %s", ticker, exc)
        warnings.append(f"desk evidence unavailable: {exc}")
    headlines = (headlines_fn or fetch_headlines)(ticker)
    return Evidence(
        ticker=ticker, as_of=as_of,
        gathered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        quote=quote_fn(ticker), headlines=headlines, technicals=technicals, desk=desk,
        fundamentals_text=fundamentals_text, warnings=warnings,
    )
