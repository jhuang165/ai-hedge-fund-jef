from datetime import datetime, timezone

from hedge_fund.reporter.evidence import (
    Evidence,
    Headline,
    Quote,
    fetch_headlines,
    gather_evidence,
    google_news_url,
    parse_rss,
    render_headlines,
    yahoo_rss_url,
)
from hedge_fund.research.dossier import Dossier

GOOGLE = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Nvidia falls after OpenAI numbers - Axios</title><link>https://a.example/1</link>
<pubDate>Thu, 08 Oct 2026 21:00:00 GMT</pubDate><source url="https://axios.com">Axios</source></item>
<item><title>Nvidia falls after OpenAI numbers - Yahoo Finance</title><link>https://a.example/2</link>
<pubDate>Thu, 08 Oct 2026 22:00:00 GMT</pubDate></item>
<item><title>Old news - Reuters</title><link>https://a.example/3</link>
<pubDate>Mon, 05 Oct 2026 09:00:00 GMT</pubDate></item>
<item><title>Should You Buy Nvidia? - Motley Fool</title><link>https://a.example/4</link>
<pubDate>Fri, 09 Oct 2026 01:00:00 GMT</pubDate></item>
</channel></rss>"""

YAHOO = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Nvidia to invest in d-Matrix</title><link>https://y.example/1</link>
<pubDate>Fri, 09 Oct 2026 02:00:00 +0000</pubDate></item>
<item><title>No date item</title><link>https://y.example/2</link></item>
<item><title></title><link>https://y.example/3</link></item>
</channel></rss>"""

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def test_parse_rss_google_splits_source_from_title():
    items = parse_rss(GOOGLE, "google")
    assert items[0].title == "Nvidia falls after OpenAI numbers" and items[0].source == "Axios"
    assert items[1].source == "Yahoo Finance"
    assert items[0].published == "2026-10-08T21:00+00:00"


def test_parse_rss_tolerates_bad_xml_and_empty_items():
    assert parse_rss(b"not xml", "yahoo") == []
    items = parse_rss(YAHOO, "yahoo")
    assert [i.title for i in items] == ["Nvidia to invest in d-Matrix", "No date item"]
    assert items[1].published is None


def test_fetch_headlines_merges_dedupes_windows_and_orders():
    def fetcher(url: str) -> bytes:
        return GOOGLE if "news.google.com" in url else YAHOO

    got = fetch_headlines("NVDA", lookback_hours=48, limit=10, fetcher=fetcher, now=NOW)
    titles = [h.title for h in got]
    # Duplicate title kept once, the 4-day-old item dropped, the undated one kept.
    assert titles.count("Nvidia falls after OpenAI numbers") == 1
    assert "Old news" not in titles
    assert "No date item" in titles
    # Newest first among dated items.
    dated = [h.published for h in got if h.published]
    assert dated == sorted(dated, reverse=True)


def test_fetch_headlines_limit_prefers_substance_over_listicles():
    def fetcher(url: str) -> bytes:
        return GOOGLE if "news.google.com" in url else YAHOO

    got = fetch_headlines("NVDA", lookback_hours=48, limit=2, fetcher=fetcher, now=NOW)
    assert all("Should You Buy" not in h.title for h in got)
    assert len(got) == 2


def test_fetch_headlines_survives_a_dead_feed():
    def fetcher(url: str) -> bytes:
        if "yahoo" in url:
            raise OSError("down")
        return GOOGLE

    got = fetch_headlines("NVDA", fetcher=fetcher, now=NOW)
    assert got and all(h.feed == "google" for h in got)


def test_feed_urls_encode_ticker():
    assert "q=NVDA+stock+when%3A2d" in google_news_url("NVDA")
    assert "s=BRK-B" in yahoo_rss_url("BRK-B")


def test_quote_render_and_headlines_render():
    q = Quote(price=100.0, previous_close=90.0, change_pct=100 / 90 - 1, day_low=95, day_high=101,
              volume=2e6, avg_volume_3m=1e6, as_of="t")
    text = q.render()
    assert "100.00" in text and "+11.11%" in text and "200% of normal" in text
    assert Quote().render().startswith("Live quote: unavailable")
    rendered = render_headlines([Headline(title="T", url="u", published="p", source="S", feed="google")])
    assert "[H0] (p) T — S" in rendered and "     u" in rendered
    assert render_headlines([]).endswith("none found.")


def test_gather_evidence_tolerates_dossier_failure():
    def bad_dossier(ticker, as_of, client):
        raise RuntimeError("yahoo is down")

    ev = gather_evidence(
        "NVDA", "2026-10-09", object(),
        quote_fn=lambda t: Quote(price=1.0, as_of="x"),
        headlines_fn=lambda t: [Headline(title="h", url="u", feed="yahoo")],
        dossier_fn=bad_dossier,
    )
    assert ev.technicals is None and ev.desk == []
    assert ev.warnings == ["desk evidence unavailable: yahoo is down"]
    assert "Price history: unavailable." in ev.render()
    assert "[H0]" in ev.render()


def test_gather_evidence_uses_dossier_blocks():
    def dossier(ticker, as_of, client):
        return Dossier(ticker=ticker, as_of=as_of, warnings=["fundamentals unavailable: thin"])

    ev = gather_evidence("NVDA", "2026-10-09", object(), quote_fn=lambda t: Quote(), headlines_fn=lambda t: [],
                         dossier_fn=dossier)
    assert isinstance(ev, Evidence)
    assert "Data warnings: fundamentals unavailable: thin" in ev.render()
    assert "no systematic models consulted" in ev.render()
