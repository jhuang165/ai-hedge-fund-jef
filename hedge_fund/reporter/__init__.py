"""The hourly reporter: a web app that keeps a cited brief on every name in
a watchlist fresh, continuously, from live prices, live headlines, and a
reasoning model that can search the web itself.

    aihf reporter                   # http://0.0.0.0:8788, scheduler on
    aihf reporter --once            # one cycle on stdout-free exit, for cron
    aihf reporter --no-scheduler    # serve the history only

How a brief is made (one per symbol per cycle, see brief.py):

1. evidence.py gathers what the fund already knows about the name — the
   same `Dossier` the research desk builds (fundamentals snapshot, price
   action, every quant model's reading) — plus a live quote from Yahoo and
   the last two days of headlines from Google News and Yahoo Finance RSS.
2. brief.py renders that into a prompt, together with the previous brief
   so the model can say what changed, and sends it to an OpenAI-compatible
   endpoint. That endpoint is meant to be circlemouth/Codex-Wrapper in
   front of the Codex CLI, signed in with the user's ChatGPT account, with
   Codex's own web search switched on — so the model reads today's news
   itself and cites URLs.
3. The answer is validated (signal, confidence, action vocabulary) and
   stored in SQLite (store.py). The scheduler (scheduler.py) runs a cycle
   every REPORTER_INTERVAL_MINUTES during market hours and less often
   outside them; server.py serves the dashboard and a JSON API.

Nothing here places an order or touches a mandate: the reporter is a
reading desk, and `action` follows the research desk's standing policy
(dossier.derive_action) when the model leaves it out.
"""

from hedge_fund.reporter.brief import Brief, CodexClient, produce_brief
from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.scheduler import Reporter
from hedge_fund.reporter.store import ReportStore

__all__ = ["Brief", "CodexClient", "Reporter", "ReporterConfig", "ReportStore", "produce_brief"]
