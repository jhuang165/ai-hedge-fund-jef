"""aihf web — a local web app over the engine.

A thin client, like the TUI and the CLI: it composes requests, hands them to
the same `diagnose`, `run_cycle` and `backtest_fund` the other surfaces call,
and renders the records they return. The server is FastAPI serving one static
page; long calls (LLM research, backtests) run as background jobs the page
polls. See hedge_fund/web/server.py.
"""
