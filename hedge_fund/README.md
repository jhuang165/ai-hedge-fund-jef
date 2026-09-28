# v2 — AI Hedge Fund core

> **Status: Work in progress.** A ground-up rebuild of the engine, developed
> alongside the shipped v1 app (`src/`, `app/`) but not yet wired into it.
> See [`../VISION.md`](../VISION.md) and [`../ROADMAP.md`](../ROADMAP.md) for
> where this is headed.

v2 rebuilds the fund as a persistent, point-in-time-honest system, mirroring a
real shop's hierarchy:

```
FUND      =  capital slices over STRATEGIES   (master risk on the netted book)
STRATEGY  =  a blend policy over MODELS       (a "pod")
MODEL     =  an alpha model → a Signal        (conviction in [-1,+1] + thesis)
```

A fund is the **desk**, not a watchlist: the mandate names no tickers. Which
names to trade is supplied per run (`--tickers`, or the app's ticker prompt)
and recorded on every `CycleRecord` — so one fund can be pointed at anything,
and every run remembers what it traded.

A fund runs two kinds of pods, like a real shop. **Discretionary** strategies
are staffed by **agents** — LLM investor personas (Warren Buffett, Charlie
Munger, Benjamin Graham, Peter Lynch, Stanley Druckenmiller) whose judgment
is the edge; blend them long-biased or market-neutral. **Systematic**
strategies are powered by quant models (post-earnings drift, 12-1 momentum,
short-term reversal, insider flow, quality-value) — the model *is* the
strategy, no persona attached. Both kinds implement one interface and plug
into the same engine unchanged.

Run a fund two ways: **one cycle** (today's data → today's target book) or a
**backtest** — the same cycle looped over history at the mandate's rebalance
cadence, producing an equity curve against your benchmark and a full
`CycleRecord` for every tick. Same code path, so a backtest is honest by
construction: it's the fund, replayed, not a separate simulator.

## Quickstart

```bash
poetry install                          # dependencies

# .env needs (at repo root):
#   FINANCIAL_DATASETS_API_KEY=...      # market/fundamentals data
#   ANTHROPIC_API_KEY=...               # only for LLM agents (Buffett)
#   TAVILY_API_KEY=...                  # only for `aihf research` (web search)

# THE command. No arguments: launch the interactive app (a Textual TUI).
# Build a fund — pick stocks, strategies, rebalance cadence — or backtest a
# saved fund and watch its equity curve draw against its benchmark.
poetry run aihf       # or, equivalently: python -m hedge_fund.tui

# With a mandate: run one cycle non-interactively (data → strategies →
# netting → risk → execution), full CycleRecord as JSON on stdout. A mandate
# carries no tickers — --tickers says what to point the fund at this run.
poetry run aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT,NVDA

# Backtest a mandate: the same run_cycle looped over history at the
# mandate's rebalance cadence, full result JSON (every CycleRecord) on stdout.
poetry run aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest

# Backtest over a dated universe (point-in-time membership) instead of a
# hand-picked list, which the result otherwise flags as survivorship-biased.
poetry run aihf ~/.hedge-fund/mandates/example.yaml --universe ~/.hedge-fund/universes/example.yaml --backtest

# The web app: research desk + fund runner in a browser, same engine.
poetry run aihf web

# Research: web search + point-in-time fundamentals + price action + every
# quant model's reading → a cited bullish/neutral/bearish diagnosis with an
# action. Bare ticker: buy/watch/avoid. With a position (SHARES@COST):
# add/hold/trim/exit. Several names come back ranked. Read-only — no
# mandate, no trades, no ledger writes. JSON on stdout, summary on stderr.
poetry run aihf research AAPL
poetry run aihf research AAPL:100@150.25 MSFT --portfolio holdings.yaml

# Grade every saved research call against what the market did next.
poetry run aihf scorecard --horizon 90

# Stress a backtest: hold-out split, and a sweep over any mandate field.
poetry run aihf validate ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --split 2025-01-01
poetry run aihf validate ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --sweep execution.slippage_bps=0,5,20

# Tests
poetry run pytest hedge_fund/
```

All API responses cache to disk (`~/.hedge-fund/cache/`), so reruns are fast,
free, and work offline once warmed.

## Architecture

```
Data (point-in-time) → Alpha models → Portfolio → Risk → Execution → Ledger
```

| Module | What | Status |
|--------|------|--------|
| `data/` | `DataClient` protocol, Financial Datasets client, disk cache | ✅ |
| `signals/` | `AlphaModel` interface; quant models (PEAD, momentum, mean-reversion, insider-flow, quality-value); `LLMAgent` + 5 investor personas | ✅ |
| `llm/` | LLM provider protocol, Anthropic client, prompt cache | ✅ |
| `features/` | Point-in-time snapshots: fundamentals (`snapshot.py`, valuation re-struck at the current price) and price action (`technicals.py`); `valuation.py` reprices filed multiples and derives the trailing dividend | ✅ |
| `fund/` | `FundSpec`/`StrategySpec` — mandates as YAML data (strategies, risk, execution policy) — the `Fund` object, and `DatedUniverse` for point-in-time membership | ✅ |
| `strategies/` | Strategy library (fundamental-ls, deep-value, inflections, earnings-drift, trend, insider-flow, quality-value, multifactor) — add yours as a YAML | ✅ |
| `portfolio/` | View blending → target weights (conviction-weighted, risk-scaled by realized vol, optional market-neutral) | ✅ |
| `risk/` | Hard limits — per-position, net-exposure and gross-exposure clamps; drawdown kill-switch | ✅ |
| `brokers/` | `Broker` protocol + `SimBroker` with a commission + slippage cost model (paper/live brokers planned) | ◐ |
| `pipeline/` | `run_cycle` — one code path for backtest/paper/live; dividend accrual on the book; delta orders under a no-trade band; `CycleRecord` | ✅ |
| `backtesting/` | `backtest_fund` — the whole fund over history on `run_cycle`, with per-strategy attribution (paper sleeve NAVs + residual) — plus the per-model engine | ✅ |
| `event_study/` | Market-model abnormal returns (CARs) | ✅ |
| `validation/` | Hold-out split and parameter sweep (`aihf validate`); CPCV and PBO planned | ◐ |
| `tui/` | The interactive app (Textual): fund builder + live backtest board | ✅ |
| `web/` | `aihf web` — FastAPI + one static page: research desk, fund runner (cycle/backtest with live curve), mandate builder, saved reports, key settings | ✅ |
| `research/` | `SearchClient` protocol + Tavily client; `diagnose()` — read-only cited research with a desk readout of every quant model and position-aware actions, outside the fund pipeline; every report saved, and `grade()` scores them against forward returns (`aihf scorecard`) | ✅ |

✅ built · ◐ partial · ⬜ planned

## Principles (non-negotiable)

- **Point-in-time by construction.** On any simulated date, only data actually
  filed by then is visible — the data layer filters on filing date, not report
  period. No lookahead, ever.
- **Fail loud.** Infrastructure failures raise; only genuine "no data" returns
  empty. A silent empty would poison a backtest as a fake "no signal."
- **The LLM never touches the trade.** Agents form *views* and *narrate*;
  deterministic code sizes and places orders; risk limits are hard gates.
- **One interface for every analyst.** Implement `AlphaModel.predict(ticker,
  date, data_client) -> Signal` and it plugs into the engine unchanged.

## Data contracts (`models.py`)

- `Signal` — an alpha model's output: `value` in `[-1, +1]`, plus `reasoning`,
  `components`, and `metadata`.
- `QuantSignals` — all signals for a ticker on a date.

## Contributing

Two high-leverage contributions:

- **A new agent or quant model** (code): read `signals/base.py` for the
  `AlphaModel` interface, use `signals/buffett.py` (an agent is just a system
  prompt) or `signals/momentum.py` (quant, over the price snapshot) as a
  template, register it, add a test. A registered quant model is
  automatically read out in `aihf research`.
- **A new strategy** (no code): drop a YAML in `strategies/` bundling existing
  models with a blend policy — the fund builder picks it up automatically.

See [`../ROADMAP.md`](../ROADMAP.md) for the open list.
