# Roadmap

Where the project is headed and where you can help. This is a living list — open a
PR to add an item, claim one, or update status. For the bigger picture behind it,
read [VISION.md](./VISION.md).

> **Educational use only.** Not investment advice; not intended for real trading.

## Status legend

✅ Shipped · 🚧 In progress · ⬜ Planned

**Current focus:** the scheduler. The ledger carries the book between runs, and a
fund can now paper-trade through an Alpaca paper account (`execution.broker:
alpaca-paper`): real quotes and fills, the account as the book of record, checked
against the receipts on every run. What's left between "run it by hand" and a fund
that is genuinely always-on is the scheduler that runs it each market day. In parallel: retiring the v1 CLI, which needs Ollama
(the free, local, no-key path) and the remaining investor personas ported.

The tables below are a capability map, not a strict order; where items depend on
each other, the dependency is noted.

## The engine

The core: a **fund** as a persistent object, and one pipeline (`run_cycle`) that runs
it in backtest, paper, or live mode (see [VISION.md](./VISION.md)).

| Item | Status |
|------|--------|
| `AlphaModel` / `Signal` interface — the contract every analyst implements | ✅ |
| Backtesting engine — `backtest_fund`: the whole fund over history on `run_cycle`, equity curve vs the mandate's benchmark (plus the per-model harness) | ✅ |
| Event-study engine — market-model abnormal returns (CARs) | ✅ |
| `run_cycle` — one pipeline (data → analysts → portfolio → risk → execution → ledger), three modes | 🚧 (backtest and paper ship: the backtest loop, and run-today on a carried book through the simulated broker or an Alpaca paper account; live is the remaining mode) |
| Fund object — persistent mandate, staff, capital, books | ✅ (mandates, staffing, per-run receipts, and a book carried between runs; tickers are a run-time input, not part of the mandate) |
| Persistent ledger — positions, every decision + thesis, orders, fills, NAV history | ✅ (every run saves a full `CycleRecord` receipt and the next opens on it — positions, cash, the high-water mark for the kill-switch, and the prior date dividends accrue from; runs move forward only; CLI, TUI, and web runner all go through `pipeline/ledger.py`) |
| LLM provider layer — one client factory (`make_llm`) routed by the model registry: Anthropic · OpenAI · DeepSeek · Google · xAI · Kimi | ✅ (Ollama next — the free local path, and the last blocker v1 holds over v2) |
| Point-in-time data correctness — as-of / filing-date queries, no lookahead | 🚧 (filing-date queries and dated universes ship; filed multiples are re-struck at the as-of price; dividends accrue from the filed payout since the price feed is unadjusted; a hand-picked universe is flagged as survivorship-biased on every backtest; delisting handling is next) |
| Validation gate — CPCV, probability of backtest overfitting (PBO) | 🚧 (`aihf validate` ships a hold-out split and a parameter sweep over any mandate field; CPCV and PBO next) |

## Analysts (alpha models) — the main contribution surface

Implement the `AlphaModel` interface, return a `Signal`, and it plugs straight into
the engine. Two flavors:

**Quantitative models** (pure math/data):

| Model | Status |
|-------|--------|
| Post-Earnings Announcement Drift (PEAD) | ✅ |
| Momentum (12-1, risk-adjusted) | ✅ |
| Mean reversion (short-term reversal, z-scored + RSI) | ✅ |
| Value / quality factors (own-history value percentiles + quality levels) | ✅ |
| Insider flow (Form 4 net purchase ratio) | ✅ |
| Market-regime detection (HMM / regime-switching) | ⬜ |
| Statistical arbitrage | ⬜ |
| *Your model here* | ⬜ |

**LLM investor agents** (reason over fundamentals in a famous investor's voice, emit
a conviction + thesis). Porting these personas to the alpha-model interface — so each
can be backtested and combined — is a great first contribution:

| Agent | Status |
|-------|--------|
| Warren Buffett | ✅ |
| Charlie Munger · Benjamin Graham · Peter Lynch · Stanley Druckenmiller | ✅ |
| Cathie Wood · Michael Burry · Bill Ackman · Aswath Damodaran | ⬜ |
| Phil Fisher · Mohnish Pabrai · Nassim Taleb · Rakesh Jhunjhunwala | ⬜ |
| *Your agent here* | ⬜ |

## Strategies & allocation

| Item | Status |
|------|--------|
| Strategy — bundle models + a blend policy + capital slice (a "pod") | ✅ (`StrategySpec` + library: fundamental-ls, deep-value, inflections, earnings-drift, trend, insider-flow, quality-value, multifactor) |
| Portfolio construction — blend model views → target weights | ✅ (conviction-weighted, risk-scaled by each name's realized vol; optional market-neutral sleeves) |
| Multi-strategy fund — many pods running at once, netted into one book | ✅ (`run_cycle` nets every sleeve into one target book, then master risk clamps it) |
| Allocator (CIO) — pluggable capital allocation across strategies | 🚧 (static slices ship, and every backtest now attributes the return to each strategy's paper sleeve — the track record an allocator reads; the pluggable interface is next) |
| ↳ Static (human-set dial) | ✅ (capital slices in the mandate) |
| ↳ Risk-parity / inverse-vol | ⬜ |
| ↳ Dynamic — feed winners, cut drawdowns (Millennium-style) | ⬜ |
| ↳ LLM CIO — reasons over regime + each pod's track record | ⬜ |

## Risk & execution

| Item | Status |
|------|--------|
| Risk model — hard caps (pod-level budgets + fund-level limits) | 🚧 (fund-level position, net and gross caps plus a drawdown kill-switch ship; pod budgets with pods) |
| Broker protocol — pluggable, mirrors the `DataClient` pattern | ✅ |
| ↳ Simulated broker (backtest) | ✅ (commission + slippage cost model from the mandate's `execution` block; orders pass a no-trade band first) |
| ↳ Paper broker | ✅ (`AlpacaBroker` on Alpaca's paper API: market orders polled to a complete fill or an error, long↔short flips split in two, the account as the book of record; runs are market-hours and today-only, a first run needs a flat account, drift from the last receipt is flagged) |
| ↳ Live broker (Interactive Brokers / Alpaca) — opt-in plugin, off by default | ⬜ (Alpaca's live endpoint takes the same requests as paper; what's missing is the deliberate opt-in) |

## Autonomy

| Item | Status |
|------|--------|
| Scheduler / daemon — market-calendar cron, idempotent ticks, kill-switch | ⬜ |
| Observability — per-cycle events, notifications, heartbeat | ⬜ |
| Research lab — backtest candidate strategies/allocators alongside the live fund | ⬜ |
| Strategy generator — composes candidate strategies from the building blocks (analysts × policies × parameters), driven by the fund's mandate | ⬜ |
| Auto-promotion — winners graduate into the live fund through the validation gate (CPCV/PBO), human-approved by default (depends: research lab, validation gate) | ⬜ |

## Interfaces

Thin clients over the engine — pick the surface, the core stays the same.

| Item | Status |
|------|--------|
| TUI — the main interface (Textual): build a fund, run it as of today, backtest it, browse every signal's thesis, fund history + delete, model picker, in-app API-key setup | 🚧 (ships and is the default `python -m v2.run`; streaming reasoning + watch mode remain) |
| CLI — thin machine client over the engine: `python -m v2.run mandate.yaml --tickers … [--backtest]`, JSON on stdout | ✅ |
| Web dashboard — replayable, time-scrubbable reasoning ledger | 🚧 (`aihf web` ships on the v2 engine: research desk with ranked, position-aware reports; fund runner with live backtest curve and per-cycle theses; mandate builder; saved reports. Time-scrubbing the ledger is next, now that runs chain) |
| Conversational control plane — operate the fund in natural language | ⬜ |

## Data

| Item | Status |
|------|--------|
| Data layer — pluggable `DataClient` protocol + provider client | ✅ |
| Alternative data connectors — satellite imagery, web & social-media search, app-download trends, shipping data, etc. | 🚧 (web search ships via `aihf research`: a `SearchClient` protocol + Tavily client behind a read-only, cited diagnosis that also reads out every quant model and judges a declared position; insider filings feed the insider-flow model; every research call is saved and `aihf scorecard` grades it against forward excess return — hit rate, bull/bear spread, rank IC; feeding search into an alpha model is next) |

## Contributing

The easiest ways to make an impact:

1. **Add an analyst (alpha model).** Pick an unchecked row above (or invent one),
   implement the `AlphaModel` interface so `predict(...)` returns a `Signal`, and add
   a test. The engine runs it without any other changes.
2. **Add a strategy or an allocator.** Bundle analysts into a strategy, or contribute
   a new capital-allocation policy (CIO). Every analyst, strategy, and allocator also
   becomes a building block the strategy generator can compose — contributions
   compound.
3. **Add an alternative data source.** [Financial Datasets](https://financialdatasets.ai)
   provides the core market, fundamentals, and earnings data. Analysts get more
   powerful with *complementary* datasets — satellite imagery, web & social-media
   search, app-download trends, shipping data, and the like. Add a connector that brings a new,
   unique signal into the mix.
4. **Build out a planned component** (portfolio construction, risk, brokers, scheduler,
   validation, an interface).

Before starting something large, open an issue to claim it so work isn't duplicated.
PRs that update this roadmap's status are encouraged.
