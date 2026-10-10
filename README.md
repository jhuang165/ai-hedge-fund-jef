# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund. The goal of this project is to explore the use of AI to make trading decisions. This project is for **educational** purposes only and is not intended for real trading or investment.

> **🚧 The project is evolving.** We're rebuilding it into a persistent, always-on AI hedge fund — a *fund* as a first-class entity you can backtest, paper-trade, and (opt-in) run live, with the investor agents reimagined as pluggable, backtestable "alpha models." Read the **[Vision →](VISION.md)** and the **[Roadmap →](ROADMAP.md)**.

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## How to Install

```bash
pipx install aihf
```

(or `uv tool install aihf`, or `pip install aihf` into an environment of your choice)

Then run it from anywhere:

```bash
aihf
```

### API keys

The app asks for keys the first time it needs them and saves them to `~/.hedge-fund/.env` — nothing to configure up front. It can use:

- A [Financial Datasets](https://financialdatasets.ai) API key, for point-in-time prices, fundamentals, and earnings. Optional: without it the fund reads Yahoo Finance (no key), which is fine for research as of today but only approximates filing dates, so backtests you intend to believe should use the keyed feed. `HEDGE_FUND_DATA_SOURCE=fd|yfinance` forces one or the other.
- One LLM API key for the LLM-powered alpha models. Supported providers: Anthropic, OpenAI, DeepSeek, Google, xAI, Kimi.
- Optionally, a [Tavily](https://tavily.com) API key (`TAVILY_API_KEY`) — only needed for `aihf research` when it searches and reasons by itself; see "Research from Claude Code" below for a path that needs no key at all.

Keys exported in your shell always win over the saved file.

## How to Run

### Interactive app

```bash
aihf
```

With no arguments, this launches the interactive terminal app. Build a fund — pick stocks, strategies, rebalance cadence — or backtest a saved fund and watch its equity curve draw against its benchmark. Funds you build are saved as mandate files in `~/.hedge-fund/mandates/`.

### Web app

The same engine in a browser, on your machine:

```bash
aihf web
```

This starts a local server at `http://127.0.0.1:8787` and opens it. The **Research** page ranks any list of names (with positions, as above) and shows each full report: the action and its rationale, the thesis, the cited catalysts and risks, the desk readout of every quant model, and the price-action snapshot. The **Fund** page runs a saved mandate for one cycle or backtests it with a live equity curve against the benchmark, and can compose a new mandate from the strategy library. Every research run is saved under `~/.hedge-fund/research/` and browsable on the **Reports** page; **Settings** shows which API keys are set and lets you save them. It binds to localhost only and has no authentication.

### Hourly reporter

A second web app that runs itself: every hour it writes a short, cited brief on each name in a watchlist, from the live quote, the last two days of headlines, the quant desk's readings, and a model that searches the web for what happened since the last brief. It is built to live on a small server and be read from anywhere:

```bash
aihf reporter                      # http://0.0.0.0:8788, scheduler on
aihf reporter --once --tickers NVDA  # one cycle, then exit
```

The model is reached through any OpenAI-compatible endpoint; the intended one is [circlemouth/Codex-Wrapper](https://github.com/circlemouth/Codex-Wrapper) in front of the Codex CLI signed in with a ChatGPT account, with Codex's own live web search on, so no API key is involved. Briefs are kept in SQLite with their evidence; the page shows a card per name (price, signal, confidence, action, headline, what changed since last time) and the full brief with its sources, catalysts, risks, what to watch next, and history. Symbols are added and removed on the page. Configuration is by environment variable (`hedge_fund/reporter/config.py`); the two-container stack and free hosting options are in [`deploy/`](deploy/README.md). The dashboard can also be mirrored to a static host after every cycle (`REPORTER_PUBLISH_DIR`, `aihf reporter --publish`): a read-only copy of the page and its data as files, which is how it runs on Firebase Hosting at a fixed free URL.

### Non-interactive

Run one fund cycle from a mandate file. The full cycle record prints to stdout as JSON; a short human summary goes to stderr:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT
```

Backtest the mandate over history at its rebalance cadence:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
```

A mandate is the desk — strategies, staff, risk, capital, cadence — and never names tickers; `--tickers` says what to point it at for this run.

A fund keeps its book between runs. Every run saves a receipt in `~/.hedge-fund/mandates/`, and the next run opens on the newest one: its positions, its cash, and its high-water mark for the drawdown kill-switch. So NAV builds into a track record instead of resetting to the mandate's capital. The first run opens on the capital. A run can repeat the last run's date or move forward, but never go back; history is what `--backtest` is for, and a backtest always starts fresh.

To paper-trade at real prices, point the fund at an Alpaca paper account. Put the paper keys in `~/.hedge-fund/.env` (or the web app's Settings page) and set the broker on the mandate:

```yaml
execution:
  broker: alpaca-paper        # APCA_API_KEY_ID / APCA_API_SECRET_KEY
```

Orders then go to Alpaca as market orders and fill at real quotes, and the account is the book of record: each run reads its positions and cash from Alpaca, checks them against the last receipt, and flags any difference on the new one. A paper run trades today, during market hours only. The fund treats everything in the account as its own and will sell what it doesn't want, so give each fund its own paper account; its first run refuses an account that isn't flat. To hand an account's existing positions to the fund on purpose (or to recover when a first run failed after some orders had filled), pass `--adopt-account`. Paper runs keep their own track record, separate from the fund's simulated runs, and backtests always simulate.

A backtest is charged for trading. The mandate's `execution` block sets the simulated broker's commission and slippage (five basis points of slippage per side by default) and a no-trade band that skips rebalances too small to be worth their costs (half a percent of equity by default; exits always trade in full). The costs paid show up on every cycle record and in the backtest metrics, so the equity curve is net of friction.

A backtest over tickers you type in today is survivorship-biased: every name on the list is one you already know made it. The result carries a warning saying so, on the receipt and on every screen. To remove the bias, backtest over a dated universe file instead, which lists which names were investable and since when, so names join and leave the book as they did in history. An example lives at `~/.hedge-fund/universes/example.yaml`:

```bash
aihf ~/.hedge-fund/mandates/example.yaml --universe ~/.hedge-fund/universes/example.yaml --backtest
```

Master risk has four hard limits: a per-name cap, a gross-exposure cap, a net-exposure cap on how far the book may lean long or short, and a drawdown kill-switch. When equity falls that far below its peak, the fund closes to flat and stays flat, and the backtest reports the date it happened.

Dividends count on both sides. The price feed is unadjusted and carries no dividend events, so the fund accrues each held name's trailing dividend per share, the filed payout ratio times EPS, pro-rated between cycles, and a short owes it. The benchmark is an ETF that files nothing, so its yield is a stated assumption in the mandate's `dividends` block. The personas and the quality-value model also judge valuation at the current price rather than the price the filing was struck at: P/E, P/B, and free-cash-flow yield are recomputed from the filed per-share figures and the last close, snapped to a ten-percent price grid so a persona re-reasons when the valuation has moved a tenth, not on every tick.

Every backtest also says who earned it. Each strategy gets a paper sleeve, its own target weights compounded on its capital slice at the fund's marks, with its own return, Sharpe, and drawdown, and the residual reports what the sleeves do not explain: master risk clamps, whole-share sizing, cash drag, and costs. Sizing inside a sleeve is risk-scaled by default, so two names the desk likes equally get equal risk rather than equal dollars; a strategy can switch that off with `vol_scaled: false` in its blend policy.

### Research a stock, or your positions

Diagnose a ticker from live web search, its point-in-time fundamentals, its price action, and a readout of every systematic model on the desk (momentum, short-term reversal, insider flow, quality-value, post-earnings drift). The report JSON prints to stdout; a summary with the action, the desk readout, and the cited sources goes to stderr:

```bash
aihf research AAPL
```

Tell it what you hold and it answers the question you are actually asking. A bare ticker gets **buy / watch / avoid**; a ticker with a position gets **add / hold / trim / exit**, judged on the position as it stands (never anchored on your cost basis):

```bash
aihf research AAPL:100@150.25
```

Several names come back ranked, best-liked first, in one table. Combine direct tickers with a portfolio file of `{ticker, shares, cost_basis}` entries (negative shares for a short):

```bash
aihf research NVDA MSFT:20@400 --portfolio ~/holdings.yaml
```

This is read-only and stands apart from the fund: it touches no mandate, trades nothing, and writes nothing to the ledger. Every run is saved under `~/.hedge-fund/research/`. Run this way it needs `TAVILY_API_KEY` and an LLM key in addition to a data source. Every catalyst and risk cites the sources it rests on, so you can check the claim against the page it came from, and the desk readout shows each model's arithmetic so you can check the numbers too.

### Research from Claude Code

The research pipeline is split at the LLM, so an agent that is already running inside the checkout can be the analyst instead of an API call. `--dossier` gathers the evidence and stops, needing no search or LLM key; `--ingest` takes the finished diagnosis back, validates it exactly as an LLM's answer would be, saves it, and prints the report:

```bash
aihf research AAPL:100@150.25 --dossier --out dossier.json
```

The dossier carries the fundamentals snapshot, the price-action snapshot, the desk readout, the marked position, the four search queries the desk would run, and the exact prompt it would send. The reasoner searches, writes the answer JSON the prompt asks for (plus its `sources` and the untouched `dossier`), and hands it back:

```bash
aihf research --ingest answer.json
```

An answer that leaves `action` out gets one from the project's standing policy (`derive_action` in `hedge_fund/research/dossier.py`): a confident bullish call on a name not held is "buy", a weak one "watch", bearish "avoid"; on a held name, a confident call with the position is "add", against it "exit", weak ones "hold" and "trim". The trade action is then the fund's rule, not the reasoner's opinion. In this checkout `/research AAPL` in Claude Code runs this whole loop with the session as the analyst (see `.claude/skills/research/`), and `/scorecard` grades the calls later like any other.

### Grade the research desk

Nothing above checks whether "buy" beat "avoid". The scorecard does. It takes every saved report old enough to judge, measures the stock's return over the horizon after it less the benchmark's, and grades whether it went the way the call said:

```bash
aihf scorecard --horizon 90 --benchmark SPY
```

Three numbers say whether the desk has an edge: the hit rate of its directional calls, the spread between bullish and bearish calls' average excess return, and the rank correlation between conviction and outcome. Calls whose horizon has not elapsed are pending, not wrong. The web app shows the same scorecard on its Reports page.

### Validate a backtest before believing it

Every knob in a mandate was chosen while looking at some backtest. Two cheap checks catch most fitted results. A hold-out split backtests the window in two halves, the one you tuned on and the one you did not, and reports how much the Sharpe decayed. A parameter sweep re-runs the backtest across values of any mandate field, reached by dotted path, and says whether the result depends on the choice:

```bash
aihf validate ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --split 2025-01-01
```

```bash
aihf validate ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --sweep strategies.0.models.0.weight=1,2,4 --sweep execution.slippage_bps=0,5,20
```

A surface whose Sharpe changes sign across the grid, or swings by more than a full point, is reported as fragile.

## Development

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
poetry run aihf
poetry run pytest hedge_fund
```

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused. This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
