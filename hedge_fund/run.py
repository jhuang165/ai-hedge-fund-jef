"""Run the AI hedge fund.

Usage::

    aihf
        No arguments: the interactive app — a Textual TUI (the same app as
        `aihf` with no arguments). Build a fund — pick stocks, strategies, rebalance
        cadence — or backtest a saved fund and watch its equity curve draw
        against its benchmark.

    aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT
        With a mandate: run one cycle non-interactively. The full CycleRecord
        prints to stdout as JSON (pipe it anywhere); a short human summary
        goes to stderr. Add --out record.json to also write it to a file.
        The run opens on the fund's last saved book and saves its receipt
        to ~/.hedge-fund/mandates/ — see hedge_fund/pipeline/ledger.py.

    aihf ~/.hedge-fund/mandates/example.yaml --tickers AAPL,MSFT --backtest
        Backtest the mandate: run_cycle looped over history at the mandate's
        rebalance cadence; the full result JSON prints to stdout.

    aihf web
        The local web app (http://127.0.0.1:8787): the research desk and the
        fund runner in a browser, over the same engine — see hedge_fund/web/.

    aihf research AAPL [MSFT:20@400 ...] [--portfolio holdings.yaml]
        Read-only research: a cited bullish/neutral/bearish diagnosis with a
        concrete action (buy/watch/avoid, or add/hold/trim/exit for a name
        you say you hold), informed by fundamentals, price action, web
        search, and every systematic model's reading. Several tickers come
        back ranked. Does not touch mandates, strategies, or backtesting —
        see hedge_fund/research/.

A mandate is the desk — strategies, staff, risk, capital, cadence — and never
names tickers; --tickers says what to point it at for this run.

Both paths run the same engine underneath. The interactive app is a thin
client: it only *composes a FundSpec* — the same machine-facing YAML this
CLI reads. Humans click, machines write, the engine reads one thing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

from rich.console import Console
from rich.table import Table

from hedge_fund.backtesting import backtest_fund
from hedge_fund.brokers import AlpacaError
from hedge_fund.data import CachedDataClient, FDClient
from hedge_fund.fund import Fund, load_spec, load_universe, normalize_universe
from hedge_fund.paths import ENV_PATH, ensure_mandates_dir, ensure_universes_dir
from hedge_fund.pipeline.ledger import LedgerError, run_carried
from hedge_fund.research.models import ResearchReport
from hedge_fund.tui.keys import apply_credentials
from hedge_fund.tui.shared import _BACKTEST_WEEKS


def main() -> None:
    # A mandate path is always a .yaml file, so this can't collide with a
    # real mandate. Handled before the main parser so the existing
    # aihf / aihf <mandate.yaml> ... behaviors stay byte-for-byte unchanged.
    if len(sys.argv) > 1 and sys.argv[1] == "research":
        _research_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "scorecard":
        _scorecard_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        _validate_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Imported lazily: the CLI and TUI never pay to load FastAPI/uvicorn.
        from hedge_fund.web.server import main as web_main

        web_main(sys.argv[2:])
        return

    apply_credentials()
    ensure_mandates_dir()
    ensure_universes_dir()
    parser = argparse.ArgumentParser(
        prog="aihf",
        description="Run the AI hedge fund. No arguments: launch the "
        "interactive app. With a mandate YAML: run one cycle and print the "
        "record.",
        epilog="other commands:\n"
        "  aihf research TICKER [TICKER:SHARES@COST ...] [--portfolio FILE]\n"
        "                         read-only cited diagnosis + action per stock\n"
        "                         (see `aihf research --help`)\n"
        "  aihf scorecard         grade every saved research call against what\n"
        "                         the market did next (see `aihf scorecard --help`)\n"
        "  aihf validate MANDATE  hold-out split and parameter sweep of a\n"
        "                         mandate's backtest (see `aihf validate --help`)\n"
        "  aihf web [--port 8787] the local web app: research desk + fund runner\n"
        "                         (see `aihf web --help`)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mandate", nargs="?",
                        help="path to a fund spec YAML, e.g. "
                        "~/.hedge-fund/mandates/example.yaml "
                        "(omit to launch the interactive app)")
    parser.add_argument(
        "--tickers",
        help="what to trade this run, comma or space separated, e.g. "
        "AAPL,MSFT,NVDA — required with a mandate (a fund carries no "
        "watchlist; the universe is a run-time input)",
    )
    parser.add_argument(
        "--universe",
        help="a dated universe YAML (entries of {from: YYYY-MM-DD, tickers: "
        "[...]}) instead of --tickers: point-in-time membership, so a "
        "backtest trades the names that were investable on each date rather "
        "than today's survivors",
    )
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="as-of date YYYY-MM-DD (default: today); models only see data "
        "filed by this date",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="backtest the mandate instead of running one cycle: one run_cycle "
        "per rebalance date from --start to --date, full result JSON on stdout",
    )
    parser.add_argument(
        "--start",
        help=f"backtest start date YYYY-MM-DD (default: {_BACKTEST_WEEKS} weeks "
        "before --date)",
    )
    parser.add_argument(
        "--model",
        help="LLM the investor agents reason with, e.g. claude-opus-5 "
        "(default: HEDGE_FUND_LLM_MODEL env, else the built-in default); quant models "
        "ignore it",
    )
    parser.add_argument(
        "--adopt-account", action="store_true",
        help="alpaca-paper funds only: let the fund's first paper run take over "
        "an account that already holds positions — they become the fund's to "
        "keep or sell",
    )
    parser.add_argument("--out", help="also write the record JSON to this file")
    args = parser.parse_args()

    if args.model:
        os.environ["HEDGE_FUND_LLM_MODEL"] = args.model

    if args.mandate is None:
        # The interactive experience is the Textual app. Import it lazily so
        # the non-interactive path never pays to load Textual.
        from hedge_fund.tui.app import HedgeFundApp

        HedgeFundApp().run()
        return

    if bool(args.tickers) == bool(args.universe):
        parser.error("give exactly one of --tickers AAPL,MSFT or --universe FILE")
    if args.universe:
        dated = load_universe(args.universe)
        # A backtest asks the file for each date's members; a single cycle
        # is one date, so resolve it here.
        universe = dated if args.backtest else dated.as_of(args.date)
    else:
        universe = normalize_universe(args.tickers.replace(",", " ").split())

    console = Console(stderr=True)  # status + summary on stderr; stdout stays pure JSON
    spec = load_spec(args.mandate)
    fund = Fund(spec)

    if args.backtest:
        start = args.start or (
            _date.fromisoformat(args.date) - timedelta(weeks=_BACKTEST_WEEKS)
        ).isoformat()
        with FDClient() as raw:
            fd = CachedDataClient(raw)
            with console.status(
                f"[cyan]{spec.name}: backtesting {start} → {args.date} "
                f"({spec.rebalance} rebalance vs {spec.benchmark}) "
                f"over {_describe(universe)}…",
                spinner="dots",
            ):
                result = backtest_fund(fund, start, args.date, fd, universe)
        print(result.model_dump_json(indent=2))
        if args.out:
            Path(args.out).write_text(result.model_dump_json(indent=2))
        m = result.metrics
        console.print(
            f"[bold]{spec.name}[/] {result.start} → {result.end}  ·  "
            f"{m.n_cycles} cycles  ·  return {m.total_return_pct:+.1%} "
            f"vs {spec.benchmark} {m.benchmark_return_pct:+.1%}  ·  "
            f"sharpe {m.sharpe_ratio:.2f}  ·  max drawdown {m.max_drawdown_pct:.1%}  ·  "
            f"costs ${m.total_costs:,.0f} ({m.costs_pct:.2%})  ·  "
            f"dividends ${m.total_dividends:,.0f}"
        )
        for sa in result.attribution.strategies:
            console.print(
                f"[dim]  {sa.name} ({sa.slice:.0%} of capital): "
                f"sleeve {sa.total_return_pct:+.1%}  ·  sharpe {sa.sharpe_ratio:.2f}  ·  "
                f"max drawdown {sa.max_drawdown_pct:.1%}  ·  "
                f"contributed {sa.contribution_pct:+.1%}[/]"
            )
        console.print(
            f"[dim]  residual (risk clamps, sizing, cash, costs): "
            f"{result.attribution.residual_pct:+.1%}[/]"
        )
        if m.kill_switch_date:
            console.print(
                f"[bold red]  kill-switch: drawdown limit hit on {m.kill_switch_date}; "
                "the fund closed to flat and stayed there[/]"
            )
        for warning in result.warnings:
            console.print(f"[bold yellow]  ⚠ {warning}[/]")
        return

    with FDClient() as raw:
        fd = CachedDataClient(raw)
        n_models = sum(len(staff) for _, staff in fund.strategies)
        with console.status(
            f"[cyan]{spec.name}: running one cycle as of {args.date} — "
            f"{len(universe)} tickers x {n_models} models "
            f"across {len(fund.strategies)} strategies…",
            spinner="dots",
        ):
            try:
                ran = run_carried(fund, args.date, fd, universe,
                                  adopt_account=args.adopt_account)
            except LedgerError as exc:
                parser.error(str(exc))
            except AlpacaError as exc:
                console.print(f"[bold red]Alpaca: {exc}[/]")
                sys.exit(1)
    record = ran.record

    print(record.model_dump_json(indent=2))
    if args.out:
        Path(args.out).write_text(record.model_dump_json(indent=2))

    for sr in record.strategies:
        abstained = sum(1 for s in sr.signals if s.metadata.get("abstained") is True)
        console.print(
            f"[dim]  {sr.name} ({sr.slice:.0%} of capital): "
            f"{len(sr.signals)} signals ({abstained} abstained)[/]"
        )
    n_signals = sum(len(sr.signals) for sr in record.strategies)
    console.print(
        f"[bold]{spec.name}[/] @ {record.as_of}  ·  "
        f"{len(record.strategies)} strategies  ·  {n_signals} signals  ·  "
        f"{len(record.clamps)} risk clamps  ·  "
        f"{len(record.orders)} orders  ·  costs ${record.costs:,.2f}  ·  NAV ${record.nav:,.2f}"
    )
    if record.skipped:
        console.print(f"[dim]skipped: {', '.join(s.ticker for s in record.skipped)}[/]")
    for warning in record.warnings:
        console.print(f"[bold yellow]  ⚠ {warning}[/]")
    console.print(f"[dim]{ran.carry.describe()}  ·  saved to {ran.path}[/]")


def _research_main(argv: list[str]) -> None:
    apply_credentials()
    parser = argparse.ArgumentParser(
        prog="aihf research",
        description="Read-only research on one or more tickers: a cited "
        "bullish/neutral/bearish diagnosis with a concrete action, informed "
        "by fundamentals, price action, web search, and a readout of every "
        "systematic model on the desk. Does not touch mandates, strategies, "
        "or backtesting.",
        epilog="examples:\n"
        "  aihf research AAPL                      one name, not held: buy / watch / avoid\n"
        "  aihf research AAPL:100@150.25           held 100 sh at $150.25: add / hold / trim / exit\n"
        "  aihf research AAPL MSFT NVDA            several names, ranked\n"
        "  aihf research --portfolio holdings.yaml every position in the file\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets", nargs="*", metavar="TICKER[:SHARES@COST]",
        help="tickers to research; append :SHARES@COST to describe a position "
        "you hold (negative shares for a short), e.g. AAPL:100@150.25",
    )
    parser.add_argument(
        "--portfolio", metavar="FILE",
        help="YAML of positions to research (positions: [{ticker, shares, "
        "cost_basis}, ...]); combines with any tickers given directly",
    )
    parser.add_argument(
        "--model",
        help="LLM to reason with, e.g. claude-opus-5 (default: "
        "HEDGE_FUND_LLM_MODEL env, else the built-in default)",
    )
    parser.add_argument(
        "--universe",
        help="a dated universe YAML (entries of {from: YYYY-MM-DD, tickers: "
        "[...]}) instead of --tickers: point-in-time membership, so a "
        "backtest trades the names that were investable on each date rather "
        "than today's survivors",
    )
    parser.add_argument(
        "--date",
        default=_date.today().isoformat(),
        help="as-of date YYYY-MM-DD (default: today)",
    )
    parser.add_argument("--out", help="also write the report JSON to this file")
    args = parser.parse_args(argv)

    from hedge_fund.research import load_portfolio, merge_targets, parse_target

    try:
        given = [parse_target(t) for t in args.targets]
        from_file = load_portfolio(args.portfolio) if args.portfolio else []
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    targets = merge_targets(from_file, given)
    if not targets:
        parser.error("nothing to research — give at least one ticker or --portfolio FILE")

    # Check the search key up front. Without it every query comes back 401
    # and the user sees four stacked HTTP errors instead of the one sentence
    # that actually fixes it.
    if not os.environ.get("TAVILY_API_KEY"):
        raise SystemExit(
            "TAVILY_API_KEY is not set — `aihf research` needs a web-search key.\n"
            "Get one at https://tavily.com/ and either export it or add it to "
            f"{ENV_PATH}:\n"
            "    TAVILY_API_KEY=your-tavily-api-key"
        )

    from hedge_fund.llm import make_llm
    from hedge_fund.research import TavilyClient, diagnose

    llm = make_llm(args.model)
    console = Console(stderr=True)
    reports: list[ResearchReport] = []
    failures: list[tuple[str, Exception]] = []

    with FDClient() as raw, TavilyClient() as search:
        fd = CachedDataClient(raw)
        for i, target in enumerate(targets, 1):
            label = target.ticker
            if target.position is not None:
                label += f" ({target.position.side} {abs(target.position.shares):g} @ {target.position.cost_basis:.2f})"
            with console.status(
                f"[cyan]researching {label} as of {args.date}"
                + (f"  [{i}/{len(targets)}]" if len(targets) > 1 else "") + "…",
                spinner="dots",
            ):
                try:
                    reports.append(diagnose(
                        target.ticker, args.date, fd, search, llm, position=target.position,
                    ))
                except Exception as exc:
                    if len(targets) == 1:
                        raise
                    # A batch keeps going: one bad name should not cost the
                    # other reports. The failure is surfaced and the exit
                    # code says so.
                    failures.append((target.ticker, exc))
                    console.print(f"[red]{target.ticker}: {type(exc).__name__}: {exc}[/]")

    # Every run is saved, so the scorecard can grade it later.
    from hedge_fund.research import save_report

    for report in reports:
        save_report(report)

    ranked = sorted(reports, key=lambda r: r.score, reverse=True)
    payload = ranked[0].model_dump_json(indent=2) if len(targets) == 1 else (
        "[" + ",\n".join(r.model_dump_json(indent=2) for r in ranked) + "]"
    )
    if reports:
        print(payload)
        if args.out:
            Path(args.out).write_text(payload)

    if len(targets) == 1 and reports:
        _print_research_summary(console, reports[0])
    elif reports:
        _print_research_table(console, ranked)
    if failures:
        raise SystemExit(f"{len(failures)} of {len(targets)} names failed: "
                         + ", ".join(t for t, _ in failures))


def _scorecard_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aihf scorecard",
        description="Grade every saved research report (~/.hedge-fund/research/) "
        "old enough to judge: each call's forward return over the horizon, less "
        "the benchmark's, and whether it went the way the call said. JSON on "
        "stdout; the summary on stderr.",
    )
    parser.add_argument("--horizon", type=int, default=90,
                        help="calendar days after the report to measure over (default 90)")
    parser.add_argument("--benchmark", default="SPY",
                        help="what excess return is measured against (default SPY)")
    parser.add_argument("--date", default=_date.today().isoformat(),
                        help="grade as of this date YYYY-MM-DD (default: today)")
    parser.add_argument("--dir", help="read reports from this directory instead")
    parser.add_argument("--out", help="also write the scorecard JSON to this file")
    args = parser.parse_args(argv)

    apply_credentials()
    from hedge_fund.research import grade, load_reports

    console = Console(stderr=True)
    reports = load_reports(Path(args.dir) if args.dir else None)
    if not reports:
        raise SystemExit("no saved research reports to grade — run `aihf research` first")
    with FDClient() as raw:
        fd = CachedDataClient(raw)
        with console.status(f"[cyan]grading {len(reports)} reports…", spinner="dots"):
            card = grade(reports, fd, horizon_days=args.horizon,
                         benchmark=args.benchmark.upper(), as_of=args.date)
    print(card.model_dump_json(indent=2))
    if args.out:
        Path(args.out).write_text(card.model_dump_json(indent=2))
    _print_scorecard(console, card)


def _print_scorecard(console: Console, card) -> None:
    def pct(v):
        return "-" if v is None else f"{v:+.1%}"

    console.print(
        f"[bold]scorecard[/] {card.horizon_days}-day excess vs {card.benchmark} "
        f"as of {card.graded_as_of}  ·  {card.n_graded} graded  ·  "
        f"{card.n_pending} pending  ·  {card.n_skipped} skipped"
    )
    if not card.n_graded:
        console.print("[dim]  nothing old enough to grade yet[/]")
        return
    console.print(
        f"  hit rate {'-' if card.hit_rate is None else f'{card.hit_rate:.0%}'}  ·  "
        f"bull − bear spread {pct(card.bull_bear_spread_pct)}  ·  "
        f"rank IC {'-' if card.rank_ic is None else f'{card.rank_ic:+.2f}'}"
    )
    table = Table(title="by action", title_justify="left")
    for col, justify in (("action", "left"), ("calls", "right"), ("hit rate", "right"),
                         ("avg excess", "right"), ("median excess", "right")):
        table.add_column(col, justify=justify)
    for g in card.by_action:
        table.add_row(g.key, str(g.n),
                      "-" if g.hit_rate is None else f"{g.hit_rate:.0%}",
                      pct(g.avg_excess_pct), pct(g.median_excess_pct))
    console.print(table)


def _validate_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aihf validate",
        description="Stress a mandate's backtest before believing it. A hold-out "
        "split backtests the window in two halves and compares them; a "
        "parameter sweep re-runs the backtest across values of any mandate field "
        "and reports how much the result depends on the choice. JSON on stdout; "
        "the summary on stderr.",
        epilog="examples:\n"
        "  aihf validate example.yaml --tickers AAPL,MSFT --split 2025-01-01\n"
        "  aihf validate example.yaml --tickers AAPL,MSFT \\\n"
        "      --sweep strategies.0.models.0.weight=1,2,4 --sweep execution.slippage_bps=0,5,20",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mandate", help="path to a fund spec YAML")
    parser.add_argument("--tickers", help="what to trade, comma or space separated")
    parser.add_argument("--universe", help="a dated universe YAML instead of --tickers")
    parser.add_argument("--date", default=_date.today().isoformat(),
                        help="end of the window YYYY-MM-DD (default: today)")
    parser.add_argument("--start",
                        help=f"start of the window (default: {_BACKTEST_WEEKS} weeks before --date)")
    parser.add_argument("--split",
                        help="hold-out boundary YYYY-MM-DD: train on [start, split), "
                        "test on [split, date] (default: the midpoint)")
    parser.add_argument("--sweep", action="append", default=[], metavar="PATH=V1,V2,...",
                        help="dotted path into the mandate and the values to try; repeat "
                        "for a grid, e.g. strategies.0.models.0.params.scale=0.5,1,2")
    parser.add_argument("--model", help="LLM the investor agents reason with")
    parser.add_argument("--out", help="also write the validation JSON to this file")
    args = parser.parse_args(argv)

    if args.model:
        os.environ["HEDGE_FUND_LLM_MODEL"] = args.model
    if bool(args.tickers) == bool(args.universe):
        parser.error("give exactly one of --tickers AAPL,MSFT or --universe FILE")
    apply_credentials()
    from hedge_fund.validation import holdout, parse_sweep, sweep

    universe = (load_universe(args.universe) if args.universe
                else normalize_universe(args.tickers.replace(",", " ").split()))
    spec = load_spec(args.mandate)
    start = args.start or (
        _date.fromisoformat(args.date) - timedelta(weeks=_BACKTEST_WEEKS)
    ).isoformat()
    split = args.split or _midpoint(start, args.date)
    overrides = parse_sweep(args.sweep)

    console = Console(stderr=True)
    payload: dict = {}
    with FDClient() as raw:
        fd = CachedDataClient(raw)
        with console.status(f"[cyan]{spec.name}: hold-out {start} | {split} | {args.date}…",
                            spinner="dots"):
            payload["holdout"] = holdout(Fund(spec), start, split, args.date, fd, universe)
        if overrides:
            with console.status(f"[cyan]{spec.name}: sweeping {', '.join(overrides)}…",
                                spinner="dots"):
                payload["sweep"] = sweep(spec, overrides, start, args.date, fd, universe,
                                         split=split)
    text = json.dumps({k: v.model_dump() for k, v in payload.items()}, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    _print_validation(console, payload)


def _midpoint(start: str, end: str) -> str:
    a, b = _date.fromisoformat(start), _date.fromisoformat(end)
    return (a + (b - a) / 2).isoformat()


def _print_validation(console: Console, payload: dict) -> None:
    h = payload["holdout"]
    console.print(
        f"[bold]hold-out[/] train {h.train_start} → {h.train_end}: "
        f"return {h.train.total_return_pct:+.1%} sharpe {h.train.sharpe_ratio:.2f}  ·  "
        f"test {h.test_start} → {h.test_end}: "
        f"return {h.test.total_return_pct:+.1%} sharpe {h.test.sharpe_ratio:.2f}  ·  "
        f"decay {h.sharpe_decay:+.2f}"
    )
    console.print(f"[bold {'green' if h.holds else 'red'}]  {h.verdict}[/]")
    sw = payload.get("sweep")
    if sw is None:
        return
    table = Table(title=f"sweep over {', '.join(sw.paths)}", title_justify="left")
    for path in sw.paths:
        table.add_column(path.split(".")[-1])
    for col in ("return", "sharpe", "max dd", "test sharpe"):
        table.add_column(col, justify="right")
    for pt in sw.points:
        table.add_row(*[str(pt.overrides[p]) for p in sw.paths],
                      f"{pt.metrics.total_return_pct:+.1%}", f"{pt.metrics.sharpe_ratio:.2f}",
                      f"{pt.metrics.max_drawdown_pct:.1%}",
                      "-" if pt.test is None else f"{pt.test.sharpe_ratio:.2f}")
    console.print(table)
    console.print(
        f"  sharpe {sw.sharpe_min:.2f} … {sw.sharpe_max:.2f} (median {sw.sharpe_median:.2f})  ·  "
        f"{sw.share_positive:.0%} of points positive"
    )
    console.print(f"[bold {'red' if sw.fragile else 'green'}]  {sw.verdict}[/]")


_SIGNAL_COLOR = {"bullish": "green", "bearish": "red", "neutral": "yellow"}
_ACTION_COLOR = {
    "buy": "green", "add": "green",
    "watch": "yellow", "hold": "yellow",
    "trim": "red", "exit": "red", "avoid": "red",
}


def _print_research_summary(console: Console, report: ResearchReport) -> None:
    color = _SIGNAL_COLOR[report.signal]
    console.print(
        f"[bold]{report.ticker}[/] @ {report.as_of}  ·  "
        f"[{color}]{report.signal.upper()}[/] ({report.confidence:.0f}% confidence)  ·  "
        f"model {report.model}"
    )
    if report.position is not None:
        p = report.position
        pnl = f"  ·  unrealized {p.unrealized_pnl_pct:+.1%}" if p.unrealized_pnl_pct is not None else ""
        console.print(
            f"[dim]position: {p.side} {abs(p.shares):g} sh @ {p.cost_basis:.2f}{pnl}[/]"
        )
    console.print(
        f"[bold {_ACTION_COLOR[report.action]}]{report.action.upper()}[/]  {report.action_rationale}"
    )
    console.print(f"[dim]{report.thesis.splitlines()[0] if report.thesis else ''}[/]")
    for s in report.desk:
        if s.metadata.get("abstained") is True:
            console.print(f"[dim]  desk  {s.model_name:<15} abstained[/]")
        else:
            c = "green" if s.value > 0.05 else "red" if s.value < -0.05 else "yellow"
            console.print(f"[dim]  desk  {s.model_name:<15} [{c}]{s.value:+.2f}[/]  {s.reasoning or ''}[/]")
    console.print(
        f"[dim]{len(report.catalysts)} catalysts  ·  {len(report.risks)} risks  ·  "
        f"{len(report.sources)} sources across {len(report.queries)} queries[/]"
    )
    for i, s in enumerate(report.sources):
        console.print(f"[dim]  [{i}] {s.title} — {s.url}[/]")
    for w in report.warnings:
        console.print(f"[yellow]warning: {w}[/]")


def _print_research_table(console: Console, ranked: list[ResearchReport]) -> None:
    """One row per name, best-liked first — the desk's rank order."""
    table = Table(title=f"research as of {ranked[0].as_of}", title_justify="left")
    table.add_column("ticker", style="bold")
    table.add_column("signal")
    table.add_column("conf", justify="right")
    table.add_column("action")
    table.add_column("position")
    table.add_column("P&L", justify="right")
    table.add_column("mom", justify="right")
    table.add_column("q-val", justify="right")
    table.add_column("insiders", justify="right")
    table.add_column("thesis")

    for r in ranked:
        desk = {s.model_name: s for s in r.desk}
        p = r.position
        table.add_row(
            r.ticker,
            f"[{_SIGNAL_COLOR[r.signal]}]{r.signal}[/]",
            f"{r.confidence:.0f}",
            f"[{_ACTION_COLOR[r.action]}]{r.action}[/]",
            f"{p.side} {abs(p.shares):g} @ {p.cost_basis:.2f}" if p else "-",
            f"{p.unrealized_pnl_pct:+.1%}" if p and p.unrealized_pnl_pct is not None else "-",
            _desk_cell(desk.get("momentum")),
            _desk_cell(desk.get("quality-value")),
            _desk_cell(desk.get("insider-flow")),
            (r.thesis.splitlines()[0][:70] if r.thesis else ""),
        )
    console.print(table)
    for r in ranked:
        for w in r.warnings:
            console.print(f"[yellow]{r.ticker}: warning: {w}[/]")


def _desk_cell(signal) -> str:
    if signal is None or signal.metadata.get("abstained") is True:
        return "[dim]-[/]"
    c = "green" if signal.value > 0.05 else "red" if signal.value < -0.05 else "yellow"
    return f"[{c}]{signal.value:+.2f}[/]"


def _describe(universe) -> str:
    if isinstance(universe, list):
        return ", ".join(universe)
    return f"a dated universe of {len(universe.tickers)} names from {universe.start}"


if __name__ == "__main__":
    main()
