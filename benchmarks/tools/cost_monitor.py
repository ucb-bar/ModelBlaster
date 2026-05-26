"""Live LLM cost monitor.

Tails one or more ``llm_calls.jsonl`` files in real time and renders a
rich TUI showing running tokens + dollars per model and an overall
total. The display resizes with the terminal automatically.

By default the monitor watches every ``llm_calls.jsonl`` under
``benchmarks/results/``, so launching one instance covers every
in-flight arm driver run. Point ``--paths`` at specific files to
focus on a single cell.

Cost computation reads ``benchmarks/config/pricing.yaml``; entries
flagged ``placeholder: true`` are skipped (their $ contribution is
shown as ``--``) so the running total never silently includes
guessed rates.

Layout (refreshes at ``--refresh-hz``, default 4 Hz):

    +---- Summary -----+----- Recent calls ------+
    |   total $        |  hh:mm:ss  cell  phase  |
    |   total tokens   |  hh:mm:ss  ...          |
    +------------------+-------------------------+
    |   Per-model breakdown (input/cached/output tokens, $) |
    +-------------------------------------------------------+
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = BENCHMARKS_ROOT / "results"
DEFAULT_PRICING = BENCHMARKS_ROOT / "config" / "pricing.yaml"


# ───────────────────── pricing ─────────────────────


def load_pricing(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def price_call(
    model_id: str,
    rec: dict,
    pricing: dict,
) -> Optional[float]:
    """Return USD cost for one call record, or None when the model has
    no concrete pricing entry (placeholder, missing, or missing rates)."""
    models_table = pricing.get("models", {})
    rates = models_table.get(model_id)
    if rates is None or rates.get("placeholder"):
        return None
    # Bedrock Converse usage block semantics (per AWS prompt-caching
    # docs): `inputTokens` is the non-cached portion only.
    #   total input billable = inputTokens         @ uncached rate
    #                        + cacheReadInputTokens @ cache_read rate
    #                        + cacheWriteInputTokens @ cache_write_5m rate
    # Anthropic first-party API uses the same convention. So we DO NOT
    # subtract cached/write from input_tokens to derive uncached -- the
    # API already split them. An earlier version did subtract; that
    # underbilled cached calls by ~28% and would silently drift on the
    # beam-search loop where prompt caching is common.
    uncached = int(rec.get("input_tokens", 0) or 0)
    cached_read = int(rec.get("cache_read_input_tokens", 0) or 0)
    cached_write = int(rec.get("cache_write_input_tokens", 0) or 0)
    out_t = int(rec.get("output_tokens", 0) or 0)

    r_in = rates.get("input_uncached")
    r_cache_read = rates.get("cache_read")
    r_cache_write_5m = rates.get("cache_write_5m")
    r_out = rates.get("output")
    if r_in is None or r_out is None:
        return None

    total = 0.0
    total += uncached * r_in / 1_000_000.0
    if cached_read and r_cache_read is not None:
        total += cached_read * r_cache_read / 1_000_000.0
    elif cached_read:
        total += cached_read * r_in / 1_000_000.0
    if cached_write and r_cache_write_5m is not None:
        total += cached_write * r_cache_write_5m / 1_000_000.0
    elif cached_write:
        total += cached_write * r_in / 1_000_000.0
    total += out_t * r_out / 1_000_000.0
    return total


# ───────────────────── aggregation state ─────────────────────


@dataclass
class ModelTally:
    n_calls: int = 0
    input_uncached: int = 0
    input_cached_read: int = 0
    input_cached_write: int = 0
    output: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True  # flips to False if any priced row was None


@dataclass
class RecentCall:
    ts: str
    cell: str
    model_id: str
    phase: Optional[str]
    in_tok: int
    out_tok: int
    cost_usd: Optional[float]


@dataclass
class WatcherState:
    # Per-file byte offset so we only read new lines on each poll.
    offsets: dict[Path, int] = field(default_factory=dict)
    # Aggregated tallies.
    by_model: dict[str, ModelTally] = field(default_factory=dict)
    recent: list[RecentCall] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    total_calls: int = 0
    # Cells whose pricing is incomplete (placeholder); displayed in summary.
    unknown_cost_models: set[str] = field(default_factory=set)


# ───────────────────── file watching ─────────────────────


def discover_jsonls(root: Path) -> list[Path]:
    """Find every ``llm_calls.jsonl`` under root. Cheap to call each
    refresh -- the harness writes a small number of these per arm
    driver invocation."""
    if not root.exists():
        return []
    return sorted(root.rglob("llm_calls.jsonl"))


def derive_cell_label(path: Path) -> str:
    """Compact 'arm/workload/run' label for the recent-calls table.
    Falls back to the parent directory name if the layout diverges."""
    parts = path.parts
    for marker in ("results",):
        if marker in parts:
            i = parts.index(marker)
            tail = parts[i + 1: -1]  # drop "results" prefix and the file
            if tail:
                return "/".join(tail[-3:])
    return path.parent.name


def poll_files(
    files: list[Path],
    state: WatcherState,
    pricing: dict,
    max_recent: int,
) -> int:
    """Read any new lines appended since the last poll. Returns the
    number of new call records ingested."""
    new_calls = 0
    for path in files:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        offset = state.offsets.get(path, 0)
        if size < offset:
            # File was truncated / rotated; restart from zero.
            offset = 0
        if size == offset:
            continue
        try:
            with open(path) as f:
                f.seek(offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _ingest(rec, path, state, pricing, max_recent)
                    new_calls += 1
                state.offsets[path] = f.tell()
        except OSError:
            continue
    return new_calls


def _ingest(rec: dict, path: Path, state: WatcherState, pricing: dict,
            max_recent: int) -> None:
    model_id = str(rec.get("model_id", "unknown"))
    # See price_call() docstring: Bedrock's `inputTokens` is the
    # non-cached portion only; cached_read / cached_write are reported
    # separately and are NOT included in inputTokens.
    uncached = int(rec.get("input_tokens", 0) or 0)
    cached_read = int(rec.get("cache_read_input_tokens", 0) or 0)
    cached_write = int(rec.get("cache_write_input_tokens", 0) or 0)
    out_t = int(rec.get("output_tokens", 0) or 0)

    tally = state.by_model.setdefault(model_id, ModelTally())
    tally.n_calls += 1
    tally.input_uncached += uncached
    tally.input_cached_read += cached_read
    tally.input_cached_write += cached_write
    tally.output += out_t

    cost = price_call(model_id, rec, pricing)
    if cost is None:
        tally.cost_known = False
        state.unknown_cost_models.add(model_id)
    else:
        tally.cost_usd += cost

    state.total_calls += 1
    state.recent.append(RecentCall(
        ts=str(rec.get("ts", "")),
        cell=derive_cell_label(path),
        model_id=model_id,
        phase=rec.get("phase"),
        in_tok=in_uncached,
        out_tok=out_t,
        cost_usd=cost,
    ))
    if len(state.recent) > max_recent:
        del state.recent[: len(state.recent) - max_recent]


# ───────────────────── rendering ─────────────────────


def _fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_usd(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v < 0.01:
        return f"${v * 100:.2f}¢"   # sub-cent precision for tiny calls
    if v < 1:
        return f"${v:.3f}"
    if v < 100:
        return f"${v:.2f}"
    return f"${v:,.0f}"


def _fmt_short_ts(ts: str) -> str:
    """ISO 8601 -> 'HH:MM:SS'. Returns the original on parse failure."""
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ts[-8:] if len(ts) >= 8 else ts


def _summary_panel(state: WatcherState, watching: int) -> Panel:
    """Big-total summary. The default view's job is to make the running
    $ spend the focal point at any terminal size, so the dollar amount
    gets a dedicated centered line in bold green and the supporting
    counters live as a single subtitle line beneath it."""
    total_cost = 0.0
    cost_known = True
    total_in_uncached = 0
    total_in_cached = 0
    total_out = 0
    for tally in state.by_model.values():
        total_cost += tally.cost_usd
        cost_known = cost_known and tally.cost_known
        total_in_uncached += tally.input_uncached
        total_in_cached += tally.input_cached_read + tally.input_cached_write
        total_out += tally.output

    elapsed = max(time.time() - state.started_at, 0.001)
    rate = state.total_calls / elapsed * 60.0  # calls/min

    # Hero line: the running spend, centered, oversized via bold + padding.
    spend_text = _fmt_usd(total_cost)
    if not cost_known:
        spend_text += "+"   # the "+" hints that some models lack rates
    hero_style = "bold green" if cost_known else "bold yellow"
    hero = Text(f"  {spend_text}  ", style=hero_style)

    # Subtitle line(s): everything else, dim but readable.
    sub = Text(justify="center")
    sub.append(f"{state.total_calls} call{'s' if state.total_calls != 1 else ''}",
               style="bold white")
    sub.append("  •  ", style="dim")
    sub.append(f"{rate:.1f}/min", style="dim")
    sub.append("  •  ", style="dim")
    sub.append(f"in {_fmt_tok(total_in_uncached)}", style="dim")
    if total_in_cached:
        sub.append(f" (+{_fmt_tok(total_in_cached)} cached)", style="dim")
    sub.append(f"  •  out {_fmt_tok(total_out)}", style="dim")
    sub.append(f"  •  watching {watching} file{'s' if watching != 1 else ''}",
               style="dim")

    body = Group(
        Text(""),
        Align.center(hero),
        Text(""),
        Align.center(sub),
        Text(""),
    )
    if state.unknown_cost_models:
        warn = Text(
            f"Pricing missing for: {', '.join(sorted(state.unknown_cost_models))}",
            style="yellow",
        )
        body = Group(body, Align.center(warn), Text(""))

    title = "[bold]LLM SPEND[/bold]" if cost_known \
        else "[bold yellow]LLM SPEND  (rates incomplete)[/bold yellow]"
    return Panel(body, title=title, border_style="green" if cost_known
                 else "yellow", padding=(0, 1))


def _per_model_table(state: WatcherState) -> Table:
    table = Table(title="Per-model breakdown",
                  show_lines=False, expand=True,
                  border_style="dim")
    table.add_column("Model", overflow="fold", no_wrap=False)
    table.add_column("Calls", justify="right")
    table.add_column("In (uncached)", justify="right")
    table.add_column("In (cache read)", justify="right")
    table.add_column("In (cache write)", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("USD", justify="right")
    for model_id, tally in sorted(state.by_model.items(),
                                  key=lambda kv: -kv[1].cost_usd):
        cost_repr = _fmt_usd(tally.cost_usd)
        if not tally.cost_known:
            cost_repr += "+"
        table.add_row(
            model_id,
            str(tally.n_calls),
            _fmt_tok(tally.input_uncached),
            _fmt_tok(tally.input_cached_read),
            _fmt_tok(tally.input_cached_write),
            _fmt_tok(tally.output),
            cost_repr,
        )
    if not state.by_model:
        table.add_row("(none yet)", "0", "0", "0", "0", "0", "—")
    return table


def _recent_table(state: WatcherState) -> Table:
    table = Table(title="Recent calls (most recent first)",
                  show_lines=False, expand=True,
                  border_style="dim")
    table.add_column("Time", no_wrap=True)
    table.add_column("Cell", overflow="fold")
    table.add_column("Model", overflow="fold")
    table.add_column("Phase")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("USD", justify="right")
    for rc in reversed(state.recent):
        table.add_row(
            _fmt_short_ts(rc.ts),
            rc.cell,
            rc.model_id,
            rc.phase or "—",
            _fmt_tok(rc.in_tok),
            _fmt_tok(rc.out_tok),
            _fmt_usd(rc.cost_usd),
        )
    if not state.recent:
        table.add_row("—", "—", "—", "—", "0", "0", "—")
    return table


def render(state: WatcherState, watching: int) -> Group:
    return Group(
        _summary_panel(state, watching),
        _per_model_table(state),
        _recent_table(state),
    )


# ───────────────────── CLI ─────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Live LLM cost monitor (tails llm_calls.jsonl).",
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT,
                    help="results root to watch (default: benchmarks/results)")
    ap.add_argument("--paths", nargs="*", type=Path, default=None,
                    help="specific llm_calls.jsonl files to watch (overrides --root)")
    ap.add_argument("--pricing", type=Path, default=DEFAULT_PRICING,
                    help="pricing.yaml path")
    ap.add_argument("--refresh-hz", type=float, default=4.0,
                    help="redraw + file-poll rate in Hz (default: 4)")
    ap.add_argument("--max-recent", type=int, default=12,
                    help="max recent calls to keep on screen (default: 12)")
    args = ap.parse_args(argv)

    if not args.pricing.exists():
        print(f"pricing file not found: {args.pricing}", file=sys.stderr)
        return 2
    pricing = load_pricing(args.pricing)

    state = WatcherState()
    console = Console()
    poll_interval = 1.0 / max(args.refresh_hz, 0.1)

    def _files() -> list[Path]:
        if args.paths:
            return [p for p in args.paths if p.exists() or True]
        return discover_jsonls(args.root)

    # One eager poll so the screen isn't empty for the first
    # refresh_interval seconds.
    files = _files()
    poll_files(files, state, pricing, args.max_recent)

    try:
        with Live(render(state, len(files)),
                  console=console,
                  refresh_per_second=args.refresh_hz,
                  screen=False) as live:
            while True:
                files = _files()
                poll_files(files, state, pricing, args.max_recent)
                live.update(render(state, len(files)))
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        # Clean exit; one final dump so the user keeps the totals.
        console.print(render(state, len(files)))
        console.print(
            f"\nTotal: {_fmt_usd(sum(t.cost_usd for t in state.by_model.values()))} "
            f"across {state.total_calls} call(s).",
            style="bold",
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
