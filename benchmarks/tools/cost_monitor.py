"""Live LLM cost monitor + session-aware ledger.

Tails every ``llm_calls.jsonl`` under ``benchmarks/results/`` and
renders a multi-window rich TUI showing:

  * CUMULATIVE      lifetime spend across every record on disk
  * THIS MONTH      spend with ts in the current UTC month
  * SESSION         spend during the currently-active named session
                    (managed via ``mb-cost session start/end``)
  * PER-MODEL       cross-cutting table of spend + tokens per model id

The display takes over the terminal (alternate screen buffer) so it
stays at the top + restores cleanly on quit. Resizes with the window.
Math matches AWS Bedrock prompt-caching semantics exactly (see
``price_call``).

Keyboard (when stdin is a tty):
    q / Ctrl-C     quit
    p              pause / resume polling
    s              cycle per-model sort: cost -> calls -> name
    j / k          scroll recent-calls list down / up
    ?              toggle key-hints overlay
    r              reset on-screen state (re-read everything from disk)

CLI subcommands:

  uv run mb-cost                              live TUI (default)
  uv run mb-cost live --budget-usd N          live TUI with budget alarm
  uv run mb-cost session start NAME [--label TEXT]
  uv run mb-cost session end
  uv run mb-cost session list
  uv run mb-cost report                       one-shot text dump, no TUI

Everything stays gitignored (ledger at ``benchmarks/results/.sessions.json``;
results dirs already in ``.gitignore``). No external state.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from rich.align import Align
from rich.box import DOUBLE, HEAVY, ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from modelblaster.benchmarks.tools.sessions import (
    SessionLedger, is_within_current_month,
)


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
    no concrete pricing entry (placeholder, missing, or missing rates).

    Bedrock Converse usage block semantics (per AWS prompt-caching
    docs): ``inputTokens`` is the non-cached portion only.
        total input billable = inputTokens         @ uncached rate
                             + cacheReadInputTokens @ cache_read rate
                             + cacheWriteInputTokens @ cache_write_5m rate
    Anthropic first-party API uses the same convention. So we DO NOT
    subtract cached/write from input_tokens to derive uncached -- the
    API already split them. An earlier version did subtract; that
    underbilled cached calls by ~28% and would silently drift on the
    beam-search loop where prompt caching is common.
    """
    models_table = pricing.get("models", {})
    rates = models_table.get(model_id)
    if rates is None or rates.get("placeholder"):
        return None
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


# ───────────────────── ingested state ─────────────────────


@dataclass
class ModelTally:
    n_calls: int = 0
    input_uncached: int = 0
    input_cached_read: int = 0
    input_cached_write: int = 0
    output: int = 0
    cost_usd: float = 0.0
    cost_known: bool = True


@dataclass
class IngestedRecord:
    """One JSONL record reduced to the fields aggregation needs.
    Per-record retention is bounded by total LLM calls across all
    results dirs (typically <1e5 — a few MB)."""
    ts: str
    model_id: str
    cost_usd: Optional[float]   # None when model lacks pricing
    input_uncached: int
    input_cached_read: int
    input_cached_write: int
    output: int
    cell: str
    phase: Optional[str]


@dataclass
class WatcherState:
    offsets: dict[Path, int] = field(default_factory=dict)
    records: list[IngestedRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    unknown_cost_models: set[str] = field(default_factory=set)
    # Mascot state -- tracks distinct kernels seen so the blaster
    # animates exactly once per new kernel. The animation frame
    # counter ticks down each render so the shot only lasts a few
    # frames before settling back to idle.
    seen_kernels: set[str] = field(default_factory=set)
    blaster_anim_frame: int = 0   # >0 means actively animating

    @property
    def total_calls(self) -> int:
        return len(self.records)


@dataclass
class WindowTotals:
    """Aggregated metrics over a subset of records."""
    label: str
    total_cost: float = 0.0
    cost_known: bool = True
    n_calls: int = 0
    input_uncached: int = 0
    input_cached: int = 0
    output: int = 0
    by_model: dict[str, ModelTally] = field(default_factory=dict)
    # Per-kernel rollup (e.g. conv2d_s8, linear_s8). Populated from
    # records whose phase is `synth:<op>` or `optimize:<op>` --
    # generate_kernels tags every LLM call that way. Empty when no
    # records carry a kernel-scoped phase.
    by_kernel: dict[str, ModelTally] = field(default_factory=dict)


# ───────────────────── file watching ─────────────────────


def discover_jsonls(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("llm_calls.jsonl"))


def derive_cell_label(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        i = parts.index("results")
        tail = parts[i + 1: -1]
        if tail:
            return "/".join(tail[-3:])
    return path.parent.name


def poll_files(files: list[Path], state: WatcherState,
               pricing: dict) -> int:
    """Read any new lines appended since the last poll. Returns the
    number of new call records ingested."""
    new = 0
    for path in files:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            continue
        offset = state.offsets.get(path, 0)
        if size < offset:
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
                    _ingest(rec, path, state, pricing)
                    new += 1
                state.offsets[path] = f.tell()
        except OSError:
            continue
    return new


def _ingest(rec: dict, path: Path, state: WatcherState,
            pricing: dict) -> None:
    model_id = str(rec.get("model_id", "unknown"))
    uncached = int(rec.get("input_tokens", 0) or 0)
    cached_read = int(rec.get("cache_read_input_tokens", 0) or 0)
    cached_write = int(rec.get("cache_write_input_tokens", 0) or 0)
    out_t = int(rec.get("output_tokens", 0) or 0)
    cost = price_call(model_id, rec, pricing)
    if cost is None:
        state.unknown_cost_models.add(model_id)
    phase = rec.get("phase")
    state.records.append(IngestedRecord(
        ts=str(rec.get("ts", "")),
        model_id=model_id,
        cost_usd=cost,
        input_uncached=uncached,
        input_cached_read=cached_read,
        input_cached_write=cached_write,
        output=out_t,
        cell=derive_cell_label(path),
        phase=phase,
    ))
    # Fire the blaster mascot when we encounter a new kernel.
    # _BLASTER_ANIM_FRAMES is the number of redraw frames the shot
    # animation persists before settling back to idle.
    kernel = _kernel_from_phase(phase)
    if kernel is not None and kernel not in state.seen_kernels:
        state.seen_kernels.add(kernel)
        state.blaster_anim_frame = _BLASTER_ANIM_FRAMES


def _reset_offsets(state: WatcherState) -> None:
    """Force a full re-scan from disk on the next poll."""
    state.offsets.clear()
    state.records.clear()
    state.unknown_cost_models.clear()


# ───────────────────── window aggregation ─────────────────────


def _kernel_from_phase(phase: Optional[str]) -> Optional[str]:
    """Extract the kernel/op name from the LLM call's `phase` field.
    generate_kernels tags converse() calls as `synth:<op>` or
    `optimize:<op>` (e.g. `synth:conv2d_s8`). Returns None for calls
    that lack a kernel-scoped phase (older runs, smoke tests, etc)."""
    if not phase:
        return None
    if ":" in phase:
        prefix, op = phase.split(":", 1)
        if prefix in ("synth", "optimize") and op:
            return op
    return None


def _aggregate(records: list[IngestedRecord], label: str
               ) -> WindowTotals:
    w = WindowTotals(label=label)
    for r in records:
        w.n_calls += 1
        w.input_uncached += r.input_uncached
        w.input_cached += r.input_cached_read + r.input_cached_write
        w.output += r.output
        if r.cost_usd is None:
            w.cost_known = False
        else:
            w.total_cost += r.cost_usd
        slot = w.by_model.setdefault(r.model_id, ModelTally())
        slot.n_calls += 1
        slot.input_uncached += r.input_uncached
        slot.input_cached_read += r.input_cached_read
        slot.input_cached_write += r.input_cached_write
        slot.output += r.output
        if r.cost_usd is None:
            slot.cost_known = False
        else:
            slot.cost_usd += r.cost_usd
        # Per-kernel rollup, keyed on phase prefix.
        kernel = _kernel_from_phase(r.phase)
        if kernel is not None:
            kslot = w.by_kernel.setdefault(kernel, ModelTally())
            kslot.n_calls += 1
            kslot.input_uncached += r.input_uncached
            kslot.input_cached_read += r.input_cached_read
            kslot.input_cached_write += r.input_cached_write
            kslot.output += r.output
            if r.cost_usd is None:
                kslot.cost_known = False
            else:
                kslot.cost_usd += r.cost_usd
    return w


def compute_windows(state: WatcherState, ledger: SessionLedger
                    ) -> dict[str, WindowTotals]:
    """Compute the four aggregation windows. Returns a dict keyed by
    'cumulative' / 'monthly' / 'session' (session is None when no
    active session)."""
    cumul = _aggregate(state.records, "CUMULATIVE")
    monthly_recs = [r for r in state.records
                    if is_within_current_month(r.ts)]
    monthly = _aggregate(monthly_recs, "THIS MONTH")
    out: dict[str, WindowTotals] = {
        "cumulative": cumul, "monthly": monthly, "session": None,
    }
    if ledger.active is not None:
        sess_recs = [r for r in state.records
                     if ledger.active.contains(r.ts)]
        sess = _aggregate(sess_recs, f"SESSION  {ledger.active.id}")
        out["session"] = sess
    return out


# ───────────────────── formatting ─────────────────────


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
        return f"${v * 100:.2f}¢"
    if v < 1:
        return f"${v:.3f}"
    if v < 100:
        return f"${v:.2f}"
    return f"${v:,.0f}"


def _fmt_short_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S")
    except ValueError:
        return ts[-8:] if len(ts) >= 8 else ts


def _budget_style(frac: float, cost_known: bool) -> tuple[str, str]:
    """(hero_style, border_style) for a budget fraction.
    Six-tier semantic palette: muted green (0-30%), bright green
    (30-50%), cyan (50-65%), yellow (65-80%), bright red (80-100%),
    red blink (>=100%)."""
    if not cost_known:
        return "bold yellow", "yellow"
    if frac >= 1.0:
        return "bold red blink", "red"
    if frac >= 0.8:
        return "bold bright_red", "bright_red"
    if frac >= 0.65:
        return "bold yellow", "yellow"
    if frac >= 0.5:
        return "bold cyan", "cyan"
    if frac >= 0.3:
        return "bold bright_green", "bright_green"
    return "bold green", "green"


# Unicode block ramp for inline sparklines. Each char is roughly
# one-eighth of the column height; combined they sketch a small
# trend chart of recent call costs in O(N chars).
_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 20) -> str:
    """Render the last `width` floats as a single-line unicode chart.
    Returns "" when the input is empty. Each value gets exactly one
    char; small values map to lower blocks, large values to higher
    blocks. Normalization is independent of absolute magnitude --
    just relative shape."""
    if not values:
        return ""
    tail = values[-width:]
    lo = min(tail)
    hi = max(tail)
    if hi <= lo:
        # Constant series -- middle block.
        return _SPARK_BLOCKS[len(_SPARK_BLOCKS) // 2] * len(tail)
    span = hi - lo
    out = []
    for v in tail:
        idx = int((v - lo) / span * (len(_SPARK_BLOCKS) - 1))
        out.append(_SPARK_BLOCKS[idx])
    return "".join(out)


# ───────────────────── mascot: 🔫 BLASTER the blaster ─────────────────────
#
# Tiny 3-line ASCII blaster pinned to the top-right corner. Idle most
# of the time; fires a pew-pew across the panel when a new kernel
# (synth:<op> / optimize:<op>) appears in the JSONL records. Decays
# back to idle over _BLASTER_ANIM_FRAMES redraws so the firing
# animation is visible at 4 Hz without being epileptic.
#
# Why three frames: idle / recoil / aftermath gives a perceptible
# "shot" without needing complex sprite work. The projectile chars
# add the pew-pew across the rest of the line.

_BLASTER_ANIM_FRAMES = 8  # at 4 Hz: 2 seconds total


# Sci-fi blaster ASCII -- riff on the classic "hjw" revolver silhouette
# scaled down to 3 lines and stylized with a power cell + barrel vent.
# The PROJECTILE is rendered as a starburst sparkle pattern inspired by
# old-school sci-fi laser art:
#
#         .              .   .'.     \   /
#       \   /      .'. .' '.'   '  -=  o  =-
#     -=  o  =-  .'   '              /   \
#       /   \                          '
#         '
#
# Each frame: the blaster sits in columns 0-13. The "shot" is a
# starburst centered some columns to the right of the muzzle, with
# sparkle dots `.` and tick marks `'` around it. As the animation
# progresses, the starburst drifts further from the muzzle and decays
# (bright `-= o =-` -> thin `'.'` -> just `.` sparkles -> gone).
#
# Composition rules:
#   * Top line  -- power cell + barrel housing
#   * Mid line  -- barrel + muzzle (where the projectile launches)
#   * Bottom    -- grip + trigger
#
# Whichever frame is rendered, the silhouette stays 3 lines tall + the
# same column count so the surrounding layout doesn't twitch.

_BLASTER_FRAMES = {
    # Idle: dim outline, no glow. Power cell visible.
    "idle": [
        "  ╔═══╦═════╕",
        "  ║▒▒▒║──── ·",
        "   `═╧╗_ ",
    ],
    # Charging: power cell pulses ▓, muzzle tightens to ◆.
    "charge": [
        "  ╔═══╦═════╕",
        "  ║▓▓▓║════ ◆",
        "   `═╧╗_ ",
    ],
    # Muzzle flash: starburst right at the muzzle tip. The `-= o =-`
    # signature borrowed straight from the sci-fi art inspiration.
    "recoil": [
        "  ╔═══╦═════╕  \\   /",
        "  ║▓▓▓║═════►-= o =-",
        "   `═╧╗_       /   \\",
    ],
    # Trail: starburst has drifted right, fading into sparkles.
    # The actual sparkles are positioned dynamically below via the
    # projectile string so the trail can move further on each frame.
    "trail": [
        "  ╔═══╦═════╕",
        "  ║▒▒▒║═════►",
        "   `═╧╗_ ",
    ],
    # Aftermath: vent puff (° rises above barrel) + lingering ' .
    "puff": [
        "  ╔═══╦═════╕ °",
        "  ║▒▒▒║──── °  '",
        "   `═╧╗_      .",
    ],
}


# Starburst variants used for the moving projectile, in decay order
# (most-intense first). Each entry is (top, mid, bottom) -- three
# lines vertically centered on the bolt path. After the brightest
# starburst we step down to thinner sparkles + finally just dots.
_STARBURST_DECAY = [
    ("\\   /",
     "-= o =-",
     " /   \\"),
    ("  .'.  ",
     " '.o.' ",
     "  '.'  "),
    ("       ",
     " .'.   ",
     "  .    "),
    ("       ",
     "  .    ",
     "       "),
]


def _blaster_frame_for(frame: int) -> tuple[str, int]:
    """(frame_key, decay_index) for the countdown frame. `frame` is
    _BLASTER_ANIM_FRAMES at the start of the shot, ticking down to 0.
    Sub-frames: charge -> recoil -> trail*N (with decaying starburst
    intensity) -> puff -> idle. ``decay_index`` is the index into
    ``_STARBURST_DECAY`` for the trail variant; -1 means no starburst."""
    if frame <= 0:
        return "idle", -1
    if frame >= _BLASTER_ANIM_FRAMES:
        return "charge", -1
    if frame == _BLASTER_ANIM_FRAMES - 1:
        return "recoil", -1
    if frame >= _BLASTER_ANIM_FRAMES - 4:
        # Trail phase: 3 frames worth of starburst drift + decay.
        # decay_index = 0 (brightest) -> len(_STARBURST_DECAY)-1.
        travelled = _BLASTER_ANIM_FRAMES - 1 - frame  # 1..4
        decay_idx = min(travelled - 1, len(_STARBURST_DECAY) - 1)
        return "trail", decay_idx
    return "puff", -1


def _mascot_panel(state: WatcherState) -> Panel:
    """The blaster mascot panel. Always 5 lines tall so the layout
    above doesn't reflow when the trail starburst extends below the
    barrel.

    Color palette is sci-fi: cyan power cell housing, orange muzzle
    glow on recoil, bright red starburst on trail frames, grey vent
    smoke on puff. The trail frames overlay the starburst from
    _STARBURST_DECAY drifting right + fading."""
    key, decay_idx = _blaster_frame_for(state.blaster_anim_frame)
    art = _BLASTER_FRAMES[key]
    body = Text()
    if key == "idle":
        for line in art:
            body.append(line + "\n", style="cyan")
        body.append("\n\n", style="dim")  # pad to fixed height
        title = "[dim cyan]⌬ BLASTER  standby[/dim cyan]"
        border = "cyan"
    elif key == "charge":
        for line in art:
            colored = line.replace("▓▓▓",
                                   "[bold bright_cyan]▓▓▓[/bold bright_cyan]")
            body.append(Text.from_markup(colored, style="cyan"))
            body.append("\n")
        body.append("\n\n", style="dim")
        title = "[bold bright_cyan]⌬ BLASTER  charging…[/bold bright_cyan]"
        border = "bright_cyan"
    elif key == "recoil":
        # Recoil frame -- starburst already embedded in the art lines.
        for line in art:
            body.append(line + "\n", style="bold yellow")
        body.append("\n\n", style="dim")
        title = "[bold yellow]⌬ BLASTER  ▸ NEW KERNEL[/bold yellow]"
        border = "yellow"
    elif key == "trail":
        # 3 lines of blaster + 3 lines of drifting starburst below.
        # Total = 6 lines but we crop to 5 for layout consistency
        # by overlaying the top starburst line on the blaster's last
        # row (the grip line gets the starburst's leading char).
        for line in art:
            body.append(line + "\n", style="bold red")
        burst = _STARBURST_DECAY[max(0, decay_idx)]
        # Offset the starburst to the right of the muzzle and further
        # each frame so it visibly drifts.
        offset = 6 + decay_idx * 3
        pad = " " * offset
        body.append(pad + burst[0] + "\n", style="bold bright_red")
        body.append(pad + burst[1] + "\n", style="bold bright_yellow")
        title = "[bold bright_red]⌬ BLASTER  ▸ FIRING ▸ pew![/bold bright_red]"
        border = "bright_red"
    else:  # puff
        for line in art:
            body.append(line + "\n", style="grey62")
        body.append("\n\n", style="dim")
        title = "[dim cyan]⌬ BLASTER  venting…[/dim cyan]"
        border = "grey50"
    return Panel(body, title=title, border_style=border, box=ROUNDED,
                 padding=(0, 1), width=28)


# ───────────────────── rendering ─────────────────────


def _summary_panel(state: WatcherState, ledger: SessionLedger,
                   windows: dict[str, WindowTotals],
                   budget_usd: Optional[float]) -> Panel:
    """Multi-window summary: cumulative is the hero; monthly + session
    are second-line callouts. Budget colors the hero when set."""
    cumul = windows["cumulative"]
    monthly = windows["monthly"]
    session = windows.get("session")

    elapsed = max(time.time() - state.started_at, 0.001)
    rate = cumul.n_calls / elapsed * 60.0

    # Hero is the cumulative dollar value.
    if budget_usd is not None and budget_usd > 0 and cumul.cost_known:
        frac = cumul.total_cost / budget_usd
        hero_style, border_style = _budget_style(frac, cumul.cost_known)
    else:
        frac = None
        hero_style = "bold green" if cumul.cost_known else "bold yellow"
        border_style = "green" if cumul.cost_known else "yellow"

    spend_text = _fmt_usd(cumul.total_cost)
    if not cumul.cost_known:
        spend_text += "+"
    hero = Text(f"  {spend_text}  ", style=hero_style)

    # Window rows -- centered, color-coded.
    win_rows = Text(justify="center")
    win_rows.append("CUMULATIVE  ", style="dim")
    win_rows.append(_fmt_usd(cumul.total_cost), style=hero_style)
    win_rows.append(f"   ({cumul.n_calls} calls)", style="dim")
    win_rows.append("\n")
    win_rows.append("THIS MONTH  ", style="dim")
    win_rows.append(_fmt_usd(monthly.total_cost),
                    style="bold cyan" if monthly.cost_known else "bold yellow")
    win_rows.append(f"   ({monthly.n_calls} calls)", style="dim")
    win_rows.append("\n")
    if session is not None:
        win_rows.append(f"SESSION  ", style="dim")
        win_rows.append(ledger.active.id, style="bold magenta")
        win_rows.append("   ", style="dim")
        win_rows.append(_fmt_usd(session.total_cost),
                        style="bold magenta")
        win_rows.append(f"   ({session.n_calls} calls)", style="dim")
        if ledger.active.label:
            win_rows.append(f"\n  {ledger.active.label}", style="dim italic")
    else:
        win_rows.append("SESSION     ", style="dim")
        win_rows.append("no active session", style="dim italic")
        win_rows.append("   (mb-cost session start NAME to open one)",
                        style="dim")

    # Footer counters.
    sub = Text(justify="center")
    sub.append(f"{rate:.1f} calls/min", style="dim")
    sub.append("  •  ", style="dim")
    sub.append(f"in {_fmt_tok(cumul.input_uncached)}", style="dim")
    if cumul.input_cached:
        sub.append(f" (+{_fmt_tok(cumul.input_cached)} cached)",
                   style="dim")
    sub.append(f"  •  out {_fmt_tok(cumul.output)}", style="dim")

    parts = [
        Text(""),
        Align.center(hero),
        Text(""),
        Align.center(win_rows),
        Text(""),
        Align.center(sub),
    ]

    if frac is not None:
        # Two-line budget visual: a real progress bar (colored by tier)
        # plus the numeric breakdown beneath.
        bar_text = Text(justify="center")
        bar_text.append("BUDGET ", style="dim")
        bar_text.append(_fmt_usd(cumul.total_cost), style=hero_style)
        bar_text.append(" / ", style="dim")
        bar_text.append(_fmt_usd(budget_usd), style="bold")
        bar_text.append(f"   ({frac * 100:.1f}%)", style=hero_style)
        if frac >= 1.0:
            bar_text.append("   ⚠ OVER BUDGET ⚠",
                            style="bold red blink")
        parts.append(Text(""))
        parts.append(Align.center(bar_text))
        pb = ProgressBar(
            total=100.0,
            completed=min(100.0, frac * 100.0),
            width=50,
            complete_style=hero_style,
            finished_style="bold red",
            style="dim",
        )
        parts.append(Align.center(pb))

    # Sparkline of recent call costs -- visual heartbeat showing
    # whether spending is steady, ramping, or bursty.
    recent_costs = [(r.cost_usd or 0.0)
                    for r in state.records[-20:]]
    if recent_costs:
        spark = _sparkline(recent_costs, width=20)
        spark_line = Text(justify="center")
        spark_line.append("recent ", style="dim")
        spark_line.append(spark, style=hero_style)
        spark_line.append(f"  Δ {_fmt_usd(recent_costs[-1])}",
                          style="dim")
        parts.append(Text(""))
        parts.append(Align.center(spark_line))

    parts.append(Text(""))
    body = Group(*parts)

    if state.unknown_cost_models:
        warn = Text(
            f"Pricing missing for: "
            f"{', '.join(sorted(state.unknown_cost_models))}",
            style="yellow",
        )
        body = Group(body, Align.center(warn), Text(""))

    if frac is not None and frac >= 1.0:
        title = "[bold red blink]💸 LLM SPEND — OVER BUDGET 💸[/bold red blink]"
    elif not cumul.cost_known:
        title = "[bold yellow]💵 LLM SPEND  (rates incomplete)[/bold yellow]"
    else:
        title = "[bold green]💵 LLM SPEND[/bold green]"
    return Panel(body, title=title, border_style=border_style,
                 padding=(0, 2), box=HEAVY)


def _per_model_table(by_model: dict[str, ModelTally],
                     sort_mode: str) -> Table:
    table = Table(
        title=f"[bold cyan]🤖 Per-model breakdown[/bold cyan]  "
              f"[dim](sort: {sort_mode})[/dim]",
        show_lines=False, expand=True, border_style="cyan",
        header_style="bold cyan", row_styles=["", "dim"],
        box=ROUNDED, title_justify="left",
    )
    table.add_column("Model", overflow="fold", no_wrap=False,
                     style="white")
    table.add_column("Calls", justify="right", style="cyan")
    table.add_column("In (uncached)", justify="right")
    table.add_column("In (cache read)", justify="right", style="green")
    table.add_column("In (cache write)", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("USD", justify="right", style="bold green")

    items = list(by_model.items())
    if sort_mode == "calls":
        items.sort(key=lambda kv: -kv[1].n_calls)
    elif sort_mode == "name":
        items.sort(key=lambda kv: kv[0])
    else:  # cost (default)
        items.sort(key=lambda kv: -kv[1].cost_usd)

    for model_id, tally in items:
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
    if not items:
        table.add_row("(none yet)", "0", "0", "0", "0", "0", "—")
    return table


def _per_kernel_table(by_kernel: dict[str, ModelTally],
                      sort_mode: str) -> Optional[Table]:
    """Per-op cost breakdown. Returns None when no records carry a
    kernel-scoped phase so the panel doesn't render empty on smoke
    tests / non-arm-B runs."""
    if not by_kernel:
        return None
    table = Table(
        title=f"[bold magenta]⚙  Per-kernel breakdown[/bold magenta]  "
              f"[dim](sort: {sort_mode}; phase: synth/optimize:<op>)[/dim]",
        show_lines=False, expand=True, border_style="magenta",
        header_style="bold magenta", row_styles=["", "dim"],
        box=ROUNDED, title_justify="left",
    )
    table.add_column("Kernel", overflow="fold", style="bold white")
    table.add_column("Calls", justify="right", style="magenta")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("USD", justify="right", style="bold green")

    items = list(by_kernel.items())
    if sort_mode == "calls":
        items.sort(key=lambda kv: -kv[1].n_calls)
    elif sort_mode == "name":
        items.sort(key=lambda kv: kv[0])
    else:
        items.sort(key=lambda kv: -kv[1].cost_usd)

    for kernel, tally in items:
        cost_repr = _fmt_usd(tally.cost_usd) + (
            "+" if not tally.cost_known else "")
        in_tok = (tally.input_uncached + tally.input_cached_read
                  + tally.input_cached_write)
        table.add_row(
            kernel, str(tally.n_calls),
            _fmt_tok(in_tok), _fmt_tok(tally.output), cost_repr,
        )
    return table


def _recent_table(records: list[IngestedRecord],
                  scroll: int, max_visible: int) -> Table:
    """Scrollable per-call detail. ``scroll`` is the offset from the
    newest record (0 = newest visible at top)."""
    title = (f"[bold blue]📞 Recent calls[/bold blue]  "
             f"[dim](newest first; j/k to scroll")
    if scroll > 0:
        title += f", offset={scroll}"
    title += ")[/dim]"
    table = Table(
        title=title,
        show_lines=False, expand=True, border_style="blue",
        header_style="bold blue", row_styles=["", "dim"],
        box=ROUNDED, title_justify="left",
    )
    table.add_column("Time", no_wrap=True, style="cyan")
    table.add_column("Cell", overflow="fold")
    table.add_column("Model", overflow="fold", style="white")
    table.add_column("Phase", style="magenta")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("USD", justify="right", style="bold green")
    # records are append-order (oldest first); newest is at the end.
    # Reverse + slice from scroll.
    reversed_recs = list(reversed(records))
    slice_ = reversed_recs[scroll: scroll + max_visible]
    for r in slice_:
        table.add_row(
            _fmt_short_ts(r.ts),
            r.cell,
            r.model_id,
            r.phase or "—",
            _fmt_tok(r.input_uncached + r.input_cached_read
                     + r.input_cached_write),
            _fmt_tok(r.output),
            _fmt_usd(r.cost_usd),
        )
    if not slice_:
        table.add_row("—", "—", "—", "—", "0", "0", "—")
    return table


def _status_bar(*, paused: bool, sort_mode: str, watching: int,
                budget_usd: Optional[float],
                ledger: SessionLedger,
                spinner: Optional[Spinner] = None) -> Panel:
    """Status bar at the very bottom. Spinner is the live-state
    heartbeat; freezes when paused so the eye registers the change."""
    line = Text()
    if paused:
        line.append("⏸  PAUSED  ", style="bold yellow on grey23")
    else:
        line.append("●  LIVE  ", style="bold green")
    line.append(f"📂 {watching}  ", style="cyan")
    line.append(f"sort:{sort_mode}  ", style="dim")
    if budget_usd is not None:
        line.append(f"budget:{_fmt_usd(budget_usd)}  ", style="dim")
    if ledger.active is not None:
        line.append(f"⚑ {ledger.active.id}  ",
                    style="bold magenta")
    else:
        line.append("no session  ", style="dim italic")
    line.append("│  ", style="grey42")
    line.append("[", style="dim")
    line.append("q", style="bold red")
    line.append("]uit  [", style="dim")
    line.append("p", style="bold")
    line.append("]ause  [", style="dim")
    line.append("s", style="bold")
    line.append("]ort  [", style="dim")
    line.append("j", style="bold")
    line.append("/", style="dim")
    line.append("k", style="bold")
    line.append("] scroll  [", style="dim")
    line.append("r", style="bold")
    line.append("]eset  [", style="dim")
    line.append("?", style="bold cyan")
    line.append("] help", style="dim")
    return Panel(line, border_style="grey42", padding=(0, 1),
                 box=ROUNDED)


def _help_overlay() -> Panel:
    text = Text()
    text.append("⌨  Keyboard\n", style="bold underline cyan")
    text.append("  q / Ctrl-C    ", style="bold")
    text.append("quit (terminal restored)\n", style="dim")
    text.append("  p             ", style="bold")
    text.append("pause / resume polling\n", style="dim")
    text.append("  s             ", style="bold")
    text.append("cycle per-model sort (cost → calls → name)\n",
                style="dim")
    text.append("  j / k         ", style="bold")
    text.append("scroll recent calls down / up\n", style="dim")
    text.append("  r             ", style="bold")
    text.append("reset state + re-scan all files\n", style="dim")
    text.append("  ?             ", style="bold")
    text.append("toggle this help\n", style="dim")
    text.append("\n⚑ Sessions ", style="bold underline magenta")
    text.append("(run in a separate terminal)\n",
                style="dim italic")
    text.append("  uv run mb-cost session start NAME [--label TEXT]\n",
                style="green")
    text.append("  uv run mb-cost session end\n", style="green")
    text.append("  uv run mb-cost session list\n", style="green")
    text.append("  uv run mb-cost run NAME -- <command...>",
                style="bold green")
    text.append("   (auto-scoped session)\n", style="dim italic")
    text.append("\n📊 Report ", style="bold underline yellow")
    text.append("(no TUI; one-shot text)\n", style="dim italic")
    text.append("  uv run mb-cost report\n", style="green")
    text.append("\n💰 Budget guards\n", style="bold underline yellow")
    text.append("  --budget-usd N", style="bold")
    text.append("   visual alarm (this monitor)\n", style="dim")
    text.append("  --max-usd N", style="bold")
    text.append("      hard kill (arm_b_* drivers)\n", style="dim")
    return Panel(text, title="[bold cyan]❔ HELP[/bold cyan]",
                 border_style="cyan", padding=(1, 2), box=DOUBLE)


def render(state: WatcherState, ledger: SessionLedger,
           *, sort_mode: str, scroll: int, max_recent: int,
           watching: int, paused: bool, show_help: bool,
           budget_usd: Optional[float]) -> Group:
    windows = compute_windows(state, ledger)
    cumul = windows["cumulative"]

    # Top row: big summary panel on the left + the BLASTER mascot
    # tucked into the upper-right. rich.Table with no borders gives
    # us a side-by-side that reflows on resize -- the summary panel
    # gets the rest of the width, the mascot stays its compact 28
    # columns.
    top_row = Table.grid(expand=True)
    top_row.add_column(ratio=1)
    top_row.add_column(width=30, justify="right")
    top_row.add_row(
        _summary_panel(state, ledger, windows, budget_usd),
        _mascot_panel(state),
    )

    parts: list = [top_row]
    if show_help:
        parts.append(_help_overlay())
    else:
        parts.append(_per_model_table(cumul.by_model, sort_mode))
        kernel_table = _per_kernel_table(cumul.by_kernel, sort_mode)
        if kernel_table is not None:
            parts.append(kernel_table)
        parts.append(_recent_table(state.records, scroll, max_recent))
    parts.append(_status_bar(paused=paused, sort_mode=sort_mode,
                             watching=watching, budget_usd=budget_usd,
                             ledger=ledger))
    # Tick the blaster animation countdown so each redraw moves it
    # one step closer to idle. Doing it here (post-render) means the
    # NEXT frame uses the decremented value.
    if state.blaster_anim_frame > 0:
        state.blaster_anim_frame -= 1
    return Group(*parts)


# ───────────────────── interactive control ─────────────────────


class _StdinReader:
    """Best-effort single-keystroke reader. Switches stdin to cbreak
    on enter, restores on exit. read_key() returns one char or None
    (non-blocking, ~0ms select). Falls back to no-op when stdin is
    not a tty (so the same code works in pipes / CI)."""

    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._old_attrs = None

    def __enter__(self):
        if self._enabled:
            import termios, tty
            self._old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, *exc):
        if self._old_attrs is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN,
                              self._old_attrs)

    def read_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        return sys.stdin.read(1)


_SORT_CYCLE = ["cost", "calls", "name"]


def _cycle_sort(current: str) -> str:
    try:
        idx = _SORT_CYCLE.index(current)
    except ValueError:
        return _SORT_CYCLE[0]
    return _SORT_CYCLE[(idx + 1) % len(_SORT_CYCLE)]


# ───────────────────── subcommand: live ─────────────────────


def cmd_live(args) -> int:
    if not args.pricing.exists():
        print(f"pricing file not found: {args.pricing}", file=sys.stderr)
        return 2
    pricing = load_pricing(args.pricing)
    ledger = SessionLedger.load()
    state = WatcherState()
    console = Console()
    poll_interval = 1.0 / max(args.refresh_hz, 0.1)

    def _files() -> list[Path]:
        if args.paths:
            return [p for p in args.paths if p.exists()]
        return discover_jsonls(args.root)

    files = _files()
    poll_files(files, state, pricing)

    sort_mode = "cost"
    scroll = 0
    paused = False
    show_help = False
    bell_rung = False

    try:
        with _StdinReader() as keyreader, \
             Live(render(state, ledger, sort_mode=sort_mode,
                         scroll=scroll, max_recent=args.max_recent,
                         watching=len(files), paused=paused,
                         show_help=show_help,
                         budget_usd=args.budget_usd),
                  console=console,
                  refresh_per_second=args.refresh_hz,
                  screen=True) as live:
            while True:
                # Drain any pending keystrokes.
                while True:
                    ch = keyreader.read_key()
                    if ch is None:
                        break
                    if ch in ("q", "Q", "\x03"):  # q / Q / Ctrl-C
                        raise KeyboardInterrupt
                    if ch == "p":
                        paused = not paused
                    elif ch == "s":
                        sort_mode = _cycle_sort(sort_mode)
                    elif ch == "j":
                        scroll = min(scroll + 1,
                                     max(0, len(state.records) - 1))
                    elif ch == "k":
                        scroll = max(0, scroll - 1)
                    elif ch == "?":
                        show_help = not show_help
                    elif ch == "r":
                        _reset_offsets(state)
                        ledger = SessionLedger.load()
                        scroll = 0

                if not paused:
                    files = _files()
                    poll_files(files, state, pricing)
                    # Re-read ledger periodically so session
                    # start/end in another terminal lands within ~1s.
                    ledger = SessionLedger.load()

                # Budget alarm: ring bell on the transition into >=100%.
                if (args.budget_usd is not None and not bell_rung):
                    total = sum((r.cost_usd or 0.0)
                                for r in state.records)
                    if total >= args.budget_usd:
                        console.bell()
                        bell_rung = True

                live.update(render(state, ledger, sort_mode=sort_mode,
                                    scroll=scroll,
                                    max_recent=args.max_recent,
                                    watching=len(files), paused=paused,
                                    show_help=show_help,
                                    budget_usd=args.budget_usd))
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass

    # Post-exit summary in the restored terminal buffer.
    windows = compute_windows(state, ledger)
    cumul = windows["cumulative"]
    console.print(
        f"\nTotal: {_fmt_usd(cumul.total_cost)} across "
        f"{cumul.n_calls} call(s).",
        style="bold",
    )
    return 0


# ───────────────────── subcommand: session ─────────────────────


def cmd_session_start(args) -> int:
    ledger = SessionLedger.load()
    try:
        s = ledger.start(args.name, label=args.label)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"started session {s.id!r}"
          f"{' — ' + s.label if s.label else ''}"
          f" at {s.started_at}")
    return 0


def cmd_session_end(args) -> int:
    ledger = SessionLedger.load()
    s = ledger.end()
    if s is None:
        print("no active session to end")
        return 1
    print(f"ended session {s.id!r} at {s.ended_at}")
    return 0


def cmd_session_list(args) -> int:
    ledger = SessionLedger.load()
    pricing = load_pricing(args.pricing)
    state = WatcherState()
    files = discover_jsonls(args.root)
    poll_files(files, state, pricing)

    sessions = ledger.list_all()
    if not sessions:
        print("(no sessions)")
        return 0

    rows = []
    for s in sessions:
        recs = [r for r in state.records if s.contains(r.ts)]
        cost = sum((r.cost_usd or 0.0) for r in recs)
        cost_known = all(r.cost_usd is not None for r in recs)
        marker = " ACTIVE" if s.is_active else ""
        cost_repr = (_fmt_usd(cost) if cost_known
                     else f"{_fmt_usd(cost)}+ (rates incomplete)")
        rows.append(
            f"  {s.id:30s}  "
            f"started={s.started_at[:19]}  "
            f"{'ongoing' if s.is_active else 'ended  =' + s.ended_at[:19]}  "
            f"calls={len(recs):4d}  cost={cost_repr}{marker}"
        )
    print("Sessions (newest first):")
    print("\n".join(rows))
    if ledger.active:
        print(f"\nActive: {ledger.active.id}")
    else:
        print("\nNo active session.")
    return 0


# ───────────────────── subcommand: run (command-wrapped session) ─────────────────────


def cmd_run(args) -> int:
    """Wrap an arbitrary command with session start/end markers. The
    session lives for the duration of the command (whatever the
    exit code). This is the right hook when a wrapper (e.g. Claude
    Code, a shell script, CI) drives the benchmark -- you don't have
    to thread session_id env vars; the time-window does it.

    Usage:
        mb-cost run baseline-v1 [--label TEXT] -- <command> [args ...]

    The double-dash is required to separate mb-cost's args from the
    wrapped command's. The session is started JUST before exec and
    ended JUST after, so an exact time window for the run is on file.
    """
    import subprocess
    if not args.command:
        print("error: pass the command to run after `--`", file=sys.stderr)
        print("  example: mb-cost run baseline-v1 --label 'first run' "
              "-- uv run python -m ...", file=sys.stderr)
        return 2

    ledger = SessionLedger.load()
    try:
        s = ledger.start(args.name, label=args.label)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[mb-cost] session {s.id!r} opened at {s.started_at}",
          file=sys.stderr)

    rc = -1
    try:
        rc = subprocess.call(args.command)
    except KeyboardInterrupt:
        rc = 130
    except FileNotFoundError as e:
        print(f"[mb-cost] command not found: {e}", file=sys.stderr)
        rc = 127
    except OSError as e:
        print(f"[mb-cost] command failed to spawn: {e}", file=sys.stderr)
        rc = 126
    finally:
        # Re-read in case `mb-cost session end` was called in another
        # terminal; only end if this is still the active session id.
        ledger2 = SessionLedger.load()
        if ledger2.active is not None and ledger2.active.id == s.id:
            ledger2.end()
            print(f"[mb-cost] session {s.id!r} closed "
                  f"(command rc={rc})",
                  file=sys.stderr)
        else:
            print(f"[mb-cost] session {s.id!r} was already closed "
                  f"(out-of-band); command rc={rc}",
                  file=sys.stderr)
    return rc


# ───────────────────── subcommand: report ─────────────────────


def cmd_report(args) -> int:
    """One-shot text report. Useful in CI / cron / Slack pipes where
    a TUI is not appropriate."""
    if not args.pricing.exists():
        print(f"pricing file not found: {args.pricing}", file=sys.stderr)
        return 2
    pricing = load_pricing(args.pricing)
    ledger = SessionLedger.load()
    state = WatcherState()
    files = discover_jsonls(args.root)
    poll_files(files, state, pricing)
    windows = compute_windows(state, ledger)

    def _print_window(w: WindowTotals):
        if w.n_calls == 0:
            return
        cost = (_fmt_usd(w.total_cost) if w.cost_known
                else f"{_fmt_usd(w.total_cost)}+ (rates incomplete)")
        print(f"\n{w.label}: {cost}")
        print(f"  calls:  {w.n_calls}")
        print(f"  input:  uncached={_fmt_tok(w.input_uncached)}  "
              f"cached={_fmt_tok(w.input_cached)}")
        print(f"  output: {_fmt_tok(w.output)}")
        if w.by_model:
            print("  by model:")
            for mid, t in sorted(w.by_model.items(),
                                 key=lambda kv: -kv[1].cost_usd):
                cr = _fmt_usd(t.cost_usd) + ("+" if not t.cost_known else "")
                print(f"    {mid:50s}  calls={t.n_calls:4d}  {cr}")
        if w.by_kernel:
            print("  by kernel:")
            for kn, t in sorted(w.by_kernel.items(),
                                key=lambda kv: -kv[1].cost_usd):
                cr = _fmt_usd(t.cost_usd) + ("+" if not t.cost_known else "")
                print(f"    {kn:30s}  calls={t.n_calls:4d}  {cr}")

    print(f"Cost report  ({datetime.now(timezone.utc).isoformat()})")
    print(f"Source: {args.root}  ({len(files)} llm_calls.jsonl file(s))")
    _print_window(windows["cumulative"])
    _print_window(windows["monthly"])
    if windows["session"] is not None:
        _print_window(windows["session"])
    if state.unknown_cost_models:
        print(f"\nMissing pricing for: "
              f"{', '.join(sorted(state.unknown_cost_models))}")
    return 0


# ───────────────────── CLI ─────────────────────


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--root", type=Path, default=DEFAULT_RESULTS_ROOT,
                    help="results root to watch "
                         "(default: benchmarks/results)")
    ap.add_argument("--paths", nargs="*", type=Path, default=None,
                    help="specific llm_calls.jsonl files to watch "
                         "(overrides --root)")
    ap.add_argument("--pricing", type=Path, default=DEFAULT_PRICING,
                    help="pricing.yaml path")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="mb-cost",
        description=("Live LLM cost monitor + session ledger. "
                     "Default action (no subcommand) is the live TUI."),
    )
    sub = ap.add_subparsers(dest="cmd")

    # live (default)
    live = sub.add_parser("live", help="live TUI (default)")
    _add_common_args(live)
    live.add_argument("--refresh-hz", type=float, default=4.0,
                      help="redraw rate (default: 4)")
    live.add_argument("--max-recent", type=int, default=15,
                      help="recent calls visible (default: 15)")
    live.add_argument("--budget-usd", type=float, default=None,
                      help="visual budget alarm threshold")

    # session start / end / list
    sess = sub.add_parser("session", help="manage named sessions")
    sess_sub = sess.add_subparsers(dest="session_cmd", required=True)
    s_start = sess_sub.add_parser("start", help="open a new session")
    s_start.add_argument("name", help="session id (unique)")
    s_start.add_argument("--label", default=None,
                         help="human-readable description")
    sess_sub.add_parser("end", help="close the active session")
    s_list = sess_sub.add_parser("list", help="list sessions + spend")
    _add_common_args(s_list)

    # report
    rep = sub.add_parser("report", help="one-shot text report (no TUI)")
    _add_common_args(rep)

    # run NAME [--label TEXT] -- <command...>
    #
    # argparse.REMAINDER consumes EVERYTHING after the positional
    # `name`, including flags like --label. So we put --label BEFORE
    # the name as a global pre-flag (still works as a kwarg on the
    # subparser; argparse parses it before the positional). The
    # canonical invocation is:
    #     mb-cost run --label TEXT NAME -- <command...>
    # or just:
    #     mb-cost run NAME -- <command...>
    run = sub.add_parser("run",
                         help="wrap a command with a named session")
    run.add_argument("--label", default=None,
                     help="human-readable description (place BEFORE name)")
    run.add_argument("name", help="session id (unique)")
    run.add_argument("command", nargs=argparse.REMAINDER,
                     help="command to run after `--`; "
                          "session opens just before exec, "
                          "closes just after")

    # Pre-parse so bare `mb-cost` (no subcommand) routes to `live`.
    if not argv:
        argv = sys.argv[1:]
    if not argv or argv[0] not in ("live", "session", "report", "run",
                                    "-h", "--help"):
        argv = ["live"] + list(argv)
    args = ap.parse_args(argv)

    if args.cmd == "live":
        return cmd_live(args)
    if args.cmd == "session":
        if args.session_cmd == "start":
            return cmd_session_start(args)
        if args.session_cmd == "end":
            return cmd_session_end(args)
        if args.session_cmd == "list":
            return cmd_session_list(args)
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "run":
        # argparse leaves a leading "--" in REMAINDER; strip it.
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        return cmd_run(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
