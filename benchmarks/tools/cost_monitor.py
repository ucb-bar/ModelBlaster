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
import re
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
from rich.layout import Layout
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


def discover_claude_code_sessions(cwd: Optional[Path] = None
                                  ) -> list[Path]:
    """Find Claude Code session JSONLs scoped to ``cwd`` (default:
    current working directory). Claude Code stores its conversation
    logs at ``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl``
    where the encoded path replaces ``/`` with ``-``. Returns an
    empty list when Claude Code hasn't created any sessions for this
    project. Files in that directory are typically only readable by
    the user (mode 600); we let the caller surface permission
    errors via the normal poll path."""
    here = (cwd or Path.cwd()).resolve()
    # Encoding matches Claude Code: leading '-' + path parts joined by '-'.
    encoded = "-" + "-".join(here.parts[1:])
    home = Path.home()
    session_dir = home / ".claude" / "projects" / encoded
    if not session_dir.exists():
        return []
    return sorted(session_dir.glob("*.jsonl"))


def derive_cell_label(path: Path) -> str:
    """Compact label for the recent-calls table. For Claude Code
    session JSONLs (which live outside benchmarks/results/), label as
    ``claude-code/<uuid-prefix>`` so they're distinguishable from
    benchmark cells."""
    parts = path.parts
    if "results" in parts:
        i = parts.index("results")
        tail = parts[i + 1: -1]
        if tail:
            return "/".join(tail[-3:])
    if ".claude" in parts and "projects" in parts:
        return f"claude-code/{path.stem[:8]}"
    return path.parent.name


def _normalize_record(rec: dict) -> Optional[dict]:
    """Return a record in our canonical ``llm_calls.jsonl`` shape, or
    None if the source record has no token usage we can attribute.

    Handles two source schemas:

    1. ``llm_calls.jsonl`` (our format, written by bedrock_client /
       gemini_client / claude_code_client). Top-level keys include
       ``model_id``, ``input_tokens``, ``output_tokens``, etc.
    2. Claude Code session JSONL (``~/.claude/projects/<enc>/*.jsonl``).
       Token usage lives on records where ``type == "assistant"`` in
       ``message.usage.{input_tokens, output_tokens,
       cache_read_input_tokens, cache_creation_input_tokens}``. The
       model id is at ``message.model``; the timestamp at
       ``timestamp``; the request id at ``requestId``. Non-assistant
       records (user messages, snapshots, permission events) are
       skipped (None return).
    """
    # Our own ``llm_calls.jsonl`` shape -- pass through.
    if "model_id" in rec and (
        "input_tokens" in rec or "output_tokens" in rec
    ):
        return rec
    # Claude Code session record.
    if rec.get("type") == "assistant":
        msg = rec.get("message") or {}
        usage = msg.get("usage") if isinstance(msg, dict) else None
        if not isinstance(usage, dict):
            return None
        # Skip records with zero useful usage (occurs on cache-only
        # iterations Claude Code logs for housekeeping).
        if not any(usage.get(k) for k in (
            "input_tokens", "output_tokens",
            "cache_read_input_tokens", "cache_creation_input_tokens",
        )):
            return None
        return {
            "ts": rec.get("timestamp", ""),
            "provider": "claude_code",
            "model_id": msg.get("model") or "unknown",
            "request_id": rec.get("requestId"),
            "phase": None,   # no per-kernel attribution for prompts
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens", 0) or 0
            ),
            # Claude Code uses `cache_creation_input_tokens`; our
            # internal field is `cache_write_input_tokens`. Same
            # semantics, both billed at the cache-write rate.
            "cache_write_input_tokens": int(
                usage.get("cache_creation_input_tokens", 0) or 0
            ),
            "stop_reason": "",
        }
    return None


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
                    # Normalize across source schemas. Unsupported /
                    # uninteresting records (user messages, snapshots)
                    # return None and get skipped.
                    norm = _normalize_record(rec)
                    if norm is None:
                        continue
                    _ingest(norm, path, state, pricing)
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
    # Fire the blaster mascot on every new LLM call (but don't restart
    # an already-running shot animation so overlapping calls look like
    # rapid fire instead of a single jittery frame). seen_kernels stays
    # populated for the "first-time-seen" highlight elsewhere if needed.
    kernel = _kernel_from_phase(phase)
    if kernel is not None:
        state.seen_kernels.add(kernel)
    if state.blaster_anim_frame == 0:
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


def _short_model(model_id: str) -> str:
    """Strip provider/version cruft so the Model column doesn't wrap to
    three rows. ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` →
    ``claude-sonnet-4-5``. Conservative: only collapses well-known
    Anthropic / Google patterns, falls back to raw id otherwise."""
    s = model_id
    for prefix in ("us.anthropic.", "anthropic.", "us.", "eu."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Drop trailing date (-YYYYMMDD) and version (-vN:N).
    s = re.sub(r"-\d{8}(-v\d+(:\d+)?)?$", "", s)
    s = re.sub(r"-v\d+(:\d+)?$", "", s)
    return s


def _short_cell(cell: str) -> str:
    """Compact arm/workload/run_id into something that fits one line.
    Keeps the leading arm + workload, shortens the trailing run-id
    timestamp to just HHMMSSZ."""
    parts = cell.split("/")
    if len(parts) >= 3:
        run_id = parts[-1]
        m = re.search(r"T(\d{2})-(\d{2})-(\d{2})Z", run_id)
        if m:
            parts[-1] = f"{m.group(1)}{m.group(2)}{m.group(3)}Z"
    return "/".join(parts)


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


# ───────────────────── splash screen ─────────────────────
#
# Big sci-fi blaster ASCII for the startup splash. Codepage-437-ish
# shading (`#`/`+`/`-`/`.`) -- the inspiration the user picked from
# the four reference variants. Plain ASCII so it renders in any
# terminal regardless of font / locale.

_BLASTER_BIG = r"""
                                ########+--+########
                            ###+                    +###
                         ###                            ###
                      ###                #                 ##.
                    -##                  ##                  ##
          ---      ##                                          ##
        --.  -                                                  ##
         -.. .                           -+###.                  .#
           ###               #######-         #####
            #### +#  ######. #  #           ##   .+######                  -.
           ##+#####      ## ## -#          #   ###---## +###            +  - +
           ###                            #.  #+----#+ ##++##         -
         ##           ##############      #  #+----+# ###+.###    --.+    -
        #       #####++ - + # #++--#     #  ##-----#  ##    ##   -- -.+   -.
     ###    ####++++ +++####### #+-+#    #  #------# ##  -      -.  .-+
  ## #+#  ##+----#  #+  #    +#   +-+#####  #------# ## .       -.
 #  .+#  #+------#  ++#########   +------#  #------# ## . ----- -. +-++--+---++
+#  #-#  #-------+# +#  #    +# ##+------#  #------# ## .       -.
#-  #+#  #--------+ -#  #  #.+# #---######  #------# ##+ ..  ## --  --     +
 #  .+#+ #+-------- +########## #---#    #  ##-----#. ##    ##   ----   +--++
  ## #-#  ####-----+           ++--+#     #  #+-----# ########    --
    #####     -++###################      ##  #+----+# ######                +
          #####-                           #   ####++##  ##                . -
               #+#######################+-  +#     .#+##
               #--------+#     .###-###############-              ##
              ##--------+  .      ###                            +#
              #-+#####+-#  .      ##                            ##
             #+-+     +-+#  .   ###                            ##
            ##--######++#-######                    +-       .##
            #--+    +--#                            .       ##
           #+--+####+-+#                                  ##
          ##---.   +--#                                ###
           ##+--+#++--#                             ###
             #####+--+##   ######             #####.
                 .####            ###########
"""


def render_splash(console: Console) -> None:
    """Print the BLASTER splash screen + title block + tagline. Caller
    decides how long to leave it on screen before transitioning to the
    live TUI. Skipped automatically when stdout isn't a tty (CI / pipe)
    or when the user passed --no-splash."""
    art = Text(_BLASTER_BIG.strip("\n"), style="bright_red")
    title = Text(
        "  ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     \n"
        "  ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     \n"
        "  ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     \n"
        "  ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     \n"
        "  ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗\n"
        "  ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝\n"
        "      ██████╗ ██╗      █████╗ ███████╗████████╗███████╗██████╗\n"
        "      ██╔══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗\n"
        "      ██████╔╝██║     ███████║███████╗   ██║   █████╗  ██████╔╝\n"
        "      ██╔══██╗██║     ██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██╗\n"
        "      ██████╔╝███████╗██║  ██║███████║   ██║   ███████╗██║  ██║\n"
        "      ╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝\n",
        style="bold bright_cyan",
    )
    tagline = Text("live LLM cost monitor   •   sessions   •   "
                   "per-kernel + per-model spend",
                   style="dim italic")
    console.clear()
    console.print()
    console.print(Align.center(title))
    console.print(Align.center(art))
    console.print()
    console.print(Align.center(tagline))
    console.print()


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

_BLASTER_ANIM_FRAMES = 30  # at 20 Hz: 1.5s per shot


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

# Blaster art -- frozen, character-exact, taken verbatim from the
# user's spec. The silhouette is the same across all phases; only
# COLORS and side-effects (firework / smoke / idle sparkles) change.
#
# Anatomy (rows are 0-indexed, columns 0-19):
#   row 0    top fin / sight
#   row 1    fin connecting into body
#   row 2    top of body
#   row 3    barrel detail + muzzle (extends to col 19, the rightmost █)
#   row 4    bottom of body
#   row 5    bottom of body break
#   row 6    grip top
#   row 7    grip bottom

_BLASTER_ART = [
    "   █▓███            ",
    "   █▓▓▓▓▓▓█         ",
    " █▒▒▒▒▒▒▒▒▒▒█ █     ",
    "██▒█  █▒ █▒▒█░░█░░░█",
    " █▒▒▒▒▒▒▒▒▒▒█ █     ",
    "   ▓▓▓▓ ▓█          ",
    "  █▓▓▓              ",
    "  █▓▓█              ",
]


# Firework explosion overlay -- styled exactly like the user's brief:
#
#                                 .
#    .              .   .'.     \   /
#  \   /      .'. .' '.'   '  -=  o  =-
# -=  o  =-  .'   '              /   \
#   /   \                          '
#     '
#
# Drawn to the right of the muzzle (column 21+) during recoil + trail
# frames. Frame 0 is brightest; subsequent frames decay through the
# user's exact star → sparkle → dot progression.
_FIREWORK_FRAMES = [
    # Frame 0 -- peak: the user's signature `-=  o  =-` starburst.
    [
        r"   \   /",
        r"    .",
        r" -=  o  =-",
        r"    '",
        r"   /   \ ",
    ],
    # Frame 1 -- still bright but contracting; opening diagonals
    # have shrunk to single chars + sparkle dots emerging.
    [
        r"   \ /",
        r"  .'.",
        r" -= o =-",
        r"  '.'",
        r"   / \ ",
    ],
    # Frame 2 -- sparkle phase: starburst gone, only `.'. '.'` left.
    [
        r"    .",
        r"   .'.",
        r"  '. .'",
        r"   '.'",
        r"    '",
    ],
    # Frame 3 -- dot phase: just a few drifting sparkles.
    [
        r"",
        r"    .",
        r"   . .",
        r"    .",
        r"",
    ],
    # Frame 4 -- almost gone: lone dot.
    [
        r"",
        r"",
        r"     .",
        r"",
        r"",
    ],
]


# Color palette for the firework lifecycle. Index matches
# _FIREWORK_FRAMES; the brightest frame uses white-hot rays,
# decaying through yellow → red → grey.
_FIREWORK_PALETTE = [
    "bold bright_white",
    "bold bright_yellow",
    "bold yellow",
    "bold bright_red",
    "red",
]


# Idle subtle sparkle patterns -- cycled by wall-clock time so the
# gun isn't 100% static when nothing's happening. Each pattern is a
# list of (row, col, char) overlays painted AROUND the gun -- never
# on top of it.
_IDLE_SPARKLES = [
    [(0, 22, "·"), (3, 24, "."), (7, 23, "·")],
    [(2, 23, "·"), (5, 21, "."), (1, 25, "·")],
    [(1, 21, "·"), (4, 24, "."), (6, 25, "·")],
    [(0, 24, "·"), (6, 22, "."), (3, 26, "·")],
]


# Positions inside the silhouette that LIGHT UP during charging.
# The "two squares" are the interior gaps in row 3 of the base art:
# columns 4-5 (left window) and column 8 (right window). Filling
# them with progressively brighter chars makes the gun visibly
# build energy before the shot.
_CHARGE_FILLS = {
    # row -> dict of {col: char}
    3: {4: "▒", 5: "▒", 8: "▒"},
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


def _mascot_render(state: WatcherState) -> Group:
    """Render the BLASTER mascot as RAW art (no panel border) so it
    sits cleanly in the top-right corner without competing with the
    spend panel's heavy box. One small caption line above the gun
    indicates state (standby / charging / firing / venting).

    Color palette is sci-fi: cyan power cell housing, orange muzzle
    glow on recoil, bright red starburst on firing frames, grey vent
    smoke on puff. Total height is fixed at 8 lines (caption + 4
    gun + 3 trail-or-pad) so the layout above doesn't reflow."""
    key, decay_idx = _blaster_frame_for(state.blaster_anim_frame)
    body = Text()

    titles = {
        "idle":   (" ⌬ BLASTER  standby",         "dim cyan"),
        "charge": (" ⌬ BLASTER  charging…",        "bold bright_cyan"),
        "recoil": (" ⌬ BLASTER  ▸ NEW KERNEL",     "bold bright_yellow"),
        "trail":  (" ⌬ BLASTER  ▸ FIRING ▸ pew!",  "bold bright_yellow"),
        "puff":   (" ⌬ BLASTER  venting…",         "grey62"),
    }
    title, title_style = titles[key]
    body.append(title + "\n", style=title_style)

    # Per-row gun color. Same style for every row -- shading is in
    # the art itself (▓ ▒ ░), so we just tint it.
    gun_style = {
        "idle":   "cyan",
        "charge": "bold bright_cyan",
        "recoil": "bold bright_yellow",
        "trail":  "bold yellow",
        "puff":   "grey62",
    }[key]

    # Mutable copy so we can paint the interior charging windows.
    grid = [list(row) for row in _BLASTER_ART]

    if key in ("charge", "recoil"):
        # Light up the two interior "windows" the user pointed at:
        # row 3 cols 4-5 and col 8 (the gaps inside the body).
        for r, cols in _CHARGE_FILLS.items():
            for c, ch in cols.items():
                if r < len(grid) and c < len(grid[r]):
                    grid[r][c] = ch

    # Build the overlay table: (row, col) -> (char, style). These
    # paint OVER the art / empty space without changing the gun
    # silhouette itself.
    overlays: dict[tuple[int, int], tuple[str, str]] = {}
    if key == "idle":
        # Subtle wall-clock-cycled sparkles around the gun. These
        # are painted OFF the silhouette (in the empty area to the
        # right of the body) so the gun chars themselves never
        # change between idle frames.
        sparkle_idx = int(time.time() * 2) % len(_IDLE_SPARKLES)
        for r, c, ch in _IDLE_SPARKLES[sparkle_idx]:
            overlays[(r, c)] = (ch, "bright_cyan")
    elif key == "charge":
        # Energy particles flowing INTO the muzzle from off-screen
        # right -- visual cue that the gun is gathering power.
        for c, ch in [(22, "·"), (25, "."), (28, "·")]:
            overlays[(3, c)] = (ch, "bold bright_cyan")
    elif key == "puff":
        # Lingering vent smoke to the right of the muzzle. Kept
        # entirely OFF the gun silhouette so the gun stays
        # character-identical to the user's spec.
        overlays.update({
            (2, 23): ("°", "grey70"),
            (3, 25): ("'", "grey62"),
            (4, 27): (".", "grey50"),
        })

    # Firework block (5 rows tall, vertically centered on row 3 of
    # the gun -> occupies gun rows 1..5).
    firework_block: Optional[list[str]] = None
    firework_offset = 0
    firework_style = ""
    if key == "recoil":
        firework_block = _FIREWORK_FRAMES[0]
        firework_offset = 2
        firework_style = _FIREWORK_PALETTE[0]
    elif key == "trail":
        idx = max(0, min(decay_idx + 1, len(_FIREWORK_FRAMES) - 1))
        firework_block = _FIREWORK_FRAMES[idx]
        firework_offset = 2 + decay_idx * 3
        firework_style = _FIREWORK_PALETTE[idx]

    GUN_WIDTH = 20
    fw_start_col = GUN_WIDTH + firework_offset

    # Render each row of the gun + any overlays in its band.
    # CRITICAL: every row is padded to exactly _MASCOT_CELL_WIDTH
    # chars before the newline. Without this, rich's renderer can
    # strip / re-align trailing whitespace per line, and since each
    # art row has different visible widths the gun appears to
    # "morph" between frames as lines drift left/right.
    _MASCOT_CELL_WIDTH = 44
    for r, row_chars in enumerate(grid):
        col_cursor = 0
        # Base art cells (with overlays for cells inside the art).
        for c, ch in enumerate(row_chars):
            if (r, c) in overlays:
                och, ostyle = overlays[(r, c)]
                body.append(och, style=ostyle)
            elif ch == " ":
                body.append(" ")
            else:
                body.append(ch, style=gun_style)
            col_cursor += 1
        # Overlays right of the art (sparkles, smoke).
        beyond = sorted(
            (c, ch, st) for (rr, c), (ch, st) in overlays.items()
            if rr == r and c >= len(row_chars)
        )
        for c, ch, st in beyond:
            if c > col_cursor:
                body.append(" " * (c - col_cursor))
                col_cursor = c
            body.append(ch, style=st)
            col_cursor += 1
        # Firework overlay (only on rows 1..5; row offsets 0..4).
        if firework_block is not None and 1 <= r <= 5:
            fw_line = firework_block[r - 1]
            if fw_line.strip():
                pad_needed = fw_start_col - col_cursor
                if pad_needed > 0:
                    body.append(" " * pad_needed)
                    col_cursor = fw_start_col
                body.append(fw_line, style=firework_style)
                col_cursor += len(fw_line)
        # Pad to fixed cell width so rich doesn't strip trailing
        # whitespace + re-align per line (which made the silhouette
        # "morph" between frames in narrow / right-justified cells).
        if col_cursor < _MASCOT_CELL_WIDTH:
            body.append(" " * (_MASCOT_CELL_WIDTH - col_cursor))
        body.append("\n")

    # Pad to a fixed height (1 title + 8 gun + 4 trailing pad lines)
    # so the layout above doesn't reflow when the animation fires.
    for _ in range(4):
        body.append("\n", style="dim")
    return body


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
                     sort_mode: str,
                     *, max_rows: Optional[int] = None,
                     scroll: int = 0,
                     highlighted: bool = False) -> Table:
    title_label = (
        "[bold cyan]🤖 Per-model breakdown[/bold cyan]"
        if not highlighted
        else "[bold reverse cyan]🤖 Per-model breakdown[/bold reverse cyan]"
    )
    border = "bright_cyan" if highlighted else "cyan"
    table = Table(
        title=f"{title_label}  [dim](sort: {sort_mode})[/dim]",
        show_lines=False, expand=True, border_style=border,
        header_style="bold cyan", row_styles=["", "dim"],
        box=ROUNDED, title_justify="left",
    )
    table.add_column("Model", overflow="ellipsis", no_wrap=True,
                     style="white", max_width=28)
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

    # Same scroll semantics as _per_kernel_table.
    total = len(items)
    if scroll > 0:
        scroll = min(scroll, max(0, total - 1))
        items = items[scroll:]
    if max_rows is not None and len(items) > max_rows:
        truncated_below = len(items) - max_rows
        items = items[:max_rows]
    else:
        truncated_below = 0
    truncated_above = scroll if scroll > 0 else 0
    if truncated_above > 0:
        table.add_row(f"[dim]↑ {truncated_above} more above…[/dim]",
                      "", "", "", "", "", "")
    for model_id, tally in items:
        cost_repr = _fmt_usd(tally.cost_usd)
        if not tally.cost_known:
            cost_repr += "+"
        table.add_row(
            _short_model(model_id),
            str(tally.n_calls),
            _fmt_tok(tally.input_uncached),
            _fmt_tok(tally.input_cached_read),
            _fmt_tok(tally.input_cached_write),
            _fmt_tok(tally.output),
            cost_repr,
        )
    if not items and total == 0:
        table.add_row("(none yet)", "0", "0", "0", "0", "0", "—")
    if truncated_below > 0:
        table.add_row(f"[dim]↓ {truncated_below} more below…[/dim]",
                      "", "", "", "", "", "")
    return table


def _per_kernel_table(by_kernel: dict[str, ModelTally],
                      sort_mode: str,
                      *, max_rows: Optional[int] = None,
                      scroll: int = 0,
                      highlighted: bool = False) -> Optional[Table]:
    """Per-op cost breakdown. Returns None when no records carry a
    kernel-scoped phase so the panel doesn't render empty on smoke
    tests / non-arm-B runs."""
    if not by_kernel:
        return None
    title_label = (
        "[bold magenta]⚙  Per-kernel breakdown[/bold magenta]"
        if not highlighted
        else "[bold reverse magenta]⚙  Per-kernel breakdown[/bold reverse magenta]"
    )
    border = "bright_magenta" if highlighted else "magenta"
    table = Table(
        title=f"{title_label}  [dim](sort: {sort_mode}; phase: synth/optimize:<op>)[/dim]",
        show_lines=False, expand=True, border_style=border,
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

    # Apply scroll offset (clamped) then slice to max_rows. scroll > 0
    # means the user has scrolled DOWN past the top of the sorted list.
    total = len(items)
    if scroll > 0:
        scroll = min(scroll, max(0, total - 1))
        items = items[scroll:]
    if max_rows is not None and len(items) > max_rows:
        truncated_below = len(items) - max_rows
        items = items[:max_rows]
    else:
        truncated_below = 0
    truncated_above = scroll if scroll > 0 else 0
    if truncated_above > 0:
        table.add_row(f"[dim]↑ {truncated_above} more above…[/dim]",
                      "", "", "", "")
    for kernel, tally in items:
        cost_repr = _fmt_usd(tally.cost_usd) + (
            "+" if not tally.cost_known else "")
        in_tok = (tally.input_uncached + tally.input_cached_read
                  + tally.input_cached_write)
        table.add_row(
            kernel, str(tally.n_calls),
            _fmt_tok(in_tok), _fmt_tok(tally.output), cost_repr,
        )
    if truncated_below > 0:
        table.add_row(f"[dim]↓ {truncated_below} more below…[/dim]",
                      "", "", "", "")
    return table


def _recent_table(records: list[IngestedRecord],
                  scroll: int, max_visible: int,
                  *, frozen: bool = False,
                  highlighted: bool = False) -> Table:
    """Scrollable per-call detail. ``scroll`` is the offset from the
    newest record (0 = newest visible at top). When ``frozen`` is
    True (explore mode) the title indicates the user is navigating
    history; ``highlighted`` thickens the border to show this is the
    active pane for arrow-key input."""
    if frozen:
        marker = " 🔍 EXPLORE"
    else:
        marker = ""
    title_label = (
        f"[bold blue]📞 Recent calls{marker}[/bold blue]"
        if not highlighted
        else f"[bold reverse blue]📞 Recent calls{marker}[/bold reverse blue]"
    )
    border = "bright_blue" if highlighted else "blue"
    hint = "(newest first; "
    hint += "↑↓ or j/k to scroll" if frozen else "j/k to scroll, e to explore"
    if scroll > 0:
        hint += f", offset={scroll}"
    hint += ")"
    table = Table(
        title=f"{title_label}  [dim]{hint}[/dim]",
        show_lines=False, expand=True, border_style=border,
        header_style="bold blue", row_styles=["", "dim"],
        box=ROUNDED, title_justify="left",
    )
    table.add_column("Time", no_wrap=True, style="cyan")
    table.add_column("Cell", overflow="ellipsis", no_wrap=True,
                     max_width=34)
    table.add_column("Model", overflow="ellipsis", no_wrap=True,
                     style="white", max_width=22)
    table.add_column("Phase", style="magenta", overflow="ellipsis",
                     no_wrap=True, max_width=20)
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
            _short_cell(r.cell),
            _short_model(r.model_id),
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
                explore_mode: bool = False,
                active_pane: str = "recent",
                spinner: Optional[Spinner] = None) -> Panel:
    """Two-row keybindings panel that's ALWAYS visible at the bottom
    of the TUI. Row 1: state + counters. Row 2: keys as labeled
    chips, each key in a bordered cell. Even if the terminal is
    narrow and rich wraps the cells, every key stays legible.

    The keys themselves never change, so the eye learns where each
    one lives -- much better than burying them in a single dim
    status line that scrolls off-screen when the terminal is small."""
    # --- Row 1: state line ---
    line1 = Text()
    if paused:
        line1.append(" ⏸  PAUSED ", style="bold yellow on grey23")
    elif explore_mode:
        line1.append(" 🔍 EXPLORE ", style="bold black on yellow")
    else:
        line1.append(" ●  LIVE ", style="bold black on green")
    line1.append("  📂 ", style="dim")
    line1.append(f"{watching}", style="bold cyan")
    line1.append("  file" + ("s" if watching != 1 else "") + "  ",
                 style="dim")
    line1.append("  ⇅ ", style="dim")
    line1.append(f"sort:{sort_mode}", style="bold")
    if explore_mode:
        line1.append("   pane:", style="dim")
        line1.append(f"{active_pane}", style="bold yellow")
    if budget_usd is not None:
        line1.append(f"   💵 budget:{_fmt_usd(budget_usd)}", style="dim")
    if ledger.active is not None:
        line1.append("   ⚑ ", style="bold magenta")
        line1.append(ledger.active.id, style="bold magenta")
    else:
        line1.append("   ⚑ no session", style="dim italic")

    # --- Row 2: keybindings, each in a labeled "chip" ---
    if explore_mode:
        key_chips = [
            ("e",    "exit explore", "bold white on yellow"),
            ("↑↓",   "scroll rows",  "bold white on grey35"),
            ("←→",   "switch pane",  "bold white on grey35"),
            ("q",    "quit",         "bold white on red"),
            ("?",    "help",         "bold white on cyan"),
        ]
    else:
        key_chips = [
            ("q",   "quit",      "bold white on red"),
            ("p",   "pause",     "bold white on grey35"),
            ("s",   "sort",      "bold white on grey35"),
            ("j/k", "scroll",    "bold white on grey35"),
            ("e",   "explore",   "bold white on yellow"),
            ("r",   "reset",     "bold white on grey35"),
            ("?",   "help",      "bold white on cyan"),
        ]
    line2 = Text()
    for i, (key, label, chip_style) in enumerate(key_chips):
        if i > 0:
            line2.append("  ", style="dim")
        line2.append(f" {key} ", style=chip_style)
        line2.append(f" {label}", style="bold")

    body = Group(line1, Text(""), line2)
    return Panel(body, title="[bold]controls[/bold]",
                 border_style="grey50", padding=(0, 1),
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
           explore_mode: bool, active_pane: str,
           budget_usd: Optional[float],
           terminal_height: int = 40) -> Layout:
    """rich.Layout with three rows pinned proportionally:
      TOP    -- summary panel + mascot (fixed ~14 lines)
      MIDDLE -- per-model + per-kernel + recent-calls (fills rest)
      BOTTOM -- controls bar (fixed 6 lines, ALWAYS visible)

    The middle section is height-budgeted to whatever vertical space
    remains after top+bottom. When content exceeds that, the per-model
    and per-kernel tables cap their row counts and the recent-calls
    table caps to the remainder. Press `e` for explore mode: arrow
    keys navigate the active pane through history without auto-scroll.
    """
    windows = compute_windows(state, ledger)
    cumul = windows["cumulative"]

    # ─── TOP: summary + mascot, side-by-side ───
    top_row = Table.grid(expand=True)
    top_row.add_column(ratio=1)
    top_row.add_column(width=44, justify="left")
    top_row.add_row(
        _summary_panel(state, ledger, windows, budget_usd),
        _mascot_render(state),
    )

    # ─── BOTTOM: controls bar -- always visible, exactly 6 lines ───
    bottom = _status_bar(paused=paused, sort_mode=sort_mode,
                         watching=watching, budget_usd=budget_usd,
                         ledger=ledger, explore_mode=explore_mode,
                         active_pane=active_pane)

    # ─── MIDDLE: dynamic tables, height-budgeted ───
    TOP_LINES = 14
    BOTTOM_LINES = 6
    MARGIN = 2
    middle_h = max(8, terminal_height - TOP_LINES - BOTTOM_LINES - MARGIN)

    if show_help:
        middle = _help_overlay()
    else:
        has_kernels = bool(cumul.by_kernel)
        per_model_rows = min(4, max(1, middle_h // 4))
        per_kernel_rows = min(4, max(1, middle_h // 5)) if has_kernels else 0
        per_model_h = per_model_rows + 4    # title + header + border padding
        per_kernel_h = (per_kernel_rows + 4) if has_kernels else 0
        recent_h = max(5, middle_h - per_model_h - per_kernel_h)
        recent_rows = max(1, recent_h - 4)

        # Route the scroll offset to whichever pane is currently active.
        # Inactive panes always show from row 0 so the user's mental
        # "current position per pane" is preserved across switches: only
        # one scroll var so leaving and returning resets, but the active
        # pane is always the one being navigated.
        model_scroll = scroll if active_pane == "model" else 0
        kernel_scroll = scroll if active_pane == "kernel" else 0
        recent_scroll = scroll if active_pane == "recent" else 0
        middle_parts: list = [
            _per_model_table(cumul.by_model, sort_mode,
                              max_rows=per_model_rows,
                              scroll=model_scroll,
                              highlighted=(explore_mode
                                            and active_pane == "model")),
        ]
        if has_kernels:
            kt = _per_kernel_table(cumul.by_kernel, sort_mode,
                                    max_rows=per_kernel_rows,
                                    scroll=kernel_scroll,
                                    highlighted=(explore_mode
                                                  and active_pane == "kernel"))
            if kt is not None:
                middle_parts.append(kt)
        middle_parts.append(
            _recent_table(state.records, recent_scroll, recent_rows,
                          frozen=explore_mode,
                          highlighted=(explore_mode
                                        and active_pane == "recent"))
        )
        middle = Group(*middle_parts)

    # ─── compose with Layout so bottom STAYS PINNED ───
    layout = Layout()
    layout.split_column(
        Layout(top_row, name="top", size=TOP_LINES),
        Layout(middle, name="middle", ratio=1),
        Layout(bottom, name="bottom", size=BOTTOM_LINES),
    )

    # Tick blaster animation countdown post-render.
    if state.blaster_anim_frame > 0:
        state.blaster_anim_frame -= 1
    return layout


# ───────────────────── interactive control ─────────────────────


class _StdinReader:
    """Best-effort keystroke reader with multi-byte escape-sequence
    handling. Switches stdin to cbreak on enter, restores on exit.

    read_key() returns:
        * a single printable char ("q", "p", "e", ...)
        * a logical name for arrow keys ("UP", "DOWN", "LEFT", "RIGHT")
        * None when nothing is pending (non-blocking poll)

    Arrow keys arrive as 3 bytes: 0x1b 0x5b 0x41..0x44. We read the
    introducer (ESC), then peek a few more bytes within a tiny
    timeout to assemble the full sequence -- otherwise the ESC by
    itself would be treated as a literal keypress.
    """

    _ARROWS = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}

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

    def _read_with_timeout(self, timeout: float) -> Optional[str]:
        # os.read bypasses Python's TextIOWrapper line-buffering. sys.stdin.read(1)
        # was waiting on Python's io layer to fill its internal buffer even AFTER
        # the TTY had bytes ready -- so held arrows / fast typing produced bursts
        # that visibly dropped between renders. os.read(fd, 1) returns immediately
        # when a single byte is in the kernel TTY buffer (which cbreak guarantees).
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        try:
            data = os.read(sys.stdin.fileno(), 1)
        except (BlockingIOError, OSError):
            return None
        if not data:
            return None
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    def read_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        ch = self._read_with_timeout(0)
        if ch is None:
            return None
        if ch != "\x1b":
            return ch
        # Possible escape sequence: peek up to 2 more bytes with a
        # short timeout. If nothing follows quickly, treat the ESC
        # itself as a keypress (some terminals send a bare ESC for
        # the Escape key).
        bracket = self._read_with_timeout(0.01)
        if bracket != "[":
            return "ESC"
        code = self._read_with_timeout(0.01)
        if code in self._ARROWS:
            return self._ARROWS[code]
        # Unknown escape -- swallow + ignore.
        return None


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

    # Startup splash: big ASCII blaster + ModelBlaster banner. Skipped
    # automatically when stdout isn't a tty (CI / pipes / no-tty
    # subprocesses) or when --no-splash is passed.
    if (not args.no_splash) and sys.stdout.isatty():
        render_splash(console)
        # 1.5s is enough to read the title without delaying real work.
        time.sleep(1.5)
        console.clear()

    def _files() -> list[Path]:
        if args.paths:
            paths = [p for p in args.paths if p.exists()]
        else:
            paths = list(discover_jsonls(args.root))
            # Optionally tail Claude Code session JSONLs. Default OFF
            # because the logs include any interactive conversation
            # (design / debug chats, not benchmark calls) which would
            # double-count any session that ALSO ran an arm_b_claude
            # cell. Benchmark Claude Code calls always land in
            # benchmarks/results/**/llm_calls.jsonl via the arm
            # driver -- those paths are watched unconditionally.
            if args.watch_claude_code:
                paths.extend(discover_claude_code_sessions())
        return paths

    files = _files()
    poll_files(files, state, pricing)

    sort_mode = "cost"
    scroll = 0
    paused = False
    show_help = False
    bell_rung = False
    explore_mode = False
    active_pane = "recent"   # one of: "recent", "model", "kernel"
    PANE_CYCLE = ["recent", "model", "kernel"]

    def _initial_layout():
        return render(state, ledger, sort_mode=sort_mode,
                      scroll=scroll, max_recent=args.max_recent,
                      watching=len(files), paused=paused,
                      show_help=show_help, explore_mode=explore_mode,
                      active_pane=active_pane,
                      budget_usd=args.budget_usd,
                      terminal_height=console.size.height)

    # Three independent rates, decoupled so each one's slow path doesn't
    # block the others. Keystrokes get 60 Hz polling (≤16 ms latency)
    # regardless of how heavy a render or a filesystem walk takes.
    KEY_INTERVAL  = 1.0 / 60.0                    # ~16 ms
    render_interval = 1.0 / max(args.refresh_hz, 0.1)
    POLL_INTERVAL = max(render_interval * 4, 0.25)  # disk poll: ~4 Hz
    last_render = 0.0
    last_poll = 0.0
    needs_render = True   # render once on entry

    try:
        with _StdinReader() as keyreader, \
             Live(_initial_layout(), console=console,
                  refresh_per_second=args.refresh_hz,
                  screen=True) as live:
            while True:
                now = time.monotonic()

                # ---- 60 Hz: drain pending keystrokes ----
                # Navigation keys (arrows + j/k) are rate-limited to one
                # action per drain so that terminal key auto-repeat (which
                # streams many escape sequences while the key is held)
                # doesn't whip through panes / scroll positions faster
                # than the eye can follow. We coalesce trailing repeats
                # of the same nav key into a single effect for this
                # drain pass. Toggle keys (p, ?, e) and one-shots (s, r,
                # q) still process every event.
                nav_consumed = False
                while True:
                    ch = keyreader.read_key()
                    if ch is None:
                        break
                    is_nav = ch in ("UP", "DOWN", "LEFT", "RIGHT", "j", "k")
                    if is_nav and nav_consumed:
                        continue   # already navigated this drain; drop repeat
                    if ch in ("q", "Q", "\x03"):  # q / Q / Ctrl-C
                        raise KeyboardInterrupt
                    if ch == "p":
                        paused = not paused
                    elif ch == "s":
                        sort_mode = _cycle_sort(sort_mode)
                    elif ch == "?":
                        show_help = not show_help
                    elif ch == "e":
                        explore_mode = not explore_mode
                        # Reset to "recent" pane when entering.
                        if explore_mode:
                            active_pane = "recent"
                            scroll = 0
                    elif ch == "r":
                        _reset_offsets(state)
                        ledger = SessionLedger.load()
                        scroll = 0
                    # Scroll keys: arrows are explore-only; j/k work
                    # in both modes so the legacy behavior still
                    # functions if you don't know about explore.
                    elif ch in ("UP", "k"):
                        scroll = max(0, scroll - 1)
                        nav_consumed = True
                    elif ch in ("DOWN", "j"):
                        # Bound by the active pane's row count so we
                        # can't scroll past the end of whichever pane
                        # we're in.
                        _cumul = compute_windows(state, ledger)["cumulative"]
                        if active_pane == "model":
                            _len = len(_cumul.by_model)
                        elif active_pane == "kernel":
                            _len = len(_cumul.by_kernel)
                        else:
                            _len = len(state.records)
                        scroll = min(scroll + 1, max(0, _len - 1))
                        nav_consumed = True
                    elif ch == "LEFT" and explore_mode:
                        idx = PANE_CYCLE.index(active_pane)
                        active_pane = PANE_CYCLE[(idx - 1) % len(PANE_CYCLE)]
                        scroll = 0   # reset to top of newly-active pane
                        nav_consumed = True
                    elif ch == "RIGHT" and explore_mode:
                        idx = PANE_CYCLE.index(active_pane)
                        active_pane = PANE_CYCLE[(idx + 1) % len(PANE_CYCLE)]
                        scroll = 0   # reset to top of newly-active pane
                        nav_consumed = True
                    else:
                        continue   # unknown key: don't trigger render
                    # Any handled keypress should re-render immediately
                    # so the UI feels live.
                    needs_render = True

                # ---- ~4 Hz: filesystem walk + JSONL polls ----
                if not paused and (now - last_poll) >= POLL_INTERVAL:
                    files = _files()
                    n_new = poll_files(files, state, pricing)
                    ledger = SessionLedger.load()
                    last_poll = now
                    if n_new:
                        needs_render = True
                # In explore mode we DON'T auto-scroll; the user is
                # navigating frozen history. In live mode the scroll
                # offset stays 0 so new records appear at the top.
                if not explore_mode:
                    scroll = 0

                # Budget alarm: ring bell on the transition into >=100%.
                if (args.budget_usd is not None and not bell_rung):
                    total = sum((r.cost_usd or 0.0)
                                for r in state.records)
                    if total >= args.budget_usd:
                        console.bell()
                        bell_rung = True

                # ---- ~refresh_hz Hz: render. Also ticks the mascot
                # animation, so we want a heartbeat even when idle. ----
                if (needs_render or state.blaster_anim_frame > 0
                        or (now - last_render) >= render_interval):
                    live.update(render(
                        state, ledger, sort_mode=sort_mode,
                        scroll=scroll,
                        max_recent=args.max_recent,
                        watching=len(files), paused=paused,
                        show_help=show_help,
                        explore_mode=explore_mode,
                        active_pane=active_pane,
                        budget_usd=args.budget_usd,
                        terminal_height=console.size.height))
                    last_render = now
                    needs_render = False

                time.sleep(KEY_INTERVAL)
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
    files = list(discover_jsonls(args.root))
    if getattr(args, "watch_claude_code", False):
        files.extend(discover_claude_code_sessions())
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


# ───────────────────── shareable report bundle ─────────────────────
#
# A self-contained snapshot of cost / call / kernel data that teammates
# can swap via git commit, Slack attachment, S3 sync, or email. Contains
# only AGGREGATES (no raw prompts/responses), so it's safe to share
# publicly.

_REPORT_SCHEMA_VERSION = 1

REPORTS_DIR = BENCHMARKS_ROOT / "reports"


def _git_identity() -> dict[str, Optional[str]]:
    """Capture git sha + branch for reproducibility. Silent fallback to
    Nones when git isn't available (so the export still works)."""
    import subprocess
    out: dict[str, Optional[str]] = {"sha": None, "branch": None}
    try:
        out["sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(BENCHMARKS_ROOT.parent), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(BENCHMARKS_ROOT.parent), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return out


def _build_report(state: WatcherState, ledger: SessionLedger,
                  pricing: dict, *, report_id: str,
                  session_filter: Optional[str] = None,
                  since: Optional[str] = None,
                  ) -> dict:
    """Aggregate the ingested state into a shareable JSON bundle.
    Filters by session id (only that session's calls) or by ts >= since
    (ISO 8601). When both are omitted, the bundle covers everything
    cumulative.
    """
    import socket
    git = _git_identity()
    records = list(state.records)

    if session_filter is not None:
        sess = ledger.get(session_filter)
        if sess is None:
            raise ValueError(
                f"session {session_filter!r} not found in ledger"
            )
        records = [r for r in records if sess.contains(r.ts)]

    if since is not None:
        records = [r for r in records if r.ts >= since]

    # Aggregate.
    win = _aggregate(records, "EXPORT")
    monthly = _aggregate(
        [r for r in records if is_within_current_month(r.ts)],
        "MONTHLY",
    )

    # Per-cell rollup.
    by_cell: dict[str, dict] = {}
    for r in records:
        slot = by_cell.setdefault(r.cell, {
            "n_calls": 0, "cost_usd": 0.0, "cost_known": True,
            "input_uncached": 0, "input_cached_read": 0,
            "input_cached_write": 0, "output": 0,
        })
        slot["n_calls"] += 1
        if r.cost_usd is None:
            slot["cost_known"] = False
        else:
            slot["cost_usd"] += r.cost_usd
        slot["input_uncached"] += r.input_uncached
        slot["input_cached_read"] += r.input_cached_read
        slot["input_cached_write"] += r.input_cached_write
        slot["output"] += r.output

    sessions_payload = []
    for s in ledger.list_all():
        s_recs = [r for r in records if s.contains(r.ts)]
        if not s_recs and session_filter is None:
            continue
        s_cost = sum((r.cost_usd or 0.0) for r in s_recs)
        sessions_payload.append({
            "id": s.id, "label": s.label,
            "started_at": s.started_at, "ended_at": s.ended_at,
            "is_active": s.is_active,
            "n_calls": len(s_recs), "cost_usd": round(s_cost, 4),
        })

    def _tally_to_dict(t: ModelTally) -> dict:
        return {
            "n_calls": t.n_calls,
            "input_uncached": t.input_uncached,
            "input_cached_read": t.input_cached_read,
            "input_cached_write": t.input_cached_write,
            "output": t.output,
            "cost_usd": round(t.cost_usd, 6),
            "cost_known": t.cost_known,
        }

    def _window_to_dict(w: WindowTotals) -> dict:
        return {
            "label": w.label, "n_calls": w.n_calls,
            "total_cost_usd": round(w.total_cost, 6),
            "cost_known": w.cost_known,
            "input_uncached": w.input_uncached,
            "input_cached": w.input_cached,
            "output": w.output,
        }

    return {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": {
            "user": os.environ.get("USER", "?"),
            "hostname": socket.gethostname(),
            "git_sha": git["sha"], "git_branch": git["branch"],
        },
        "filter": {
            "session": session_filter, "since": since,
        },
        "windows": {
            "cumulative": _window_to_dict(win),
            "monthly": _window_to_dict(monthly),
        },
        "by_model": {k: _tally_to_dict(v) for k, v in win.by_model.items()},
        "by_kernel": {k: _tally_to_dict(v) for k, v in win.by_kernel.items()},
        "by_cell": by_cell,
        "sessions": sessions_payload,
        "config": {
            "pricing_last_updated": pricing.get("last_updated"),
        },
    }


def _render_report_markdown(bundle: dict) -> str:
    """Human-readable markdown rendering of an exported bundle."""
    lines = []
    rid = bundle["report_id"]
    g = bundle["generated_by"]
    lines.append(f"# Cost report — `{rid}`")
    lines.append("")
    lines.append(f"_Generated at {bundle['generated_at']} by_  "
                 f"`{g['user']}@{g['hostname']}`  "
                 f"_on git_  `{g.get('git_sha') or 'n/a'}`  "
                 f"_branch_  `{g.get('git_branch') or 'n/a'}`.")
    flt = bundle.get("filter", {})
    if flt.get("session"):
        lines.append(f"\n_Filter:_ session = `{flt['session']}`")
    if flt.get("since"):
        lines.append(f"\n_Filter:_ since = `{flt['since']}`")
    lines.append("")
    win = bundle["windows"]["cumulative"]
    monthly = bundle["windows"]["monthly"]
    plus = "" if win["cost_known"] else "+"
    lines.append(f"## Totals")
    lines.append(f"- **Spend:** ${win['total_cost_usd']:.4f}{plus} "
                 f"across {win['n_calls']} calls")
    lines.append(f"- This month: ${monthly['total_cost_usd']:.4f} "
                 f"({monthly['n_calls']} calls)")
    lines.append(f"- Input (uncached / cached): {win['input_uncached']:,} / "
                 f"{win['input_cached']:,}")
    lines.append(f"- Output: {win['output']:,}")
    lines.append("")

    if bundle.get("by_model"):
        lines.append("## Per-model")
        lines.append("| Model | Calls | In (uncached) | In (cached) | Out | USD |")
        lines.append("|---|---|---|---|---|---|")
        for mid, t in sorted(bundle["by_model"].items(),
                              key=lambda kv: -kv[1]["cost_usd"]):
            cached = t["input_cached_read"] + t["input_cached_write"]
            lines.append(f"| `{mid}` | {t['n_calls']} | "
                         f"{t['input_uncached']:,} | {cached:,} | "
                         f"{t['output']:,} | ${t['cost_usd']:.4f} |")
        lines.append("")

    if bundle.get("by_kernel"):
        lines.append("## Per-kernel (by op)")
        lines.append("| Kernel | Calls | In | Out | USD |")
        lines.append("|---|---|---|---|---|")
        for k, t in sorted(bundle["by_kernel"].items(),
                            key=lambda kv: -kv[1]["cost_usd"]):
            in_tok = (t["input_uncached"]
                       + t["input_cached_read"]
                       + t["input_cached_write"])
            lines.append(f"| `{k}` | {t['n_calls']} | {in_tok:,} | "
                         f"{t['output']:,} | ${t['cost_usd']:.4f} |")
        lines.append("")

    if bundle.get("by_cell"):
        lines.append("## Per-cell")
        lines.append("| Cell | Calls | USD |")
        lines.append("|---|---|---|")
        for cell, slot in sorted(bundle["by_cell"].items(),
                                  key=lambda kv: -kv[1]["cost_usd"]):
            lines.append(f"| `{cell}` | {slot['n_calls']} | "
                         f"${slot['cost_usd']:.4f} |")
        lines.append("")

    if bundle.get("sessions"):
        lines.append("## Sessions covered")
        lines.append("| Session | Label | Started | Ended | Calls | USD |")
        lines.append("|---|---|---|---|---|---|")
        for s in bundle["sessions"]:
            ended = "ongoing" if s["is_active"] else s["ended_at"][:19]
            lines.append(
                f"| `{s['id']}` | {s.get('label') or ''} | "
                f"{s['started_at'][:19]} | {ended} | "
                f"{s['n_calls']} | ${s['cost_usd']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


# ───────────────────── subcommand: export ─────────────────────


def _methodology_template(name: str, bundle: dict) -> str:
    """Opinionated methodology.md template. The author fills it in
    BEFORE sharing -- without context, numbers alone aren't enough to
    compare approaches."""
    g = bundle["generated_by"]
    cells = ", ".join(sorted(bundle.get("by_cell", {}).keys())[:5])
    return f"""# Methodology: {name}

_Fill this in before sharing — without prose, numbers alone don't tell
the reader WHY you tried something or how to reproduce it._

## Approach
What did you try? (e.g. "Switch Arm B-bedrock from beam=2 exp=3 iter=2
to beam=1 exp=4 iter=1 to spend more on diversity per parent and less
on iteration depth.")

## Hypothesis
What did you expect would work, and why? (e.g. "Wider expansions per
parent should land a better kernel earlier without paying for extra
iterations on already-converged ops.")

## Knobs changed
| Knob | Default | This run |
|---|---|---|
| arm | A | (e.g. B-bedrock) |
| LLM model | — | (e.g. claude-sonnet-4-5) |
| beam | 2 | |
| expansions | 3 | |
| iterations | 2 | |
| FIRESIM_EVAL | 0 | |
| max_usd | none | |

## Result interpretation
What did the numbers show? Which kernels improved, by how much, and
why? Reference specific cells in this report:

  cells covered (first 5): {cells}

See `report.md` for the cost + cycle table; `kernels/` for the actual
generated C code; `per-cell/<cell>/cycles_per_op.json` for per-op cycle
breakdowns.

## Reproducing this report
```bash
git checkout {g.get('git_sha') or '<sha>'}
# ... your commands here ...
mb-cost export --full {name}
```

## Next steps
What would you try next based on these results?
"""


def _collect_experiment_artifacts(bundle: dict, out_dir: Path,
                                  results_root: Path) -> dict:
    """Copy the per-cell + per-(model,quant,target) artifacts into
    ``out_dir/`` so the recipient sees full context: actual kernel C
    source, IR graphs, beam-search trajectory, etc.

    Discovers run dirs DIRECTLY from the filesystem (any directory
    under results_root/ containing a run.json), so Arm A runs that
    made zero LLM calls are still included. Returns a manifest dict.
    """
    import shutil
    manifest: dict = {
        "per_cell": [], "kernels": [], "missing": [],
    }
    per_cell_root = out_dir / "per-cell"
    kernels_root = out_dir / "kernels"
    per_cell_root.mkdir(parents=True, exist_ok=True)
    kernels_root.mkdir(parents=True, exist_ok=True)

    PER_CELL_FILES = [
        "run.json", "accuracy.json", "cycles_per_op.json",
        "kernel_picks.json", "stage_timings.json", "binary_size.json",
        "graph_summary.json", "passes_applied.json",
        "wall_cycles.txt", "env.txt",
        "profile_spike.csv", "profile_firesim.csv",
        "llm_calls.jsonl", "llm_tokens.json",
        "optimize_summary.json", "beam_search_trajectory.jsonl",
        "xpurt_trace.csv", "cross_tile_estimate.json",
    ]
    seen_models: set[tuple[str, str, str]] = set()

    # Filesystem discovery: every dir with a run.json under
    # results_root/ is a candidate cell (Arm A has no llm_calls.jsonl
    # but still has artifacts worth sharing).
    cells: list[tuple[str, Path]] = []
    if results_root.exists():
        for rj_path in results_root.rglob("run.json"):
            run_dir = rj_path.parent
            # Skip results/dashboard.csv etc; only proper cell dirs.
            rel = run_dir.relative_to(results_root)
            cell_key = "/".join(rel.parts)
            cells.append((cell_key, run_dir))

    for cell_key, src in cells:
        dst = per_cell_root / cell_key.replace("/", "__")
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for fname in PER_CELL_FILES:
            sp = src / fname
            if sp.exists():
                shutil.copy2(sp, dst / fname)
                copied.append(fname)
        manifest["per_cell"].append({"cell": cell_key, "files": copied})

        rj_path = src / "run.json"
        if rj_path.exists():
            try:
                rj = json.loads(rj_path.read_text())
                key = (rj.get("model"), rj.get("quant"), rj.get("target"))
                if all(key) and key not in seen_models:
                    seen_models.add(key)
            except json.JSONDecodeError:
                pass

    # Per-(model, quant, target) artifacts: the actual generated C
    # source + IR. ONE copy regardless of how many cells share the
    # tuple. The recipient can read kernel.c byte-for-byte.
    KERNEL_FILES = [
        "kernels.c", "kernels.h", "graph.json",
        "passes_applied.json", "kernel_picks.json",
        "optimize_summary.json", "beam_search_trajectory.jsonl",
    ]
    for (model, quant, target) in seen_models:
        gen_dir = (results_root.parent.parent  # repo root
                   / "examples" / model / quant / "generated")
        # Some artifacts (graph.json, passes_applied.json) live at the
        # generated/ level; kernels.c + target-specific stuff lives at
        # generated/<target>/.
        dst = kernels_root / model / quant / target
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for fname in ("graph.json", "passes_applied.json"):
            sp = gen_dir / fname
            if sp.exists():
                shutil.copy2(sp, dst / fname)
                copied.append(fname)
        target_dir = gen_dir / target
        for fname in KERNEL_FILES:
            sp = target_dir / fname
            if sp.exists():
                shutil.copy2(sp, dst / fname)
                copied.append(fname)
        manifest["kernels"].append({
            "model": model, "quant": quant, "target": target,
            "files": copied,
        })
    return manifest


def cmd_export(args) -> int:
    """Aggregate current state into a portable JSON + Markdown bundle
    suitable for sharing with teammates.

    Two modes:
      * default (lightweight) -- writes <name>.json + <name>.md only.
        Best for cost-only summaries (Slack / email).
      * --full                -- writes a DIRECTORY <name>/ containing
        report.json + report.md + methodology.md template + the actual
        per-cell artifacts (run.json, accuracy.json, cycles_per_op.json,
        kernel_picks.json, etc.) + the generated kernel C source +
        graph.json + optimize_summary.json + beam_search_trajectory.jsonl.
        Best for methodology comparison: the recipient can read your
        actual kernels and re-derive the numbers.
    """
    if not args.pricing.exists():
        print(f"pricing file not found: {args.pricing}", file=sys.stderr)
        return 2
    pricing = load_pricing(args.pricing)
    ledger = SessionLedger.load()
    state = WatcherState()
    files = list(discover_jsonls(args.root))
    if getattr(args, "watch_claude_code", False):
        files.extend(discover_claude_code_sessions())
    poll_files(files, state, pricing)

    try:
        bundle = _build_report(
            state, ledger, pricing,
            report_id=args.name,
            session_filter=args.session,
            since=args.since,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.full:
        bundle["kind"] = "experiment"
        out_root = REPORTS_DIR / args.name
        out_root.mkdir(parents=True, exist_ok=True)
        manifest = _collect_experiment_artifacts(
            bundle, out_root, args.root,
        )
        bundle["bundle_contents"] = manifest
        report_blob = json.dumps(bundle, indent=2, default=str)
        (out_root / "report.json").write_text(report_blob)
        (out_root / "report.md").write_text(
            _render_report_markdown(bundle))
        meth = out_root / "methodology.md"
        if not meth.exists():
            meth.write_text(_methodology_template(args.name, bundle))
        print(f"wrote experiment bundle: {out_root}/")
        print(f"  - report.json ({len(report_blob)/1024:.1f} KB)")
        print(f"  - report.md")
        print(f"  - methodology.md  (TEMPLATE -- edit before sharing)")
        print(f"  - per-cell/  ({len(manifest['per_cell'])} cell(s))")
        print(f"  - kernels/   ({len(manifest['kernels'])} "
              f"(model,quant,target) tuple(s))")
        if manifest["missing"]:
            print(f"  WARN missing run dirs (skipped): "
                  f"{manifest['missing']}")
        return 0

    bundle["kind"] = "report"
    out_json = REPORTS_DIR / f"{args.name}.json"
    out_md = REPORTS_DIR / f"{args.name}.md"
    out_json.write_text(json.dumps(bundle, indent=2, default=str))
    out_md.write_text(_render_report_markdown(bundle))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(f"\nShare either file. JSON is machine-readable + diff-able; "
          f"Markdown is human-friendly for Slack / PRs / email.")
    return 0


# ───────────────────── subcommand: import ─────────────────────


def cmd_import(args) -> int:
    """Read a teammate's exported bundle and print a text summary.
    Read-only: does NOT merge anything into the local ledger or state.

    Accepts either a single .json file (lightweight bundle) OR a
    directory produced by `mb-cost export --full`. For directory
    bundles, prints the methodology.md + report + a listing of the
    included kernels, per-cell artifacts, and graphs.
    """
    src = Path(args.path)
    if not src.exists():
        print(f"path not found: {src}", file=sys.stderr)
        return 2

    bundle_path = src / "report.json" if src.is_dir() else src
    if not bundle_path.exists():
        print(f"no report.json found at {bundle_path}", file=sys.stderr)
        return 2
    try:
        bundle = json.loads(bundle_path.read_text())
    except json.JSONDecodeError as e:
        print(f"not valid JSON: {e}", file=sys.stderr)
        return 2
    sv = bundle.get("schema_version")
    if sv != _REPORT_SCHEMA_VERSION:
        print(f"warning: bundle schema_version={sv} differs from "
              f"current ({_REPORT_SCHEMA_VERSION}); "
              f"some fields may be missing.",
              file=sys.stderr)

    # For full-experiment bundles, show methodology first (it's the
    # author's intent + the only context the numbers belong to).
    if src.is_dir():
        meth = src / "methodology.md"
        if meth.exists():
            print("═" * 76)
            print(meth.read_text())
            print("═" * 76)
        print()

    print(_render_report_markdown(bundle))

    if src.is_dir():
        contents = bundle.get("bundle_contents") or {}
        print()
        print("## Artifacts in this bundle")
        cells = contents.get("per_cell") or []
        kerns = contents.get("kernels") or []
        print(f"- per-cell artifacts: {len(cells)} cell(s)")
        for c in cells[:5]:
            print(f"    {c['cell']}  ({len(c['files'])} files)")
        if len(cells) > 5:
            print(f"    ... +{len(cells) - 5} more")
        print(f"- generated kernels: {len(kerns)} (model,quant,target) tuple(s)")
        for k in kerns:
            print(f"    {k['model']}/{k['quant']}/{k['target']}  "
                  f"({len(k['files'])} files)")
        print()
        print(f"Read the actual kernels at: "
              f"{src / 'kernels'}/<model>/<quant>/<target>/kernels.c")
    return 0


# ───────────────────── subcommand: diff ─────────────────────


def _load_bundle(p: Path) -> tuple[dict, Optional[Path]]:
    """Load report.json from a file OR a directory bundle. Returns
    (bundle_dict, kernels_root_or_None) so the caller can find the
    .c source files if it's a directory bundle."""
    if p.is_dir():
        bp = p / "report.json"
        return json.loads(bp.read_text()), (p / "kernels")
    return json.loads(p.read_text()), None


def cmd_diff(args) -> int:
    """Side-by-side comparison of two report bundles. When both are
    directory bundles (from `mb-cost export --full`), also surfaces
    kernel-source diffs so you can see HOW the approaches differ at
    the C level, not just the bottom-line numbers."""
    a_path = Path(args.a)
    b_path = Path(args.b)
    if not a_path.exists() or not b_path.exists():
        print("both A and B paths must exist", file=sys.stderr)
        return 2
    a, a_kernels = _load_bundle(a_path)
    b, b_kernels = _load_bundle(b_path)

    def _g(d, *keys, default=None):
        for k in keys:
            d = d.get(k, {}) if isinstance(d, dict) else default
        return d if d != {} else default

    print(f"DIFF  {a['report_id']}  vs  {b['report_id']}\n")
    aw = a["windows"]["cumulative"]; bw = b["windows"]["cumulative"]
    print(f"  spend:   ${aw['total_cost_usd']:.4f}  vs  ${bw['total_cost_usd']:.4f}"
          f"  (Δ ${bw['total_cost_usd'] - aw['total_cost_usd']:+.4f})")
    print(f"  calls:   {aw['n_calls']}  vs  {bw['n_calls']}"
          f"  (Δ {bw['n_calls'] - aw['n_calls']:+d})")
    print(f"  output:  {aw['output']:,}  vs  {bw['output']:,}"
          f"  (Δ {bw['output'] - aw['output']:+,})")

    am = set(a.get("by_model", {}).keys())
    bm = set(b.get("by_model", {}).keys())
    common = sorted(am & bm)
    only_a = sorted(am - bm)
    only_b = sorted(bm - am)
    if common:
        print("\n  per-model (common):")
        for m in common:
            ac = a["by_model"][m]["cost_usd"]
            bc = b["by_model"][m]["cost_usd"]
            print(f"    {m[:60]:60s}  ${ac:.4f}  vs  ${bc:.4f}  "
                  f"(Δ ${bc-ac:+.4f})")
    if only_a:
        print(f"\n  only in {a['report_id']!r}: {only_a}")
    if only_b:
        print(f"\n  only in {b['report_id']!r}: {only_b}")

    # Kernel-source diff: only available when both sides are full
    # directory bundles. Lists shared (model,quant,target) tuples
    # and reports byte-size + LOC diffs of the actual kernels.c.
    if a_kernels and b_kernels:
        print("\n  KERNEL SOURCE DIFF (both bundles are --full):")
        a_keys = set()
        b_keys = set()
        for root, keys in ((a_kernels, a_keys), (b_kernels, b_keys)):
            if root.exists():
                for mdir in root.iterdir():
                    if not mdir.is_dir(): continue
                    for qdir in mdir.iterdir():
                        if not qdir.is_dir(): continue
                        for tdir in qdir.iterdir():
                            if tdir.is_dir():
                                keys.add(
                                    f"{mdir.name}/{qdir.name}/{tdir.name}"
                                )
        common_k = sorted(a_keys & b_keys)
        for k in common_k:
            ap = a_kernels / k / "kernels.c"
            bp = b_kernels / k / "kernels.c"
            if not (ap.exists() and bp.exists()):
                continue
            a_sz, b_sz = ap.stat().st_size, bp.stat().st_size
            a_lo = sum(1 for _ in open(ap))
            b_lo = sum(1 for _ in open(bp))
            same = ap.read_bytes() == bp.read_bytes()
            tag = " (identical)" if same else ""
            print(f"    {k}/kernels.c{tag}")
            print(f"      size:  {a_sz:>6,}B vs {b_sz:>6,}B  "
                  f"(Δ {b_sz-a_sz:+,}B)")
            print(f"      LOC:   {a_lo:>6,}  vs {b_lo:>6,}  "
                  f"(Δ {b_lo-a_lo:+,})")
            if not same:
                print(f"      diff: diff -u {ap} {bp}")
        only_a_k = sorted(a_keys - b_keys)
        only_b_k = sorted(b_keys - a_keys)
        if only_a_k:
            print(f"    only in A: {only_a_k}")
        if only_b_k:
            print(f"    only in B: {only_b_k}")
    elif (a_kernels and not b_kernels) or (b_kernels and not a_kernels):
        print("\n  (one bundle is --full and the other isn't; "
              "kernel-source diff requires both)")
    return 0


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
    files = list(discover_jsonls(args.root))
    if getattr(args, "watch_claude_code", False):
        files.extend(discover_claude_code_sessions())
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
    ap.add_argument("--watch-claude-code", action="store_true",
                    help="ALSO tail Claude Code session JSONLs at "
                         "~/.claude/projects/<encoded-cwd>/*.jsonl. "
                         "Default: OFF -- those logs include any "
                         "interactive Claude Code conversation in this "
                         "project directory (e.g. design / debug chats), "
                         "which inflates the dashboard with non-benchmark "
                         "spend. Benchmark LLM calls always land in "
                         "benchmarks/results/**/llm_calls.jsonl via the "
                         "arm drivers; that path is watched unconditionally.")


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
    live.add_argument("--refresh-hz", type=float, default=20.0,
                      help="redraw rate Hz (default: 20). At 20 Hz the "
                           "main loop sleeps 50ms per cycle, which is "
                           "also the keystroke-to-effect latency. Drop "
                           "to 4 for low-bandwidth ssh sessions or when "
                           "another live TUI is sharing the terminal.")
    live.add_argument("--max-recent", type=int, default=15,
                      help="recent calls visible (default: 15)")
    live.add_argument("--budget-usd", type=float, default=None,
                      help="visual budget alarm threshold")
    live.add_argument("--no-splash", action="store_true",
                      help="skip the startup BLASTER splash screen")

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

    # export: shareable bundle for teammates
    exp = sub.add_parser("export",
                         help="export current state as a portable JSON+MD "
                              "bundle for sharing with teammates")
    _add_common_args(exp)
    exp.add_argument("name", help="report id (used as filename; should "
                                   "be unique per snapshot)")
    exp.add_argument("--session", default=None,
                     help="filter to a specific session id (default: all)")
    exp.add_argument("--since", default=None,
                     help="filter records to ts >= YYYY-MM-DD or ISO 8601")
    exp.add_argument("--full", action="store_true",
                     help="produce a DIRECTORY bundle <name>/ with "
                          "per-cell artifacts + generated kernel C source "
                          "+ IR graphs + optimize_summary.json + a "
                          "methodology.md template. For methodology "
                          "comparison: teammates can read your actual "
                          "kernels and reproduce the numbers.")

    # import: read someone else's bundle
    imp = sub.add_parser("import",
                         help="render a teammate's exported bundle as text "
                              "(read-only; doesn't merge into local state)")
    imp.add_argument("path", help="path to a .json bundle from `mb-cost export`")

    # diff: compare two bundles
    df = sub.add_parser("diff", help="side-by-side diff of two bundles")
    df.add_argument("a", help="path to first bundle")
    df.add_argument("b", help="path to second bundle")

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
                                    "export", "import", "diff",
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
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "import":
        return cmd_import(args)
    if args.cmd == "diff":
        return cmd_diff(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
