"""ModelBlaster as the graph optimiser: KernelBlaster rounds to a fixpoint.

One round, for a set of IRs sharing a work dir (hints, kernel cache and
KernelBlaster knowledge carry over between rounds and runs):

1. **profile** every IR in its current state -- accepted graph hints applied,
   cached winners in place (``kb_plan.profile_ir``);
2. **MB proposals** from the whole-model profiles
   (``graph_hints.propose_from_profile``: elementwise consumers worth fusing,
   including consumers of already-fused ops, so chains grow);
3. **plan** -- rank op instances by (entry, signature, shape) across the IRs,
   select under the budget, skipping items sent in earlier rounds
   (``kb_plan``); export the problems seeded from the cache, run
   KernelBlaster on them, ingest its winners; its ``proposals.json`` (fuse,
   split) become hints;
4. **decide** -- for the undecided hints with the largest estimated gain
   (at most ``max_variants``): export the affected ops as an *original*
   problem and, with the hint applied, as a *variant* problem, both seeded
   with the same current-best kernels; KernelBlaster optimises both with the
   same budget; both composite bests are rebuilt and re-timed back to back
   on one GPU (``kb_proposals.remeasure``); the hint is accepted if the
   variant is ``accept_gain`` faster. Accepted hints apply from the next
   round on, and the variant's new slots (the fused kernel, the split
   variants) go into the cache. Hints that cannot affect each other
   (``footprint`` / ``conflicts``) are decided together in *waves*
   (``schedule``): one KernelBlaster run for all their problems, re-timing
   as soon as a hint's two problems finish; conflicting hints wait for the
   wave before them, so the outcome is that of deciding them one by one in
   rank order (``--parallel-hints 1``);
5. stop when a round accepts nothing and has nothing new to send (the
   fixpoint), or after ``rounds``.

Finally every IR is verified at model level with everything applied
(``kb_ingest.verify_ir``), the model's result is timed through KernelBlaster's
evaluator on the runs' pool and GPU model (``kb_model_run``) -- with the
PyTorch module timed against it on the same GPU (``--torch-baseline``,
``--torch-python``; ``torch_us`` / ``speedup_vs_torch`` in the summary) -- and
``summary.json`` lists first vs final times.

Where the GPU steps run: KernelBlaster's runs, the re-measurement and the
model's result use KernelBlaster's pool (``--gpus`` here, or the
``--ssh-host`` GPUs). The profiles (step 1) and the final verification go
there too with ``--ssh-host`` (or ``--mb-evaluator kb``): through
KernelBlaster's evaluator (``kb_eval.run_problem``, ``scripts/time_problems.py``),
the references computed here at export; without it they are built and run
on this host's default GPU, as before. Every GPU step records its device and
must run on one GPU model: ``--gpu-model``, else the model the first profile
ran on (``state.json`` ``gpu_model``), which then also pins the
KernelBlaster runs (``--gpu-model``); another model anywhere is an error
unless ``--allow-gpu-mismatch``. ``summary.json`` ``gpu`` lists each step's
devices.

    python -m modelblaster.pipeline.kb_graph_opt IR_DIR [...] --work W --kb KB_CHECKOUT \\
        [--model claude-code:sonnet] [--plugin stubs.py] [--rounds 4] \\
        [--coverage 0.8 --top-k 6] [--rollouts 2 --steps 4 --variant-steps 3] \\
        [--max-variants 3 --remeasure 5 --accept-gain 0.03] [--arch 75] \\
        [--parallel-hints auto|N] [--gpus auto|all|0,2] \\
        [--ssh-host H [--ssh-host H2] --cuda-visible-devices auto|ids [--ssh-key K]] \\
        [--runtime auto|host|process] [--gpu-model M] [--allow-gpu-mismatch] [--mb-evaluator auto|kb|builtin] \\
        [--knowledge v1|v2] [--ncu off|best|all] [--reward e2e|slot] [--model-simple M --model-hard M] \\
        [--hetero-policy P] [--models-config F] [--upkeep-model M] [--kb-arg=--flag --kb-arg=value] \\
        [--efficiency-rules on|safe|off] [--stop-patience N] [--plan-min-share-late X] \\
        [--knowledge-global off|read|readwrite --knowledge-global-seed SEED_DIR \\
         --knowledge-global-store STORE --knowledge-global-part cuda:sm_89|fp16]

Efficiency rules (``kb_efficiency.py``; ``l3_farm/analysis/efficiency_rules.md``), ON by default since X-129 --
results before and after are not directly comparable: ``--efficiency-rules on`` (default) = R1 patience (the
plan's KernelBlaster runs stop a slot problem after ``--stop-patience`` 8 consecutive committed steps without a
new slot best; never the hint-decision runs) + R2 late min-share (rounds >= 1 only: plan items below
``--plan-min-share-late`` 0.005 of the round's model time are not sent; round 0 keeps ``--plan-min-share`` /
``--min-share``); ``safe`` = late min-share 0.01, no patience; ``off`` = the behaviour before X-129 (ablations:
pass ``--efficiency-rules off``). A per-rule flag overrides the preset (0 = that rule off). Without any of these
flags a work dir keeps the rules it was started with (``state.json`` ``efficiency``); a work dir from before
X-129 resumes with the rules off and its round key unchanged. Recorded per round (``round.json`` /
``state.json`` ``efficiency``: late-skipped items with their shares, patience stops with their commit) and in
``summary.json`` ``efficiency``.

Global knowledge (``kb_global.py``, off by default): ``read`` starts a fresh ``knowledge/`` from a
global snapshot's seed instead of the repo seed (recorded in ``state.json`` / ``summary.json``
``knowledge_global`` and in the round key); ``readwrite`` also merges the run back into the store.

Work dir: ``graph_hints.json`` (modelblaster.graph_hints/v1), ``cache/``
(mb-cuda-cache/1), ``knowledge/`` (KernelBlaster), ``state.json``,
``rNN/{profile,plan.json,problems,run,winners,variants,round.json}``,
``summary.json``. One process at a time owns a work dir (``.lock``).

Resume (``kb_resume.py``): ``--rounds`` is the total for the work dir, IR set and
settings (``--more-rounds N`` adds N); a killed run continues inside the
interrupted round from ``rNN/progress.json`` (the KernelBlaster runs from their
journals); SIGTERM / SIGINT / SIGHUP stop cleanly, exit 75.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import graph_hints as gh
from . import kb_barrier
from . import kb_efficiency
from . import kb_global
from . import kb_halfway
from . import kb_resume
from . import kb_telemetry as kbt
from . import kb_trace
from .kb_export import base_slot_id, export_problem
from .kb_ingest import CudaCache, cache_plan, ingest, verify_ir
from .kb_plan import export_plan, make_plan, profile_ir
from .kb_proposals import (ACCEPT_GAIN, _all_specs, _median, best_sources, find_chains, is_infra_result,
                           remeasure, with_fused_specs, write_variant_ir)
from .kernel_abi import parse_signature


@dataclass
class Config:
    kb: Path
    model: str = "claude-code:sonnet"
    reflect_model: str | None = None
    plugins: list = field(default_factory=list)
    python: str = sys.executable
    arch: str = "75"
    rounds: int = 4                     # rounds in total for this work dir, IR set and configuration
    more_rounds: int = 0                # run this many rounds more than were run (even past a fixpoint)
    rollouts: int = 2
    steps: int = 4
    variant_steps: int | None = None
    max_fix: int = 2
    coverage: float = 0.8
    top_k: int | None = 6
    min_share: float = 0.01
    min_gain_pct: float = 2.0
    # what a round sends to KernelBlaster: "budget" = the cost-ranked selection (coverage / top_k / min_share),
    # one problem per IR; "all" = every op kind x shape of the IRs not sent before, one problem per kernel,
    # run in parallel (plan_parallel at a time, 0 = all of them): each kernel gets its own rollouts in one round
    plan_mode: str = "budget"
    plan_parallel: int = 0
    # plan_mode "all": cap the per-item problems to the plan_top_k largest ops (0 = no cap) above plan_min_share of the
    # model time -- large nets have hundreds of op/shape items, which one GPU cannot work through in a round
    plan_top_k: int = 0
    plan_min_share: float = 0.0
    # E7 cost-efficiency rules (kb_efficiency.py; l3_farm/analysis/efficiency_rules.md), ON by default since X-129:
    # efficiency_rules on (patience 8 + late min-share 0.005) | safe (late min-share 0.01) | off (the behaviour
    # before X-129); None = auto: on for a new work dir, else the rules the work dir was started with (off for a
    # work dir from before X-129). stop_patience: KernelBlaster --stop-patience for the plan's runs (never the hint
    # decisions'), 0 = off; plan_min_share_late: the plan's min share in rounds >= 1 (round 0 keeps min_share /
    # plan_min_share), 0 = off. An explicit per-rule value overrides the preset's.
    efficiency_rules: str | None = None
    stop_patience: int | None = None
    plan_min_share_late: float | None = None
    max_variants: int = 3
    remeasure: int = 5
    # a re-measurement lost to an infrastructure failure (the copy to / the link to / the GPU of the
    # leased device failed, not a candidate) is retried on a fresh lease this many times in all, the
    # k-th retry after k x remeasure_retry_s, and once more when the wave's KernelBlaster run has
    # finished; still failing, the hint is left undecided (a candidate again next round)
    remeasure_attempts: int = 3
    remeasure_retry_s: float = 20.0
    accept_gain: float = ACCEPT_GAIN
    profile_runs: int = 3
    seed: int = 0
    # hints decided at once (0 = auto: max(4, 2 x GPUs in the pool)); 1 = one by one
    parallel_hints: int = 0
    # KernelBlaster's GPU pool (run_multislot --gpus / --ssh-host / --cuda-visible-devices)
    gpus: str = "auto"
    ssh_hosts: list = field(default_factory=list)
    cuda_visible_devices: list = field(default_factory=list)
    ssh_key: str | None = None
    remote_dir: str | None = None
    # KernelBlaster's runtime (run_multislot / time_problems --runtime): auto | host | process
    runtime: str = "auto"
    # KernelBlaster run_multislot --eval-timeout-s / --eval-timeout-mode (fixed | adaptive)
    eval_timeout_s: int = 900
    eval_timeout_mode: str = "fixed"
    # the GPU model every timing must run on (default: the model the KernelBlaster runs recorded);
    # allow_gpu_mismatch: time anyway where it is not available (loud warnings)
    gpu_model: str | None = None
    allow_gpu_mismatch: bool = False
    # where ModelBlaster's own GPU steps run (the per-round profiles, the final verification, the
    # local re-measurement): kb = KernelBlaster's evaluator on the runs' pool (local GPUs or the
    # --ssh-host GPUs); builtin = built and run by ModelBlaster here (profiles / verification on the
    # default GPU, re-measurement on a leased local GPU); auto = kb with ssh_hosts, else builtin
    mb_evaluator: str = "auto"
    # KernelBlaster's knowledge and search (run_multislot flags of the same names): v2 also syncs
    # this work dir's graph_hints.json into its graph hypotheses at the start of every KB run
    knowledge: str = "v2"
    ncu: str = "off"
    reward: str | None = None
    model_simple: str | None = None
    model_hard: str | None = None
    hetero_policy: str | None = None
    models_config: str | None = None
    upkeep_model: str | None = None
    # the model's result against PyTorch on the same GPU (kb_model_run --torch-*; docs/gpu_harness_perf.md
    # "Against PyTorch"): auto | on | off, the torch python of the timing host ("" = $MB_TORCH_PYTHON /
    # the local python), the model's PyTorch source ("" = found next to the IR set), ABBA rounds, variants
    torch_baseline: str = "auto"
    torch_python: str = ""
    torch_source: str = ""
    torch_rounds: int = 4
    torch_variants: str = "eager,graph,compile_mna,compile_ro"
    # global knowledge across problems / sweeps (kb_global.py): off (default: knowledge/ starts from the
    # repo seed) | read (a fresh knowledge/ starts from a global snapshot's seed) | readwrite (and the run
    # is merged back into the store at the end); the seed dir to start from (as `merge seed` / the farm
    # writes it), the store root (read without a seed: materialize from its current snapshot), the
    # partition (default cuda:sm_<arch>|<IR dtype>) and the provenance name (default: the work dir name)
    knowledge_global: str = "off"
    knowledge_global_seed: str = ""
    knowledge_global_store: str = ""
    knowledge_global_part: str = ""
    knowledge_global_problem: str = ""
    # mid-run sharing (kb_global.refresh_at_round, KernelBlaster knowledge/v2/refresh.py): a directory where a
    # controller drops refresh packages (*.tar.gz or dirs with refresh.json); at the start of every round
    # the newest one is imported into knowledge/ (other runs' claims, statistics, impl winners since this
    # run's last sync; never its own). "" (default) = off: no import and the round key is unchanged
    knowledge_global_refresh: str = ""
    # halfway sync (kb_halfway.py; l3_farm/analysis/halfway_sync.md): "on" pauses the round's plan run once every rollout
    # completed half its steps, waits (bounded) for the controller's refresh package, imports it and goes on with each
    # rollout's own state. Needs a seed and --knowledge-global-refresh. "off" (default) = nothing changes, round key too
    knowledge_global_halfway: str = "off"
    knowledge_global_halfway_wait_s: float = 180.0
    # barrier scheduling (kb_barrier.py; l3_farm/analysis/barrier_scheduling.md): "on" parks the run after the characterization pass
    # (writes <work>/parked.json, exits 78) so the farm can resume every problem once all have parked; rerun the same command
    # without the flag to resume. Not part of the round key. "off" (default) = nothing changes
    park_after_char: str = "off"
    # characterization sync (kb_barrier.py): after A4's characterization pass, write <round>/char_sync.json (stage paused), wait (bounded)
    # for the controller's package (ack char-r00.ack.json in --knowledge-global-refresh), import it (why="char") and go on to the
    # re-profile / plan. Needs a seed + the refresh inbox. Superseded by --park-after-char. "off" (default) = nothing changes
    knowledge_global_char_sync: str = "off"
    knowledge_global_char_sync_wait_s: float = 180.0
    # anything else for run_multislot, verbatim (e.g. --kb-arg=--explore-frac --kb-arg=0.2)
    kb_args: list = field(default_factory=list)
    # CUDA-only-baseline learnings (cuda_learnings.py; l3_farm/analysis/cuda_learnings_impl.md), all off by default:
    # winner reuse across split variants / problems (off|siblings|global; reuse_canon: flag fields that are not
    # shape, "default" = kb_plan.CANON_FLAGS; rollout_accept: auto = local with reuse on | local | composite),
    # concurrent CUDA-graph branches (off|on) with an SM budget per branch (split|none|strict) and a deadlock watchdog,
    # fusion beyond producer->elementwise (off|siblings|epilogue|attention|all; fused variants keep per-shape kernels)
    reuse_winners: str = "off"
    reuse_canon: str = "default"
    rollout_accept: str = "auto"
    graph_concurrency: str = "off"
    sm_budget_policy: str = "split"
    conc_timeout_s: float = 30.0
    fusion_extended: str = "off"
    # search fixes (search_fixes.py; l3_farm/analysis/search_fixes.md): fallback_repair on (route + re-queue the
    # instances a profile fallback left on a seed; default, kill switch off) | route | requeue | off, at most
    # fallback_requeue_max re-queues per item; final_guard warn (default) | repair (route, zero LLM, and verify again)
    # | off: dominant ops (>= final_guard_share of the model) on or near (< final_guard_min_speedup x) their seed;
    # seed_nearest off | on: nearest-shape verified kernels as trial seeds; rollout_resync X (0 = off): KernelBlaster
    # rollouts lagging the composite by > X after a non-improving step restart from it (rollout_accept lineage = local
    # acceptance + resync 1.5)
    fallback_repair: str = "on"
    fallback_requeue_max: int = 1
    final_guard: str = "warn"
    final_guard_share: float = 0.10
    final_guard_min_speedup: float = 1.25
    seed_nearest: str = "off"
    rollout_resync: float = 0.0
    # fused-slot winners of the plan runs are ingested (their specs carry the fused kinds of the round's graphs; X-137):
    # on (default, correctness fix) | off (kill switch, also env MB_INGEST_FUSED=0: the pre-fix behaviour)
    ingest_fused: str = "on"
    # technique-only knowledge seed: off = KERNELBLASTER_SEED_MINED=0 for this process and every KernelBlaster run it starts
    # (the repo's mined.json evidence is not folded into a fresh knowledge.db); on = unchanged (default)
    seed_mined: str = "on"
    # graph search redesign (graph_search.py; l3_farm/analysis/graph_search_redesign.md), all off by default
    # (rounds = the round-based flow; nothing else runs and the round key is unchanged): interleaved = hint
    # decisions inside the plan's KernelBlaster run on successive-halving rungs, decided and applied mid-round;
    # dataset priors (auto|off|rank|preapply), round-0 characterization pass, survival-weighted plan budgets,
    # graph-diverse beam (members), fold_layout proposals
    graph_search: str = "rounds"
    graph_priors: str = "auto"
    graph_prior_conf: float = 0.55
    graph_prior_min_n: int = 3
    graph_explore: float = 0.1
    graph_prior_min_gain: float = 0.0
    graph_characterize: int = -1
    graph_char_top: int = 8
    graph_char_rollouts: int = 1
    graph_survival: str = "auto"
    graph_survival_head: float = 0.25
    graph_rung_steps: int = 0
    graph_eta: float = 2.0
    graph_cap_steps: int = 8
    graph_defer: int = 1
    graph_diverse_beam: int = 0
    graph_layout_fold: str = "off"
    # unapply audit of pre-applied hints (preapply_audit.py; l3_farm/analysis/preapply_audit.md): which classes may be
    # pre-applied (auto|high|all|none), the audit (auto|off|final|rounds), its revert margin (-1 = accept_gain) and
    # the zero-LLM nearest-shape routing of stage kernels (on|off)
    graph_preapply_classes: str = "auto"
    graph_preapply_audit: str = "auto"
    graph_preapply_margin: float = -1.0
    graph_preapply_route: str = "on"
    # graph and kernel capabilities (capabilities.py; l3_farm/analysis/capabilities_impl.md): dead-op pruning at
    # export on (kill switch off); recurrent op class off|on; precision policy off|auto (acts at extraction);
    # region slots off|rnn_stack,attention_block,all; multi-turn codegen with an experiment tool off|on and its
    # per-step budget "turns,runs,seconds"; whole-model context in slot problems off|on
    prune_dead: str = "on"
    recurrent_class: str = "off"
    precision_policy: str = "off"
    region_slots: str = "off"
    multiturn: str = "off"
    multiturn_budget: str = "3,4,120"
    # X-145 experiment recording (needs --multiturn on): off | record | ingest | ingest+prompt
    experiments: str = "off"
    model_context: str = "off"
    view_fusion: str = "off"
    # trace store (KernelBlaster docs/trace.md): <work>/trace for ModelBlaster's steps and each
    # KernelBlaster run's <out>/trace (prompts, responses, candidates, builds, driver output):
    # full | meta (hashes and sizes only) | off; env KB_TRACE
    trace: str = field(default_factory=lambda: os.getenv("KB_TRACE") or "full")


_LIST_FIELDS = ("plugins", "ssh_hosts", "cuda_visible_devices", "kb_args")
_MANUAL_FIELDS = ("kb", "model", "reflect_model", "python", "ssh_key", "remote_dir", "gpus", "parallel_hints",
                  "runtime", "gpu_model", "allow_gpu_mismatch", "mb_evaluator", *_LIST_FIELDS)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


# ---------------------------------------------------------------- locks

def _flock_off() -> bool:
    return os.getenv("KERNELBLASTER_GPU_FLOCK", "1").lower() in ("0", "false", "no", "off")


def gpu_lock_path(ident: str) -> Path:
    """KernelBlaster's per-GPU lock file (``resources.gpu_lock_wrap``): every
    KernelBlaster timed run holds an flock on it, so MB's own timed runs
    holding it too take turns with KernelBlaster runs of any process."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", ident)
    return Path(os.getenv("KERNELBLASTER_GPU_LOCK_DIR", "/dev/shm")) / f"kernelblaster_gpu_{safe}.lock"


@contextlib.contextmanager
def _flock(path: Path, *, blocking: bool = True, timeout: float = 3600.0):
    """flock(2) on ``path`` (created if missing). Blocking waits up to
    ``timeout`` and then raises TimeoutError; non-blocking raises
    BlockingIOError when the lock is held."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if not blocking:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            t_end = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > t_end:
                        raise TimeoutError(f"lock {path} held for more than {timeout:.0f} s")
                    time.sleep(0.05)
        yield fd
    finally:
        os.close(fd)                                  # closing releases the lock


@contextlib.contextmanager
def work_lock(work: Path):
    """One kb_graph_opt process per work dir: it is the only writer of the
    hints file, the cache index, state.json and (through the one
    KernelBlaster process it runs at a time) the knowledge dir."""
    work.mkdir(parents=True, exist_ok=True)
    try:
        with _flock(work / ".lock", blocking=False):
            yield
    except BlockingIOError:
        raise RuntimeError(f"{work} is in use by another kb_graph_opt run (lock {work / '.lock'})") from None


class GpuLeases:
    """Exclusive leases on this host's GPUs for ModelBlaster's own timed runs
    (the re-measurement that decides a hint): one lease per device in this
    process, least recently used first, and while leased the device's
    KernelBlaster lock file is held, so KernelBlaster runs (this flow's or
    another process's) wait instead of timing on it at the same time.

    ``devices``: [{"ident": UUID or index, "name": ...}]; empty = one
    unpinned device (the environment's default GPU, no lock file). A device
    with a ``host`` (``--ssh-host``) is leased in this process only: the
    KernelBlaster evaluator that times on it takes that host's GPU lock per
    run (``lease_device``)."""

    def __init__(self, devices: list[dict] | None = None, *, lock_timeout: float = 3600.0):
        self.devices = list(devices) or [{"ident": None, "name": "default"}]
        self.lock_timeout = lock_timeout
        self._cv = threading.Condition()
        self._busy: set[int] = set()
        self._last: dict[int, float] = {}
        self.log: list[dict] = []                     # (device, t0, t1) per lease, for tests and stats

    @contextlib.contextmanager
    def lease(self):
        """-> the environment of the leased local GPU (``CUDA_VISIBLE_DEVICES``)."""
        with self.lease_device() as dev:
            ident = dev.get("ident")
            yield {"CUDA_VISIBLE_DEVICES": str(ident)} if ident is not None else {}

    @contextlib.contextmanager
    def lease_device(self, lock: bool = True):
        """-> the leased device dict (local: its KernelBlaster lock held, unless ``lock`` is
        False -- a KernelBlaster evaluator pinned to the device takes it per run)."""
        with self._cv:
            while len(self._busy) == len(self.devices):
                self._cv.wait()
            i = min((k for k in range(len(self.devices)) if k not in self._busy),
                    key=lambda k: (self._last.get(k, 0.0), k))
            self._busy.add(i)
        dev = self.devices[i]
        ident = dev.get("ident")
        try:
            with contextlib.ExitStack() as stack:
                if lock and ident is not None and not dev.get("host") and not _flock_off():
                    try:
                        stack.enter_context(_flock(gpu_lock_path(str(ident)), timeout=self.lock_timeout))
                    except PermissionError:            # another user's lock file: run unlocked, as KB does
                        pass
                t0 = time.time()
                try:
                    yield dev
                finally:
                    self.log.append({"device": i, "t0": t0, "t1": time.time()})
        finally:
            with self._cv:
                self._busy.discard(i)
                self._last[i] = time.monotonic()
                self._cv.notify_all()


# ---------------------------------------------------------------- GPUs

_DETECT = r'''
import asyncio, json, sys
sys.path.insert(0, sys.argv[1] + "/src")
from kernelblaster.servers.utils import resources as r
a = json.loads(sys.argv[2])
out = {"local": [], "remote": []}
if not a["ssh_hosts"]:
    inv = r.gpu_inventory()
    chosen, why = r.select_gpus(a["gpus"], inv)
    out["local"] = [{"ident": g.uuid or str(g.index), "index": g.index, "name": g.name} for g in chosen]
    out["why"] = why
else:
    cvd = a["cvd"] or [None]
    for k, h in enumerate(a["ssh_hosts"]):
        spec = cvd[k] if len(cvd) == len(a["ssh_hosts"]) else cvd[0]
        devs = asyncio.run(r.remote_devices(h, spec, ssh_key=a["ssh_key"]))
        out["remote"] += [{"host": h, "ident": d.ident, "index": d.index, "name": d.name} for d in devs]
print(json.dumps(out))
'''


def detect_gpus(cfg: Config) -> dict:
    """KernelBlaster's own GPU selection (``resources.select_gpus``: auto =
    the idle GPUs, else the least busy one) for the pool the KernelBlaster
    runs will use -> {"local": [...], "remote": [...]}; on failure, one
    unpinned local device."""
    args = {"gpus": cfg.gpus, "ssh_hosts": list(cfg.ssh_hosts), "cvd": list(cfg.cuda_visible_devices),
            "ssh_key": cfg.ssh_key}
    try:
        cp = subprocess.run([cfg.python, "-c", _DETECT, str(Path(cfg.kb).resolve()), json.dumps(args)],
                            capture_output=True, text=True, timeout=120)
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"local": [], "remote": [], "why": f"detection failed: {e!r}"}


def auto_parallel_hints(n_gpus: int) -> int:
    """Hints per wave by default: max(4, 2 x GPUs). The wave width does not
    decide how many runs share a GPU -- KernelBlaster's pool gives each timed
    run a GPU of its own and its build limiter bounds the compiles -- it
    decides how much LLM latency and compile time overlaps other hints'
    runs. Measured (docs/graph_optimizer.md): 4 hints per wave beat 2 on one
    TITAN RTX and on two L40S, most with LLM latency."""
    return max(4, 2 * int(n_gpus))


# ---------------------------------------------------------------- KernelBlaster

class KBRunner:
    """KernelBlaster's ``scripts/run_multislot.py`` as a subprocess."""

    def __init__(self, cfg: Config, knowledge: Path):
        self.cfg, self.knowledge = cfg, knowledge
        self.calls = 0
        self.telemetry: list = []                     # each run's run-level telemetry (kb_telemetry)
        self.gpus: list = []                          # the GPU each problem's run recorded (kb_eval.recorded_gpus)
        self.commands: dict = {}                      # out dir -> its command line
        self.halfway = None                           # kb_halfway.Halfway (--knowledge-global-halfway on) or None
        self._kw: dict = {}                           # out dir -> the start() arguments (a paused run restarts alike)

    def command(self, problems: list[Path], out: Path, *, steps: int | None = None,
                winners: Path | None = None, parallel_problems: int | None = None,
                stop_patience: int | None = None) -> list[str]:
        c = self.cfg
        cmd = [c.python, str(Path(c.kb) / "scripts" / "run_multislot.py"), *map(str, problems), "--out", str(out),
               "--knowledge-dir", str(self.knowledge), "--model", c.model, "--arch", c.arch,
               "--rollouts", str(c.rollouts), "--steps", str(steps if steps is not None else c.steps),
               "--max-fix", str(c.max_fix), "--seed", str(c.seed)]
        if kb_resume.kb_supports_resume(c.kb):         # a killed run's out dir continues (its journals)
            cmd += ["--resume"]
        if c.reflect_model:
            cmd += ["--reflect-model", c.reflect_model]
        cmd += ["--knowledge", c.knowledge, "--ncu", c.ncu]
        if c.knowledge == "v2":
            cmd += ["--graph-hints", str(self.knowledge.parent / "graph_hints.json")]
        for flag, v in (("--reward", c.reward), ("--model-simple", c.model_simple), ("--model-hard", c.model_hard),
                        ("--hetero-policy", c.hetero_policy), ("--models-config", c.models_config),
                        ("--upkeep-model", c.upkeep_model)):
            if v:
                cmd += [flag, str(v)]
        if winners is not None:
            cmd += ["--export-winners", str(winners)]
        if parallel_problems:
            cmd += ["--parallel-problems", str(parallel_problems)]
        if c.ssh_hosts:
            cmd += ["--evaluator", "ssh"]
            for h in c.ssh_hosts:
                cmd += ["--ssh-host", h]
            for v in c.cuda_visible_devices:
                cmd += ["--cuda-visible-devices", v]
            if c.ssh_key:
                cmd += ["--ssh-key", c.ssh_key]
            if c.remote_dir:
                cmd += ["--remote-dir", c.remote_dir]
        elif c.gpus and c.gpus != "auto":
            cmd += ["--gpus", c.gpus]
        if c.runtime and c.runtime != "auto":
            cmd += ["--runtime", c.runtime]
        if c.eval_timeout_s != 900:
            cmd += ["--eval-timeout-s", str(c.eval_timeout_s)]
        if c.eval_timeout_mode != "fixed":
            cmd += ["--eval-timeout-mode", c.eval_timeout_mode]
        if c.gpu_model:
            cmd += ["--gpu-model", c.gpu_model]
        for p in c.plugins:
            cmd += ["--plugin", str(p)]
        if kb_trace.kb_supports_trace(c.kb):
            cmd += ["--trace", kb_trace._level(c.trace)]
        if kb_efficiency.kb_supports_patience(c.kb):   # always explicit: KernelBlaster's own default is on
            sp = c.stop_patience if stop_patience is None else stop_patience
            cmd += ["--stop-patience", str(int(sp or 0))]
        if self.halfway is not None:                  # halfway sync: the plan run's first half carries --pause-spec
            cmd += self.halfway.flags(out)
        cmd += list(c.kb_args)
        self.commands[str(out)] = cmd                 # the trace's mb.kb_run record
        return cmd

    def start(self, problems: list[Path], out: Path, _append: bool = False, **kw) -> subprocess.Popen:
        out.mkdir(parents=True, exist_ok=True)
        self._kw[str(out)] = dict(kw)
        if self.halfway is not None and self.halfway.mode(out) == "paused":   # a crash during the pause: no run to start
            from .kb_halfway import _PausedProc
            return _PausedProc()
        log = open(out.with_suffix(".log"), "a" if _append else "w")
        try:
            # own session: a terminal's Ctrl-C reaches it once, forwarded by kb_resume.STOP
            p = subprocess.Popen(self.command(problems, out, **kw), stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(self.cfg.kb), start_new_session=True)
        finally:
            log.close()                               # the child holds its own descriptor
        kb_resume.STOP.register(p)
        return p

    def collect(self, problems: list[Path], out: Path, proc: subprocess.Popen) -> list[dict]:
        rc = proc.wait()
        kb_resume.STOP.unregister(proc)
        if rc == kb_halfway.EXIT_PAUSED and self.halfway is not None and self.halfway.active(out) \
                and kb_resume.STOP.requested is None:
            # halfway sync: the plan's problems are parked at their pause step (journals flushed, no process has the
            # knowledge db open): wait for the controller's package, import it, then the same command without the pause
            self.halfway.sync(out)
            proc = self.start(problems, out, _append=True, **self._kw.get(str(out), {}))
            rc = proc.wait()
            kb_resume.STOP.unregister(proc)
        kb_resume.kb_exit(rc, out.with_suffix(".log"))  # 75: stopped on a signal -> Interrupted
        if rc != 0:
            raise RuntimeError(f"KernelBlaster failed ({rc}); see {out.with_suffix('.log')}")
        res = []
        for d in problems:
            r = read_result(out, Path(d)) or {"problem": problem_name(Path(d)), "final_ok": False}
            r["_dir"] = str(out / problem_name(Path(d)))
            self.calls += r.get("llm_calls", 0) or 0
            if r.get("final_ok"):
                self.gpus.append({"run": r["_dir"], "gpu": r.get("gpu"), "arch": r.get("arch")})
            res.append(r)
        self.telemetry.append(kbt.kb_run_summary(out))
        # the run as a whole; its own trace store has the steps inside it
        kb_trace.record("kb_run", run_dir=str(out), rc=rc, cmd=self.commands.get(str(out)),
                        kb_trace=str(out / "trace") if (out / "trace" / "index.jsonl").exists() else None,
                        files=kb_trace.files({"log": out.with_suffix(".log"), "summary.json": out / "summary.json"}),
                        problems=[{k: r.get(k) for k in ("problem", "run_id", "final_ok", "baseline_e2e_us",
                                                          "final_e2e_us", "e2e_speedup", "llm_calls", "_dir")}
                                  for r in res])
        return res

    def run(self, problems: list[Path], out: Path, *, steps: int | None = None,
            winners: Path | None = None, parallel_problems: int | None = None) -> list[dict]:
        p = self.start(problems, out, steps=steps, winners=winners, parallel_problems=parallel_problems)
        return self.collect(problems, out, p)


def problem_name(d: Path) -> str:
    return json.loads((Path(d) / "manifest.json").read_text())["name"]


def read_result(out: Path, problem: Path) -> dict | None:
    """A problem's result.json in a (possibly still running) KernelBlaster
    run; None until it is complete. KernelBlaster writes the best slot
    sources before result.json, so once it parses the bests are final."""
    f = out / problem_name(problem) / "result.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def _usage(results: list[dict]) -> dict:
    tot = {"llm_calls": 0, "cost_usd": 0.0}
    for r in results:
        tot["llm_calls"] += r.get("llm_calls", 0) or 0
        tot["cost_usd"] += (r.get("llm_usage") or {}).get("cost_usd", 0.0) or 0.0
    return tot


def _est_gain(h: dict) -> float:
    return max((s.get("est_gain_pct") or 0.0) for s in h.get("source", [{}]) or [{}])


def _hint_ir(h: dict, profiles: list[dict]) -> dict | None:
    irs = [s.get("ir") for s in h.get("source", []) if s.get("ir")]
    return next((p for ir in irs for p in profiles if p["ir"] == ir), None)


def _scope_ops(h: dict, graph: dict, specs: dict) -> list[str]:
    """Current op names a hint touches in ``graph``."""
    if h["kind"] == "fuse":
        chains = h.get("chains") or find_chains(graph, h["fused_kind"], specs)
        names = []
        for ch in chains:
            try:
                names += gh._locate(graph, ch)
            except ValueError:
                continue
        return list(dict.fromkeys(names))
    if h["kind"] == "split_slot":
        return [o["name"] for o in graph["ops"] if o["op"] in specs and base_slot_id(o["op"], specs) == h["slot"]]
    from . import cuda_learnings as cl
    if cl.ext_kind(h) or h["kind"] in ("fold_layout", "fuse_region", "fuse_view"):   # ext fusion / fold / A2 kinds
        from .fusion_ext import scope_ops
        return scope_ops(h, graph, specs)
    return []


# ---------------------------------------------------------------- conflicts

def _entry(kind: str, specs: dict) -> str:
    """The C entry a kind's kernels are cached under (kinds can share one)."""
    try:
        spec = specs.get(kind)
        if spec is None and kind.startswith("fused__"):
            from .kb_proposals import spec_from_kind
            spec = spec_from_kind(kind, specs)
        return parse_signature(spec.signature)[0] if spec is not None else kind
    except Exception:  # noqa: BLE001
        return kind


def footprint(h: dict, ir: str, graph: dict, specs: dict) -> dict:
    """What deciding ``h`` on ``graph`` (its IR's current state) reads and
    writes of the shared state.

    - ``ops``: (IR, original op name) of every op the hint's problems hold;
    - ``reads``: cache entries its problems are seeded from -- the entries of
      those ops and of their stages (fused ops), plus the variant's own new
      slot (a cached fused kernel of that kind, a split variant's base slot);
    - ``writes``: cache entries an acceptance ingests -- the fused kind
      (``fuse``), or the base slot's entry, guarded on its shapes (``split``).

    Entries, not (entry, shape): a slot's cache lookup and coverage split look
    at every instance of the slot in the IR, not only the touched ones."""
    specs = with_fused_specs(graph, specs)
    names = _scope_ops(h, graph, specs)
    by_name = {o["name"]: o for o in graph["ops"]}
    ops, reads = set(), set()
    for n in names:
        ops.update((ir, a) for a in gh.atoms(n))
        o = by_name.get(n)
        if o is None:
            continue
        reads.add(_entry(o["op"], specs))
        reads.update(_entry(s["op"], specs) for s in o.get("sub_ops", []))
    if h["kind"] == "fuse":
        writes = {_entry(h["fused_kind"], specs)}
    elif h["kind"] == "split_slot":
        writes = {_entry(by_name[names[0]]["op"], specs)} if names else {h["slot"]}
    elif h["kind"] in ("fuse_siblings", "fuse_attention", "fold_layout"):   # extended kinds: the new ops' entries
        from .fusion_ext import new_kinds
        writes = {_entry(k, specs) for k in new_kinds(h, graph, specs)}
    else:
        writes = set()
    return {"ops": ops, "reads": reads | writes, "writes": writes}


def conflicts(a: dict, b: dict) -> bool:
    """Two hints conflict when deciding one can change the other's inputs or
    outcome: they share an op (one's rewrite changes the other's graph;
    a chain and its prefix fusion), or one's acceptance writes a cache entry
    the other's problems are seeded from."""
    return bool(a["ops"] & b["ops"] or a["writes"] & b["reads"] or b["writes"] & a["reads"])


def schedule(fps: list[dict], width: int) -> list[list[int]]:
    """Waves of hint indices (``fps`` in rank order): hint i goes to the
    first wave after every earlier hint it conflicts with, and that has
    room (``width``). Every conflicting pair is thus decided in rank order,
    so the result equals deciding the hints one by one in rank order;
    independent hints share a wave. Deterministic: depends only on the
    order and the footprints."""
    width = max(1, int(width))
    wave_of: list[int] = []
    waves: list[list[int]] = []
    for i, f in enumerate(fps):
        w = 1 + max((wave_of[j] for j in range(i) if conflicts(fps[j], f)), default=-1)
        while w < len(waves) and len(waves[w]) >= width:
            w += 1
        while len(waves) <= w:
            waves.append([])
        waves[w].append(i)
        wave_of.append(w)
    return waves


# ---------------------------------------------------------------- decisions

def _ingest_slots(cache: CudaCache, problem: Path, result: dict, slots: list[str], specs: dict, tag: str) -> list:
    """Put a run's best sources for ``slots`` into the cache (guarded on the
    problem's instance shapes), whatever their speedup over their seed: the
    hint's acceptance already measured them against the unchanged graph."""
    man = json.loads((Path(problem) / "manifest.json").read_text())
    best = best_sources(Path(result["_dir"]))
    out = []
    for s in man["slots"]:
        if s["id"] not in slots or s["id"] not in best:
            continue
        spec = specs.get(s["op"])
        if spec is None:
            from .kb_proposals import spec_from_kind
            spec = spec_from_kind(s["op"], specs)
        entry, _ = parse_signature(spec.signature)
        sr = result.get("slots", {}).get(s["id"], {})
        e, st = cache.add(op=s["op"], entry=entry, signature=spec.signature, arch=f"sm_{man_arch(result)}",
                          shapes=[i["shape"] for i in s["instances"]], code=best[s["id"]],
                          slot_us=sr.get("best_us") or sr.get("final_us") or 0.0,
                          speedup=(sr["baseline_us"] / sr["best_us"]) if sr.get("baseline_us") and sr.get("best_us") else None,
                          impl_id=f"{result.get('run_id', 'run')}:{s['id']}", source={"hint": tag})
        out.append({"slot": s["id"], "status": st})
    cache.save()
    return out


def man_arch(result: dict) -> str:
    return str(result.get("arch", "sm_75")).removeprefix("sm_")


def prepare_hint(h: dict, prof: dict, out: Path, *, cache: CudaCache, cfg: Config, specs: dict) -> dict:
    """Export a hint's original and variant problems from the current state
    (reads the cache, writes only under ``out``). -> a job; ``job["early"]``
    is set when there is nothing to run (the hint does not occur / apply)."""
    base_graph = json.loads((Path(prof["ir_dir"]) / "graph.json").read_text())
    specs = with_fused_specs(base_graph, specs)
    base = cache_plan(base_graph, cache, f"sm_{cfg.arch}", specs)
    g0 = base["graph"]
    names = _scope_ops(h, g0, specs)
    job = {"hint": h, "specs": specs, "names": names,
           "ev": {"problem": None, "ir": prof["ir"], "ts": gh._now()}}
    if not names:
        job["early"] = "hint does not occur in this IR"
        return job
    hh = dict(h)
    if h["kind"] == "fuse":                             # scope the variant to this IR's instances
        hh["chains"] = [c for c in (h.get("chains") or [])
                        if all(a in {x for n in names for x in gh.atoms(n)} for a in c)] or None
    vg, _, prov = gh.apply_hints(g0, [hh], specs, statuses=None)
    if not prov["applied"]:
        job["early"] = "; ".join(s["why"] for s in prov["skipped"])[:500]
        return job
    from . import cuda_learnings as cl
    # --fusion-extended: a fused slot is split by its stages' per-shape winners (each split slot seeded with them)
    vplan = cache_plan(vg, cache, f"sm_{cfg.arch}", specs, split=cl.variant_split())
    out.mkdir(parents=True, exist_ok=True)
    src_ir = Path(prof["source_ir"])
    oir = write_variant_ir(src_ir, g0, out / "orig_ir")
    vir = write_variant_ir(src_ir, vplan["graph"], out / "variant_ir")
    stem = _safe(f"{prof['ir']}__{h['id']}")
    oidx = [i for i, o in enumerate(g0["ops"]) if o["name"] in set(names)]
    vnames = {prov["renamed"].get(n, n) for n in names}
    vidx = [i for i, o in enumerate(vplan["graph"]["ops"]) if o["name"] in vnames]
    seeds = {**base["sources"], **vplan["sources"]}
    tags = {**base["tags"], **vplan["tags"]}
    from . import graph_search as gsm                   # a promising hint's next rung continues from its best
    gsm.variant_seed_override(h, vplan, seeds, tags, specs)
    if h["kind"] == "fold_layout":                      # --graph-layout-fold: composite seed (parity start)
        specs = with_fused_specs(vplan["graph"], specs)
        job["specs"] = specs
        job["composite"] = gsm.fold_seeds(h, g0, vplan, base, seeds, tags, specs)
    if h["kind"] == "fuse_region":                      # region slots: specs of the region kinds (parity seed in export)
        specs = with_fused_specs(vplan["graph"], specs)
        job["specs"] = specs
    if cl.ext_kind(h):                                  # composite seeds for structural kinds (attention)
        from .fusion_ext import variant_seeds
        specs = with_fused_specs(vplan["graph"], specs)
        job["specs"] = specs
        job["composite"] = variant_seeds(h, g0, vplan, base, seeds, tags, specs)
    job["orig"] = export_problem(oir, out / "orig", op_indices=oidx, specs=specs, name=stem + "__orig",
                                 strategy="hint-original", slot_sources=base["sources"],
                                 slot_source_tags=base["tags"], sub_sources=base["sub_sources"])
    job["variant"] = export_problem(vir, out / "variant", op_indices=vidx, specs=specs, name=stem + "__variant",
                                    strategy="hint-variant", slot_sources=seeds, slot_source_tags=tags,
                                    sub_sources=vplan["sub_sources"])
    if job.get("composite"):
        from .fusion_ext import note_composite
        note_composite(job["variant"], job["composite"])
    job["stem"], job["prov"] = stem, prov
    return job


def kb_eval_for(cfg: Config):
    """KernelBlaster's evaluator for ModelBlaster's own timings (the model run; re-measurement
    over ssh) on the KernelBlaster runs' pool: ssh with ``ssh_hosts``, else local ``gpus``.
    None when the checkout has no ``scripts/time_problems.py`` (an older pin)."""
    from .kb_eval import KBEval

    ev = KBEval(evaluator="ssh" if cfg.ssh_hosts else "local", kb=Path(cfg.kb), python=cfg.python, gpus=cfg.gpus,
                ssh_hosts=list(cfg.ssh_hosts), cuda_visible_devices=list(cfg.cuda_visible_devices),
                ssh_key=cfg.ssh_key, remote_dir=cfg.remote_dir, runtime=cfg.runtime)
    return ev if ev.available() else None


def use_kb_eval(cfg: Config) -> bool:
    """Do ModelBlaster's own GPU steps (profiles, verification, re-measurement) go through
    KernelBlaster's evaluator (``cfg.mb_evaluator``)?"""
    if cfg.mb_evaluator not in ("auto", "kb", "builtin"):
        raise ValueError(f"mb_evaluator must be auto|kb|builtin, not {cfg.mb_evaluator!r}")
    return cfg.mb_evaluator == "kb" or (cfg.mb_evaluator == "auto" and bool(cfg.ssh_hosts))


def mb_kb_eval(cfg: Config):
    """KernelBlaster's evaluator for the profiles and the verification (one GPU of the pool),
    or None for ModelBlaster's builtin runs here. Asked for (``kb`` / ``--ssh-host``) but the
    checkout has no ``scripts/time_problems.py``: an error, never a silent local run."""
    if not use_kb_eval(cfg):
        return None
    ev = kb_eval_for(cfg)
    if ev is None:
        raise RuntimeError(f"{cfg.kb} has no scripts/time_problems.py: the profiles and the verification cannot "
                           "run on KernelBlaster's pool (--ssh-host / --mb-evaluator kb); use a newer checkout "
                           "or --mb-evaluator builtin")
    return ev


def _device_record(dev: dict | None, arch: str) -> dict | None:
    """A leased local device (``detect_gpus``) as a result's ``gpu`` record; the
    environment's default GPU (nvidia-smi) when the lease is unpinned."""
    from .kb_eval import builtin_device

    if not dev or dev.get("ident") is None:
        return builtin_device(arch)
    ident = str(dev["ident"])
    return {"host": dev.get("host"), "label": str(dev.get("index", ident)), "index": dev.get("index"),
            "uuid": ident if ident.startswith("GPU-") else None, "name": dev.get("name"), "arch": f"sm_{arch}"}


@contextlib.contextmanager
def _recorded_lease(leases: GpuLeases | None, info: dict, arch: str):
    """``GpuLeases.lease`` that also records the leased device in ``info["device"]``."""
    if leases is None:
        info["device"] = _device_record(None, arch)
        yield {}
        return
    with leases.lease_device() as dev:
        info["device"] = _device_record(dev, arch)
        ident = dev.get("ident")
        yield {"CUDA_VISIBLE_DEVICES": str(ident)} if ident is not None else {}


def measure_hint(job: dict, ro: dict, rv: dict, cfg: Config, leases: GpuLeases | None,
                 attempts: int | None = None) -> list | None:
    """Re-time both composite bests interleaved on one leased GPU ->
    [original times, variant times], or None (a side failed / no
    re-measurement). Thread-safe: writes only under the job's problem dirs.

    With ``ssh_hosts`` the leased GPU is a remote one (the KernelBlaster
    runs' pool) and the runs go through KernelBlaster's ssh evaluator pinned
    to it, on the GPU model the two runs recorded; locally, ModelBlaster's
    own runs on the leased local GPU (its lock held).

    An infrastructure failure is retried on a fresh lease (``attempts``, default
    ``cfg.remeasure_attempts``); the failures' texts are kept in ``job``
    (``remeasure_failures``, ``remeasure_attempts``) for the hint's evidence, and
    ``job["remeasure_infra"]`` tells :func:`finish_hint` to leave the hint undecided."""
    if not (ro.get("final_ok") and rv.get("final_ok")) or not cfg.remeasure:
        return None
    with kb_trace.get().scope(mb_stage="remeasure", hint=(job.get("hint") or {}).get("id")):   # its runs' driver output
        return _measure_hint(job, ro, rv, cfg, leases, attempts=attempts)


def _remeasure_once(pair, cfg: Config, leases: GpuLeases | None, ev, hw: dict | None, info: dict, work: Path):
    """One re-measurement on a fresh lease (``info``: device, failures)."""
    if ev is None:
        return remeasure(pair, arch=cfg.arch, reps=cfg.remeasure, lease=_recorded_lease(leases, info, cfg.arch),
                         info=info)
    # a local device's lock is taken per run by the KernelBlaster evaluator pinned to it
    with (leases.lease_device(lock=False) if leases is not None else contextlib.nullcontext(None)) as dev:
        return remeasure(pair, arch=cfg.arch, reps=cfg.remeasure, kbeval=ev.for_device(dev),
                         gpu_model=hw["gpu_model"], info=info, work=work)


def _failure_text(fails: list) -> str:
    return "; ".join(f"{f.get('side') or 'run'}: " + (f"[{f['stage']}] " if f.get("stage") else "") +
                     str(f.get("error") or "?")[:600] for f in fails)


def _measure_hint(job: dict, ro: dict, rv: dict, cfg: Config, leases: GpuLeases | None,
                  attempts: int | None = None) -> list | None:
    """The attempts: an infrastructure failure (every failing side's error is one) is
    retried on a fresh lease; a candidate's own failure is not. Records them in ``job``:
    ``remeasure_attempts`` (each one's device, failures and work dir),
    ``remeasure_failures`` (the last one's, with the error texts) and ``remeasure_infra``."""
    from .kb_eval import HardwareMismatch, recorded_gpus, required_hardware
    from .kb_proposals import is_infra_text

    pair = [(job["orig"], best_sources(Path(ro["_dir"]))), (job["variant"], best_sources(Path(rv["_dir"])))]
    sides = {str(job["orig"]): "original", str(job["variant"]): "variant"}
    ev = kb_eval_for(cfg) if use_kb_eval(cfg) else None
    hw = None if ev is None else required_hardware(
        recorded_gpus([Path(ro["_dir"]), Path(rv["_dir"])]), gpu_model=cfg.gpu_model, arch=cfg.arch,
        allow_mismatch=cfg.allow_gpu_mismatch)
    log = job.setdefault("remeasure_attempts", [])
    hid = (job.get("hint") or {}).get("id")
    T = kbt.get()
    times, fails, infra = None, [], False
    for a in range(max(1, int(attempts or cfg.remeasure_attempts))):
        if a:
            time.sleep(cfg.remeasure_retry_s * a)
        k = len(log) + 1
        work = Path(job["variant"]).parent / ("remeasure" if k == 1 else f"remeasure_a{k}")
        info: dict = {}
        with T.span("mb.remeasure.attempt", attempt=k, hint=hid) as sp:
            try:
                times = _remeasure_once(pair, cfg, leases, ev, hw, info, work)
            except HardwareMismatch:
                raise
            except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as e:
                # time_problems did not run / write its result (it says why), a timeout, a lost link
                msg = " ".join(f"{type(e).__name__}: {e}".split())[:1500]
                times, info["failures"] = None, [{"problem": None, "stage": "time", "error": msg,
                                                  "infra": is_infra_text(msg) or is_infra_result({"final_error": msg}) or "time_problems failed" in msg or
                                                  isinstance(e, (OSError, subprocess.SubprocessError))}]
            ok = bool(times and len(times) == 2 and times[0] and times[1])
            fails = [] if ok else [dict(f, side=sides.get(str(f.get("problem")))) for f in info.get("failures") or []]
            if not ok and not fails:
                fails = [{"problem": None, "side": None, "stage": "run", "error": "no passing timed run "
                          f"({[len(x) for x in times or []]} of {cfg.remeasure}; the evaluator reported no error)",
                          "infra": False}]
            infra = not ok and all(f.get("infra") for f in fails)
            dev = info.get("device")
            log.append({"attempt": k, "ok": ok, "infra": infra, "work": str(work), "failures": fails,
                        "device": dev.get("label") if isinstance(dev, dict) else dev})
            if sp is not None and hasattr(sp, "set"):
                sp.set(ok=ok, infra=infra, error=_failure_text(fails)[:800] if fails else None,
                       device=log[-1]["device"])
            kb_trace.record("remeasure", rid_parts=(hid or "?", k), hint=hid, attempt=k, ok=ok, infra=infra,
                            failures=fails, times=times, device=dev, work=str(work))
        if dev:
            job["remeasure_device"] = dev
        else:
            job.setdefault("remeasure_device", None)
        if ok or not infra:
            break
        print(f"[kb_graph_opt] {hid}: re-measurement attempt {k}: infrastructure failure: "
              f"{_failure_text(fails)[:400]}", file=sys.stderr, flush=True)
    job["remeasure_failures"], job["remeasure_infra"] = fails, infra
    return times


def finish_hint(job: dict, ro: dict | None, rv: dict | None, times: list | None, *, cache: CudaCache, doc: dict,
                cfg: Config) -> dict:
    """Decide from the measurements, record the evidence, ingest an accepted
    variant's new slots. Writes the hints doc and the cache: call from one
    thread, in rank order."""
    h, ev, names = job["hint"], job["ev"], job["names"]
    if "early" in job:
        ev.update(decision="rejected", why=job["early"])
        gh.record_decision(doc, h, ev)
        kb_trace.record("hint", rid_parts=(h["id"],), hint=h["id"], hint_kind=h.get("kind"), mb_stage="decide",
                        decision="rejected", why=ev["why"], evidence=ev)
        return {"hint": h["id"], "status": "rejected", "why": ev["why"]}
    ev.update(problem=job["stem"], original_us=ro.get("final_partition_us"), variant_us=rv.get("final_partition_us"),
              kb_original_us=ro.get("final_partition_us"), kb_variant_us=rv.get("final_partition_us"),
              llm_calls=(ro.get("llm_calls") or 0) + (rv.get("llm_calls") or 0))
    ok = bool(ro.get("final_ok") and rv.get("final_ok"))
    fails = []
    bad = [(side, r) for side, r in (("original", ro), ("variant", rv)) if not r.get("final_ok")]
    if bad and all(is_infra_result(r) for _, r in bad):
        # KernelBlaster's final run lost to the GPU / link / staging, not to the composite (the result's own tag, or an
        # infra signature in its error when the tag is missing: scp, NoDevice, relay 503 / queue_full)
        ev["kb_errors"] = {side: r.get("final_error") for side, r in bad}
        return _undecided(job, ro, rv, times, ev, doc, "KernelBlaster: infrastructure failure of the final run, "
                          "left undecided: " + "; ".join(f"{side}: {r.get('final_error') or '?'}" for side, r in bad))
    if ok and cfg.remeasure:
        ok = bool(times and times[0] and times[1])
        if job.get("remeasure_attempts"):
            ev["remeasure_attempts"] = len(job["remeasure_attempts"])
        if ok:
            ev.update(original_us=_median(times[0]), variant_us=_median(times[1]),
                      remeasured={"original_us": times[0], "variant_us": times[1],
                                  **({"device": job["remeasure_device"]} if job.get("remeasure_device") else {})})
        else:
            fails = job.get("remeasure_failures") or []
            ev["remeasure_errors"] = fails
            if job.get("remeasure_infra"):
                return _undecided(job, ro, rv, times, ev, doc,
                                  f"re-measurement: infrastructure failure after "
                                  f"{len(job.get('remeasure_attempts') or [])} attempt(s), left undecided: "
                                  f"{_failure_text(fails)}")
    if not ok:
        ev["why"] = ("KernelBlaster: " + ", ".join(
                         f"{side} not final_ok ({r.get('error') or r.get('final_error') or 'see result.json'})"
                         for side, r in bad)
                     if not (ro.get("final_ok") and rv.get("final_ok")) else
                     f"re-measurement: no passing run on a side ({[len(x) for x in times or []]} of {cfg.remeasure})"
                     + (f": {_failure_text(fails)}" if fails else ""))
    gain = (ev["original_us"] - ev["variant_us"]) / ev["original_us"] if ok else None
    status = "accepted" if ok and gain >= cfg.accept_gain else "rejected"
    ev.update(gain=gain, decision=status, applied=job["prov"]["applied"])
    rec = gh.record_decision(doc, h, ev)
    res = {"hint": h["id"], "status": status, "hint_status": rec["status"], "gain": gain,
           "original_us": ev["original_us"], "variant_us": ev["variant_us"], "ops": names}
    if job.get("remeasure_device"):
        res["remeasure_device"] = job["remeasure_device"]
    if rec["status"] == "accepted":
        man = json.loads((Path(job["variant"]) / "manifest.json").read_text())
        changed = [s["id"] for s in man["slots"]
                   if s["op"].startswith("fused__") or s.get("variant_of")
                   or h["kind"] in ("fuse_siblings", "fuse_attention", "fold_layout")   # only new ops there
                   or s["op"].startswith("region__")]                                   # region slots (A2)
        res["cached"] = _ingest_slots(cache, job["variant"], rv, changed, job["specs"], h["id"])
    _trace_hint(job, ro, rv, times, ev, res)
    return res


INFRA_UNRESOLVED = "infra_unresolved"     # hint status: lost to infrastructure after its re-evaluations; not accepted,
#                                           not rejected, never a candidate again (the candidate filters want "proposed")


def infra_retry_budget() -> int:
    """Automatic re-evaluations of a hint whose decision was lost to infrastructure (``MB_INFRA_RETRY``, default 1: the
    hint stays ``proposed`` and is decided again in the next round; a second loss makes it ``infra_unresolved``).
    0 = the old behaviour (undecided every time, no bound)."""
    try:
        return max(0, int(os.environ.get("MB_INFRA_RETRY", "1")))
    except ValueError:
        return 1


def _infra_failures(rec: dict) -> int:
    return sum(1 for x in rec.get("evidence") or [] if x.get("decision") == "undecided" and x.get("infra"))


def _undecided(job: dict, ro: dict, rv: dict, times, ev: dict, doc: dict, why: str) -> dict:
    """Record an infrastructure failure's evidence without a decision on the hint's merit:
    it does not vote, and the hint stays a candidate for a later round."""
    h = job["hint"]
    ev.update(decision="undecided", infra=True, gain=None, applied=job["prov"]["applied"], why=why)
    rec = gh.record_decision(doc, h, ev)
    status = "undecided"
    n, budget = _infra_failures(rec), infra_retry_budget()
    if budget > 0 and n > budget:                  # re-evaluated ``budget`` time(s) and lost again: stop, no verdict
        rec["status"] = status = INFRA_UNRESOLVED
        why += f" [infra_unresolved: {n} infrastructure failures, {budget} re-evaluation(s) allowed]"
        ev["why"] = why
    res = {"hint": h["id"], "status": status, "hint_status": rec["status"], "infra": True, "why": why,
           "gain": None, "original_us": ev.get("original_us"), "variant_us": ev.get("variant_us"),
           "ops": job["names"], "infra_failures": n}
    _trace_hint(job, ro, rv, times, ev, res)
    return res


def _trace_hint(job: dict, ro: dict | None, rv: dict | None, times, ev: dict, res: dict) -> None:
    """The hint decision's record: both problems (IR graphs, manifests), KernelBlaster's two
    results and run dirs (their traces), the re-measured times, the decision."""
    if not kb_trace.get().enabled():
        return
    o, v = Path(job["orig"]), Path(job["variant"])
    kb_trace.record("hint", rid_parts=(job["hint"]["id"],), hint=job["hint"]["id"], hint_kind=job["hint"].get("kind"),
                    mb_stage="decide", decision=res["status"], why=ev.get("why"), gain=res.get("gain"),
                    evidence=ev, remeasured=times, provenance=job.get("prov"),
                    kb_dirs={"original": (ro or {}).get("_dir"), "variant": (rv or {}).get("_dir")},
                    files=kb_trace.files({"orig_ir/graph.json": o.parent / "orig_ir" / "graph.json",
                                          "variant_ir/graph.json": v.parent / "variant_ir" / "graph.json",
                                          "orig/manifest.json": o / "manifest.json",
                                          "variant/manifest.json": v / "manifest.json"}))


def evaluate_hints(items: list[tuple[dict, dict, Path]], run_dir: Path, *, kb: KBRunner, cache: CudaCache,
                   doc: dict, cfg: Config, specs: dict, leases: GpuLeases | None = None,
                   poll_s: float = 1.0) -> list[dict]:
    """Decide a wave of independent hints ((hint, profile, out dir) each).

    Every hint's original and variant problems go to one KernelBlaster run
    (``--parallel-problems`` = all of them: KernelBlaster's GPU pool gives each
    timed run one GPU, its build limiter bounds the compiles, and its store /
    outcome log / claims stay single-writer in that one process); a hint is
    re-timed as soon as both its results are written, on one leased GPU, while
    KernelBlaster works on the others. Decisions and cache ingests are then
    made in the given order, in this thread."""
    T = kbt.get()
    with T.span("mb.prepare", hints=len(items)):
        jobs = [kb_resume.prepared(prepare_hint, h, prof, out, cache=cache, cfg=cfg, specs=specs)
                for h, prof, out in items]
    live = [k for k, j in enumerate(jobs) if "early" not in j]
    results: dict[int, tuple] = {}
    if live:
        problems = [p for k in live for p in (jobs[k]["orig"], jobs[k]["variant"])]
        parent = T.current()
        wait = kbt.section("mb.kb_wait", kind="hints", problems=len(problems))
        # no patience in the hint decisions: the original and the variant get the same budget (E7 replayed it so)
        proc = kb.start(problems, run_dir, steps=cfg.variant_steps, parallel_problems=len(problems), stop_patience=0)
        futs, sides = {}, {}
        # re-timing workers: two per GPU (one builds while the other times), at most one per hint
        n_dev = len(leases.devices) if leases is not None else 1
        with ThreadPoolExecutor(max_workers=max(1, min(len(live), 2 * n_dev))) as ex:
            try:
                while True:
                    done = proc.poll() is not None
                    for k in live:
                        if k in futs:
                            continue
                        ro, rv = read_result(run_dir, jobs[k]["orig"]), read_result(run_dir, jobs[k]["variant"])
                        if (ro is not None and rv is not None) or done:
                            _lost = {"final_ok": False, "final_infra": True,      # missing result.json at exit: infra/crash,
                                     "final_error": "no result.json when the KernelBlaster process exited"}   # not a verdict
                            ro = dict(ro or _lost, _dir=str(run_dir / problem_name(jobs[k]["orig"])))
                            rv = dict(rv or _lost,
                                      _dir=str(run_dir / problem_name(jobs[k]["variant"])))
                            sides[k] = (ro, rv)
                            futs[k] = ex.submit(kbt.run_within, parent, "mb.remeasure", measure_hint, jobs[k],
                                                ro, rv, cfg, leases)
                    if done:
                        break
                    time.sleep(poll_s)
            except BaseException:
                proc.kill()
                raise
            res = kb.collect(problems, run_dir, proc)          # final result.json (llm calls, proposals)
            wait.end()
            times = {k: f.result() for k, f in futs.items()}
        # lost to an infrastructure failure while the wave's KernelBlaster run still shared the GPUs (and
        # the remote dir): once more, on a fresh lease, now that the run has finished
        for k in sorted(times):
            if jobs[k].get("remeasure_infra"):
                times[k] = kbt.run_within(parent, "mb.remeasure", measure_hint, jobs[k], *sides[k], cfg, leases,
                                          attempts=1)
        for n, k in enumerate(live):
            results[k] = (res[2 * n], res[2 * n + 1], times.get(k))
    with T.span("mb.finish"):
        return [finish_hint(j, *results.get(k, (None, None, None)), cache=cache, doc=doc, cfg=cfg)
                for k, j in enumerate(jobs)]


def evaluate_hint(h: dict, prof: dict, out: Path, *, kb: KBRunner, cache: CudaCache, doc: dict,
                  cfg: Config, specs: dict, leases: GpuLeases | None = None) -> dict:
    """Build original + variant problems for one hint on one IR, optimise
    both with the same budget, re-measure, decide, record."""
    return evaluate_hints([(h, prof, out)], out / "run", kb=kb, cache=cache, doc=doc, cfg=cfg, specs=specs,
                          leases=leases)[0]


def decide_hints(cands: list[tuple[dict, dict]], graphs: dict, out: Path, width: int, *, kb: KBRunner,
                 cache: CudaCache, doc: dict, cfg: Config, specs: dict, leases: GpuLeases | None = None,
                 hints_path: Path | None = None) -> tuple[list[list[str]], list[dict]]:
    """A round's decisions: ``cands`` = (hint, profile) in rank order,
    ``graphs`` = {IR: its current graph}. Schedules them into waves of at
    most ``width`` independent hints, evaluates wave after wave (the hints
    file is saved after each) -> (waves as hint ids, decisions in rank order)."""
    fps = [footprint(h, p["ir"], graphs[p["ir"]], specs) for h, p in cands]
    waves = schedule(fps, width)
    decided: dict[int, dict] = {}
    for k, w in enumerate(waves):
        items = [(cands[i][0], cands[i][1], out / _safe(cands[i][0]["id"])) for i in w]
        run_dir = items[0][2] / "run" if len(w) == 1 else out / f"wave{k:02d}"
        for i, d in zip(w, evaluate_hints(items, run_dir, kb=kb, cache=cache, doc=doc, cfg=cfg, specs=specs,
                                          leases=leases)):
            decided[i] = d
        if hints_path is not None:
            gh.save(doc, hints_path)
    return [[cands[i][0]["id"] for i in w] for w in waves], [decided[i] for i in sorted(decided)]


# ---------------------------------------------------------------- rounds

def run(irs: list[Path], work: Path, cfg: Config) -> dict:
    work = Path(work).resolve()
    kb_resume.STOP.install()                      # SIGTERM / SIGINT / SIGHUP: stop cleanly, resumable
    with work_lock(work):
        tel = kbt.load(cfg.kb)                    # telemetry (kb_telemetry.py); a no-op without it
        tel.configure(work / "telemetry.jsonl")
        kb_trace.configure(cfg.kb, work, cfg.trace)   # trace store (kb_trace.py); a no-op when off
        try:
            with tel.span("mb.run", collect=True), kb_trace.get().scope(mb="graph_opt"):
                return _run(irs, work, cfg)
        finally:
            kb_trace.close()
            tel.close()


class FlowHardware:
    """The one GPU model every GPU step of a run uses, and the device each step ran on.

    The model is ``--gpu-model``, else the model the first profile ran on (kept in
    ``state.json`` across resumed calls). Once it is known it pins the KernelBlaster runs
    (their ``--gpu-model``), the re-measurement and the model's result, and every step's
    recorded device is checked against it (:meth:`check`): another model is a
    ``HardwareMismatch`` unless ``allow_gpu_mismatch`` (then a warning)."""

    STEPS = ("profile", "kb", "remeasure", "verify", "model_run")

    def __init__(self, cfg: Config, state: dict):
        self.explicit = cfg.gpu_model
        self.model = cfg.gpu_model or state.get("gpu_model")
        self.allow = cfg.allow_gpu_mismatch
        self.devices: dict[str, list] = {k: [] for k in self.STEPS}
        self.warnings: list[str] = []

    def check(self, step: str, device: dict | None, what: str) -> None:
        from .kb_eval import check_device, gpu_model_key

        d = dict(device or {})
        self.devices[step].append({"what": what, **{k: d.get(k) for k in ("host", "label", "uuid", "name", "arch")}})
        if self.model is None and gpu_model_key(d.get("name")):
            self.model = d["name"]                      # the first step that ran fixes the model
            return
        check_device(d, self.model, what=f"{step} {what}", allow_mismatch=self.allow, warnings=self.warnings)

    def pin(self, cfg: Config) -> Config:
        """``cfg`` with the flow's model as the required one (KernelBlaster runs, re-measurement,
        model run); with ``allow_gpu_mismatch`` only an explicit ``--gpu-model`` pins."""
        from dataclasses import replace

        if self.model and not (self.allow and not self.explicit) and cfg.gpu_model != self.model:
            return replace(cfg, gpu_model=self.model)
        return cfg

    def summary(self) -> dict:
        from .kb_eval import gpu_model_key

        names = sorted({d["name"] for v in self.devices.values() for d in v if gpu_model_key(d.get("name"))})
        hosts = sorted({d["host"] or "local" for v in self.devices.values() for d in v if d.get("name")})
        return {"gpu_model": self.model, "explicit": bool(self.explicit), "models": names,
                "same_model": len({gpu_model_key(n) for n in names + ([self.model] if self.model else [])}) <= 1,
                "hosts": hosts,
                "local_gpu_used": any(d.get("name") and not d.get("host") for v in self.devices.values() for d in v),
                "steps": {k: v for k, v in self.devices.items() if v}, "warnings": self.warnings}


def _restrict_leases(leases: GpuLeases, model: str | None, allow: bool) -> None:
    """Re-timing leases only on GPUs of the flow's model (when the pool knows its models)."""
    from .kb_eval import HardwareMismatch, gpu_model_key, same_gpu_model

    if not model or not any(gpu_model_key(d.get("name")) for d in leases.devices):
        return
    keep = [d for d in leases.devices if same_gpu_model(d.get("name"), model)]
    if keep:
        leases.devices = keep
    elif not allow:
        raise HardwareMismatch(f"no GPU of the flow's model {model} in the re-timing pool "
                               f"{[d.get('name') for d in leases.devices]} (--allow-gpu-mismatch to go on)")


def _run(irs: list[Path], work: Path, cfg: Config) -> dict:
    hints_path, state_path = work / "graph_hints.json", work / "state.json"
    doc, cache = gh.load(hints_path), CudaCache(work / "cache")
    state = json.loads(state_path.read_text()) if state_path.exists() else {"sent": [], "rounds": []}
    # E7 efficiency rules (kb_efficiency.py): resolved against the work dir before the round key is made, so a
    # resumed work dir keeps its rules (a pre-X-129 one: off, its round key unchanged)
    eff = kb_efficiency.resolve(cfg, work, state)
    cfg = kb_efficiency.apply(cfg, eff)
    rec_eff = {k: eff[k] for k in ("rules", "stop_patience", "plan_min_share_late", "source")}
    if {k: v for k, v in (state.get("efficiency") or {}).items() if k != "source"} != \
            {k: v for k, v in rec_eff.items() if k != "source"}:
        state["efficiency"] = rec_eff
        kb_resume.atomic_write_json(state_path, state)
    print(json.dumps({"efficiency": rec_eff, "note": eff.get("note")}), flush=True)
    print(f"[kb_graph_opt] {kb_efficiency.describe(eff)}", flush=True)
    if str(cfg.seed_mined).lower() in ("0", "off", "false", "no"):
        os.environ["KERNELBLASTER_SEED_MINED"] = "0"
    # global knowledge (off by default): seed a fresh knowledge/ from a snapshot, once
    gk = kb_global.apply_seed(work, cfg, irs, state)
    if gk is not None:
        kb_resume.atomic_write_json(state_path, state)
    hw = FlowHardware(cfg, state)
    cfg = hw.pin(cfg)
    kb = KBRunner(cfg, work / "knowledge")
    bar = kb_barrier.Barrier(cfg, work, state, state_path, gk, irs)    # barrier scheduling: park after the characterization pass
    if kb_halfway.enabled(cfg, gk):                      # --knowledge-global-halfway on (needs seed + refresh inbox)
        kb.halfway = kb_halfway.Halfway(cfg, work, state, state_path, gk)
    elif str(cfg.knowledge_global_halfway).lower() == "on":
        print(json.dumps({"knowledge_global_halfway": "inactive (needs a seed, --knowledge-global-refresh and a "
                                                      "KernelBlaster with --pause-spec)"}), flush=True)
    specs = dict(_all_specs())
    from . import cuda_learnings as cl
    ext = cl.CURRENT["ext"] = cl.Ext(cfg, work)          # all off by default (no effect)
    ext.apply_env()
    from . import search_fixes as sfx
    sf = sfx.Fixes(cfg, work)                            # fallback repair + final guard on by default (kill switches)
    sf.apply_env()
    from . import capabilities as capm
    caps = capm.Caps(cfg, work)                          # A2 switches (defaults: prune on, the rest off)
    caps.apply_env()
    from . import graph_search as gsm
    gs = gsm.CURRENT["gs"] = gsm.GraphSearch(cfg, work, irs)   # all off by default (no effect)
    if kb_barrier.enabled(cfg) or kb_barrier.char_sync_enabled(cfg, gk):
        gs.on_char_done = bar.after_char                     # --park-after-char on: park once the characterization has run

    def _reprofile(R, sub, hints_doc=None):              # graph search: profiles redone mid-round
        return [profile_ir(Path(d), R / sub / Path(d).name, hints_doc=hints_doc or doc, cache=cache, arch=cfg.arch,
                           specs=specs, runs=cfg.profile_runs, kbeval=mbev, gpu_model=hw.model,
                           allow_gpu_mismatch=cfg.allow_gpu_mismatch, on_fallback=sf.on_fallback(mbev, hw.model))
                for d in irs]
    gpus = detect_gpus(cfg)
    n_gpus = len(gpus["remote"]) if cfg.ssh_hosts else len(gpus["local"])
    width = cfg.parallel_hints or auto_parallel_hints(n_gpus)
    # MB's own timed runs (re-measurement) lease from the KernelBlaster runs' pool: remote with --ssh-host
    leases = GpuLeases(gpus["remote"] if cfg.ssh_hosts else gpus["local"])
    _restrict_leases(leases, hw.model, hw.allow)
    # the profiles and the final verification: KernelBlaster's evaluator with --ssh-host / --mb-evaluator kb
    mbev = mb_kb_eval(cfg)
    state["gpus"] = {"pool": gpus, "parallel_hints": width,
                     "mb_evaluator": mbev.describe() if mbev is not None else {"evaluator": "builtin"}}
    first = None
    n_gpu_rec = len(kb.gpus)
    T = kbt.get()
    run_span = T.current()
    # resume (kb_resume.py): --rounds counts the rounds of this IR set and configuration in the
    # work dir; a round killed midway continues in the phase it was in (rNN/progress.json)
    key = kb_resume.round_key(cfg, irs)
    key.update(kb_global.key_fields(cfg, gk))       # only when enabled: existing work dirs keep their key
    key.update(kb_halfway.key_fields(cfg, gk))      # halfway sync: a constant marker, only when it is on
    key.update(ext.round_key())                     # cuda_learnings: likewise
    key.update(sf.round_key())                      # search fixes: only non-default switches
    key.update(caps.round_key())                    # capabilities: only non-default switches
    if str(cfg.ingest_fused).lower() in ("0", "off", "false", "no"):
        key["ingest_fused"] = "off"                 # kill switch: only when set
    if str(cfg.seed_mined).lower() in ("0", "off", "false", "no"):
        key["seed_mined"] = "off"                   # technique-only seed: only when set
    key.update(gs.round_key())                      # graph search: only when a --graph-* flag is set
    key.update(kb_efficiency.key_fields(cfg))       # efficiency rules: only when on (off = the legacy key)
    kh = kb_resume.key_hash(key)
    n_rounds = (cfg.more_rounds if cfg.more_rounds else
                0 if state.get("fixpoint_key") == kh else
                max(0, cfg.rounds - sum(1 for x in state["rounds"] if x.get("key") == kh)))
    if not n_rounds and state.get("complete") == kh and (work / "summary.json").exists():
        print(json.dumps({"resume": "work dir already complete for these IRs and settings"}), flush=True)
        return json.loads((work / "summary.json").read_text())
    if (work / kb_barrier.PARKED_FILE).exists() and not kb_barrier.enabled(cfg):   # a parked run resumes: import the post-barrier package once
        bar.resume_import()
    for r in range(len(state["rounds"]), len(state["rounds"]) + n_rounds):
        kb_resume.STOP.check()
        state.pop("complete", None)
        R = work / f"r{r:02d}"
        prog = kb_resume.RoundProgress(R, key)
        if prog.resumed:
            print(json.dumps({"round": r, "resume": sorted(prog.data["phases"])}), flush=True)
        elif gk is not None and cfg.knowledge_global_refresh:   # a round boundary: no KernelBlaster run is open
            rf = kb_global.refresh_at_round(work, cfg, state, r)
            if rf is not None:
                kb_resume.atomic_write_json(state_path, state)
                print(json.dumps({"round": r, "knowledge_global_refresh": {k: rf.get(k) for k in
                                                                           ("status", "counts", "reason", "error")}}),
                      flush=True)
        t0 = time.time()
        rec = {"round": r, "key": kh, "kb_runs": [], "decisions": []}
        rsec, n_kb = kbt.section("mb.round", collect=True, laps=True, round=r), len(kb.telemetry)
        T.lap("mb.profile")
        profiles = []
        saved = prog.get("profile")
        for i, d in enumerate(irs):
            p = saved[i] if saved else profile_ir(
                Path(d), R / "profile" / Path(d).name, hints_doc=doc, cache=cache, arch=cfg.arch,
                specs=specs, runs=cfg.profile_runs, kbeval=mbev, gpu_model=hw.model,
                allow_gpu_mismatch=cfg.allow_gpu_mismatch, on_fallback=sf.on_fallback(mbev, hw.model))
            had = hw.model
            hw.check("profile", p.get("device"), f"r{r:02d}/{p['ir']}")
            if hw.model != had:                          # the first profile fixed the flow's model
                state["gpu_model"] = hw.model
                cfg = hw.pin(cfg)
                kb.cfg = cfg
                _restrict_leases(leases, hw.model, hw.allow)
            profiles.append(p)
        rec["profile_devices"] = {p["ir"]: (p.get("device") or {}).get("label") for p in profiles}
        bad = [p["ir"] for p in profiles if not p["passed"]]
        if bad:
            raise RuntimeError(f"round {r}: profile failed for {bad}; see {R / 'profile'}")
        if not saved:
            prog.mark("profile", profiles)
        kb_resume.STOP.check()
        first = first or {p["ir"]: p["partition_us"] for p in profiles}
        rec["partition_us"] = {p["ir"]: p["partition_us"] for p in profiles}
        if gs.active():                                      # round 0: characterization pass (re-profiled)
            profiles = gs.before_propose(r, R, prog, profiles, kb=kb, cache=cache, specs=specs,
                                         reprofile=lambda sub, hd=None, R=R: _reprofile(R, sub, hd))
        if r == 0 and kb_barrier.enabled(cfg) and not prog.done("gs_char"):
            bar.skipped("characterization is off (--graph-characterize 0 / no --graph-* flag): nothing to park after")
        T.lap("mb.propose")
        graphs = {}
        for p in profiles:                                  # MB's own proposals
            g = graphs[p["ir"]] = json.loads((Path(p["ir_dir"]) / "graph.json").read_text())
            gh.merge(doc, gh.propose_from_profile(g, {o["name"]: o["us"] or 0.0 for o in p["ops"]}, specs,
                                                   doc=doc, min_gain_pct=cfg.min_gain_pct, ir=p["ir"],
                                                   launch_us=p.get("launch_us") or 0.0, total_us=p.get("e2e_us")))
        ext.propose(graphs, profiles, doc, specs)             # --fusion-extended (no-op when off)
        caps.propose(graphs, profiles, doc, specs)            # --region-slots (no-op when off)
        if gs.active():                                      # priors pre-apply, fold proposals, beam members
            profiles, graphs = gs.after_propose(r, R, prog, profiles, graphs, doc, specs, hints_path=hints_path,
                                                reprofile=lambda sub, hd=None, R=R: _reprofile(R, sub, hd))
        T.lap("mb.plan")
        if cfg.plan_mode == "all":
            plan_kw = dict(coverage=2.0, top_k=cfg.plan_top_k or None, per_item=True,
                           min_share=kb_efficiency.round_min_share(cfg.plan_min_share, cfg.plan_min_share_late, r))
        else:
            plan_kw = dict(coverage=cfg.coverage, top_k=cfg.top_k,
                           min_share=kb_efficiency.round_min_share(cfg.min_share, cfg.plan_min_share_late, r))
        plan_kw.update(ext.plan_kw(specs))                   # --reuse-winners: canonical item keys
        plan = prog.once("plan", lambda: make_plan(gs.plan_input(profiles), **plan_kw, exclude=sf.exclude(
            set(state["sent"]), profiles, state, r, plan_kw, specs)))        # + re-queued items (search fixes)
        (R / "plan.json").write_text(json.dumps(plan, indent=1))
        rec["plan"] = {"selected_share": plan["selected_share"],
                       "problems": [(p["ir"], len(p["op_names"])) for p in plan["problems"]]}
        base_ms = cfg.plan_min_share if cfg.plan_mode == "all" else cfg.min_share
        if kb_efficiency.active(eff):
            rec["efficiency"] = {"stop_patience": cfg.stop_patience, "min_share": plan_kw["min_share"],
                                 "late_min_share": cfg.plan_min_share_late if r >= 1 else None,
                                 "late_skipped": kb_efficiency.late_skipped(plan, base_ms) if r >= 1 else []}

        def _cands():
            if gs.active():                                 # graph search: prior ranking; beam: members decide
                return gs.candidates(doc, profiles, ("fuse", "split_slot") + ext.decide_kinds() + caps.decide_kinds() + gs.decide_kinds(),
                                     _est_gain, _hint_ir, cfg)
            todo = [h for h in doc["hints"] if h.get("status") == "proposed" and h.get("target", "cuda") == "cuda"
                    and h["kind"] in ("fuse", "split_slot") + ext.decide_kinds() + caps.decide_kinds() and _est_gain(h) >= cfg.min_gain_pct]
            todo.sort(key=lambda h: -_est_gain(h))          # stable: ties keep the hints file's order
            return {"n_todo": len(todo), "cands": [(h["id"], p["ir"]) for h in todo[:cfg.max_variants]
                                                   if (p := _hint_ir(h, profiles)) is not None]}

        def _inter_kw():                                    # --graph-search interleaved: the engine's inputs
            return dict(cands_fn=_cands, doc=doc, profiles=profiles, graphs=graphs, kb=kb, cache=cache, cfg=cfg,
                        specs=specs, leases=leases, width=width, hints_path=hints_path)
        if plan["problems"]:
            T.lap("mb.export")
            dirs = [Path(x) for x in prog.once("export", lambda: [str(x) for x in gs.budget(sf.seed_problems(ext.seed_problems(
                export_plan(plan, R / "problems", cache=cache, arch=cfg.arch, specs=specs,
                            names={p.get("id", p["ir"]): f"{p.get('id', p['ir'])}__r{r:02d}" for p in plan["problems"]}),
                cache, specs), plan, cache, specs, r), plan, doc, graphs, specs, _scope_ops, steps=cfg.steps,
                stop_patience=cfg.stop_patience)])]
            caps.annotate(dirs, profiles)                    # --model-context (no-op when off)
            T.lap("mb.kb_wait", kind="plan")
            if kb.halfway is not None:                       # halfway sync: the plan problems park at half their steps
                kb.halfway.arm(r, R / "run", [problem_name(d) for d in dirs])
            par = (cfg.plan_parallel or len(dirs)) if cfg.plan_mode == "all" and len(dirs) > 1 else None
            results = gs.run_interleaved(r, R, prog, plan_dirs=dirs, plan=plan, **_inter_kw())["plan_results"] \
                if gs.inter else kb.run(dirs, R / "run", winners=R / "winners", parallel_problems=par)
            if "efficiency" in rec:
                rec["efficiency"].update(kb_efficiency.kb_patience(results))
            T.lap("mb.ingest")
            rec["kb_runs"] = [{k: x.get(k) for k in ("problem", "baseline_partition_us", "final_partition_us",
                                                      "speedup", "final_ok", "llm_calls")} for x in results]
            if (R / "winners" / "winners.json").exists():
                rec["ingest"] = prog.once("ingest", lambda: ingest(gs.winners(R), cache, gs.ingest_specs(specs, graphs)))
            if (rq := sf.ingest_requeued(results, cache, specs, r)):      # search fixes: re-queued problems' bests
                rec["requeue_ingest"] = rq
            sf.mark_sent(plan, results, state, r)        # search fixes: sent = measured (lost / infra items stay eligible)
            for x in results:
                pf = Path(x["_dir"]) / "proposals.json"
                if pf.exists():
                    props = json.loads(pf.read_text())
                    for h in gh.from_kb_proposals(props, specs=specs):
                        for s in h["source"]:
                            s["ir"] = json.loads((Path(props["problem_dir"]) / "manifest.json").read_text())["ir_name"]
                        gh.merge(doc, [h])
        if ext.wants_after_round():                          # zero-LLM: winner cross-application, concurrency auto
            rec["reuse"] = prog.once("reuse", lambda: ext.after_round(R, profiles, doc, cache, specs, mbev, hw.model))
        gh.save(doc, hints_path)
        T.lap("mb.decide")
        kb_resume.STOP.check()

        cs = prog.once("cands", _cands)                     # the candidates before any was decided
        todo = [None] * cs["n_todo"]
        cands = [(gh.get(doc, hid), next(p for p in profiles if p["ir"] == ir)) for hid, ir in cs["cands"]]
        t_dec = time.time()

        def _wave(k, items):
            run_dir = items[0][2] / "run" if len(items) == 1 else R / "variants" / f"wave{k:02d}"
            return evaluate_hints(items, run_dir, kb=kb, cache=cache, doc=doc, cfg=cfg, specs=specs, leases=leases)
        waves, rec["decisions"] = gs.interleaved_decide(r, R, prog, **_inter_kw()) if gs.inter else kb_resume.decide_hints(
            prog, cands, graphs, R / "variants", width, decide_wave=_wave, schedule=schedule, footprint=footprint,
            specs=specs, hints_path=hints_path, save_hints=lambda: gh.save(doc, hints_path))
        rec["parallel_hints"], rec["waves"] = width, waves
        for g in kb.gpus[n_gpu_rec:]:
            hw.check("kb", g.get("gpu"), Path(g["run"]).name)
        n_gpu_rec = len(kb.gpus)
        for dec in rec["decisions"]:
            if dec.get("remeasure_device"):
                hw.check("remeasure", dec["remeasure_device"], dec["hint"])
        rec["decide_seconds"] = round(time.time() - t_dec, 1)
        rec["seconds"] = round(time.time() - t0, 1)
        rec["llm_calls_total"] = kb.calls
        if (sfr := sf.round_record()) is not None:
            rec["search_fixes"] = sfr
        if gs.active() and gs.audit_mode == "rounds":        # --graph-preapply-audit rounds: unapply audit at the round end
            if (au := gs.audit("round", irs=irs, doc=doc, hints_path=hints_path, cache=cache, specs=specs, cfg=cfg,
                               leases=leases, recorded=list(kb.gpus))):
                rec["preapply_audit"] = {k: au.get(k) for k in ("reverted", "kept", "skipped", "audit_error")}
                rec["decisions"] += au.get("decisions") or []
        if gs.active():                                      # beam checkpoint; graph decisions of the round
            rec["graph_search"] = gs.end_round(r, R, prog, profiles=profiles, doc=doc, cache=cache, specs=specs,
                                               kbeval=mbev, gpu_model=hw.model, hints_path=hints_path)
        T.end_laps(rsec.span)
        rsec.end()
        tr = {"mb": kbt.compact(T.full_summary(rsec.span)), "kb": kbt.kb_rollup(kb.telemetry[n_kb:])}
        if tr["mb"] or tr["kb"]:
            rec["telemetry"] = tr
        state["rounds"].append(rec)
        kb_resume.atomic_write_json(state_path, state)
        (R / "round.json").write_text(json.dumps(rec, indent=1))
        accepted = [d for d in rec["decisions"] if d.get("hint_status") == "accepted"]
        print(json.dumps({"round": r, "partition_us": rec["partition_us"], "plan": rec["plan"]["problems"],
                          "waves": rec["waves"],
                          "decisions": [(d["hint"], d["status"], d.get("gain")) for d in rec["decisions"]]}),
              flush=True)
        if "efficiency" in rec:
            ef = rec["efficiency"]
            print(json.dumps({"round": r, "efficiency": {
                "late_skipped": [(x["op"], x["share"]) for x in ef["late_skipped"]],
                "patience_stopped": [(x["problem"], x["at_commit"]) for x in ef.get("patience_stopped", [])],
                "steps": [ef.get("steps_committed"), ef.get("steps_planned")]}}), flush=True)
        # graph search: a promising hint (next rung next round) or a beam still deciding is not a fixpoint
        undecided = [d for d in rec["decisions"] if d.get("status") in ("undecided", "promising")]
        if not accepted and not undecided and not plan["problems"] and len(todo) <= cfg.max_variants \
                and not gs.blocks_fixpoint(r, doc):
            state["fixpoint"], state["fixpoint_key"] = r, kh
            kb_resume.atomic_write_json(state_path, state)
            break
    if gs.active() and gs.audit_mode in ("final", "rounds"):   # unapply audit of the pre-applied hints, before the final model
        gs.audit("final", irs=irs, doc=doc, hints_path=hints_path, cache=cache, specs=specs, cfg=cfg, leases=leases,
                 recorded=list(kb.gpus), state=state, state_path=state_path)
    final = {}
    with T.span("mb.verify", evaluator=mbev.evaluator if mbev is not None else "builtin"):
        for d in irs:
            v = final[Path(d).name] = verify_ir(Path(d), cache, arch=cfg.arch, specs=specs, hints=hints_path,
                                                work=work / "final" / Path(d).name, kbeval=mbev,
                                                gpu_model=hw.model, allow_gpu_mismatch=cfg.allow_gpu_mismatch)
            if "device" in v:
                hw.check("verify", v.get("device"), v.get("ir", Path(d).name))
    # search fixes: dominant ops left on / near their seed kernel (warn; repair = route them, zero LLM, verify again)
    guard = sf.final_guard(irs, final, state, cache, specs, kbeval=mbev, gpu_model=hw.model, hints_doc=doc,
                           reverify=lambda d: verify_ir(Path(d), cache, arch=cfg.arch, specs=specs, hints=hints_path,
                                                        work=work / "final" / Path(d).name, kbeval=mbev,
                                                        gpu_model=hw.model, allow_gpu_mismatch=cfg.allow_gpu_mismatch))
    summary = {"irs": {}, "rounds": len(state["rounds"]), "fixpoint": state.get("fixpoint"),
               **({"cuda_learnings": ext.summary()} if ext.round_key() else {}),
               **({"capabilities": caps.summary()} if caps.round_key() else {}),
               "accepted": [h["id"] for h in doc["hints"] if h.get("status") == "accepted"],
               "rejected": [h["id"] for h in doc["hints"] if h.get("status") == "rejected"],
               "llm_calls": kb.calls, "parallel_hints": width,
               "efficiency": kb_efficiency.summarize(rec_eff, state["rounds"]),
               **({"graph_search": gs.summary()} if gs.active() else {})}
    first0 = state["rounds"][0]["partition_us"] if state["rounds"] else (first or {})
    # the model's result: the whole model on the final cache and hints, as deployed (CUDA graph
    # replay in the persistent runtime), against the hand-seeded original timed the same way
    # -- through KernelBlaster's evaluator on the KernelBlaster runs' pool (local or ssh), on the GPU
    # model those runs recorded (kb_eval), so the model and its components share the hardware
    from .kb_model_run import model_runtime
    models = {}
    kbeval = kb_eval_for(cfg)
    with T.span("mb.model_run", evaluator=kbeval.evaluator if kbeval else "builtin"):
        for d in irs:
            try:
                models[Path(d).name] = model_runtime(Path(d), work=work / "final_model" / Path(d).name, cache=cache,
                                                     hints=hints_path, arch=cfg.arch, specs=specs, kbeval=kbeval,
                                                     gpu_model=cfg.gpu_model, recorded=kb.gpus,
                                                     allow_gpu_mismatch=cfg.allow_gpu_mismatch,
                                                     torch_baseline=cfg.torch_baseline,
                                                     torch_opts={"python": cfg.torch_python or None,
                                                                 "source": cfg.torch_source or None,
                                                                 "rounds": cfg.torch_rounds,
                                                                 "variants": [v for v in cfg.torch_variants.split(",")
                                                                              if v]})
            except Exception as e:                   # the verification above stands; report why
                models[Path(d).name] = {"error": repr(e)}
            else:
                try:
                    hw.check("model_run", models[Path(d).name].get("device"), Path(d).name)
                except Exception as e:  # noqa: BLE001 -- kb_model_run already enforces it; report
                    models[Path(d).name]["error"] = repr(e)
    for name, v in final.items():
        ir = v.get("ir", name)
        f0 = first0.get(ir)
        m = models.get(name, {})
        summary["irs"][ir] = {"first_us": f0, "final_us": v.get("partition_us"), "passed": v.get("passed"),
                              "speedup": (f0 / v["partition_us"]) if f0 and v.get("partition_us") else None,
                              "hints": v.get("hints"), "fused": v.get("fused"),
                              "cache_split": v.get("cache_split"),
                              "model_e2e_us": m.get("model_e2e_us"), "model_basis": m.get("model_basis"),
                              "seed_e2e_us": m.get("seed_e2e_us"), "model_speedup": m.get("speedup"),
                              "model_passed": m.get("passed"), "model_gpu": m.get("gpu_name"),
                              "model_device": (m.get("device") or {}).get("label"),
                              "model_hardware_match": (m.get("hardware") or {}).get("match"),
                              "verify_device": (v.get("device") or {}).get("label"),
                              "verify_gpu": (v.get("device") or {}).get("name"),
                              # kb_model_run --torch-baseline: torch_status, torch_us, torch_<variant>_us,
                              # torch_<variant>_wall_us, speedup_vs_torch(_wall, _eager), ... (the table is in
                              # model_runtime.json's torch_baseline)
                              **{k: v for k, v in m.items() if k.startswith(("torch_", "speedup_vs_torch",
                                                                              "ours_cmp_"))
                                 and k != "torch_baseline"},
                              **({"model_error": m["error"]} if "error" in m else {})}
    summary["search_fixes"] = {**sf.summary(), "final_guard": guard}
    for ir, x in ((guard or {}).get("irs") or {}).items():
        if ir in summary["irs"]:
            summary["irs"][ir]["seed_left"] = x.get("flagged")
    (work / "model_runtime.json").write_text(json.dumps(models, indent=1, default=str))
    summary["gpu"] = hw.summary()
    state["gpu_check"] = summary["gpu"]
    tr = {"mb": kbt.compact(T.full_summary(run_span)), "kb": kbt.kb_rollup(kb.telemetry)}
    if tr["mb"] or tr["kb"]:
        summary["telemetry"] = tr
    if gk is not None:
        summary["knowledge_global"] = dict(gk)
        merged = kb_global.merge_back(work, cfg, gk)
        if merged is not None:
            summary["knowledge_global"]["merge"] = {k: merged.get(k) for k in ("status", "version", "parent",
                                                                                "fingerprint", "error") if k in merged}
    kb_resume.atomic_write_json(work / "summary.json", summary)
    state["complete"] = kh                         # a rerun with nothing to do returns this summary
    kb_resume.atomic_write_json(state_path, state)
    return summary


def _parallel_arg(s: str) -> int:
    return 0 if s == "auto" else max(1, int(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--work", required=True)
    ap.add_argument("--kb", required=True, help="KernelBlaster checkout")
    ap.add_argument("--model", default="claude-code:sonnet")
    ap.add_argument("--reflect-model")
    ap.add_argument("--plugin", action="append", default=[], help="passed to run_multislot.py (stub: models)")
    ap.add_argument("--python", default=sys.executable, help="python for KernelBlaster")
    ap.add_argument("--parallel-hints", type=_parallel_arg, default=0,
                    help="hints decided at once: auto (default: max(4, 2 x GPUs in the pool)) or N (1 = one by one)")
    ap.add_argument("--gpus", default="auto", help="local GPUs for KernelBlaster and re-timing: auto|all|0,2")
    ap.add_argument("--ssh-host", action="append", default=[], help="KernelBlaster runs on these hosts (repeatable)")
    ap.add_argument("--cuda-visible-devices", action="append", default=[],
                    help="with --ssh-host: auto|ids, once for all hosts or once per host")
    ap.add_argument("--ssh-key")
    ap.add_argument("--remote-dir")
    ap.add_argument("--runtime", choices=["auto", "host", "process"], default="auto",
                    help="KernelBlaster runtime for its runs and ModelBlaster's timings through it")
    ap.add_argument("--gpu-model", help="the GPU model every timing must use (default: what the KernelBlaster runs "
                    "recorded), e.g. 'NVIDIA L40S'")
    ap.add_argument("--allow-gpu-mismatch", action="store_true",
                    help="time the model / re-measure even where no GPU of that model is available (warnings)")
    ap.add_argument("--mb-evaluator", choices=["auto", "kb", "builtin"], default="auto",
                    help="where the per-round profiles, the final verification and the re-measurement run: kb = "
                         "KernelBlaster's evaluator on the runs' pool (local --gpus or the --ssh-host GPUs); builtin "
                         "= built and run here (default GPU / a leased local GPU); auto (default): kb with "
                         "--ssh-host, else builtin")
    ap.add_argument("--kb-arg", action="append", default=[],
                    help="extra run_multislot argument, verbatim (repeatable; use --kb-arg=--flag for dashes)")
    ap.add_argument("--build-cache", metavar="DIR", default=None,
                    help="shared content-addressed nvcc build cache (KERNELBLASTER_BUILD_CACHE) for every build of this "
                         "run and its KernelBlaster subprocesses; off by default (analysis/compile_pool.md)")
    ap.add_argument("--remote-build", action="store_true",
                    help="with --ssh-host: KernelBlaster compiles candidate units on the pool hosts "
                         "(KERNELBLASTER_REMOTE_BUILD=1; the driver's CPUs stay free); off by default")
    ap.add_argument("--ncu-clock-control", choices=["base", "none"], default=None,
                    help="ncu's --clock-control for the slot profiles (KERNELBLASTER_NCU_CLOCK_CONTROL; X-149): base = "
                         "ncu locks the GPU to base clocks (a killed / timed-out ncu can leave them locked); none = "
                         "profile at the running clocks. Default: KernelBlaster's default (analysis/ncu_clock_guard.md)")
    ap.add_argument("--clock-guard", choices=["off", "on"], default=None,
                    help="on: query the GPU's SM clock before each timed step and after an ncu failure, quarantine a "
                         "locked / throttled GPU, re-time elsewhere or mark the step clock-suspect "
                         "(KERNELBLASTER_CLOCK_GUARD=1; X-149); off by default")
    for k, v in asdict(Config(kb=Path("."))).items():
        if k in _MANUAL_FIELDS:
            continue
        ap.add_argument("--" + k.replace("_", "-"), type=type(v) if v is not None else
                        (int if k in ("top_k", "variant_steps", "stop_patience") else
                         float if k == "plan_min_share_late" else str), default=v)
    a = ap.parse_args()
    if a.build_cache:                       # inherited by the KernelBlaster subprocesses (compile pool, X-148)
        os.environ["KERNELBLASTER_BUILD_CACHE"] = str(Path(a.build_cache).expanduser())
    if a.remote_build:
        os.environ["KERNELBLASTER_REMOTE_BUILD"] = "1"
    if a.ncu_clock_control:                 # inherited by the KernelBlaster subprocesses (X-149)
        os.environ["KERNELBLASTER_NCU_CLOCK_CONTROL"] = a.ncu_clock_control
    if a.clock_guard == "on":
        os.environ["KERNELBLASTER_CLOCK_GUARD"] = "1"
    elif a.clock_guard == "off":
        os.environ.pop("KERNELBLASTER_CLOCK_GUARD", None)
    cfg = Config(kb=Path(a.kb).resolve(), model=a.model, reflect_model=a.reflect_model,
                 plugins=[str(Path(p).resolve()) for p in a.plugin], python=a.python,
                 parallel_hints=a.parallel_hints, gpus=a.gpus, ssh_hosts=a.ssh_host,
                 cuda_visible_devices=a.cuda_visible_devices, ssh_key=a.ssh_key, remote_dir=a.remote_dir,
                 runtime=a.runtime, gpu_model=a.gpu_model, allow_gpu_mismatch=a.allow_gpu_mismatch,
                 mb_evaluator=a.mb_evaluator, kb_args=a.kb_arg,
                 **{k: getattr(a, k) for k in asdict(Config(kb=Path("."))) if k not in _MANUAL_FIELDS})
    try:
        print(json.dumps(run([Path(d) for d in a.irs], Path(a.work), cfg), indent=1))
    except kb_resume.Interrupted as e:
        print(f"[kb_graph_opt] interrupted: {e}; rerun the same command to resume", file=sys.stderr, flush=True)
        sys.exit(kb_resume.EXIT_INTERRUPTED)
    except kb_barrier.Parked as e:
        print(f"[kb_graph_opt] parked: {e}; rerun the same command without --park-after-char to resume", flush=True)
        sys.exit(kb_barrier.EXIT_PARKED)


if __name__ == "__main__":
    main()
