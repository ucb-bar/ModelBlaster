"""FireSim-based candidate evaluation for the optimize loop.

The base optimize loop scores candidates on spike (flat memory). This
module adds a re-rank step that takes a list of candidate kernels and
runs each one on FireSim's quad-rocket-saturn RTL, returning per-op
cycles measured against the *actual* cache hierarchy. Cache-locality
wins (which look identical to pipeline wins on spike) finally show up.

Design notes
------------

* **Re-rank, not score-every-candidate.** A single firesim build+run
  takes 60-180 s on the alveo_u250 board. Scoring every spike-survivor
  per beam expansion (typical: 12-18 candidates per op) would push
  per-op optimization to 15-50 minutes; for the dozen ops in dronet that
  exceeds a typical session budget. Instead we let spike's cheap
  cycle counter drive the inner beam, then take the top-K survivors and
  re-score those on firesim. Top-K=3-5 is a sweet spot — large enough
  that a "spike says equal, firesim says different" tie among similar
  candidates doesn't get pruned, small enough that re-rank is bounded at
  ~3-15 min per op.

* **Cache-replacement policy.** A candidate is written back to the
  per-op cache only when its firesim cycles improve on the existing
  cached kernel by at least `replacement_threshold_pct` (default 1%).
  This is a guardrail against accepting a marginal "win" that turns
  out to be measurement noise — firesim's mtime-based timing is
  deterministic but the harness wall-cycle reading can drift a few
  thousand cycles between runs of an unchanged binary.

* **FPGA queue.** The host FPGA is shared across this re-ranker and
  any other firesim-using process (the threadpool microbench, manual
  validation runs). We poll for a free FPGA before each run rather
  than failing — `wait_for_fpga` blocks up to `fpga_wait_timeout` and
  reports who's holding the slot every 30 s.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from modelblaster.pipeline.backends import Backend
from modelblaster.pipeline.reference_kernels import KernelSpec


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class FiresimEvalConfig:
    """Where firesim binaries live + how patient we are with the FPGA queue.

    Defaults match the user's `chipyard-fsim` install. Everything's
    overridable for portability.
    """
    firesim_root: str = os.environ.get(
        "FIRESIM_ROOT", "/scratch2/dima/chipyard-fsim/sims/firesim")
    firesim_env: str = os.environ.get(
        "FIRESIM_ENV", "/scratch2/dima/chipyard-fsim/env.sh")
    firesim_slot: str = os.environ.get(
        "FIRESIM_SLOT", "firesim_rundir/sim_slot_0")
    # Per-run wall-time budget for the firesim_runner.run_firesim() call.
    # Override via FIRESIM_TIMEOUT env. Default 1800 s — the per-element
    # output-dump in the harness can drain HTIF UART for several minutes
    # on kernelbench-shaped tensors (4k+ outputs each). Manager startup
    # adds another ~30s. The previous 240s default was sized for the
    # single-network sequential dronet (<15 s @ 1 GHz target) and timed
    # out on every kernelbench bench's stock-shape printf dump.
    firesim_timeout_sec: float = float(
        os.environ.get("FIRESIM_TIMEOUT", "1800"))
    # FPGA queue politeness: how long we wait for a busy FPGA before
    # giving up. Re-ranks take 1-3 min each so blocking 10-15 min on a
    # contended FPGA is reasonable.
    fpga_wait_timeout_sec: float = 900.0
    fpga_poll_interval_sec: float = 5.0
    # When promoting a candidate to the persistent cache, only do it if
    # its firesim cycles are lower than the existing best by >= this
    # fraction. Guards against measurement noise.
    replacement_threshold_pct: float = 1.0
    # FireSim board target — chipyard's quad-rocket-saturn slot.
    board_target: str = "chipyard_riscv64/rocketchip_virt_riscv64"


@dataclass
class FiresimEvalResult:
    """Outcome of a single evaluate() call."""
    ok: bool
    cycles_for_op: Optional[int] = None
    cycles_by_op: dict[str, int] = field(default_factory=dict)
    wall_cycles: Optional[int] = None
    diagnostic: str = ""
    # PyTorch-golden numerical match (the harness verifies model output
    # vs io.npz; we surface the same outcome here so callers can reject
    # candidates that compile + run but produce wrong numbers).
    golden_ok: Optional[bool] = None
    golden_max_abs_err: Optional[float] = None


# ---------------------------------------------------------------------------
# FPGA queue helper
# ---------------------------------------------------------------------------

def wait_for_fpga(*, timeout_sec: float, poll_interval_sec: float,
                  log) -> bool:
    """Block until no `FireSim-xilinx_alveo_u250-*` simulator is running.
    Returns True on FPGA-available, False on timeout.
    """
    deadline = time.monotonic() + timeout_sec
    last_msg = 0.0
    my_pid = os.getpid()
    while True:
        # `pgrep -f` is needed because the binary name is >15 chars; that
        # means `pgrep -f` itself appears in its own command line and
        # self-matches. Filter our own PID + the pgrep PID out of the
        # result so a quiet FPGA reads as "no match".
        proc = subprocess.run(
            ["pgrep", "-f", "FireSim-xilinx_alveo_u250"],
            capture_output=True, text=True,
        )
        # pgrep returns 1 when no match found — that's our happy path.
        if proc.returncode != 0:
            return True
        # Drop our own PID (rare — but the parent shell might match if
        # this scaffold is being invoked by something like `bash -c
        # 'python ... FireSim-...'`) plus any PID that isn't actually
        # the FPGA driver.
        pids = []
        for ln in proc.stdout.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                pid = int(ln)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            # Read /proc/<pid>/comm to confirm — actual driver's comm[]
            # is "FireSim-xilinx_a" (truncated at TASK_COMM_LEN-1=15).
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if comm.startswith("FireSim-"):
                pids.append(pid)
        if not pids:
            return True
        now = time.monotonic()
        if now > deadline:
            log(f"  [firesim_eval] FPGA still busy after "
                f"{timeout_sec:.0f}s; giving up "
                f"(holding pids: {pids})")
            return False
        if now - last_msg > 30.0:
            log(f"  [firesim_eval] FPGA busy, waiting "
                f"({len(pids)} sim process(es): {pids})")
            last_msg = now
        time.sleep(poll_interval_sec)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FiresimEvaluator:
    """Build + firesim-run a candidate kernel, return per-op cycles.

    Reuses the existing build_and_run path from profile_kernel.py, but
    swaps in (a) the chipyard_riscv64 board target, and (b) the
    firesim_runner instead of spike_runner. The harness output format
    is identical across runners, so cycle parsing reuses runner_common.

    Stateful: each evaluator owns the chipyard build dir (separate from
    the spike build dir so the spike beam search can rebuild while a
    firesim run is in flight). Builds are pristine on the first call
    per evaluator instance, incremental thereafter.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        impls: dict[str, str],
        specs: list[KernelSpec],
        repo_root: str,
        model_dir: str,
        firesim_build_dir: str,
        harness_dir: str,
        io_path: str,
        config: Optional[FiresimEvalConfig] = None,
        log=None,
    ) -> None:
        self.backend = backend
        self.impls = dict(impls)
        self.specs = specs
        self.repo_root = repo_root
        self.model_dir = model_dir
        self.firesim_build_dir = firesim_build_dir
        self.harness_dir = harness_dir
        self.io_path = io_path
        self.config = config or FiresimEvalConfig()
        self._log = log or (lambda m: print(m, flush=True))
        self._first_build = True

    # -- public API ---------------------------------------------------

    def evaluate(
        self,
        spec: KernelSpec,
        candidate_code: str,
    ) -> FiresimEvalResult:
        """Build a harness with `candidate_code` substituted in for
        `spec.op` and run it on FireSim. Returns the per-op cycles
        (and the wall-cycle, golden-OK status, etc.) so the caller can
        compare candidates.
        """
        # Wait for FPGA before doing the (expensive) chipyard build —
        # if FPGA's never going to free, no point compiling.
        if not wait_for_fpga(
            timeout_sec=self.config.fpga_wait_timeout_sec,
            poll_interval_sec=self.config.fpga_poll_interval_sec,
            log=self._log,
        ):
            return FiresimEvalResult(
                ok=False,
                diagnostic=(
                    f"FPGA busy for >"
                    f"{self.config.fpga_wait_timeout_sec:.0f}s; "
                    f"skipping firesim eval"
                ),
            )

        trial_impls = dict(self.impls)
        trial_impls[spec.op] = candidate_code

        ok, build_diag, elf_path = self._build_chipyard(trial_impls)
        if not ok:
            return FiresimEvalResult(ok=False, diagnostic=build_diag)

        ok, run_diag, parsed = self._run_firesim(elf_path)
        if not ok:
            return FiresimEvalResult(ok=False, diagnostic=run_diag)

        cycles_by_op = parsed.get("cycles_by_op", {}) or {}
        cycles_for_op = cycles_by_op.get(spec.op)
        wall = parsed.get("wall_cycles")
        golden_ok = parsed.get("golden_ok")
        golden_max = parsed.get("golden_max_abs_err")

        if golden_ok is False:
            return FiresimEvalResult(
                ok=False,
                cycles_for_op=cycles_for_op,
                cycles_by_op=cycles_by_op,
                wall_cycles=wall,
                golden_ok=False,
                golden_max_abs_err=golden_max,
                diagnostic=(
                    f"firesim run OK, golden mismatch "
                    f"(max_abs_err={golden_max:.3g})"
                ),
            )
        if cycles_for_op is None:
            return FiresimEvalResult(
                ok=False,
                cycles_by_op=cycles_by_op,
                wall_cycles=wall,
                golden_ok=golden_ok,
                diagnostic=(
                    f"firesim run completed but no profile entry for "
                    f"op '{spec.op}' (got: {sorted(cycles_by_op)})"
                ),
            )
        return FiresimEvalResult(
            ok=True,
            cycles_for_op=cycles_for_op,
            cycles_by_op=cycles_by_op,
            wall_cycles=wall,
            golden_ok=True,
            golden_max_abs_err=golden_max,
            diagnostic=(
                f"firesim {spec.op}={cycles_for_op} cyc "
                f"(wall={wall})"
            ),
        )

    # -- internals ----------------------------------------------------

    def _build_chipyard(
        self, trial_impls: dict[str, str]
    ) -> tuple[bool, str, str]:
        """Emit kernels.{c,h} with the trial substituted, then west-build
        for the chipyard board with the firesim_chipyard.conf overlay.
        Returns (ok, diagnostic, elf_path)."""
        # Local import — same lazy pattern profile_kernel uses to avoid
        # the generate_kernels <-> profile_kernel cycle.
        from modelblaster.pipeline.generate_kernels import (
            emit_kernels_h, emit_kernels_c,
        )

        # Need the model name for kernel-symbol mangling. Look it up from
        # graph.json in the IR dir (one level up from the per-target
        # model_dir, same logic as build_and_run).
        model_name: Optional[str] = None
        for candidate in (
            os.path.join(self.model_dir, "graph.json"),
            os.path.join(os.path.dirname(self.model_dir), "graph.json"),
        ):
            if os.path.exists(candidate):
                import json as _json
                with open(candidate) as _f:
                    model_name = _json.load(_f).get("name")
                break

        emit_kernels_h(self.specs, self.model_dir, model_name=model_name)
        emit_kernels_c(
            trial_impls, "firesim-eval", self.model_dir,
            backend=self.backend, model_name=model_name,
        )

        cmd = ["west", "build", "-b", self.config.board_target,
               self.harness_dir, "--build-dir", self.firesim_build_dir]
        if self._first_build:
            cmd.insert(2, "-p")
            self._first_build = False
        overlay = os.path.join(
            self.repo_root, "modelblaster", "harness", "backends",
            "firesim_chipyard.conf",
        )
        cmd += [
            "--",
            f"-DMODEL_DIR={self.model_dir}",
            f"-DMODELBLASTER_BACKEND={self.backend.name}",
            f"-DEXTRA_CONF_FILE={overlay}",
        ]
        if self.backend.kernel_cflags:
            cmd.append(
                f"-DMODELBLASTER_KERNEL_CFLAGS="
                f"{';'.join(self.backend.kernel_cflags)}"
            )

        env = os.environ.copy()
        # Drop the stale Vitis cmake the same way _run_lib.sh does.
        env["PATH"] = "/usr/bin:" + env.get("PATH", "")
        proc = subprocess.run(
            cmd, cwd=self.repo_root, env=env,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stdout + "\n" + proc.stderr)[-2500:]
            return False, (
                f"chipyard build failed (rc={proc.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  output (tail):\n{tail}"
            ), ""
        elf = os.path.join(self.firesim_build_dir, "zephyr", "zephyr.elf")
        if not os.path.exists(elf):
            return False, f"zephyr.elf not produced at {elf}", ""
        return True, "", elf

    def _run_firesim(
        self, elf_path: str
    ) -> tuple[bool, str, dict]:
        """Stage the elf to the firesim sim slot, run it, parse output.
        Returns (ok, diagnostic, parsed_dict). parsed_dict holds
        cycles_by_op, wall_cycles, golden_ok, golden_max_abs_err."""
        # Local imports — same lazy pattern; firesim_runner is heavy.
        from modelblaster.validation.firesim_runner import run_firesim
        from modelblaster.validation.runner_common import (
            parse_output, parse_profile, parse_verify, parse_wall_cycles,
            compare,
        )
        import numpy as np

        try:
            uart = run_firesim(
                elf_path,
                models=None,  # single-model harness
                firesim_root=self.config.firesim_root,
                firesim_env=self.config.firesim_env,
                firesim_slot=self.config.firesim_slot,
                timeout=self.config.firesim_timeout_sec,
                stage_elf=True,
                kill_first=True,
                verbose=False,
            )
        except Exception as e:
            return False, f"firesim run threw: {type(e).__name__}: {e}", {}

        # Modern harness emits a single MODELBLASTER_VERIFY summary; legacy
        # binaries ship the per-element OUTPUT block. Prefer the
        # summary on FireSim — the OUTPUT path's per-element printf
        # over HTIF UART used to dominate the wall budget.
        verify = parse_verify(uart)

        profile = parse_profile(uart) or []
        cycles_by_op: dict[str, int] = {}
        for row in profile:
            cycles_by_op[row["op"]] = (
                cycles_by_op.get(row["op"], 0) + int(row["cycles"])
            )
        wall = parse_wall_cycles(uart)

        raw_golden = np.load(self.io_path)["output"]
        is_int = raw_golden.dtype.kind in ("i", "u")
        is_fp16 = raw_golden.dtype == np.float16
        if is_int:
            atol = rtol = 0.0
        elif is_fp16:
            atol, rtol = 1e-2, 1e-2
        else:
            atol, rtol = 1e-5, 1e-4

        if verify is not None:
            n_golden = int(np.asarray(raw_golden).reshape(-1).size)
            if verify["n"] != n_golden:
                ok = False
                max_ae = float("inf")
            else:
                max_ae = verify["max_abs_err"]
                max_re = verify["max_rel_err"]
                ok = (max_ae <= atol) or (max_re <= rtol)
        else:
            try:
                actual = parse_output(uart)
            except Exception as e:
                return False, f"firesim output parse failed: {e}", {}
            golden = raw_golden.astype(np.float32).reshape(-1)
            ok, stats = compare(actual, golden, atol=atol, rtol=rtol)
            max_ae = stats.get("max_abs_err")

        return True, "ok", {
            "cycles_by_op": cycles_by_op,
            "wall_cycles": wall,
            "golden_ok": ok,
            "golden_max_abs_err": max_ae,
        }


# ---------------------------------------------------------------------------
# Top-K re-rank entry point
# ---------------------------------------------------------------------------

def evaluate_top_k(
    spec: KernelSpec,
    candidates: list[tuple[str, int]],
    *,
    evaluator: FiresimEvaluator,
    spike_baseline: tuple[str, int],
    log,
    k: int = 3,
) -> tuple[Optional[str], Optional[int], list[dict]]:
    """Take a list of (code, spike_cycles) candidates from the optimize
    beam, plus the spike baseline, and re-score the top-K + baseline on
    firesim. Returns (best_code, best_firesim_cycles, history).

    `candidates` should already be sorted by spike_cycles ascending. The
    function picks the top-K (cheapest spike cycles), prepends the
    baseline if not already in that list, and runs each on firesim.

    Returns (None, None, history) if every firesim run fails (e.g. FPGA
    unavailable or all candidates produce wrong numerics on RTL).
    """
    history: list[dict] = []
    if not candidates:
        log(f"  [{spec.op}/firesim] no candidates to re-rank")
        return None, None, history

    # Build the run list: baseline + top-K spike-best, deduplicated.
    seen_norm: set[str] = set()
    run_list: list[tuple[str, int, str]] = []  # (code, spike_cyc, label)

    base_code, base_cyc = spike_baseline
    base_key = " ".join(base_code.split())
    seen_norm.add(base_key)
    run_list.append((base_code, base_cyc, "baseline"))

    for i, (code, cyc) in enumerate(candidates[:k]):
        key = " ".join(code.split())
        if key in seen_norm:
            continue
        seen_norm.add(key)
        run_list.append((code, cyc, f"top-{i+1}"))

    log(f"  [{spec.op}/firesim] re-ranking {len(run_list)} kernel(s) "
        f"(baseline + {len(run_list)-1} unique top-K from spike beam)")

    best_code: Optional[str] = None
    best_cyc: Optional[int] = None
    for label, (code, spike_cyc, _) in zip(
        [t[2] for t in run_list], run_list,
    ):
        log(f"  [{spec.op}/firesim] {label} (spike={spike_cyc} cyc) "
            f"-> firesim build+run")
        t_start = time.monotonic()
        res = evaluator.evaluate(spec, code)
        elapsed = time.monotonic() - t_start
        entry = {
            "label": label, "spike_cycles": spike_cyc,
            "firesim_cycles": res.cycles_for_op,
            "firesim_ok": res.ok, "elapsed_sec": elapsed,
            "diagnostic": res.diagnostic[:300],
            "golden_ok": res.golden_ok,
        }
        history.append(entry)
        if not res.ok:
            log(f"  [{spec.op}/firesim] {label} FAILED in "
                f"{elapsed:.1f}s: {res.diagnostic.splitlines()[0]}")
            continue
        log(f"  [{spec.op}/firesim] {label} firesim={res.cycles_for_op} "
            f"cyc ({elapsed:.1f}s)")
        if best_cyc is None or res.cycles_for_op < best_cyc:
            best_code = code
            best_cyc = res.cycles_for_op

    if best_code is None:
        log(f"  [{spec.op}/firesim] all {len(run_list)} candidates "
            f"failed firesim re-rank")
        return None, None, history

    log(f"  [{spec.op}/firesim] BEST firesim={best_cyc} cyc")
    return best_code, best_cyc, history
