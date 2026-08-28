"""Batch kernel re-ranking on real RTL.

`generate_kernels.py` has always *called* this module from its
`--firesim-eval` path, but the module never existed -- so the moment
FIRESIM_EVAL=1 actually fired for an op it raised ImportError. That is
why FPGA-in-the-loop ranking has never run.

What it does
------------
Given N candidate implementations of ONE op, produce one guest ELF per
candidate (built locally, sequentially -- west build dirs cannot be
shared) and then run the whole wave *concurrently* on the F2 pool
through `fq`. The original design note promised "N candidates in 1
firesim boot"; a fan-out over 8 idle lanes gets the same wall-clock for
much less machinery, and degrades gracefully when a lane is busy.

Correctness and cycles both come from the guest's own uartlog, parsed by
`firesim_eval.evaluator.parse_uart_result`, so a batch result is
byte-for-byte comparable with a single `FiresimEvaluator.evaluate()`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

from modelblaster.optimize.firesim_eval.evaluator import (
    FiresimEvalConfig, FiresimEvalResult, parse_uart_result,
)
from modelblaster.pipeline.backends import Backend
from modelblaster.pipeline.reference_kernels import KernelSpec


class BatchEvaluator:
    """Build many candidate kernels, then run them all on the FPGA pool."""

    def __init__(
        self,
        *,
        shared_model_dir: str,
        build_dir: str,
        backend: Backend,
        specs: list[KernelSpec],
        io_path: str,
        repo_root: str,
        harness_dir: str,
        impls: Optional[dict[str, str]] = None,
        runner: str = "firesim",
        firesim_config: Optional[FiresimEvalConfig] = None,
        log=None,
    ) -> None:
        self.model_dir = shared_model_dir
        self.build_dir = build_dir
        self.backend = backend
        self.specs = specs
        self.io_path = io_path
        self.repo_root = repo_root
        self.harness_dir = harness_dir
        self.impls = dict(impls or {})
        self.runner = runner
        self.config = firesim_config or FiresimEvalConfig()
        self._log = log or (lambda m: print(m, flush=True))
        self._first_build = True

    # -- build ---------------------------------------------------------

    def _model_name(self) -> Optional[str]:
        import json
        for cand in (
            os.path.join(self.model_dir, "graph.json"),
            os.path.join(os.path.dirname(self.model_dir), "graph.json"),
        ):
            if os.path.exists(cand):
                with open(cand) as f:
                    return json.load(f).get("name")
        return None

    def _build_one(
        self, spec: KernelSpec, code: str, label: str
    ) -> tuple[bool, str, str]:
        """Emit kernels.{c,h} with `code` in for spec.op, west-build, and
        park the ELF at a per-label path so the next build can't clobber
        it. Returns (ok, diagnostic, staged_elf_path)."""
        from modelblaster.pipeline.generate_kernels import (
            emit_kernels_h, emit_kernels_c,
        )
        model_name = self._model_name()
        trial = dict(self.impls)
        trial[spec.op] = code

        emit_kernels_h(self.specs, self.model_dir, model_name=model_name)
        emit_kernels_c(
            trial, f"batch-eval:{label}", self.model_dir,
            backend=self.backend, model_name=model_name,
        )

        cmd = ["west", "build", "-b", self.config.board_target,
               self.harness_dir, "--build-dir", self.build_dir]
        if self._first_build:
            cmd.insert(2, "-p")
            self._first_build = False
        overlay = os.path.join(
            self.repo_root, "modelblaster", "harness", "backends",
            "firesim_chipyard.conf",
        )
        cmd += ["--",
                f"-DMODEL_DIR={self.model_dir}",
                f"-DMODELBLASTER_BACKEND={self.backend.name}",
                f"-DEXTRA_CONF_FILE={overlay}"]
        if self.backend.kernel_cflags:
            cmd.append(
                f"-DMODELBLASTER_KERNEL_CFLAGS="
                f"{';'.join(self.backend.kernel_cflags)}")

        env = os.environ.copy()
        env["PATH"] = "/usr/bin:" + env.get("PATH", "")
        proc = subprocess.run(cmd, cwd=self.repo_root, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stdout + "\n" + proc.stderr)[-2000:]
            return False, f"build failed (rc={proc.returncode}):\n{tail}", ""

        elf = os.path.join(self.build_dir, "zephyr", "zephyr.elf")
        if not os.path.exists(elf):
            return False, f"no zephyr.elf at {elf}", ""
        staged_dir = os.path.join(self.build_dir + "_staged")
        os.makedirs(staged_dir, exist_ok=True)
        staged = os.path.join(staged_dir, f"{label}.elf")
        shutil.copy(elf, staged)
        return True, "", staged

    # -- public --------------------------------------------------------

    def evaluate(
        self,
        run_list: list[tuple[str, str]],
        spec: Optional[KernelSpec] = None,
    ) -> dict[str, FiresimEvalResult]:
        """run_list is [(label, code)]. Returns {label: FiresimEvalResult}."""
        if spec is None:
            raise ValueError("BatchEvaluator.evaluate needs the KernelSpec")

        results: dict[str, FiresimEvalResult] = {}
        jobs: list[tuple[str, str]] = []
        for label, code in run_list:
            ok, diag, elf = self._build_one(spec, code, label)
            if not ok:
                self._log(f"  [{spec.op}/batch] {label}: BUILD FAIL")
                results[label] = FiresimEvalResult(ok=False, diagnostic=diag)
                continue
            jobs.append((label, elf))

        if not jobs:
            return results

        self._log(f"  [{spec.op}/batch] running {len(jobs)} candidate(s) "
                  f"on the FPGA pool ({self.config.transport})")

        if self.config.transport == "fq":
            from modelblaster.optimize.firesim_eval.fq_transport import (
                run_fq_many,
            )
            # fq tags become remote dir names -- keep them short and safe.
            tag_of = {lb: f"be-{spec.op[:12]}-{lb}".replace("_", "")[:28]
                      for lb, _ in jobs}
            uarts = run_fq_many(
                [(tag_of[lb], elf) for lb, elf in jobs],
                timeout_sec=int(self.config.firesim_timeout_sec),
            )
            per_label = {lb: uarts.get(tag_of[lb]) for lb, _ in jobs}
        else:
            from modelblaster.validation.firesim_runner import run_firesim
            per_label = {}
            for lb, elf in jobs:
                try:
                    per_label[lb] = run_firesim(
                        elf, models=None,
                        firesim_root=self.config.firesim_root,
                        firesim_env=self.config.firesim_env,
                        firesim_slot=self.config.firesim_slot,
                        timeout=self.config.firesim_timeout_sec,
                        stage_elf=True, kill_first=True, verbose=False)
                except Exception as e:  # noqa: BLE001
                    per_label[lb] = e

        for lb, _ in jobs:
            uart = per_label.get(lb)
            if isinstance(uart, Exception) or not uart:
                results[lb] = FiresimEvalResult(
                    ok=False, diagnostic=f"run failed: {uart}")
                continue
            ok, diag, parsed = parse_uart_result(uart, self.io_path)
            if not ok:
                results[lb] = FiresimEvalResult(ok=False, diagnostic=diag)
                continue
            cbo = parsed.get("cycles_by_op", {}) or {}
            cyc = cbo.get(spec.op)
            golden_ok = parsed.get("golden_ok")
            results[lb] = FiresimEvalResult(
                ok=bool(golden_ok) and cyc is not None,
                cycles_for_op=cyc,
                cycles_by_op=cbo,
                wall_cycles=parsed.get("wall_cycles"),
                golden_ok=golden_ok,
                golden_max_abs_err=parsed.get("golden_max_abs_err"),
                diagnostic=(
                    f"{spec.op}={cyc} cyc golden="
                    f"{'PASS' if golden_ok else 'FAIL'}"),
            )
        return results
