"""ModelBlaster's timed runs through KernelBlaster's evaluators.

The whole-model result (``kb_model_run``), the hint re-measurement
(``kb_proposals.remeasure``) and, with ``--ssh-host`` (or ``kb_graph_opt
--mb-evaluator kb``), the graph optimiser's per-round profiles
(``kb_plan.profile_ir``) and final verification (``kb_ingest.verify_ir``,
both through :func:`run_problem`) are run by KernelBlaster's own evaluators,
so they run where -- and the way -- the KernelBlaster runs that optimised the
components ran:

- ``local``: this host's GPU pool (``--gpus auto|all|ids``; auto = the idle
  GPUs), the per-GPU flock every KernelBlaster run takes, the persistent
  driver host;
- ``ssh``: build here, run on remote GPUs (``--ssh-host``, repeatable,
  ``--cuda-visible-devices``, ``--remote-dir``, ``--ssh-key``), with
  KernelBlaster's rsync deltas and run-time ``data.bin`` copies for problems
  above 1 GiB.

KernelBlaster is driven as a subprocess (``scripts/time_problems.py`` of the
checkout, with its python), as ``kb_graph_opt`` drives ``run_multislot.py``;
nothing of it is imported here. :func:`recorded_gpus` reads the GPU each
KernelBlaster run recorded (``result.json`` ``gpu``), which callers pass on
as the required GPU model (``gpu_model``): KernelBlaster then keeps only
GPUs of that model and fails -- never times on another model -- when there
is none.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import kb_telemetry as kbt

ROOT = Path(__file__).resolve().parents[1]


def default_kb() -> Path:
    """The KernelBlaster checkout: ``$MB_KB_ROOT``, else the submodule."""
    return Path(os.environ.get("MB_KB_ROOT") or ROOT / "third_party" / "KernelBlaster")


def gpu_model_key(name) -> str | None:
    """KernelBlaster's model normalisation (``resources.gpu_model_key``):
    "NVIDIA L40S" == "l40s"; None for unknown placeholders ("default", "gpu")."""
    if not name:
        return None
    k = " ".join(str(name).split()).lower()
    k = k[len("nvidia "):] if k.startswith("nvidia ") else k
    return None if k in ("", "default", "gpu") else k


def same_gpu_model(a, b) -> bool:
    ka, kb = gpu_model_key(a), gpu_model_key(b)
    return ka is not None and ka == kb


class HardwareMismatch(RuntimeError):
    """The required GPU model (or arch) is not available, or the inputs disagree."""


@dataclass
class KBEval:
    """Where and how KernelBlaster's evaluators time ModelBlaster's runs."""

    evaluator: str = "local"                    # local | ssh
    kb: Path | None = None                      # checkout (default: default_kb())
    python: str = sys.executable
    gpus: str = "auto"
    ssh_hosts: list = field(default_factory=list)
    cuda_visible_devices: list = field(default_factory=list)
    ssh_key: str | None = None
    remote_dir: str | None = None
    remote_cuda_bin: str | None = None
    runtime: str = "auto"                       # auto | host | process
    devices: int = 1                            # GPUs in the pool (1: every run on one GPU)
    timeout: int = 1800

    @property
    def root(self) -> Path:
        return Path(self.kb) if self.kb else default_kb()

    def available(self) -> bool:
        return (self.root / "scripts" / "time_problems.py").exists()

    def describe(self) -> dict:
        d = {"evaluator": self.evaluator, "runtime": self.runtime, "devices": self.devices}
        if self.evaluator == "ssh":
            d.update(ssh_hosts=list(self.ssh_hosts), cuda_visible_devices=list(self.cuda_visible_devices) or ["auto"],
                     remote_dir=self.remote_dir)
        else:
            d["gpus"] = self.gpus
        return d

    def for_device(self, dev: dict | None) -> "KBEval":
        """Pinned to one detected device (``{"host", "ident", ...}``, as
        ``kb_graph_opt.detect_gpus`` lists them); None: unchanged."""
        if not dev or dev.get("ident") is None:
            return self
        if dev.get("host"):
            return replace(self, evaluator="ssh", ssh_hosts=[dev["host"]], cuda_visible_devices=[str(dev["ident"])],
                           devices=1)
        return replace(self, evaluator="local", gpus=str(dev["ident"]), devices=1)

    def command(self, jobs: Path, out: Path, *, arch: str, work: Path, gpu_model: str | None = None,
                telemetry: Path | None = None) -> list[str]:
        cmd = [self.python, str(self.root / "scripts" / "time_problems.py"), str(jobs), "--out", str(out),
               "--arch", str(arch).removeprefix("sm_"), "--evaluator", self.evaluator, "--runtime", self.runtime,
               "--devices", str(self.devices), "--timeout", str(self.timeout), "--work", str(work)]
        if self.evaluator == "ssh":
            if not self.ssh_hosts:
                raise ValueError("--evaluator ssh needs --ssh-host")
            if not self.remote_dir:
                raise ValueError("--evaluator ssh needs --remote-dir")
            for h in self.ssh_hosts:
                cmd += ["--ssh-host", h]
            for v in self.cuda_visible_devices:
                cmd += ["--cuda-visible-devices", str(v)]
            if self.ssh_key:
                cmd += ["--ssh-key", self.ssh_key]
            cmd += ["--remote-dir", self.remote_dir]
            if self.remote_cuda_bin:
                cmd += ["--remote-cuda-bin", self.remote_cuda_bin]
        elif self.gpus and self.gpus != "auto":
            cmd += ["--gpus", str(self.gpus)]
        if gpu_model:
            cmd += ["--gpu-model", gpu_model]
        if telemetry is not None:
            cmd += ["--telemetry", str(telemetry)]
        return cmd

    def run(self, jobs: list[dict], *, arch: str, work: Path, gpu_model: str | None = None,
            name: str = "time") -> dict:
        """Time ``jobs`` (``time_problems.py``'s job list) -> its result
        (``kernelblaster.time_problems/v1``); raises on a failure to run it,
        :class:`HardwareMismatch` when no GPU of ``gpu_model`` is there."""
        if not self.available():
            raise FileNotFoundError(f"no KernelBlaster checkout with scripts/time_problems.py at {self.root} "
                                    "(set --kb or MB_KB_ROOT)")
        work = Path(work).resolve()                 # time_problems.py runs in the checkout: absolute paths
        work.mkdir(parents=True, exist_ok=True)
        jobs = [{**j, "problem": str(Path(j["problem"]).resolve())} for j in jobs]
        jf, of, log = work / f"{name}.jobs.json", work / f"{name}.result.json", work / f"{name}.log"
        jf.write_text(json.dumps({"jobs": jobs}))
        of.unlink(missing_ok=True)
        cmd = self.command(jf, of, arch=arch, work=work / f"{name}_kbwork", gpu_model=gpu_model,
                           telemetry=work / f"{name}.telemetry.jsonl")
        T = kbt.get()
        with T.span("mb.kb_time", evaluator=self.evaluator, jobs=len(jobs), gpu_model=gpu_model,
                    arch=f"sm_{str(arch).removeprefix('sm_')}") as sp:
            with open(log, "w") as lf:
                cp = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(self.root),
                                    timeout=self.timeout * max(4, 2 * len(jobs)) + 600)
            try:
                res = json.loads(of.read_text())
            except (OSError, ValueError):
                raise RuntimeError(f"KernelBlaster time_problems failed (rc {cp.returncode}); see {log}: "
                                   + log.read_text()[-1500:]) from None
            if sp is not None and hasattr(sp, "set"):
                sp.set(models=res.get("models"), seconds=res.get("seconds"), error=res.get("error"))
        err = res.get("error")
        if err:
            if "NoDevice" in err:
                raise HardwareMismatch(err)
            raise RuntimeError(f"KernelBlaster time_problems: {err} (see {log})")
        return res


def add_args(ap, *, prefix_help: str = "", local_default: str = "local") -> None:
    """The evaluator flags (``kb_model_run``, ``kb_proposals decide``; ``kb_plan profile`` and
    ``kb_ingest --verify`` with ``local_default="builtin"``: auto without ``--ssh-host`` keeps
    ModelBlaster's own run on the default GPU)."""
    g = ap.add_argument_group("KernelBlaster evaluator" + (f" ({prefix_help})" if prefix_help else ""))
    auto = ("local when the KernelBlaster checkout has scripts/time_problems.py, else builtin"
            if local_default == "local" else "builtin")
    g.add_argument("--evaluator", choices=["auto", "local", "ssh", "builtin"], default="auto",
                   help="local: this host's GPU pool + locks; ssh: remote GPUs (--ssh-host); builtin: ModelBlaster's "
                        f"own run on the default GPU (no pool); auto: ssh with --ssh-host, else {auto}")
    g.add_argument("--kb", help="KernelBlaster checkout (default $MB_KB_ROOT, else third_party/KernelBlaster)")
    g.add_argument("--kb-python", default=sys.executable, help="python for KernelBlaster")
    g.add_argument("--gpus", default="auto", help="local GPUs: auto (idle ones) | all | nvidia-smi indices / UUIDs")
    g.add_argument("--ssh-host", action="append", default=[], help="user@host with GPUs (repeatable)")
    g.add_argument("--cuda-visible-devices", action="append", default=[],
                   help="GPUs on the ssh host(s): auto | all | ids; once for every host or once per --ssh-host")
    g.add_argument("--ssh-key")
    g.add_argument("--remote-dir", help="writable dir on the ssh host(s)")
    g.add_argument("--remote-cuda-bin", help="CUDA bin dir on the ssh host(s) (default /usr/local/cuda/bin)")
    g.add_argument("--runtime", choices=["auto", "host", "process"], default="auto",
                   help="persistent driver host (auto: where the driver has it) or one process per run")
    g.add_argument("--devices", type=int, default=1, help="GPUs the runs may spread over (default 1: one GPU)")
    g.add_argument("--eval-timeout", type=int, default=1800, help="seconds per evaluation")


def from_args(a, *, local_default: str = "local") -> KBEval | None:
    """The evaluator the flags select; None for ``builtin``."""
    ev = a.evaluator
    if ev == "auto":
        ev = "ssh" if a.ssh_host else local_default
        if ev == "local" and not KBEval(kb=Path(a.kb) if a.kb else None).available():
            return None
    if ev == "builtin":
        return None
    return KBEval(evaluator=ev, kb=Path(a.kb) if a.kb else None, python=a.kb_python, gpus=a.gpus,
                  ssh_hosts=list(a.ssh_host), cuda_visible_devices=list(a.cuda_visible_devices), ssh_key=a.ssh_key,
                  remote_dir=a.remote_dir, remote_cuda_bin=a.remote_cuda_bin, runtime=a.runtime,
                  devices=a.devices, timeout=a.eval_timeout)


def recorded_gpus(run_dirs) -> list[dict]:
    """The GPU every KernelBlaster run recorded (``result.json`` ``gpu``:
    name, uuid, host, arch), with the run; runs from before the field have
    ``gpu`` None (their ``arch`` still counts)."""
    out = []
    for r in run_dirs:
        try:
            res = json.loads((Path(r) / "result.json").read_text())
        except (OSError, ValueError):
            continue
        out.append({"run": str(r), "gpu": res.get("gpu"), "arch": (res.get("gpu") or {}).get("arch") or res.get("arch")})
    return out


def required_hardware(recorded: list[dict], *, gpu_model: str | None = None, arch: str | None = None,
                      allow_mismatch: bool = False) -> dict:
    """The GPU model and arch the timing must use -> {"gpu_model", "arch",
    "recorded_models", "recorded_archs", "verified", "explicit", "warnings"}.

    Explicit ``gpu_model`` / ``arch`` win but must agree with what the runs
    recorded; otherwise the runs' one model and arch are taken. Runs on
    several models or archs, or an explicit value the runs contradict, are a
    :class:`HardwareMismatch` (with ``allow_mismatch``: a warning, and the
    explicit value -- else the first recorded one -- is used). ``verified`` is
    False when no run recorded its GPU (nothing to match; a loud warning)."""
    models = sorted({g["gpu"]["name"] for g in recorded if (g.get("gpu") or {}).get("name")
                     and gpu_model_key(g["gpu"]["name"])})
    keys = {gpu_model_key(m) for m in models}
    archs = sorted({str(g["arch"]).removeprefix("sm_") for g in recorded if g.get("arch")})
    a = str(arch).removeprefix("sm_") if arch else None
    problems, warnings = [], []
    if len(keys) > 1:
        problems.append(f"the KernelBlaster runs were timed on different GPU models {models}: time each model's "
                        "runs separately (--kb-runs) or pass --gpu-model")
    if len(archs) > 1:
        problems.append(f"the KernelBlaster runs were built for different archs {['sm_' + x for x in archs]}")
    if gpu_model and models and not any(same_gpu_model(gpu_model, m) for m in models):
        problems.append(f"--gpu-model {gpu_model!r} but the components were optimised on {', '.join(models)}")
    if a and archs and a not in archs:
        problems.append(f"--arch sm_{a} but the components were built for {', '.join('sm_' + x for x in archs)}")
    if problems and not allow_mismatch:
        raise HardwareMismatch("; ".join(problems) + " (--allow-gpu-mismatch to time anyway)")
    warnings += problems
    model = gpu_model or (models[0] if models else None)
    if model is None:
        warnings.append("no KernelBlaster run recorded its GPU (result.json 'gpu'): the model time cannot be "
                        "matched to the components' GPU model; pass --gpu-model to pin it")
    return {"gpu_model": model, "arch": a or (archs[0] if archs else None), "recorded_models": models,
            "recorded_archs": [f"sm_{x}" for x in archs], "verified": bool(models) and not problems,
            "explicit": bool(gpu_model), "warnings": warnings}


# ---------------------------------------------------------------- ModelBlaster's check / profile runs

def builtin_device(arch: str) -> dict | None:
    """The GPU a builtin run lands on (the first visible one), from nvidia-smi."""
    try:
        txt = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = [[x.strip() for x in line.split(",")] for line in txt.splitlines() if line.count(",") >= 2]
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if vis:
        gpus = [g for v in vis for g in gpus if v == g[0] or (v.startswith("GPU-") and g[1].startswith(v))]
    if not gpus:
        return None
    i, uuid, name = gpus[0][:3]
    return {"host": None, "label": i, "index": int(i) if i.isdigit() else None, "uuid": uuid, "name": name,
            "arch": f"sm_{arch}"}


def driver_result(row: dict) -> dict:
    """One ``time_problems`` run (a KernelBlaster ``EvalResult`` as JSON) -> the
    dict ``kb_export.parse_driver_output`` / ``build_and_run`` give for the same
    driver output (``passed``, ``slots`` = the isolated-op checks, ...), plus
    ``device`` (the GPU it ran on) and ``runtime``."""
    res = {k: row.get(k) for k in (
        "outputs", "slot_us", "op_us", "partition_us", "partition_graph_us", "graph_skip", "errors",
        "e2e_us", "e2e_basis", "ref_e2e_us", "ref_e2e_basis", "e2e_ratio", "attr_slot_us", "attr_op_us",
        "attr_sum_us", "attr_launches", "attr_slot_launches", "attr_skip", "launch_overhead_us",
        "launch_overhead_graph_us", "launch_overhead_frac", "graph_nodes", "runtime", "eval_wall_s")}
    for k in ("outputs", "slot_us", "op_us", "attr_slot_us", "attr_op_us", "attr_slot_launches"):
        res[k] = res[k] or {}
    res["errors"] = res["errors"] or []
    res.update(passed=bool(row.get("ok")), slots=row.get("slot_checks") or {}, stage=row.get("stage"),
               log=row.get("log") or "", device=row.get("gpu"))
    return res


def median_run(rows: list[dict]) -> dict:
    """``build_and_run``'s choice over ``runs``: the passing run with the median
    partition time (``partition_us_runs`` has every run's), else the first run."""
    rs = [driver_result(r) for r in rows]
    if not rs:
        return {"passed": False, "stage": "run", "log": "no run", "device": None, "partition_us_runs": []}
    ok = sorted((r for r in rs if r["passed"] and r["partition_us"]), key=lambda r: r["partition_us"])
    best = ok[len(ok) // 2] if ok else rs[0]
    return {**best, "partition_us_runs": [r["partition_us"] for r in rs if r["partition_us"]]}


def check_device(device: dict | None, gpu_model: str | None, *, what: str, allow_mismatch: bool = False,
                 warnings: list | None = None) -> bool | None:
    """Did ``what`` run on ``gpu_model``? None when there is nothing to compare (no
    required model or no recorded device name); a different model is a
    :class:`HardwareMismatch` unless ``allow_mismatch`` (then a warning)."""
    name = (device or {}).get("name")
    if not gpu_model or not gpu_model_key(name):
        return None
    if same_gpu_model(name, gpu_model):
        return True
    msg = f"{what} ran on {name!r} ({(device or {}).get('label')}), not on {gpu_model} (the flow's GPU model)"
    if not allow_mismatch:
        raise HardwareMismatch(msg + " (--allow-gpu-mismatch to go on)")
    if warnings is not None:
        warnings.append(msg)
    print(f"[kb_eval] WARNING: {msg}", file=sys.stderr, flush=True)
    return False


def run_problem(kbeval: KBEval, problem: Path, *, arch: str, work: Path, reps: int = 1,
                gpu_model: str | None = None, allow_mismatch: bool = False, name: str = "check",
                sources: dict | None = None) -> dict:
    """Check (and time) one exported problem -- the driver checks every output
    and op against the references linked into it -- through KernelBlaster's
    evaluator, ``reps`` runs on one GPU of ``gpu_model`` -> :func:`median_run`'s
    dict plus ``evaluator`` (``describe()``), ``kb_time`` (seconds, pool, models).

    No GPU of ``gpu_model`` there is a :class:`HardwareMismatch`; with
    ``allow_mismatch`` it runs on whatever the pool has, with a warning
    (``hw_warnings``)."""
    job = [{"id": "p", "problem": str(problem), "sources": sources, "reps": max(1, int(reps))}]
    warn = []
    try:
        kr = kbeval.run(job, arch=arch, work=work, gpu_model=gpu_model, name=name)
    except HardwareMismatch as e:
        if not allow_mismatch:
            raise
        warn.append(f"{e}; ran on another GPU model (--allow-gpu-mismatch)")
        print(f"[kb_eval] WARNING: {warn[-1]}", file=sys.stderr, flush=True)
        kr = kbeval.run(job, arch=arch, work=work, gpu_model=None, name=name)
    j = kr["jobs"]["p"]
    res = median_run(j.get("runs") or [])
    res["device"] = res.get("device") or j.get("gpu")
    res["devices"] = j.get("gpus")
    res["evaluator"] = kbeval.describe()
    res["kb_time"] = {k: kr.get(k) for k in ("seconds", "models", "pool")}
    if warn:
        res["hw_warnings"] = warn
    return res
