"""ModelBlaster's timed runs through KernelBlaster's evaluators.

The whole-model result (``kb_model_run``) and the hint re-measurement
(``kb_proposals.remeasure``) are timed by KernelBlaster's own evaluators, so
they run where -- and the way -- the KernelBlaster runs that optimised the
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
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
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


def add_args(ap, *, prefix_help: str = "") -> None:
    """The evaluator flags (``kb_model_run``, ``kb_proposals decide``)."""
    g = ap.add_argument_group("KernelBlaster evaluator" + (f" ({prefix_help})" if prefix_help else ""))
    g.add_argument("--evaluator", choices=["auto", "local", "ssh", "builtin"], default="auto",
                   help="local: this host's GPU pool + locks; ssh: remote GPUs (--ssh-host); builtin: ModelBlaster's "
                        "own host on the default GPU (no pool); auto: ssh with --ssh-host, else local when the "
                        "KernelBlaster checkout has scripts/time_problems.py, else builtin")
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


def from_args(a) -> KBEval | None:
    """The evaluator the flags select; None for ``builtin``."""
    ev = a.evaluator
    if ev == "auto":
        ev = "ssh" if a.ssh_host else "local"
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
