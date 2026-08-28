"""Run a guest ELF on the shared AWS F2 FireSim pool via the `fq` queue.

Why this exists
---------------
`modelblaster.validation.firesim_runner` drives a *local* FireSim install
(`FIRESIM_ROOT`, default `/scratch2/dima/chipyard-fsim`). On this host that
install is wired to `externally_provisioned` + `localhost` +
XilinxAlveoU250InstanceDeployManager -- i.e. garden's single local U250.
Using it for kernel re-ranking (a) serialises every candidate behind one
FPGA and (b) grabs a board other people are using.

The `fq` queue on the AWS manager fronts an 8-lane f2.6xlarge pool with the
same RoSE bitstream. This module speaks to it over ssh so the optimize loop
can rank candidates on 8 FPGAs at once instead of contending for one.

Transport is deliberately dumb: scp the ELF up, `fq submit --wait`, cat the
uartlog back. The uartlog format is identical to the local runner's, so all
of `runner_common`'s parsing (parse_profile / parse_verify /
parse_wall_cycles) works unchanged on the result.

Selected with FIRESIM_TRANSPORT=fq (see FiresimEvalConfig.transport).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class FqConfig:
    """Where the fq pool lives. All overridable by env."""
    host: str = os.environ.get("FQ_HOST", "ubuntu@3.88.218.39")
    key: str = os.environ.get(
        "FQ_KEY", os.path.expanduser("~/.ssh/firesim.pem"))
    # The manager has hosted more than one fq daemon (different state dirs,
    # independent job counters) and a socket has already gone missing under
    # us mid-campaign. Treat the socket as discovered, not assumed:
    # _live_socket() probes these in order for one that answers `fq status`.
    socket: str = os.environ.get("FQ_SOCKET", "")
    # ORDER MATTERS. Both /var/lib/fq and /tmp/fq have been seen running
    # daemons whose pool.yaml lists the SAME eight F2 hosts -- two
    # schedulers each believing they own all 8 lanes. That double-books
    # hosts, and since `firesim kill` is a host-wide pkill and infrasetup
    # reflashes the slot, co-scheduled jobs destroy each other and leave
    # the previous occupant's uartlog behind. /var/lib/fq is the canonical,
    # operator-maintained pool; prefer it and only fall back if it is down.
    socket_candidates: tuple = (
        "/var/lib/fq/fq.sock", "/tmp/fq/fq.sock",
    )
    queue_dir: str = os.environ.get("FQ_DIR", "/home/ubuntu/fpga_queue")
    tree: str = os.environ.get("FQ_TREE", "/home/ubuntu/chipyard-rose")
    hw_config: str = os.environ.get(
        "FQ_HW_CONFIG", "f2_dual_small_norose_tacit_q31_60mhz")
    # Remote scratch root. Kept SHORT on purpose: fq's control socket is
    # AF_UNIX and the whole path budget is ~108 bytes.
    remote_root: str = os.environ.get("FQ_REMOTE_ROOT", "/home/ubuntu/r")
    timeout_sec: int = int(os.environ.get("FQ_TIMEOUT", "3000"))
    # Max ELFs in flight. The pool has 8 lanes; leave headroom by default
    # so a campaign run does not starve interactive use.
    max_parallel: int = int(os.environ.get("FQ_MAX_PARALLEL", "6"))
    # Seconds between launches. A lane can report free while its previous
    # occupant is still tearing down (pkill FireSim-f2 racing hw_server);
    # jobs fired in the same second onto freshly-freed lanes have come back
    # with the PREVIOUS job's uartlog. Staggering avoids that race.
    stagger_sec: float = float(os.environ.get("FQ_STAGGER", "12"))


_LIVE_SOCKET: dict[str, str] = {}


def _live_socket(cfg: "FqConfig") -> str:
    """Return a socket path whose daemon actually answers.

    Cached per host: probing costs an ssh round trip and the answer only
    changes when a daemon dies (at which point every call is failing
    anyway and a re-probe is cheap relative to the FPGA run).
    """
    if cfg.socket:
        return cfg.socket
    hit = _LIVE_SOCKET.get(cfg.host)
    if hit:
        return hit
    for cand in cfg.socket_candidates:
        probe = (
            f"test -S {shlex.quote(cand)} && cd {shlex.quote(cfg.queue_dir)} && "
            f"FQ_SOCKET={shlex.quote(cand)} ./bin/fq status >/dev/null 2>&1 "
            f"&& echo LIVE"
        )
        try:
            out = subprocess.run(
                _ssh_base(cfg) + [probe],
                capture_output=True, text=True, timeout=90,
            ).stdout
        except Exception:  # noqa: BLE001
            continue
        if "LIVE" in out:
            _LIVE_SOCKET[cfg.host] = cand
            return cand
    raise RuntimeError(
        f"fq: no live daemon socket among {cfg.socket_candidates} on {cfg.host}")


def _ssh_base(cfg: FqConfig) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        "-i", cfg.key, cfg.host,
    ]


def _ssh(cfg: FqConfig, remote_cmd: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        _ssh_base(cfg) + [remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.stdout


def run_fq(
    elf_path: str,
    *,
    tag: str | None = None,
    cfg: FqConfig | None = None,
    timeout_sec: int | None = None,
    expected_model: str | None = None,
) -> str:
    """Submit one ELF to the F2 pool and return its uartlog text.

    Raises RuntimeError if the ELF is missing or nothing comes back.
    """
    cfg = cfg or FqConfig()
    if not os.path.exists(elf_path):
        raise RuntimeError(f"fq: no such ELF: {elf_path}")
    to = timeout_sec or cfg.timeout_sec
    # NEVER reuse a results dir. A failed COLLECT leaves the previous job's
    # uartlog in place, and reading it silently attributes another model's
    # cycles to this kernel. The uuid suffix makes every submission's dir
    # unique even across re-runs of the same tag.
    tag = tag or "j"
    # `stamp` is this submission's fingerprint. fq names the staged binary
    # after the workload (backends/base.py: workload or f"fq-job-{job_id}"),
    # and FireSim echoes "+prog0=<workload>0-z.elf" into the uartlog. Setting
    # it ourselves gives every submission a globally unique marker that a
    # stale log from a warm lane cannot possibly carry.
    stamp = f"{tag[:16]}-{uuid.uuid4().hex[:6]}"
    rdir = f"{cfg.remote_root}/{stamp}"
    sock = _live_socket(cfg)

    _ssh(cfg, f"mkdir -p {shlex.quote(rdir)}")
    scp = subprocess.run(
        ["scp", "-q", "-o", "StrictHostKeyChecking=no", "-i", cfg.key,
         elf_path, f"{cfg.host}:{rdir}/z.elf"],
        capture_output=True, text=True, timeout=600,
    )
    if scp.returncode != 0:
        raise RuntimeError(f"fq: scp failed: {scp.stderr[-500:]}")

    submit = (
        f"cd {shlex.quote(cfg.queue_dir)} && "
        f"FQ_SOCKET={shlex.quote(sock)} ./bin/fq submit "
        f"--tree {shlex.quote(cfg.tree)} "
        f"--hw-config {shlex.quote(cfg.hw_config)} "
        f"--elf {rdir}/z.elf --timeout {to} "
        f"--workload {shlex.quote(stamp)} "
        f"--results {rdir} --wait --quiet"
    )
    # fq's exit code tracks the *guest's* disposition; a nonzero rc still
    # leaves a perfectly good uartlog behind (e.g. golden mismatch). Read
    # the log regardless and let the caller's parser judge.
    sub = subprocess.run(
        _ssh_base(cfg) + [submit],
        capture_output=True, text=True, timeout=to + 600,
    )
    # Capture the job id fq assigned. FireSim names the staged binary
    # "fq-job-<ID>-z.elf" and echoes it into the uartlog's command line,
    # which gives us a per-submission fingerprint to authenticate the log.
    jid = None
    mj = re.search(r"job\s+(\d+)", (sub.stdout or "") + (sub.stderr or ""))
    if mj:
        jid = mj.group(1)
    uart = _ssh(
        cfg,
        f"cat {rdir}/uartlog 2>/dev/null || "
        f"find {rdir} -name 'uartlog*' -exec cat {{}} + 2>/dev/null",
        timeout=300,
    )
    if not uart.strip():
        raise RuntimeError(f"fq: empty uartlog for {tag} ({elf_path})")

    # Strongest identity guard: the uartlog must name THIS submission's
    # workload. Observed for real -- two different ELFs came back with
    # byte-identical uartlogs both stamped fq-job-30, because a warm lane
    # still held the previous occupant's log and the collect step copied
    # that. A model-name check cannot catch this when both runs are the
    # same model; a per-submission stamp can.
    if stamp not in uart:
        found = re.findall(r"\+prog0=(\S+)", uart)
        raise RuntimeError(
            f"fq: uartlog for {tag} does not carry this submission's stamp "
            f"{stamp!r} (found prog0={found or '?'}) -- stale lane/results "
            f"content, discarded")

    # Identity guard. A results dir can hold a PREVIOUS job's uartlog when
    # the collect step finds nothing to copy, so a perfectly well-formed
    # profile may belong to a different model entirely. Refuse to hand back
    # a log that does not name the model we asked for.
    if expected_model:
        m = re.search(r"harness: model=(\S+)", uart)
        if not m:
            raise RuntimeError(
                f"fq: uartlog for {tag} has no 'harness: model=' banner; "
                f"refusing to trust it")
        if m.group(1) != expected_model:
            raise RuntimeError(
                f"fq: uartlog for {tag} is model={m.group(1)!r} but we "
                f"submitted {expected_model!r} -- stale results dir, discarded")
    return uart


def run_fq_many(
    jobs: list[tuple[str, str]],
    *,
    cfg: FqConfig | None = None,
    timeout_sec: int | None = None,
    expected_model: str | None = None,
) -> dict[str, str | Exception]:
    """Run many (tag, elf_path) jobs across the pool concurrently.

    Returns {tag: uartlog} with an Exception in place of the log for any
    job that failed to produce one. Never raises for a single bad job --
    a re-rank should survive one broken candidate.
    """
    cfg = cfg or FqConfig()
    out: dict[str, str | Exception] = {}
    if not jobs:
        return out
    workers = max(1, min(cfg.max_parallel, len(jobs)))

    _live_socket(cfg)  # probe once up front, not once per thread

    def _one(item):
        idx, (tag, elf) = item
        # Stagger launches: a lane can look free while its previous
        # occupant is still being torn down, and same-second dispatch onto
        # freshly-freed lanes is what produces cross-job uartlogs.
        if idx and cfg.stagger_sec:
            time.sleep(idx * cfg.stagger_sec)
        try:
            return tag, run_fq(elf, tag=tag, cfg=cfg, timeout_sec=timeout_sec,
                               expected_model=expected_model)
        except Exception as e:  # noqa: BLE001 - reported per-job
            return tag, e

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tag, res in ex.map(_one, list(enumerate(jobs))):
            out[tag] = res
    return out
