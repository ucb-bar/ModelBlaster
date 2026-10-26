"""Shared content-addressed build cache for ModelBlaster's own nvcc builds (compile pool, X-148).

Same directory layout, negative entries, LRU eviction and stats files as KernelBlaster's
``servers/utils/build_cache.py`` (so one ``KERNELBLASTER_BUILD_CACHE=<dir>`` serves both), written synchronously for
``kb_export.build_units`` / ``build_host`` / ``build_so`` (thread pool, no asyncio). **Off unless the variable is set**
(``--build-cache DIR`` of kb_graph_opt sets it for the run and its subprocesses); with it unset :func:`cached_build`
just calls the build, so nothing changes.

A key is the sha256 of the tool command's flags, the full source text, the toolchain identity (nvcc release / build,
host compiler, compiler-relevant environment), the content of ``#include "/abs/path"`` headers and the size / mtime of
``.incbin`` data. A compile error is cached too (negative entry, ``KERNELBLASTER_BUILD_CACHE_NEG_TTL`` seconds, default
3600; never for timeouts / killed compilers / resource errors). Builds on pool hosts are KernelBlaster's
``SSHEvaluator`` (``KERNELBLASTER_REMOTE_BUILD=1``); ModelBlaster's own builds here run where the GPU runs and stay
local, they only share the cache.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

_OFF = ("", "0", "off", "false", "no")
_ABS_INPUT_RE = re.compile(r'(?:\.incbin\s+\\?"|#\s*include\s+")(/[^"\\]+)')
_ABS_INCLUDE_RE = re.compile(r'#\s*include\s+"(/[^"\\]+)')
_TOOL_ENV = ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDAFLAGS")
_INFRA_RE = re.compile(r"No space left|Cannot allocate memory|out of memory|Killed|Resource temporarily unavailable|"
                       r"Segmentation fault|internal error|Input/output error|Transport endpoint|Connection (?:reset|refused)|"
                       r"Stale file handle|Too many open files|fork:|timed out", re.I)
_TOOL: dict = {}


def shared_dir() -> Optional[Path]:
    v = os.environ.get("KERNELBLASTER_BUILD_CACHE", os.environ.get("MB_BUILD_CACHE", "")).strip()
    return None if v.lower() in _OFF else Path(os.path.expanduser(v))


def deterministic_failure(rc: int, text: str) -> bool:
    return 0 < rc < 128 and not _INFRA_RE.search(text or "")


def tool_id(nvcc: str = "nvcc") -> str:
    """nvcc release / build, host compiler and compiler environment: the toolchain part of every key (the same string
    KernelBlaster's build cache computes)."""
    if nvcc not in _TOOL:
        def run(cmd):
            try:
                cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                return cp.stdout + cp.stderr
            except (OSError, subprocess.SubprocessError):
                return ""
        out, cx = run([nvcc, "--version"]), run(["c++", "--version"])
        nv = " | ".join(l.strip() for l in out.splitlines() if "release" in l or l.strip().startswith("Build"))
        _TOOL[nvcc] = (f"nvcc[{nv or out.strip()[-200:]}] cxx[{(cx.splitlines() or [''])[0].strip()}]" if out else None)
    core = _TOOL[nvcc]
    if core is None:
        return f"nvcc-unavailable:{nvcc}"
    return core + " env[" + ";".join(f"{k}={os.environ.get(k, '')}" for k in _TOOL_ENV) + "]"


def key_of(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (list, tuple)):
            p = "\n".join(map(str, p))
        b = p if isinstance(p, bytes) else str(p).encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def external_inputs(text: str) -> list[str]:
    """Absolute ``#include "..."`` headers by content, ``.incbin`` data by size and mtime."""
    out = []
    inc = set(_ABS_INCLUDE_RE.findall(text))
    for p in sorted(set(_ABS_INPUT_RE.findall(text))):
        try:
            st = os.stat(p)
        except OSError:
            out.append(f"{p}:missing")
            continue
        if p in inc:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            out.append(f"{p}:sha256:{h.hexdigest()}")
        else:
            out.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
    return out


class Cache:
    """The synchronous face of the shared cache (same files as KernelBlaster's ``BuildCache``)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        gb = os.environ.get("KERNELBLASTER_BUILD_CACHE_MAX_GB", "").strip()
        self.max_bytes = int(float(gb) * 2**30) if gb else None
        self.neg_ttl = float(os.environ.get("KERNELBLASTER_BUILD_CACHE_NEG_TTL", "3600") or 3600)
        sd = os.environ.get("KERNELBLASTER_BUILD_CACHE_STATS_DIR", "").strip()
        self.stats_dir = Path(os.path.expanduser(sd)) if sd else self.root / ".stats"
        self.st = {"hits": 0, "misses": 0, "neg_hits": 0, "neg_stores": 0, "bytes_saved": 0, "seconds_saved": 0.0,
                   "bytes_built": 0, "seconds_built": 0.0, "lock_waits": 0, "lock_wait_s": 0.0, "evicted": 0,
                   "evicted_bytes": 0, "failed_uncached": 0}
        self._sid = f"mb-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._last_flush = 0.0
        import atexit
        atexit.register(self.flush)

    def path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / f"{key}{suffix}"

    def flush(self) -> None:
        try:
            self.stats_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.stats_dir / f".{self._sid}.tmp"
            tmp.write_text(json.dumps({**self.st, "root": str(self.root), "t": round(time.time(), 1), "pid": os.getpid()}))
            os.replace(tmp, self.stats_dir / f"{self._sid}.json")
            self._last_flush = time.time()
        except OSError:
            pass

    def _flush_soon(self) -> None:
        if time.time() - self._last_flush > 5:
            self.flush()

    def _neg(self, key: str) -> Optional[str]:
        if self.neg_ttl <= 0:
            return None
        try:
            d = json.loads((self.root / key[:2] / f"{key}.err").read_text())
        except (OSError, ValueError):
            return None
        return str(d.get("text", "")) if time.time() - float(d.get("t", 0)) <= self.neg_ttl else None

    def _flock(self, key: str, wait_s: float = 1800.0):
        d = self.root / ".locks"
        d.mkdir(parents=True, exist_ok=True)
        fd = os.open(d / key, os.O_CREAT | os.O_RDWR, 0o644)
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if time.time() - t0 > 0.05:
                    self.st["lock_waits"] += 1
                    self.st["lock_wait_s"] = round(self.st["lock_wait_s"] + time.time() - t0, 3)
                return fd
            except BlockingIOError:
                if time.time() - t0 > wait_s:
                    os.close(fd)
                    return None
                time.sleep(0.05)

    def build(self, key: str, suffix: str, out: Path, run: Callable[[Path], tuple]) -> tuple:
        """Materialise the product for ``key`` at ``out`` -> ``(rc, log)``. ``run(tmp)`` builds into ``tmp`` and returns
        ``(rc, log)``; on a hit it is not called (``log`` is ""), a stored compile error is returned as ``(1, text)``."""
        dst = self.path(key, suffix)
        meta = dst.with_name(dst.name + ".meta")

        def hit() -> bool:
            if not dst.exists():
                return False
            self.st["hits"] += 1
            try:
                self.st["bytes_saved"] += dst.stat().st_size
                self.st["seconds_saved"] = round(self.st["seconds_saved"] + float(json.loads(meta.read_text())["seconds"]), 4)
                if time.time() - dst.stat().st_mtime > 60:
                    os.utime(dst)
            except (OSError, ValueError, KeyError):
                pass
            _link(dst, out)
            self._flush_soon()
            return True

        if hit():
            return 0, ""
        if (neg := self._neg(key)) is not None:
            self.st["neg_hits"] += 1
            return 1, neg
        dst.parent.mkdir(parents=True, exist_ok=True)
        fd = self._flock(key)
        try:
            if hit():
                return 0, ""
            if (neg := self._neg(key)) is not None:
                self.st["neg_hits"] += 1
                return 1, neg
            tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
            t0 = time.time()
            try:
                rc, log = run(tmp)
                if rc == 0 and tmp.exists():
                    secs, size = time.time() - t0, tmp.stat().st_size
                    os.replace(tmp, dst)
                    try:
                        meta.write_text(json.dumps({"seconds": round(secs, 4), "bytes": size}))
                    except OSError:
                        pass
                    self.st["misses"] += 1
                    self.st["bytes_built"] += size
                    self.st["seconds_built"] = round(self.st["seconds_built"] + secs, 4)
                    _link(dst, out)
                    self._flush_soon()
                    return 0, log
                if rc != 0 and self.neg_ttl > 0 and deterministic_failure(rc, log):
                    errp = self.root / dst.parent.name / f"{key}.err"
                    t2 = errp.with_name(f".{errp.name}.{uuid.uuid4().hex[:8]}.tmp")
                    t2.write_text(json.dumps({"text": log[-12000:], "t": time.time()}))
                    os.replace(t2, errp)
                    self.st["neg_stores"] += 1
                elif rc != 0:
                    self.st["failed_uncached"] += 1
                return (rc if rc != 0 else 1), log
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)

    def evict(self, max_bytes: Optional[int] = None, min_age: float = 600.0) -> int:
        """LRU eviction down to 80% of the cap; returns the number of files removed."""
        cap = max_bytes if max_bytes is not None else self.max_bytes
        if not cap:
            return 0
        ents = []
        for d in os.scandir(self.root):
            if d.is_dir() and not d.name.startswith("."):
                for f in os.scandir(d.path):
                    if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".meta"):
                        st = f.stat()
                        ents.append((st.st_mtime, st.st_size, f.path))
        total = sum(s for _, s, _ in ents)
        n, now = 0, time.time()
        if total > cap:
            for m, s, p in sorted(ents):
                if total <= cap * 0.8:
                    break
                if now - m < min_age:
                    continue
                for q in (p, p + ".meta"):
                    try:
                        os.unlink(q)
                    except OSError:
                        pass
                total -= s
                n += 1
        self.st["evicted"] += n
        return n


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        os.link(src, tmp)
    except OSError:
        import shutil
        shutil.copyfile(src, tmp)
        os.chmod(tmp, 0o755)
    os.replace(tmp, dst)


_CACHE: dict = {}


def cache() -> Optional[Cache]:
    """The shared cache of this process, None when ``KERNELBLASTER_BUILD_CACHE`` is unset (off)."""
    root = shared_dir()
    if root is None:
        return None
    if str(root) not in _CACHE:
        _CACHE[str(root)] = Cache(root)
    return _CACHE[str(root)]


def cached_build(kind: str, parts: list, out: Path, suffix: str, run: Callable[[Path], tuple]) -> tuple:
    """``run(out)`` -> ``(rc, log)``, through the shared cache when it is on: ``parts`` (everything the product depends
    on: flags, source text, ...) + ``kind`` + the toolchain identity give the key. Off: ``run(out)`` unchanged."""
    c = cache()
    if c is None:
        return run(out)
    nvcc = next((str(p[len("nvcc="):]) for p in parts if isinstance(p, str) and p.startswith("nvcc=")), "nvcc")
    key = key_of(kind, tool_id(nvcc), *[p for p in parts if not (isinstance(p, str) and p.startswith("nvcc="))])
    rc, log = c.build(key, suffix, Path(out), run)
    if c.max_bytes and (c.st["misses"] % 20 == 0 and rc == 0):
        try:
            c.evict()
        except OSError:
            pass
    return rc, log
