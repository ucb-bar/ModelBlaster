#!/usr/bin/env python3
"""Regenerate the integer DroNet sources for the Gemmini+Saturn arty200t shells.

This is Dima's 10b50ba recipe, which `rb modelblaster generate` cannot express:
grayscale input, conv0 and maxpool fused into one Gemmini call, NHWC islands,
and the gemmini_q31_rvv kernel family compiled for a shell that has an
integer-only vector unit. Its output is committed, verified, under

    artifacts/modelblaster/dronet/int8/gemmini_q31_rvv_softfp-gray-nhwc   (Fp16At35)
    artifacts/modelblaster/dronet/int8/gemmini_q31_rvv_zve64x-gray-nhwc   (IntAt35)

and a board image is built from those with `rb modelblaster build --gen-dir`.
This script is how those directories were made, and how to check that they
still come out the same:

    python3 tools/modelblaster/dronet_gray_nhwc.py --target gemmini_q31_rvv_softfp
    diff -r .rb/modelblaster/dronet-gray-nhwc/ir_gray_nhwc/gemmini_q31_rvv_softfp \\
            artifacts/modelblaster/dronet/int8/gemmini_q31_rvv_softfp-gray-nhwc

SILICON TIER ONLY. Kernel selection is verify-gated: every curated kernel runs
on spike against the scalar reference and is rejected past MB_DRIFT_ATOL. That
needs Chipyard's spike (RB_RISCV_TOOLS, or backends/chipyard/.conda-env) AND a
libgemmini.so whose accumulator scale is Q0.31. Chipyard ships one built from
an older libgemmini with a float scale, and with it every Gemmini kernel fails
the gate at max_abs_err 111 and silently falls back to the scalar reference.
So this script builds the Q31 model itself, from ucb-bar/libgemmini at
LIBGEMMINI_COMMIT and ModelBlaster's per_config/default16x16/gemmini_params.h
(the header the gate's verify images are compiled against), and hands spike
that library with --extlib. --extlib matters: spike carries an RPATH to its own
lib/, which wins over LD_LIBRARY_PATH, so pointing LD_LIBRARY_PATH at a
different libgemmini.so changes nothing. The first run clones libgemmini from
GitHub into the work directory; later runs reuse it.

It also sets MODELBLASTER_GEMMINI_SPIKE explicitly. Left unset, ModelBlaster
falls back to a hardcoded spike under another user's home directory when that
path exists, which makes the gate's result depend on which host ran it.

Why a curated directory rather than --algorithms alone: ModelBlaster pre-seeds
its verify baseline with curated kernels and that pre-seed ignores
--algorithms, so without this it seeds gemmini_resadd (add_s8, an fp32 scale
path) and gemmini_blocked_tb32 (the relayouts) and the picks follow. The
curated directory is kernels/ minus exactly those files, plus, for the
noPermute target, the RVV linear_s8, whose compiled loop carries vmv4r.v.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from rb.environments import zephyr_env  # noqa: E402
from rb import modelblaster as mb  # noqa: E402

LIBGEMMINI_URL = "https://github.com/ucb-bar/libgemmini.git"
LIBGEMMINI_COMMIT = "5b1254a7dc72f757442280e7f28ef30542c78533"
GATE_HEADER = "default16x16"

TARGETS = ("gemmini_q31_rvv_softfp", "gemmini_q31_rvv_zve64x")
ALGORITHMS = ",".join((
    "gemmini_tiled_conv_pool_nhwc", "gemmini_tiled_conv_nhwc", "direct_nhwc",
    "rvv_seg", "direct", "scalar_lut",
))
EXCLUDE = (
    "gemmini_q31_rvv/gemmini_q31_rvv_add_s8_gemmini_resadd.c",
    "gemmini_q31_rvv/gemmini_q31_rvv_nchw_to_nhwc_s8_gemmini_blocked_tb32.c",
    "gemmini_q31_rvv/gemmini_q31_rvv_nhwc_to_nchw_s8_gemmini_blocked_tb32.c",
)
EXCLUDE_NOPERMUTE = ("gemmini_q31_rvv/gemmini_q31_rvv_linear_s8_direct.c",)
DRIFT_ATOL = 3


def run(cmd: list[str], env: dict[str, str], cwd: Path, log) -> None:
    print("+ " + " ".join(cmd), flush=True)
    log.write("+ " + " ".join(cmd) + "\n")
    log.flush()
    result = subprocess.run(cmd, env=env, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(cmd)}\nsee {log.name}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q31_spike(work: Path) -> tuple[Path, Path]:
    """Build the Q31 libgemmini and a spike wrapper that loads it. Idempotent."""
    tools = mb.CHIPYARD_RISCV_TOOLS
    spike = tools / "bin" / "spike"
    if not spike.is_file() or not (tools / "include" / "riscv").is_dir():
        raise SystemExit(
            f"no Chipyard spike at {tools} (bin/spike and include/riscv/ are both "
            "needed). Initialise backends/chipyard (setup.md step 4) or point "
            "RB_RISCV_TOOLS at a Chipyard .conda-env/riscv-tools."
        )
    out = work / "q31-spike"
    lib = out / "libgemmini.so"
    wrapper = out / "spike"
    src = out / "libgemmini"
    if not (src / ".git").is_dir():
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", LIBGEMMINI_URL, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", LIBGEMMINI_COMMIT], check=True)
    header = mb.per_config_header_dir(GATE_HEADER) / "gemmini_params.h"
    shutil.copy2(header, src / "gemmini_params.h")
    subprocess.run(
        ["g++", "-L", str(tools / "lib"), f"-Wl,-rpath,{tools / 'lib'}", "-shared",
         "-o", str(lib), "-std=c++17", "-I", str(src), "-I", str(tools / "include"),
         "-fPIC", "-O3", str(src / "gemmini.cc")],
        check=True,
    )
    wrapper.write_text(
        "#!/bin/sh\n"
        "# Chipyard spike with the Q31 libgemmini forced in; see "
        f"{HERE.relative_to(ROOT)}.\n"
        f'exec "{spike}" --extlib="{lib}" "$@"\n'
    )
    wrapper.chmod(0o755)
    print(f"q31 spike: {wrapper} (libgemmini {LIBGEMMINI_COMMIT[:7]} + "
          f"per_config/{GATE_HEADER}, sha256 {sha256(lib)[:16]})")
    return wrapper, lib


def curated_dir(work: Path, target: str) -> Path:
    """kernels/ as symlinks, minus the kernels the pre-seed must not pick."""
    exclude = set(EXCLUDE) | (set(EXCLUDE_NOPERMUTE) if target.endswith("zve64x") else set())
    out = work / f"curated-{target}"
    if out.exists():
        shutil.rmtree(out)
    kernels = mb.MODELBLASTER / "kernels"
    for path in sorted(kernels.rglob("*")):
        rel = path.relative_to(kernels)
        if path.is_dir() or str(rel) in exclude:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(path)
    missing = [e for e in exclude if not (kernels / e).is_file()]
    if missing:
        raise SystemExit(f"kernels to exclude are not in {kernels}: {missing} -- "
                         "the ModelBlaster pin moved; re-derive the exclusions")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--work", default=str(ROOT / ".rb" / "modelblaster" / "dronet-gray-nhwc"),
                    help="scratch and output root (default: %(default)s)")
    args = ap.parse_args()
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    m = mb.MODELBLASTER
    if not (m / "pipeline").is_dir():
        raise SystemExit(f"{m} is empty: git submodule update --init backends/modelblaster")

    wrapper, lib = q31_spike(work)
    curated = curated_dir(work, args.target)

    env = zephyr_env()
    env.update({
        "PYTHONPATH": str(m.parent),
        "MODELBLASTER_DRONET_CHANNELS": "1",
        "MB_REQUIRE_NHWC": "1",
        "MB_DRIFT_ATOL": str(DRIFT_ATOL),
        "MODELBLASTER_GEMMINI_SPIKE": str(wrapper),
        "MODELBLASTER_GEMMINI_LIB": str(lib),
    })
    env.pop("MODELBLASTER_GEMMINI_CONFIG", None)
    py = "python"
    fused, nhwc = work / "ir_gray_fused", work / "ir_gray_nhwc"
    # The gate's verify builds find the model name (which every kernel symbol
    # is mangled with) in <gen>/../graph.json, so the sources must be written
    # inside the IR directory, exactly where ModelBlaster's own flow puts them.
    gen = nhwc / args.target
    for d in (gen, work / "verify_build" / args.target, work / "cache" / args.target):
        if d.exists():
            shutil.rmtree(d)
    log_path = work / f"{args.target}.log"
    with log_path.open("w") as log:
        run([py, "-m", "modelblaster.pipeline.extract_graph", "--model", "dronet",
             "--quant", "int8", "--enable-fusion", "--fusion-target", "gemmini_q31_rvv",
             "--out-dir", str(fused)], env, m, log)
        nhwc.mkdir(parents=True, exist_ok=True)
        run([py, "pipeline/assign_layouts.py", str(fused / "graph.json"),
             str(nhwc / "graph.json"), "--policy", "islands",
             "--hint", "examples/dronet/int8/dronet_layout_hint.json", "--model", "dronet"],
            env, m, log)
        for name in ("weights.npz", "io.npz", "passes_applied.json"):
            shutil.copy2(fused / name, nhwc / name)
        run([py, "-m", "modelblaster.pipeline.generate_skeleton",
             "--ir", str(nhwc / "graph.json"), "--weights", str(nhwc / "weights.npz"),
             "--io", str(nhwc / "io.npz"), "--out-dir", str(gen), "--backend", args.target],
            env, m, log)
        run([py, "-m", "modelblaster.pipeline.generate_kernels",
             "--ir", str(nhwc / "graph.json"), "--out-dir", str(gen), "--backend", "reference",
             "--target", args.target, "--quant", "int8", "--io", str(nhwc / "io.npz"),
             "--repo-root", str(m), "--build-dir", str(work / "verify_build" / args.target),
             "--harness-dir", str(m / "harness"), "--cache-dir", str(work / "cache" / args.target),
             "--algorithms", ALGORITHMS, "--global-curated-dir", str(curated)], env, m, log)

    # Name each picked kernel by its path inside ModelBlaster, not by the
    # scratch symlink that happened to be used, so the directory can be
    # committed and `rb modelblaster report` can still read the source.
    picks_path = gen / "kernel_picks.json"
    picks = json.loads(picks_path.read_text())
    for pick in picks.get("picks", {}).values():
        path = pick.get("path")
        if isinstance(path, str) and Path(path).is_relative_to(curated):
            pick["path"] = str(Path("kernels") / Path(path).relative_to(curated))
    picks_path.write_text(json.dumps(picks, indent=2) + "\n")

    passed = log_path.read_text().count("verify PASS")
    (gen / mb.CODEGEN_MARKER).write_text(json.dumps({
        "verified": passed > 0,
        "spike": f"Chipyard spike --extlib libgemmini {LIBGEMMINI_COMMIT[:7]} "
                 f"+ per_config/{GATE_HEADER} (Q0.31 scale)",
        "verification_disabled_by": [],
        "drift_atol": DRIFT_ATOL,
        "recipe": str(HERE.relative_to(ROOT)),
    }, indent=2) + "\n")
    print(f"\n{passed} gate PASS line(s); log {log_path}")
    for op, pick in sorted(picks.get("picks", {}).items()):
        print(f"  {op:18s} {pick.get('source')}: {pick.get('algorithm')}")
    print(gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
