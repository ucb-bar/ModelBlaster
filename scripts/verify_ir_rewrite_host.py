#!/usr/bin/env python3
"""Prove a fuse/split IR rewrite is numerically identical -- on the BUILD HOST.

WHY THIS IS NOT A BOARD SCRIPT
------------------------------
The gate (`XPU-RT/scripts/diff_dispatch_graph.py`) proves a rewrite CHANGED the
dispatch graph. It says nothing about whether the rewrite still computes the
right answer, and the answer is what makes the timings mean anything: a
granularity rung whose correctness is unknown is not a measurement, it is the
`RVV_fused` precedent with extra steps.

`MODELBLASTER_VERIFY` answers it, but it runs in the harness binary, and until
`generate_skeleton --platform host` existed every harness binary was RISC-V --
both `linux` and `zephyr` emit `rdcycle`/`rdtime`, which do not assemble on x86.
So the only way to ask "is this rewrite correct" was to spend a board slot on
it. That is backwards: an int8 model is exact integer arithmetic end to end, so
max_abs_err is architecture-independent and the host can answer it for free.

WHAT IT CHECKS, AND WHY THE SECOND CHECK IS NEEDED
--------------------------------------------------
1. The in-binary golden verify (`max_abs_err`) for baseline and rewrite.
2. Element-for-element equality of the REWRITTEN OP'S OWN OUTPUT TENSOR, via
   `ir["inspect_tensors"]`.

(1) alone is a weak witness whenever the model's surface output is small and the
rewrite is deep in the graph: dronet's output is 2 elements and the interesting
split is at dispatch 0, so a wrong channel can in principle be swallowed by
downstream saturation. (2) compares the 100352 elements the split op actually
produced. Both are cheap; run both.

BIT-EXACTNESS IS THE EXPECTATION, NOT A TOLERANCE
-------------------------------------------------
An OC split of a conv partitions the set of output channels; each output
element's sum over (IC, KH, KW) is unchanged, and requantization is elementwise
integer. An N split of a linear does the same over output rows. So the expected
result is max_abs_err == 0 and max_abs_diff == 0, and a nonzero value is a BUG
rather than drift. A reduction-dim (K) split would be the case where accumulation
order genuinely changes and the argument has to be made instead of asserted;
`apply_split_hint` does not emit one today.

USAGE
    scripts/verify_ir_rewrite_host.py \
        --baseline-ir build/k1/dronet/int8/graph.json \
        --rewritten-ir round/graph.split_x2.json \
        --weights build/k1/dronet/int8/weights.npz \
        --io      build/k1/dronet/int8/io.npz \
        --tensor  conv_modules_0 \
        --work    build/host_verify --json out.json

`--tensor` is the PRE-rewrite name of the tensor to compare. Its post-rewrite
counterparts are discovered from the rewritten IR (`<tensor>.tile_<i>` for a
split, the same name for a fusion) and concatenated in tile order, which is the
parent buffer's own layout.

Exit 0 identical, 1 differing, 2 could not be established.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def _sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def _tile_names(rewritten: dict, tensor: str) -> list[str]:
    """Post-rewrite names for `tensor`, in tile order.

    A split renames the op's output to `<tensor>.tile_<i>`; a fusion keeps the
    final tensor's name. Ordering by tile index rather than by appearance is
    deliberate: the parent buffer is laid out in tile order and the tiles are
    aliased into it at `tile * slice_size`, so any other order would compare a
    permutation of the right data and report a false difference.
    """
    have = {t for op in rewritten.get("ops", []) for t in op.get("outputs", [])}
    tiles = sorted(
        (n for n in have if n.startswith(f"{tensor}.tile_")),
        key=lambda n: int(n.rsplit("_", 1)[1]))
    if tiles:
        return tiles
    if tensor in have:
        return [tensor]
    raise SystemExit(
        f"--tensor {tensor!r} has no counterpart in the rewritten IR "
        f"(looked for {tensor!r} and {tensor}.tile_*). Name the PRE-rewrite "
        f"tensor produced by the op that was rewritten.")


def _build_and_run(ir: dict, label: str, args, work: Path) -> tuple[float, dict]:
    gen = work / label
    if gen.exists():
        _sh(["rm", "-rf", str(gen)])
    gen.mkdir(parents=True)
    ir_path = gen / "graph.json"
    ir_path.write_text(json.dumps(ir))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}" + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    py = sys.executable
    _sh([py, "-m", "modelblaster.pipeline.generate_skeleton",
         "--ir", str(ir_path), "--weights", args.weights, "--io", args.io,
         "--out-dir", str(gen), "--backend", args.backend,
         "--platform", "host"], env=env)
    _sh([py, "-m", "modelblaster.pipeline.generate_kernels",
         "--ir", str(ir_path), "--out-dir", str(gen),
         "--target", args.backend, "--backend", "reference",
         "--quant", args.quant,
         "--global-curated-dir", str(REPO_ROOT / "kernels")], env=env)
    binary = work / f"{label}_harness"
    # CFLAGS without -static: this binary never leaves the build host, and a
    # static glibc link is not always available here.
    _sh(["make", "-s", "-C", str(REPO_ROOT / "harness_linux"),
         f"MODEL_DIR={gen}", f"OUT={binary}", "CFLAGS=-O2"])
    out = subprocess.run([str(binary)], capture_output=True, text=True).stdout
    (work / f"{label}.out").write_text(out)
    m = re.search(r"MODELBLASTER_VERIFY === max_abs_err=(\S+)", out)
    if not m:
        raise SystemExit(f"{label}: no MODELBLASTER_VERIFY line in stdout")
    return float(m.group(1)), _parse_inspect(out)


def _parse_inspect(text: str) -> dict[str, np.ndarray]:
    out, cur, name = {}, None, None
    for line in text.splitlines():
        m = re.match(r"=== MODELBLASTER_INSPECT_BEGIN \[(.+?)\] ===", line)
        if m:
            name, cur = m.group(1), []
            continue
        if line.startswith("=== MODELBLASTER_INSPECT_END"):
            out[name] = np.array(cur, dtype=np.int64)
            cur = None
            continue
        if cur is not None:
            cur.append(int(float(line.strip())))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-ir", required=True)
    ap.add_argument("--rewritten-ir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--io", required=True)
    ap.add_argument("--tensor", required=True,
                    help="pre-rewrite name of the tensor to compare")
    ap.add_argument("--backend", default="scalar",
                    help="host-compilable backend. scalar is the only one that "
                         "assembles on x86 -- the rvv kernels are intrinsics. "
                         "That is a real limit of this check: it verifies the "
                         "REWRITE, not the vector kernels, and a backend whose "
                         "conv weights are packed away from OIHW needs its own "
                         "argument (see generate_skeleton."
                         "split_conv_tile_weights).")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--work", default="build/host_verify")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    base_ir = json.loads(Path(a.baseline_ir).read_text())
    new_ir = json.loads(Path(a.rewritten_ir).read_text())

    names = _tile_names(new_ir, a.tensor)
    base_ir = {**base_ir, "inspect_tensors": [a.tensor]}
    new_ir = {**new_ir, "inspect_tensors": names}

    base_err, base_insp = _build_and_run(base_ir, "baseline", a, work)
    new_err, new_insp = _build_and_run(new_ir, "rewritten", a, work)

    if a.tensor not in base_insp:
        raise SystemExit(f"baseline run did not dump {a.tensor!r}")
    want = base_insp[a.tensor]
    got = np.concatenate([new_insp[n] for n in names])
    if got.size != want.size:
        print(f"FAIL size mismatch: {a.tensor} is {want.size} elements, the "
              f"rewrite's {len(names)} part(s) total {got.size}")
        return 1
    diff = np.abs(got - want)
    # A split whose tiles are all identical is the historical weight-offset
    # defect; report it explicitly because it can coexist with a passing
    # surface-output verify on a small output.
    tiles_identical = (len(names) > 1 and
                       all(np.array_equal(new_insp[names[0]], new_insp[n])
                           for n in names[1:]))
    res = {
        "baseline_ir": a.baseline_ir, "rewritten_ir": a.rewritten_ir,
        "backend": a.backend, "tensor": a.tensor, "parts": names,
        "n_elems": int(want.size),
        "golden_max_abs_err": {"baseline": base_err, "rewritten": new_err},
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "n_differing": int((diff != 0).sum()),
        "tiles_all_identical": bool(tiles_identical),
    }
    ok = (res["max_abs_diff"] == 0 and base_err == 0 and new_err == 0
          and not tiles_identical)
    res["bit_exact"] = bool(ok)
    res["expectation"] = (
        "bit-exact. An OC/N tile partitions outputs without reordering any "
        "accumulation, so a nonzero difference is a bug, not drift.")
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.json}")
    if tiles_identical:
        print("FAIL every tile produced identical values -- the tile weight "
              "pointers are not distinct (the tile_offset defect)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
