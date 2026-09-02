"""Apply XPU-RT IME placement hints (modelblaster.ime_hints/v1) to an IR.

The fifth applier. Like apply_shard_hint it rewrites NO graph structure — it
records, per (op-kind, shape), that those dispatches are to be COMPILED and
RUN on the K1 IME matrix engine (`ime_x60`) instead of RVV, because they are
KNOWN faster there. Same dispatch count, ids, edges; one added metadata block:

    ir["ime_placements"] = [
      {"op": "conv2d_s8", "shape": {IC:256,IH:10,IW:10,OC:128,KH:1,KW:1},
       "impl": "ime_x60", "speedup": 2.478, "source": "measured"}, ...]

WHY PER-(OP,SHAPE) AND NOT PER-OP-KIND. IME wins at large-channel GEMMs and
LOSES at small M / thin conv (attention M=8 -> 0.27x; yolo l0 IC=3 -> 0.40x).
The per-op-KIND kernel picker cannot express "IME for l6.cv2 but RVV for l0" —
so the picker keeps conv on RVV as the default (generate_kernels' shape-aware
guard) and THIS records the per-shape winners for the scheduler/runtime to
route individually. The decision is the ONLY-IF-BETTER rule from `ime_cost`
(measured conv table + measured matmul M-curve); a shape that is not KNOWN
faster is never placed on IME.

With a --hint (the advisor's emitted contract) the placement set is taken from
it (still filtered through ime_cost so a stale hint can never place a loser);
without one, the placements are derived directly from ime_cost — same result,
because the advisor derives the hint from the same measured model.

CLI:
    python -m modelblaster.pipeline.apply_ime_hint \\
        --ir examples/<net>/int8/generated/graph.json \\
        --network <net> --out examples/<net>/int8/generated/graph.ime.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from modelblaster.pipeline import ime_cost
from modelblaster.pipeline.reference_kernels import shapes_from_ir

IME_CONTRACT = "modelblaster.ime_hints/v1"
IME_CLASS = ime_cost.MATMUL_CLASS  # + any conv2d* (checked by prefix)


def _ime_ops(ir: dict) -> List[str]:
    kinds = set()
    for o in ir.get("ops", []):
        op = o.get("op", "")
        if op in IME_CLASS or op.startswith("conv2d"):
            kinds.add(op)
    return sorted(kinds)


def plan_placements(ir: dict) -> Dict[str, List[Dict[str, Any]]]:
    """Per (op, unique shape), decide IME vs RVV by the only-if-better rule.
    Returns {"placements": [...ime...], "kept_rvv": [...known/unknown losers...]}."""
    placements, kept = [], []
    for op in _ime_ops(ir):
        seen = set()
        for s in shapes_from_ir(ir, op):
            sig = tuple(sorted((k, int(v)) for k, v in s.items() if isinstance(v, int)))
            if sig in seen:
                continue
            seen.add(sig)
            sp, prov = ime_cost.ime_speedup_for(op, s)
            rec = {"op": op, "shape": {k: s[k] for k in s if isinstance(s[k], int)},
                   "speedup": (round(sp, 3) if sp is not None else None), "source": prov}
            if sp is not None and sp > 1.0:
                rec["impl"] = "ime_x60"
                placements.append(rec)
            else:
                rec["impl"] = "rvv"
                kept.append(rec)
    return {"placements": placements, "kept_rvv": kept}


def apply(ir: dict, hint: dict | None = None) -> dict:
    plan = plan_placements(ir)
    chosen = plan["placements"]
    if hint is not None:
        # take only placements the hint also asks for, but NEVER trust a hint to
        # place a loser — the ime_cost filter above already removed those.
        want = {(p.get("op"), tuple(sorted(p.get("shape", {}).items()))) for p in hint.get("placements", [])}
        if want:
            chosen = [p for p in chosen
                      if (p["op"], tuple(sorted(p["shape"].items()))) in want or not want]
    out = json.loads(json.dumps(ir))  # deep copy, structure untouched
    out["ime_placements"] = chosen
    out["ime_placement_meta"] = {
        "contract": IME_CONTRACT, "rule": "only_if_faster_than_rvv",
        "n_ime": len(chosen), "n_rvv_kept": len(plan["kept_rvv"]),
        "crossover_note": "conv from measured K1 table; matmul from measured M-curve",
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--network", default=None)
    ap.add_argument("--hint", default=None, help="optional ime_hints/v1 contract")
    a = ap.parse_args(argv)

    ir = json.loads(Path(a.ir).read_text())
    hint = None
    if a.hint:
        hint = json.loads(Path(a.hint).read_text())
        if hint.get("contract") != IME_CONTRACT:
            print(f"ERROR: {a.hint} is contract {hint.get('contract')!r}, "
                  f"expected {IME_CONTRACT!r}", file=sys.stderr)
            return 2
    out = apply(ir, hint)
    Path(a.out).write_text(json.dumps(out, indent=1))
    m = out["ime_placement_meta"]
    print(f"{a.network or a.ir}: {m['n_ime']} (op,shape) placed on IME "
          f"(only-if-better), {m['n_rvv_kept']} kept on RVV -> {a.out}")
    for p in out["ime_placements"]:
        sh = "x".join(f"{k}{v}" for k, v in p["shape"].items())
        print(f"  IME  {p['op']:26s} {sh[:52]:52s} x{p['speedup']} [{p['source']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
