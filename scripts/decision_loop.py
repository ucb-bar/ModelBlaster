"""Phase 3 — measured-grounded fuse/split decision loop.

For each round:
  1. Run the granularity advisor (predicted scoring).
  2. For each top-K candidate that's realizable by an existing
     IR-rewriter (fuse_linear_chain via apply_fusion_hint; linear_s8
     split via apply_split_hint — skip conv2d_s8 splits with a
     "not-yet-realizable" log entry, do NOT silently drop them).
  3. Invoke `scripts/measure_candidate.sh` per candidate → spike-PASS
     gate + per-dispatch measured cycles.
  4. ACCEPT iff measured improves total cycles by > epsilon AND
     verify passed; otherwise REJECT with logged reason.
  5. Persist round artifacts and render a Gantt-style summary
     (predicted-vs-measured per candidate).

The point of the loop is the DECISION QUALITY, not headline makespan
delta. We log every (candidate, predicted_Δ, measured_Δ, accepted)
quadruple so you can see, after a few rounds, where the predicted
model overfits / underfits the hardware.

Usage:
  decision_loop.py \
      --networks-json data/toplevel/networks_mlp10_dronet20_yolov8_firesim_static_q31profile.json \
      --baseline-solver decomposed \
      --network mlp_control \
      --quant int8 \
      --target rvv_opu \
      --backend llm \
      --K 3 \
      --rounds 1 \
      --out-dir artifacts/decision_loop/round_001
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
XPURT_ROOT = Path("/scratch2/agustin/XPU-RT")


# ─────────────── candidate scoping ────────────────────────────────────────

REALIZABLE_FUSE_TYPES = {"fuse_linear_chain", "fuse_producer_consumer"}
# apply_split_hint covers both linear_s8 (along N) and conv2d_s8 (along OC)
# as of the conv2d_s8 splitter addition. The candidate's `affected`
# dispatches encode the op kind only indirectly via the name; we accept
# both `linear` and `conv` patterns.
REALIZABLE_SPLIT_KINDS = {"linear_s8", "conv2d_s8"}


@dataclass
class CandidateOutcome:
    id: str
    type: str
    affected: list[str]
    predicted_delta_us: float
    realizable: bool
    realizability_reason: str
    measured_cycles_before: int | None = None
    measured_cycles_after: int | None = None
    measured_delta: int | None = None
    measured_delta_pct: float | None = None
    accepted: bool = False
    accept_reason: str = ""


# ─────────────── steps ───────────────────────────────────────────────────

def run_granularity(args, out_dir: Path) -> dict:
    """Run the predicted granularity scorer and return the result dict."""
    # granularity_loop.py writes its result file relative to its cwd
    # (joining its --out-dir under REPO inside it). Pass absolute paths
    # and resolve where the file actually ends up.
    abs_out = out_dir.resolve()
    abs_hint = (abs_out / "_granularity_hint.json").resolve()
    cmd = [
        "/scratch2/agustin/miniforge3/envs/merlin-dev/bin/python",
        str(XPURT_ROOT / "scripts" / "granularity_loop.py"),
        "--networks-json", str(Path(args.networks_json).resolve()),
        "--baseline-solver", args.baseline_solver,
        "--max-per-type", str(args.K),
        "--out-dir", str(abs_out),
        "--emit-hint", str(abs_hint),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{XPURT_ROOT/'xpu-rt'}"
    print(f"[decision_loop] granularity_loop:\n  {' '.join(shlex.quote(c) for c in cmd)}")
    p = subprocess.run(cmd, cwd=str(XPURT_ROOT), env=env,
                       capture_output=True, text=True)
    log = out_dir / "granularity.log"
    log.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"granularity_loop failed: see {log}")
    # The granularity script writes its result file via os.path.join(REPO, args.out_dir)
    # where REPO is XPU-RT. If our out_dir is absolute it still gets re-joined,
    # producing a weird path. Search common locations.
    candidates = [
        out_dir / "granularity_result.json",
        out_dir.resolve() / "granularity_result.json",
        XPURT_ROOT / "artifacts" / out_dir.name / "granularity_result.json",
    ]
    # Also try interpreting the absolute path as-given under XPU-RT root
    rel = str(out_dir).lstrip("/")
    candidates.append(XPURT_ROOT / rel / "granularity_result.json")
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    raise SystemExit(f"could not locate granularity_result.json (tried {candidates}); see {log}")


def classify_realizability(cand: dict) -> tuple[bool, str]:
    """Decide whether a candidate from granularity_result.json can be
    turned into IR by one of our existing rewriters."""
    ctype = cand["type"]
    affected = cand.get("affected") or []
    if ctype in REALIZABLE_FUSE_TYPES:
        return True, "fuse via apply_fusion_hint --pairwise (linear_s8_elu_s8 KernelSpec exists)"
    if ctype == "split_heavy_dispatch":
        # apply_split_hint covers linear_s8 (N) and conv2d_s8 (OC). The
        # candidate's `affected` is a dispatch name like "dronet1_dispatch_0"
        # — we don't know the op kind from the name alone, but both are
        # supported, so realizable for either. The downstream
        # apply_split_hint call will surface SplitHintError if the op kind
        # is something we don't support yet (e.g. matmul_s8 along N).
        return True, ("split via apply_split_hint (linear_s8 along N OR "
                      "conv2d_s8 along OC; both supported)")
    return False, f"unknown candidate type {ctype}"


def build_fuse_hint(cand: dict, network_filter: str | None = None) -> dict:
    """Turn a granularity_loop candidate into a Contract-2 fusion hint."""
    affected: list[str] = cand["affected"]
    # affected entries look like 'mlp_control3_dispatch_0' — group by network.
    groups: dict[str, list[int]] = {}
    for a in affected:
        # strip 'dispatch_' suffix
        i = a.rfind("_dispatch_")
        if i < 0:
            continue
        net_inst = a[:i]
        disp_id = int(a[i + len("_dispatch_"):])
        # collapse instance suffix to network name: mlp_control3 -> mlp_control
        net = "".join(c for c in net_inst if not c.isdigit() or c == "_").rstrip("_")
        if network_filter and net != network_filter:
            continue
        groups.setdefault(net, []).append(disp_id)
    return {
        "contract": "modelblaster.fusion_hints/v1",
        "reason": f"decision_loop: granularity candidate {cand['id']}; "
                  f"predicted Δmakespan {cand.get('makespan_delta_us')} µs",
        "networks": [{"network": n, "fuse_groups": [sorted(set(ids))],
                      "n_tiny": len(ids)}
                     for n, ids in groups.items()],
    }


def build_split_hint(cand: dict, network_filter: str | None = None,
                     n_splits: int = 2) -> dict:
    affected: list[str] = cand["affected"]
    by_net: dict[str, list[int]] = {}
    for a in affected:
        i = a.rfind("_dispatch_")
        if i < 0:
            continue
        net_inst = a[:i]; did = int(a[i + len("_dispatch_"):])
        net = "".join(c for c in net_inst if not c.isdigit() or c == "_").rstrip("_")
        if network_filter and net != network_filter:
            continue
        by_net.setdefault(net, []).append(did)
    return {
        "contract": "modelblaster.split_hints/v1",
        "reason": f"decision_loop: granularity candidate {cand['id']}; "
                  f"predicted Δmakespan {cand.get('makespan_delta_us')} µs",
        "networks": [{"network": n, "split_ops": [{"op": did, "n_splits": n_splits}
                                                  for did in dids]}
                     for n, dids in by_net.items()],
    }


def measure_one(hint_path: Path, args, out_dir: Path) -> dict | None:
    """Run measure_candidate.sh; return ingested cycles JSON or None on fail."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash", str(REPO_ROOT / "scripts" / "measure_candidate.sh"),
        "--hint", str(hint_path),
        "--model", args.network,
        "--target", args.target,
        "--quant", args.quant,
        "--backend", args.backend,
        "--runner", "spike",
        "--out-dir", str(out_dir),
    ]
    print(f"[decision_loop] measure: {hint_path.name} → {out_dir.name}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "measure.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    if p.returncode != 0:
        return None
    cycles_path = out_dir / "measured_cycles.json"
    if not cycles_path.exists():
        return None
    return json.loads(cycles_path.read_text())


# ─────────────── main ────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks-json", required=True)
    ap.add_argument("--baseline-solver", default="decomposed")
    ap.add_argument("--network", required=True,
                    help="Which network's IR to rewrite (e.g. mlp_control). "
                         "Candidates affecting other networks are logged but skipped.")
    ap.add_argument("--quant", default="int8")
    ap.add_argument("--target", default="rvv_opu")
    ap.add_argument("--backend", default="llm")
    ap.add_argument("--K", type=int, default=3,
                    help="Top-K candidates to measure (after realizability filter)")
    ap.add_argument("--rounds", type=int, default=1,
                    help="Currently informational; this driver always runs 1 round. "
                         "Multi-round iteration is a follow-up (would need to accept "
                         "and re-baseline on the accepted IR).")
    ap.add_argument("--epsilon-cycles", type=int, default=1000,
                    help="Acceptance threshold on measured Δcycles_total.")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase A: predicted candidates
    gr = run_granularity(args, out_dir)
    print(f"[decision_loop] granularity scored {len(gr.get('top_merges', []))} merges "
          f"+ {len(gr.get('top_splits', []))} splits "
          f"(verdict={gr.get('granularity_verdict')})")

    # Phase B: realizability filter, then top-K
    all_cands = []
    for c in gr.get("top_merges", []):
        all_cands.append(c)
    for c in gr.get("top_splits", []):
        all_cands.append(c)
    outcomes: list[CandidateOutcome] = []
    for c in all_cands:
        realizable, reason = classify_realizability(c)
        outcomes.append(CandidateOutcome(
            id=c["id"], type=c["type"],
            affected=c.get("affected", []),
            predicted_delta_us=float(c.get("makespan_delta_us", 0.0)),
            realizable=realizable, realizability_reason=reason,
        ))

    # Filter to candidates targeting --network. Otherwise top-K picks
    # whichever network has the most-favorable predicted score, which
    # may not be the network we set up to measure. The "candidate
    # targets other network" rejection used to happen at measure time —
    # too late, we'd already spent budget on the wrong build path.
    def _affects_network(o):
        return any(a.startswith(args.network) for a in o.affected)
    on_target = [o for o in outcomes if o.realizable and _affects_network(o)]
    print(f"[decision_loop] candidates on --network={args.network}: {len(on_target)}")

    # Sort by predicted delta (most negative first = predicted improvement).
    # Splits often score positive (predicted WORSE) under the granularity
    # advisor — that's because the advisor's model doesn't include the
    # cross-core parallelism benefit of placing tiles on different
    # accelerators. We still measure them: the agentic loop is supposed
    # to disagree with the predicted model when measurement says
    # otherwise.
    realizable_sorted = sorted(on_target, key=lambda o: o.predicted_delta_us)
    to_measure = realizable_sorted[:args.K]
    if not to_measure:
        print(f"[decision_loop] no realizable candidates targeting "
              f"--network={args.network}; nothing to measure")
        # Still write summary with the unfiltered outcomes for trace.
        (out_dir / "summary.json").write_text(json.dumps({
            "baseline_solver": args.baseline_solver,
            "network": args.network,
            "outcomes": [asdict(o) for o in outcomes],
            "note": "no on-target candidates after realizability filter",
        }, indent=2))
        return 0
    print(f"[decision_loop] realizable: {len(realizable_sorted)} / {len(outcomes)} "
          f"→ measuring top-K={len(to_measure)}")

    # Phase C: baseline measurement (no hint, current IR + chosen backend)
    baseline_dir = out_dir / "baseline"
    empty_hint = out_dir / "_empty_hint.json"
    empty_hint.write_text(json.dumps({
        "contract": "modelblaster.fusion_hints/v1",
        "reason": "baseline measurement (no rewrite)", "networks": [],
    }))
    base_cyc = measure_one(empty_hint, args, baseline_dir)
    if not base_cyc:
        raise SystemExit(f"baseline measurement failed; see {baseline_dir}/measure.log")
    base_total = base_cyc["total_dispatch_cycles"]
    print(f"[decision_loop] BASELINE: {base_total} cycles "
          f"({base_cyc['n_dispatches']} dispatches)")

    # Phase D: per-candidate measurement + accept/reject
    for i, o in enumerate(to_measure):
        cand_dir = out_dir / f"cand_{i:02d}_{o.id[:40]}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        # Find the original candidate dict
        orig = next(c for c in all_cands if c["id"] == o.id)
        if "fuse" in o.type:
            hint = build_fuse_hint(orig, network_filter=args.network)
        else:
            hint = build_split_hint(orig, network_filter=args.network)
        if not hint["networks"]:
            o.accept_reason = f"candidate targets a network other than --network={args.network}"
            continue
        hint_path = cand_dir / "hint.json"
        hint_path.write_text(json.dumps(hint, indent=2))
        cyc = measure_one(hint_path, args, cand_dir)
        if cyc is None:
            o.accept_reason = "measurement failed (build error or spike FAIL)"
            continue
        o.measured_cycles_before = base_total
        o.measured_cycles_after = cyc["total_dispatch_cycles"]
        o.measured_delta = base_total - o.measured_cycles_after
        o.measured_delta_pct = 100.0 * o.measured_delta / base_total
        if o.measured_delta > args.epsilon_cycles:
            o.accepted = True
            o.accept_reason = (f"measured improvement {o.measured_delta} cyc "
                               f"({o.measured_delta_pct:.1f}%) > epsilon")
        else:
            o.accepted = False
            o.accept_reason = (f"measured Δ={o.measured_delta} cyc "
                               f"({o.measured_delta_pct:.1f}%) — "
                               f"{'WORSE' if o.measured_delta < 0 else 'within noise'}")

    # Persist
    summary = {
        "baseline_solver": args.baseline_solver,
        "network": args.network,
        "target": args.target,
        "quant": args.quant,
        "backend": args.backend,
        "K": args.K,
        "epsilon_cycles": args.epsilon_cycles,
        "baseline_total_cycles": base_total,
        "outcomes": [asdict(o) for o in outcomes],
        "measured_outcomes": [asdict(o) for o in to_measure],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[decision_loop] wrote {out_dir}/summary.json")

    # Print human-readable table
    print("\nMeasured outcomes:")
    print(f"  {'id':<50} {'pred_Δus':>9} {'meas_Δcyc':>11} {'meas_Δ%':>8} {'accept'}")
    for o in to_measure:
        td = "—" if o.measured_delta is None else f"{o.measured_delta:+d}"
        tp = "—" if o.measured_delta_pct is None else f"{o.measured_delta_pct:+.1f}"
        print(f"  {o.id[:50]:<50} {o.predicted_delta_us:>+9.2f} "
              f"{td:>11} {tp:>8} {'ACCEPT' if o.accepted else 'reject'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
