"""Graph and kernel capabilities (A2; l3_farm/analysis/capabilities_impl.md): the flow switches and their hooks.

``kb_graph_opt`` Config fields / flags (defaults keep the flow as it was, except the dead-op pruning noted):

| flag | values (default first) | what |
|---|---|---|
| ``--prune-dead`` | ``on`` \\| ``off`` | dead-op pruning at export (graph_prune; ``$MB_PRUNE_DEAD``). DEFAULT ON (kill switch off): only ops whose outputs never reach a graph output lose their call |
| ``--recurrent-class`` | ``off`` \\| ``on`` | gru_seq / lstm_seq are class ``recurrent`` (op_classes; KernelBlaster menu, roofline state, prompt notes) |
| ``--precision-policy`` | ``off`` \\| ``auto`` | recorded only: the policy acts at extraction (``precision_policy extract``); the flow refuses nothing |
| ``--region-slots`` | ``off`` \\| ``rnn_stack,attention_block,all`` | propose ``fuse_region`` hints (region_slots): one slot owns a sub-graph, composite seed, decided like any fusion |
| ``--multiturn`` | ``off`` \\| ``on`` | KernelBlaster codegen may run experiments (agents/experiment_tool.py) over several turns; ``--multiturn-budget`` "turns,runs,seconds" |
| ``--experiments`` | ``off`` \\| ``record`` \\| ``ingest`` \\| ``ingest+prompt`` | with ``--multiturn on``: record each probe program (X-145, knowledge/v2/experiments.py): ``record`` writes ``<out>/experiments.jsonl``; ``ingest`` also turns them into experiment claims / probe templates in the run's knowledge dir; ``ingest+prompt`` also shows up to 3 prior facts (800 chars) at the start of a multi-turn step |
| ``--model-context`` | ``off`` \\| ``on`` | each exported slot problem's manifest gets ``model_context`` (whole-model op shares, sibling ops) for the prompt |
| ``--view-fusion`` | ``off`` \\| ``on`` | propose ``fuse_view`` hints: producer -> pure-reshape view -> flat elementwise pairs (view_fusion) |

Round key: only switches that differ from their defaults (existing work dirs keep their key).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULTS = {"prune_dead": "on", "recurrent_class": "off", "precision_policy": "off", "region_slots": "off",
            "multiturn": "off", "multiturn_budget": "3,4,120", "experiments": "off", "model_context": "off", "view_fusion": "off"}


class Caps:
    def __init__(self, cfg, work: Path | None = None):
        self.cfg, self.work = cfg, work
        self.v = {k: str(getattr(cfg, k, d) if getattr(cfg, k, None) is not None else d).strip().lower()
                  for k, d in DEFAULTS.items()}
        from . import graph_prune, precision_policy, region_slots
        graph_prune.enabled(self.v["prune_dead"])            # validate
        precision_policy.enabled(self.v["precision_policy"])
        self.regions = region_slots.enabled(self.v["region_slots"])
        for k in ("recurrent_class", "multiturn", "model_context", "view_fusion"):
            if self.v[k] not in ("on", "off"):
                raise ValueError(f"--{k.replace('_', '-')} must be on|off, not {self.v[k]!r}")
        parse_budget(self.v["multiturn_budget"])
        if self.v["experiments"] not in ("off", "record", "ingest", "ingest+prompt"):
            raise ValueError(f"--experiments must be off|record|ingest|ingest+prompt, not {self.v['experiments']!r}")

    # ---- environment for exports (this process) and KernelBlaster runs (children)
    def apply_env(self) -> None:
        if self.v["prune_dead"] == "off":
            os.environ["MB_PRUNE_DEAD"] = "0"
        if self.v["recurrent_class"] == "on":
            os.environ["MB_RECURRENT_CLASS"] = "1"
        if self.regions:
            os.environ["MB_REGION_SLOTS"] = ",".join(sorted(self.regions))
        if self.v["multiturn"] == "on":
            os.environ["KERNELBLASTER_MULTITURN"] = "1"
            os.environ["KERNELBLASTER_MULTITURN_BUDGET"] = self.v["multiturn_budget"]
            ex = self.v["experiments"]                       # X-145 (knowledge/v2/experiments.py); off: nothing set
            if ex != "off":
                os.environ["KERNELBLASTER_EXPERIMENT_" + ("RECORD" if ex == "record" else "INGEST")] = "1"
            if ex == "ingest+prompt":
                os.environ["KERNELBLASTER_EXPERIMENT_PROMPT"] = "3"

    def round_key(self) -> dict:
        return {f"caps_{k}": v for k, v in self.v.items() if v != DEFAULTS[k]}

    def summary(self) -> dict:
        return dict(self.v)

    # ---- proposals: fuse_region hints
    def propose(self, graphs: dict, profiles: list, doc: dict, specs: dict) -> list:
        out = self._propose_views(graphs, profiles, doc, specs) if self.v["view_fusion"] == "on" else []
        return out + self._propose_regions(graphs, profiles, doc, specs)

    def _propose_views(self, graphs, profiles, doc, specs) -> list:
        from . import graph_hints as gh
        from .view_fusion import propose
        new = []
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            try:
                new += propose(g, {o["name"]: o.get("us") or 0.0 for o in p.get("ops", [])}, specs, doc=doc,
                               min_gain_pct=getattr(self.cfg, "min_gain_pct", 2.0), ir=p["ir"],
                               launch_us=p.get("launch_us") or 0.0, total_us=p.get("e2e_us"))
            except Exception as e:  # noqa: BLE001 -- a proposer never breaks a round
                print(f"[kb_graph_opt] view-fusion proposer: {type(e).__name__}: {e}", flush=True)
        return gh.merge(doc, new) if new else []

    def _propose_regions(self, graphs, profiles, doc, specs) -> list:
        if not self.regions:
            return []
        from . import graph_hints as gh
        from .graph_hints import decided
        from .kb_proposals import with_fused_specs
        from .region_slots import find_regions
        done = decided(doc) if doc else set()
        new = []
        for p in profiles:
            g = graphs.get(p["ir"])
            if g is None:
                continue
            op_us = {o["name"]: o.get("us") or 0.0 for o in p.get("ops", [])}
            total = p.get("e2e_us") or sum(op_us.values()) or 1.0
            sp = with_fused_specs(g, specs)
            for pat in sorted(self.regions):
                hid = f"region:{pat}"
                if hid in done:
                    continue
                try:
                    found = find_regions(gh.unique_names(json.loads(json.dumps(g))), pat, sp)
                except Exception as e:  # noqa: BLE001 -- a proposer never breaks a round
                    print(f"[kb_graph_opt] region proposer {pat}: {type(e).__name__}: {e}", flush=True)
                    continue
                if not found:
                    continue
                share = sum(op_us.get(n, 0.0) for n in found[0]) / total
                new.append({"id": hid, "kind": "fuse_region", "pattern": pat, "target": "cuda", "status": "proposed",
                            "source": [{"by": "mb-region-slots", "ir": p["ir"], "members": len(found[0]),
                                        "share": round(share, 4),
                                        # a megakernel's measured upside (cuda_vs_flow.md: -8% 41, -17% 43) on the
                                        # region's share: the ranking estimate; the decision is the paired re-timing
                                        "est_gain_pct": round(100 * 0.08 * share, 2)}]})
        return gh.merge(doc, new) if new else []

    def decide_kinds(self) -> tuple:
        return (("fuse_region",) if self.regions else ()) + (("fuse_view",) if self.v["view_fusion"] == "on" else ())

    # ---- model context for each exported slot problem (multi-turn / prompt)
    def annotate(self, dirs: list, profiles: list) -> list:
        if self.v["model_context"] != "on":
            return dirs
        for d in dirs:
            try:
                add_model_context(Path(d), profiles)
            except Exception as e:  # noqa: BLE001
                print(f"[kb_graph_opt] model context {d}: {type(e).__name__}: {e}", flush=True)
        return dirs


def parse_budget(s: str) -> dict:
    """"turns,runs,seconds" -> {"turns", "runs", "seconds"} (multi-turn codegen budget per step)."""
    parts = [x.strip() for x in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"multiturn budget {s!r}: expected turns,runs,seconds")
    t, r, sec = int(parts[0]), int(parts[1]), float(parts[2])
    if t < 1 or r < 0 or sec <= 0:
        raise ValueError(f"multiturn budget {s!r}: turns >= 1, runs >= 0, seconds > 0")
    return {"turns": t, "runs": r, "seconds": sec}


def model_context(manifest: dict, profiles: list, max_ops: int = 24) -> dict | None:
    """Whole-model context of a slot problem: the model's ops by share (from the round's profile of its IR), which
    of them this problem holds, and the sibling ops of each slot (same kind / reading the same tensors)."""
    ir = manifest.get("ir_name")
    prof = next((p for p in profiles if p.get("ir") == ir), None) or (profiles[0] if len(profiles) == 1 else None)
    if prof is None:
        return None
    ops = [o for o in prof.get("ops", []) if o.get("us")]
    total = prof.get("e2e_us") or sum(o["us"] for o in ops) or 1.0
    mine = set(manifest.get("partition", {}).get("op_names", []))
    top = sorted(ops, key=lambda o: -o["us"])[:max_ops]
    by_kind = {}
    for o in ops:
        by_kind.setdefault(o["op"], []).append(o)
    return {"ir": ir, "model_us": round(total, 1), "n_ops": len(prof.get("ops", [])),
            "top_ops": [{"name": o["name"], "op": o["op"], "us": round(o["us"], 1), "share": round(o["us"] / total, 4),
                         "in_this_problem": o["name"] in mine, "shape": o.get("shape")} for o in top],
            "kinds": {k: {"count": len(v), "share": round(sum(x["us"] for x in v) / total, 4)}
                      for k, v in sorted(by_kind.items(), key=lambda kv: -sum(x["us"] for x in kv[1]))[:12]}}


def add_model_context(problem: Path, profiles: list) -> bool:
    man_p = problem / "manifest.json"
    man = json.loads(man_p.read_text())
    ctx = model_context(man, profiles)
    if ctx is None:
        return False
    man["model_context"] = ctx
    man_p.write_text(json.dumps(man, indent=1))
    return True
