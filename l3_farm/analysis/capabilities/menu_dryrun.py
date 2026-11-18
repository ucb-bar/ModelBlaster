"""A2 item 2 evidence (zero LLM): what the slot optimiser offers 41_GRUBidirectional's GRU slots with the op class
the sweep used (gemm) vs the opt-in recurrent class: v1 bandit menu, v2 ontology candidates per state chain, the
roofline state + numbers that go into the prompt, and whether the recurrent notes are in the codegen prompt.

Run: PYTHONPATH=<KernelBlaster>/src python menu_dryrun.py [out.json]
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KB = ROOT / "ModelBlaster" / "third_party" / "KernelBlaster" / "src"
sys.path.insert(0, str(KB))

from kernelblaster.agents import opt_multislot as om  # noqa: E402
from kernelblaster.knowledge.slots import SlotProblem  # noqa: E402
from kernelblaster.knowledge.v2.knowledge import KnowledgeV2 as Knowledge  # noqa: E402

PROB = ROOT / "l3_farm" / "results" / "41_GRUBidirectional" / "work" / "final" / "41_GRUBidirectional" / "problem"
# per-slot GPU time of the sweep's last round (r02 profile), us
SLOT_US = {"gru_seq_f16__v3": 889.5, "gru_seq_f16__v2": 569.4, "gru_seq_f16__v0": 2774.9, "gru_seq_f16__v1": 5025.5}
CHAINS = [["latency_bound"], ["underutilized", "underutilized_compute"], ["memory_bound"], ["compute_bound"]]


def problem_with_class(cls: str) -> SlotProblem:
    d = Path(tempfile.mkdtemp(prefix="a2menu_"))
    man = json.loads((PROB / "manifest.json").read_text())
    for s in man["slots"]:
        if s["op"].startswith("gru_seq"):
            s["op_class"] = cls
    (d / "manifest.json").write_text(json.dumps(man))
    for f in PROB.iterdir():
        if f.name != "manifest.json":
            (d / f.name).symlink_to(f)
    for f in (man["driver"], man["data"]):           # not kept with the final problem; not needed here
        if not (d / f).exists():
            (d / f).write_text("")
    return SlotProblem(d)


def main(out: str | None = None):
    kv = Knowledge(Path(tempfile.mkdtemp(prefix="a2kv_")), seed_mined=False)
    tkey = "cuda:sm_89"
    kv.registry.ensure(tkey, peaks=(91600.0, 864.0))
    res = {}
    for cls in ("gemm", "recurrent"):
        p = problem_with_class(cls)
        r = {"v1_menu": [t for t, _ in om.techniques_for(cls, om.TECHNIQUES, arch="89", dtype="fp16")]}
        r["v2_candidates"] = {"|".join(ch): [t.id for t in kv.candidates(
            target_key=tkey, op_class=cls, dtype="fp16", chain=ch, vec={}, op_kind="gru_seq_f16")[0]]
            for ch in CHAINS}
        r["roofline"] = {sid: om.slot_state(p, sid, us, "89", (91600.0, 864.0)) for sid, us in SLOT_US.items()}
        msgs = om.build_codegen_messages(problem=p, sid="gru_seq_f16__v0", source="// src", technique="t", arch="89",
                                         technique_desc="d", state="s", state_numbers={}, slot_us=1.0,
                                         partition_us=2.0, neighbours=[], examples=[], adapt=None,
                                         knowledge_text="", already_tried="", best_note="")
        r["prompt_has_recurrent_notes"] = "Recurrent op notes" in msgs[1]["content"]
        res[cls] = r
    txt = json.dumps(res, indent=1)
    if out:
        Path(out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
