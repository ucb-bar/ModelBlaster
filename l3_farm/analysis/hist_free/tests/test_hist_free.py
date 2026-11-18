import json
import os
from pathlib import Path

import pytest

from kernelblaster.agents import opt_multislot as om
from kernelblaster.agents import slot_prompt_features as spf
from kernelblaster.knowledge.outcomes import OutcomeRecord
from kernelblaster.knowledge.slots import SlotProblem
from kernelblaster.knowledge.v2.knowledge import Choice, KnowledgeV2
from kernelblaster.knowledge.v2.states import Classification

import golden_cases

HERE = Path(__file__).parent
FIX = next(p for p in [Path(os.environ.get("KB_FIXTURES", "")), HERE / "fixtures", HERE.parent / "tests" / "fixtures"]
           if (p / "mb_lenet").is_dir()) / "mb_lenet"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MB_SLOT_HISTORY", raising=False)
    monkeypatch.delenv("MB_STEP_FREEDOM", raising=False)


# ---------------------------------------------------------------- (a) golden: flags off == pre-change output
def test_golden_flags_off_equals_pristine_snapshot():
    p = SlotProblem(FIX)
    gold = json.load(open(HERE / "golden_off.json"))
    for name, (sid, kw) in golden_cases.cases(p).items():
        assert om.build_codegen_messages(problem=p, sid=sid, **kw) == gold[name], name


def test_golden_explicit_off_args_identical():
    p = SlotProblem(FIX)
    for name, (sid, kw) in golden_cases.cases(p).items():
        a = om.build_codegen_messages(problem=p, sid=sid, **kw)
        b = om.build_codegen_messages(problem=p, sid=sid, history_text="", freedom="none", **kw)
        assert a == b
    assert "## Technique to apply: vectorize_loads\nUse float4 loads.\n\nCurrent slot source" in a[1]["content"]


# ---------------------------------------------------------------- (e) env parsing
@pytest.mark.parametrize("val,k", [(None, 0), ("", 0), ("0", 0), ("-3", 0), ("abc", 0), ("2.5", 0), ("4", 4),
                                   (" 6 ", 6), ("999", spf.HISTORY_MAX_K)])
def test_history_env(monkeypatch, val, k):
    if val is not None:
        monkeypatch.setenv("MB_SLOT_HISTORY", val)
    assert spf.history_k() == k


@pytest.mark.parametrize("val,mode", [(None, "none"), ("", "none"), ("none", "none"), ("combo", "combo"),
                                      ("FREE", "free"), (" Free ", "free"), ("yes", "none"), ("1", "none")])
def test_freedom_env(monkeypatch, val, mode):
    if val is not None:
        monkeypatch.setenv("MB_STEP_FREEDOM", val)
    assert spf.step_freedom() == mode


def test_features_tag(monkeypatch):
    assert spf.features_tag() == {"slot_history": 0, "step_freedom": "none"}
    monkeypatch.setenv("MB_SLOT_HISTORY", "5")
    monkeypatch.setenv("MB_STEP_FREEDOM", "combo")
    assert spf.features_tag() == {"slot_history": 5, "step_freedom": "combo"}


# ---------------------------------------------------------------- (c) freedom blocks + parse
def test_freedom_blocks_in_prompt():
    p = SlotProblem(FIX)
    sid, kw = golden_cases.cases(p)["full"]
    off = om.build_codegen_messages(problem=p, sid=sid, **kw)[1]["content"]
    combo = om.build_codegen_messages(problem=p, sid=sid, freedom="combo", **kw)[1]["content"]
    free = om.build_codegen_messages(problem=p, sid=sid, freedom="free", **kw)[1]["content"]
    assert "## Primary technique: vectorize_loads\nUse float4 loads." in combo
    assert "ALSO apply up to two other compatible optimizations" in combo and "`// techniques: a, b, c`" in combo
    assert "## Technique to apply" not in combo and "## Technique to apply" not in free
    assert "## Suggested technique: vectorize_loads (a hint, not a constraint)\nUse float4 loads." in free
    assert "restructuring the kernel completely" in free and "signature and semantics are preserved" in free
    assert combo.endswith(off[off.index("Current slot source"):]) and free.endswith(off[off.index("Current slot source"):])
    # everything before the technique block is untouched
    pre = off[:off.index("## Technique to apply")]
    assert combo.startswith(pre) and free.startswith(pre)
    # system prompt is the same in every mode (it does not forbid combining)
    assert (om.build_codegen_messages(problem=p, sid=sid, freedom="free", **kw)[0]
            == om.build_codegen_messages(problem=p, sid=sid, **kw)[0])


def test_system_prompt_does_not_forbid_extra_changes():
    low = om.SYSTEM_PROMPT.lower()
    for bad in ("only the technique", "only apply", "do not change anything else", "apply exactly", "nothing else"):
        assert bad not in low


@pytest.mark.parametrize("code,want", [
    ("// techniques: vectorize_loads, shared_mem_tiling, unroll\n__global__ void k(){}", ["vectorize_loads", "shared_mem_tiling", "unroll"]),
    ("// Techniques : `a` ; 'b'; c.\nx", ["a", "b", "c"]),
    ("#include <x>\n  //techniques:tiling,tiling, tiling\n", ["tiling"]),
    ("__global__ void k(){}", []),
    (None, []),
    ("", []),
    ("// techniques:\nx", []),
    ("// techniques: , ;; ,\nx", []),
    ("// techniques: none\nx", []),
    ("// technique list: a, b\nx", []),
    ("/* techniques: a */\nx", []),
    ("// techniques: " + "x" * 80 + ", ok\n", ["ok"]),
    ("// techniques: " + ", ".join(f"t{i}" for i in range(20)) + "\n", [f"t{i}" for i in range(8)]),
    ("// techniques: a\n// techniques: b\n", ["a"]),
])
def test_parse_declared(code, want):
    assert spf.parse_declared_techniques(code) == want


# ---------------------------------------------------------------- (b) history
def _row(i, tech, status="improved", sp=1.1, before=100.0, after=None, impl=None, parent=None, err=None, dsum=None,
         md=None, mb=None, **v2):
    after = after if after is not None else (before / sp if sp else None)
    return {"id": f"o{i}", "seq": i, "technique": tech, "status": status, "speedup": sp, "metric_before": before,
            "metric_after": after, "impl_id": impl, "parent_impl_id": parent, "error": err, "diff_summary": dsum,
            "metrics_delta": md or {}, "v2": {"metrics_before": mb or {}, **v2}}


ROWS = [
    _row(1, "tiling", "regressed", 0.8, 100, 125, impl="aaaaaaaa11", parent=None, dsum="+10/-2 lines"),
    _row(2, "vectorize_loads", "improved", 1.25, 100, 80, impl="bbbbbbbb22", parent=None, dsum="+5/-5 lines",
         md={"dram_throughput": 8.0, "achieved_occupancy": -1.0, "l1_hit_rate": 30.0, "x": 0.0},
         mb={"dram_throughput": {"v": 40.0}, "achieved_occupancy": {"v": 50.0}, "l1_hit_rate": {"v": 300.0}}),
    _row(3, "tiling", "compile_error", None, 80, None, err="kernel.cu(12): error: expected ';'\nmore", parent="bbbbbbbb22"),
    _row(4, "unroll", "neutral", 1.0, 80, 80, impl="cccccccc33", parent="bbbbbbbb22"),
    _row(5, "unroll", "regressed", 0.9, 80, 88, impl="dddddddd44", parent="bbbbbbbb22"),
]


def test_history_off_and_empty():
    assert spf.slot_history_text(ROWS, 0) == ""
    assert spf.slot_history_text([], 5) == ""


def test_history_content_and_order():
    t = spf.slot_history_text(ROWS, 3)
    lines = t.splitlines()
    assert lines[0] == "Attempt history on this slot (most recent 3 attempts, oldest first):"
    assert lines[1].startswith("(2 older attempts omitted)")
    assert [l.split()[1] for l in lines[2:5]] == ["#3", "#4", "#5"] or lines[2].startswith("- #3")
    assert lines[2].startswith("- #3 tiling [compile_error]; no timing") and 'error: "kernel.cu(12): error: expected \';\'"' in lines[2]
    assert "more" not in lines[2] and "impl bbbbbbbb -> none" in lines[2]
    assert lines[3].startswith("- #4 unroll [neutral]; 1.00x vs parent") and "impl bbbbbbbb -> cccccccc" in lines[3]
    assert lines[4].startswith("- #5 unroll [regressed]; 0.90x vs parent")
    assert lines[5] == "Current best kernel for this slot: impl bbbbbbbb, 1.25x over the seed."
    assert lines[6] == "Dead ends (tried >=2 times, never improved): tiling (2 tries), unroll (2 tries)"
    assert len(lines) == 7


def test_history_full_row_metrics_and_diff():
    t = spf.slot_history_text(ROWS, 10)
    assert "(older" not in t
    row2 = next(l for l in t.splitlines() if l.startswith("- #2"))
    # relative ranking: l1_hit_rate (+10%) vs dram (+20%) vs occupancy (-2%): top-3, zero deltas dropped
    assert "moved: dram_throughput +8, l1_hit_rate +30, achieved_occupancy -1" in row2
    assert "diff: +5/-5 lines" in row2 and "1.25x vs parent" in row2 and "impl seed -> bbbbbbbb" in row2
    assert t.splitlines()[1].startswith("- #1 tiling [regressed]; 0.80x vs parent")


def test_history_no_improvement_yet():
    t = spf.slot_history_text(ROWS[:1], 4)
    assert "Current best kernel for this slot: the seed (no verified improvement yet)" in t
    assert "Dead ends" not in t


def test_history_deterministic_and_order_insensitive():
    a = spf.slot_history_text(ROWS, 5)
    assert a == spf.slot_history_text(list(reversed(ROWS)), 5) == spf.slot_history_text(ROWS, 5)


def test_history_budget_truncates_oldest_first():
    rows = [_row(i, f"tech{i % 3}", "regressed", 0.9, 100, 111, impl=f"{i:08d}", parent="p" * 8,
                 dsum="d" * 300, err="e" * 400) for i in range(1, 40)]
    t = spf.slot_history_text(rows, 40)
    assert len(t) <= spf.HISTORY_BUDGET_CHARS
    assert "older attempts omitted" in t
    shown = [int(l.split()[1][1:]) for l in t.splitlines() if l.startswith("- #")]
    assert shown == sorted(shown) and shown[-1] == 39 and shown[0] > 1      # newest kept, oldest dropped
    assert shown == list(range(shown[0], 40))                               # contiguous suffix
    assert "Current best kernel" in t and "Dead ends" in t                  # fixed lines survive
    assert len(spf.slot_history_text(rows, 3)) < 1500


def test_history_hypothesis_name():
    r = _row(1, "tiling", kind="hypothesis", item_id="hyp_smem_reuse", impl="x" * 10)
    assert "- #1 hypothesis hyp_smem_reuse (tiling) [improved]" in spf.slot_history_text([r], 2)


def _kv(tmp_path):
    return KnowledgeV2(tmp_path / "k", seed_mined=False)


def _rec(kv, i, tech, pid="p", slot="s", run="run1", status="improved", sp=1.2, extra=None, **kw):
    rec = OutcomeRecord(id=f"{tech}_{i}_{pid}_{slot}_{run}", technique=tech, status=status, target="cuda", arch="sm_89",
                        problem_id=pid, slot=slot, op_class="conv", run_id=run, speedup=sp, metric_before=10.0,
                        metric_after=10.0 / sp if sp else None, **kw)
    return kv.record(rec, choice=Choice(tech, "technique", ""), reward=0.1, cls=Classification(["compute"], "classified"),
                     before={}, after=None, op_class="conv", dtype="fp32", extra=extra)


def test_history_rows_scoping_and_visibility(tmp_path):
    kv = _kv(tmp_path)
    _rec(kv, 1, "tiling")
    _rec(kv, 2, "unroll", slot="other")                 # other slot
    _rec(kv, 3, "unroll", pid="q")                      # other problem
    _rec(kv, 4, "vec", run="run0")                      # other run
    rows = spf.history_rows(kv, "p", "s", "run1")
    assert [r["technique"] for r in rows] == ["tiling"]
    assert [r["technique"] for r in spf.history_rows(kv, "p", "s")] == ["tiling", "vec"]
    # committed attempts show up immediately, uncommitted ones never (same table as already_tried)
    _rec(kv, 5, "vec")
    assert [r["technique"] for r in spf.history_rows(kv, "p", "s", "run1")] == ["tiling", "vec"]
    assert "vec" in kv.already_tried("p", "s") and "unroll" not in kv.already_tried("p", "s")
    t = spf.slot_history_text(spf.history_rows(kv, "p", "s", "run1"), 5)
    assert "#1 tiling" in t and "#2 vec" in t and "unroll" not in t


# ---------------------------------------------------------------- (d) record() extra
def test_record_extra_declared_techniques(tmp_path):
    kv = _kv(tmp_path)
    v2 = _rec(kv, 1, "tiling", extra={"lineage": ["tiling"], "declared_techniques": ["tiling", "unroll"]})
    assert v2["declared_techniques"] == ["tiling", "unroll"] and v2["lineage"] == ["tiling"]
    row = kv.outcomes.rows("problem_id=?", ("p",))[0]
    assert row["v2"]["declared_techniques"] == ["tiling", "unroll"]
    assert row["technique"] == "tiling" and row["v2"]["item_id"] == "tiling"       # credit stays on the primary
    v2b = _rec(kv, 2, "unroll", extra={"declared_techniques": []})
    assert v2b["declared_techniques"] == []
    v2c = _rec(kv, 3, "vec")                                                   # flag off: key absent
    assert "declared_techniques" not in v2c
