"""KernelBlaster cache for RISC-V backends (k3-work TASK-016): kb_ingest --target riscv,
kb_riscv.codegen_source's shape guard, the board re-verify marks, and that the CUDA /
OpenCL cache paths do not see riscv entries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from modelblaster.pipeline import kb_ingest, kb_riscv  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402
from modelblaster.pipeline.cuda_dtypes import cuda_signature  # noqa: E402

SPEC = KERNEL_SPECS["conv2d_s8"]


def _shape(ic, ih, oc):
    return {"N": 1, "IC": ic, "IH": ih, "IW": ih, "OC": oc, "OH": ih, "OW": ih,
            "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1}


S1, S2, S3 = _shape(3, 16, 8), _shape(8, 8, 16), _shape(16, 4, 16)


def _winners(tmp: Path, shapes, arch="ime_k3_x100", algo="x100_vmadot_4x4x8", target="riscv"):
    (tmp / "w.c").write_text("/* algorithm: x100_vmadot_4x4x8 */\nvoid kernel_conv2d_s8(void) {}\n")
    w = {"format": "kb-winners/1", "winners": [{
        "op": "conv2d_s8", "id": "t1", "target": target, "arch": arch, "algorithm": algo,
        "signature_sha": kb_ingest.signature_sha(cuda_signature(SPEC.signature)),
        "shapes": shapes, "file": "w.c", "slot_us": 10.0, "speedup_vs_seed": 1.2}]}
    (tmp / "winners.json").write_text(json.dumps(w))
    return tmp / "winners.json"


def test_ingest_and_guard(tmp_path):
    cache = kb_ingest.CudaCache(tmp_path / "cache", target="riscv")
    rep = kb_ingest.ingest(_winners(tmp_path, [S1, S2]), cache, KERNEL_SPECS)
    assert [r["status"] for r in rep] == ["added"]
    e = cache.entries[0]
    assert e["target"] == "riscv" and e["file"].endswith(".c") and e["source"]["algorithm"] == "x100_vmadot_4x4x8"
    root = tmp_path / "cache"
    got = kb_riscv.codegen_source("ime_k3_x100", SPEC, "x100_vmadot_4x4x8", [S1, S2], "net", root)
    assert got and got.endswith(".c")
    # a shape outside the guard, another backend or algorithm: the curated kernel
    assert kb_riscv.codegen_source("ime_k3_x100", SPEC, "x100_vmadot_4x4x8", [S1, S3], "net", root) is None
    assert kb_riscv.codegen_source("rvv_k3", SPEC, "x100_vmadot_4x4x8", [S1], "net", root) is None
    assert kb_riscv.codegen_source("ime_k3_x100", SPEC, "rvv_vsmul_vnclip", [S1], "net", root) is None
    assert kb_riscv.has_entries("ime_k3_x100", root) and not kb_riscv.has_entries("rvv_k3", root)


def test_off_switch(tmp_path, monkeypatch):
    cache = kb_ingest.CudaCache(tmp_path / "cache", target="riscv")
    kb_ingest.ingest(_winners(tmp_path, [S1]), cache, KERNEL_SPECS)
    monkeypatch.setenv("MB_KB_RISCV", "0")
    assert not kb_riscv.has_entries("ime_k3_x100", tmp_path / "cache")
    assert kb_riscv.codegen_source("ime_k3_x100", SPEC, "x100_vmadot_4x4x8", [S1], "n", tmp_path / "cache") is None


def test_mark_and_rejected_entries_are_skipped(tmp_path):
    root = tmp_path / "cache"
    kb_ingest.ingest(_winners(tmp_path, [S1]), kb_ingest.CudaCache(root, target="riscv"), KERNEL_SPECS)
    src = kb_riscv.codegen_source("ime_k3_x100", SPEC, "x100_vmadot_4x4x8", [S1], "dronet", root)
    gen = tmp_path / "gen"
    gen.mkdir()
    (gen / "kernel_picks.json").write_text(json.dumps({"schema_version": 1, "target": "ime_k3_x100", "picks": {
        "conv2d_s8": {"source": kb_riscv.SOURCE_LABEL, "algorithm": "x100_vmadot_4x4x8", "path": src},
        "add_s8": {"source": "curated[rvv]", "algorithm": "x", "path": "y"}}}))
    r = kb_riscv.mark(gen, "=== MODELBLASTER_VERIFY === max_abs_err=0 max_rel_err=0 n=2\n", "dronet", root)
    assert r and r[0]["passed"]
    c = kb_ingest.CudaCache(root, target="riscv")
    assert c.entries[0]["model_verified"] == ["dronet"]
    kb_riscv.mark(gen, "=== MODELBLASTER_VERIFY === max_abs_err=3 max_rel_err=1 n=2\n", "other", root)
    assert kb_ingest.CudaCache(root, target="riscv").entries[0]["rejected"] == ["other"]
    assert kb_riscv.codegen_source("ime_k3_x100", SPEC, "x100_vmadot_4x4x8", [S1], "other", root) is None


def test_cuda_and_opencl_caches_ignore_riscv_entries(tmp_path):
    root = tmp_path / "cache"
    kb_ingest.ingest(_winners(tmp_path, [S1]), kb_ingest.CudaCache(root, target="riscv"), KERNEL_SPECS)
    for target in ("cuda", "opencl"):
        c = kb_ingest.CudaCache(root, target=target)
        assert c.lookup("kernel_conv2d_s8", SPEC.signature, "ime_k3_x100", [S1]) is None
        rep = kb_ingest.ingest(_winners(tmp_path, [S1]), c, KERNEL_SPECS)
        assert rep[0]["status"] == "other-target"
    assert kb_ingest._EXT == {"cuda": ".cu", "opencl": ".cpp", "riscv": ".c"}


def test_repo_cache_entry_is_well_formed():
    c = kb_ingest.CudaCache(kb_ingest.DEFAULT_RISCV_CACHE, target="riscv")
    for e in c.entries:
        assert (c.root / e["file"]).exists() and e["arch"] in ("ime_k3_x100", "ime_k3_a100", "rvv_k3", "rvv_k3_a100")
        assert e["shape_guard"] and (e.get("source") or {}).get("algorithm")
