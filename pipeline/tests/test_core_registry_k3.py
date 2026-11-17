"""K3 registry: distinct X100/A100 kinds, AI-thread cpus, K1 unaffected."""

import os
import subprocess
import sys

from modelblaster.pipeline import core_registry as cr

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K3 = os.path.join(REPO, "cores", "spacemit_k3.json")
K1 = os.path.join(REPO, "cores", "spacemit_k1.json")


def test_kinds_distinct_eight_harts_each():
    reg = cr.load(K3)
    assert reg.kinds() == {"x100", "a100"}
    x = [h for c in reg.by_kind["x100"] for h in c.harts]
    a = [h for c in reg.by_kind["a100"] for h in c.harts]
    assert x == list(range(0, 8))
    assert a == list(range(8, 16))


def test_capabilities_shared_and_no_ime():
    reg = cr.load(K3)
    caps = {c.capabilities for c in reg.cores}
    assert len(caps) == 1
    assert not any(c.startswith("ime_") for c in next(iter(caps)))
    k1_rvv = {c for c in cr.load(K1).by_kind["rvv_c1"][0].capabilities}
    assert set(next(iter(caps))) == k1_rvv


def test_ai_thread_cpus():
    reg = cr.load(K3)
    assert reg.ai_thread_cpus == tuple(range(8, 16))
    assert reg.ai_cpus_str() == "8-15"


def test_k1_has_no_ai_cpus():
    reg = cr.load(K1)
    assert reg.ai_thread_cpus == ()
    assert reg.ai_cpus_str() == ""


def test_compact_cpulist():
    assert cr.compact_cpulist([]) == ""
    assert cr.compact_cpulist([3, 0, 2, 8, 9]) == "0,2-3,8-9"


def test_cli_prints_compact_string():
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, "pipeline", "core_registry.py"),
         K3, "--ai-cpus"], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "8-15"


def test_rvv_k3_backend_matches_x60():
    from modelblaster.pipeline.backends import BACKENDS
    k3, x60 = BACKENDS["rvv_k3"], BACKENDS["rvv_x60"]
    assert k3.kernel_cflags == x60.kernel_cflags
    assert k3.curated_aliases == ("rvv",)
    assert k3.verify_method == x60.verify_method


def test_rvv_k3_a100_backend():
    from modelblaster.pipeline.backends import BACKENDS
    a100, k3 = BACKENDS["rvv_k3_a100"], BACKENDS["rvv_k3"]
    assert "-march=rv64gcv_zvl1024b_zfh_zvfh" in a100.kernel_cflags
    assert not any("zvl256b" in f for f in a100.kernel_cflags)
    # every other flag identical to rvv_k3
    rest = lambda b: [f for f in b.kernel_cflags if not f.startswith("-march=")]
    assert rest(a100) == rest(k3)
    assert a100.curated_aliases == ("rvv",)
    assert a100.verify_method == k3.verify_method


def test_registry_core_backends():
    reg = cr.load(K3)
    assert {c.backend for c in reg.by_kind["x100"]} == {"rvv_k3"}
    assert {c.backend for c in reg.by_kind["a100"]} == {"ime_k3_a100"}
    assert {c.backend for c in cr.load(K1).cores} == {""}


def test_registry_rejects_mixed_backends_in_one_kind(tmp_path):
    import json
    import pytest
    doc = {"cores": [
        {"name": "a", "kind": "k", "capabilities": ["x"], "backend": "b1"},
        {"name": "b", "kind": "k", "capabilities": ["x"], "backend": "b2"}]}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="different backends"):
        cr.load(str(p))


def test_a100_target_dir_and_vector_backend():
    # generate_kernels probes a backend's curated_aliases after its own
    # kernels/<backend>/ dir; without "rvv" here every op the A100 has no tuned
    # kernel for would fall back to the scalar reference.
    from modelblaster.pipeline.backends import BACKENDS
    assert "rvv" in BACKENDS["rvv_k3_a100"].curated_aliases
    assert os.path.isdir(os.path.join(REPO, "kernels", "rvv_k3_a100"))


def test_every_x60_affined_algorithm_is_k3_affined():
    # The picker matches target_affinity by exact backend name; a miss silently
    # selects the scalar reference.
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    for spec in KERNEL_SPECS.values():
        for a in spec.algorithms:
            if "rvv_x60" in a.target_affinity:
                assert "rvv_k3" in a.target_affinity, (spec.op, a.name)
                assert "rvv_k3_a100" in a.target_affinity, (spec.op, a.name)


def test_ime_k3_a100_backend():
    from modelblaster.pipeline.backends import BACKENDS, backend_lineage
    ime, a100 = BACKENDS["ime_k3_a100"], BACKENDS["rvv_k3_a100"]
    assert "-march=rv64gcv_zvl1024b_zfh_zvfh" in ime.kernel_cflags
    assert not any("zvl256b" in f for f in ime.kernel_cflags)
    assert backend_lineage("ime_k3_a100") == ("ime_k3_a100", "rvv_k3_a100", "rvv")
    assert ime.verify_method == a100.verify_method
    assert os.path.isdir(os.path.join(REPO, "kernels", "ime_k3_a100"))


def test_ime_k3_a100_algorithms_and_layout():
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    from modelblaster.pipeline.generate_skeleton import _conv_weight_layout_for_backend
    for op in ("conv2d_batchnorm2d_silu_s8", "conv2d_batchnorm2d_s8",
               "conv2d_s8", "linear_s8", "matmul_s8"):
        names = [a.name for a in KERNEL_SPECS[op].algorithms
                 if "ime_k3_a100" in a.target_affinity]
        assert names == ["ime_vmadot_8x16x8"], op
    # conv weights stay IHWOC (inherited via the rvv lineage)
    assert _conv_weight_layout_for_backend("ime_k3_a100") == \
        _conv_weight_layout_for_backend("rvv_k3_a100") == "ihwoc"


_IME_CONV_OPS = ("conv2d_batchnorm2d_silu_s8", "conv2d_batchnorm2d_s8", "conv2d_s8")


def test_ime_k3_a100_conv_kernels_exist():
    for op in _IME_CONV_OPS:
        p = os.path.join(REPO, "kernels", "ime_k3_a100",
                         f"ime_k3_a100_{op}_ime_vmadot_8x16x8.c")
        assert os.path.exists(p), p
    assert os.path.exists(os.path.join(
        REPO, "kernels", "ime_k3_a100", "ime_k3_a100_common.h"))


def test_ime_k3_a100_includes_own_dir():
    from modelblaster.pipeline.backends import BACKENDS
    flags = BACKENDS["ime_k3_a100"].resolved_kernel_cflags(REPO)
    assert f"-I{REPO}/kernels/ime_k3_a100" in flags
    # the shared rvv include dir (mb_rvv_vxrm_compat.h) is still there
    assert f"-I{REPO}/kernels/rvv" in flags


def test_ime_conv_kernels_cross_compile_without_whole_register_spills(tmp_path):
    """The IME conv kernels must compile at zvl1024b and contain no
    vs*r.v / vl*r.v whole-register spills (Phase B hazard on the A100)."""
    import re
    import shutil
    import subprocess
    cross = os.environ.get("CROSS", "")
    gcc = (cross + "gcc") if cross else shutil.which("riscv64-unknown-linux-gnu-gcc")
    if not gcc or not shutil.which(gcc):
        import pytest
        pytest.skip("no riscv cross gcc")
    from modelblaster.pipeline.backends import BACKENDS
    flags = list(BACKENDS["ime_k3_a100"].resolved_kernel_cflags(REPO))
    for op in _IME_CONV_OPS:
        src = os.path.join(REPO, "kernels", "ime_k3_a100",
                           f"ime_k3_a100_{op}_ime_vmadot_8x16x8.c")
        obj = str(tmp_path / f"{op}.o")
        subprocess.run([gcc, "-O2", *flags, "-c", src, "-o", obj], check=True)
        dis = subprocess.run([gcc.replace("gcc", "objdump"), "-d", obj],
                             check=True, capture_output=True, text=True).stdout
        assert not re.search(r"\bvs\dr\.v\b|\bvl\dre?\d*\.v\b", dis), op
