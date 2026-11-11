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


def test_every_x60_affined_algorithm_is_k3_affined():
    # The picker matches target_affinity by exact backend name; a miss silently
    # selects the scalar reference.
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    for spec in KERNEL_SPECS.values():
        for a in spec.algorithms:
            if "rvv_x60" in a.target_affinity:
                assert "rvv_k3" in a.target_affinity, (spec.op, a.name)
