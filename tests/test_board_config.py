"""Board defaults, topo tags and per-core-type hw names (no board needed)."""

import os

import pytest

from modelblaster.validation import board_config as bc


def test_k1_defaults_unchanged():
    c = bc.resolve(None, {})
    assert (c.host, c.remote_root) == ("k1", "/root/mb_k1")
    assert (c.registry, c.cpu_p_kind, c.cpu_e_kind) == (
        "cores/spacemit_k1.json", "rvv", "rvv_c1")
    assert (c.backend, c.profile_cpu) == ("rvv_x60", "spacemit_x60")
    assert (c.out_root_single, c.out_root_xpurt) == ("build/k1", "build/k1_xpurt")
    assert c.clock_mhz == 24.0 and not c.topo_relative_default
    assert c.launch_cpus == ""


def test_k3_defaults():
    c = bc.resolve("k3", {})
    assert (c.host, c.remote_root) == ("spacemitk3", "/home/sudospace/mb_k3")
    assert c.registry == "cores/spacemit_k3.json"
    assert (c.cpu_p_kind, c.cpu_e_kind) == ("x100", "a100")
    assert (c.backend, c.profile_cpu) == ("rvv_k3", "spacemit_k3")
    assert c.out_root_xpurt == "build/k3_xpurt"
    assert c.clock_mhz == 24.0 and c.topo_relative_default
    assert c.launch_cpus == "0-7"


def test_board_from_env_and_overrides():
    env = {"MODELBLASTER_BOARD": "k3", "MODELBLASTER_K3_HOST": "x",
           "MODELBLASTER_K1_HOST": "ignored"}
    c = bc.resolve(None, env)
    assert c.name == "k3" and c.host == "x"
    c1 = bc.resolve("k1", {"MODELBLASTER_K1_HOST": "h", "MODELBLASTER_K1_REMOTE_ROOT": "/r"})
    assert (c1.host, c1.remote_root) == ("h", "/r")


def test_unknown_board():
    with pytest.raises(ValueError):
        bc.resolve("k9", {})


def test_parse_cpulist_spellings_agree():
    assert bc.parse_cpulist("8-11") == bc.parse_cpulist("8,9,10,11") \
        == bc.parse_cpulist("8_9_10_11") == [8, 9, 10, 11]
    with pytest.raises(ValueError):
        bc.parse_cpulist("a")


def test_topo_tag():
    assert bc.topo_tag([8, 9, 10, 11], True) == "0_1_2_3"
    assert bc.topo_tag([8, 9, 10, 11], False) == "8_9_10_11"
    assert bc.topo_tag([4], True) == "0"
    assert bc.topo_tag(list(range(8)), True) == "0_1_2_3_4_5_6_7"
    with pytest.raises(ValueError):
        bc.topo_tag([], True)


def test_hw_name_k3():
    k3 = bc.resolve("k3", {})
    assert bc.derive_hw_name(k3, [0, 1], "rvv_k3") == "rvv_k3_x100"
    assert bc.derive_hw_name(k3, [8, 9, 10, 11], "rvv_k3") == "rvv_k3_a100"
    assert bc.derive_hw_name(k3, [8], "rvv_k3", explicit="custom") == "custom"
    with pytest.raises(ValueError):
        bc.derive_hw_name(k3, [6, 7, 8], "rvv_k3")
    with pytest.raises(ValueError):
        bc.derive_hw_name(k3, [16], "rvv_k3")


def test_hw_name_core_type_target_used_as_is():
    k3 = bc.resolve("k3", {})
    assert bc.derive_hw_name(k3, [8, 9], "rvv_k3_a100") == "rvv_k3_a100"
    assert bc.derive_hw_name(k3, [0], "rvv_k3_x100") == "rvv_k3_x100"
    assert bc.derive_hw_name(k3, [8], "rvv_k3_a100", explicit="c") == "c"


def test_hw_name_a100_target_on_x100_cpus_is_error():
    k3 = bc.resolve("k3", {})
    with pytest.raises(ValueError, match="illegal on the X100"):
        bc.derive_hw_name(k3, [0, 1], "rvv_k3_a100")
    with pytest.raises(ValueError):
        bc.derive_hw_name(k3, [8], "rvv_k3_x100")
    with pytest.raises(ValueError):  # mixed set, even with an _a100 target
        bc.derive_hw_name(k3, [7, 8], "rvv_k3_a100")


def test_k3_per_core_type_backends():
    c = bc.resolve("k3", {})
    assert bc.backend_for_core_type(c, "x100") == "rvv_k3"
    assert bc.backend_for_core_type(c, "a100") == "ime_k3_a100"
    assert bc.backend_for_core_type(bc.resolve("k1", {}), "rvv_c1") == "rvv_x60"
    out = bc.shell_exports(c)
    assert "BOARD_BACKEND_P=rvv_k3\n" in out + "\n"
    assert "BOARD_BACKEND_E=ime_k3_a100" in out
    k1 = bc.shell_exports(bc.resolve("k1", {}))
    assert "BOARD_BACKEND_P=rvv_x60" in k1 and "BOARD_BACKEND_E=rvv_x60" in k1


def test_hw_name_k1_is_none():
    assert bc.derive_hw_name(bc.resolve("k1", {}), [0, 5], "rvv_x60") is None


def test_core_types_match_registry():
    from modelblaster.pipeline import core_registry as cr
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg = cr.load(os.path.join(repo, "cores", "spacemit_k3.json"))
    cfg = bc.resolve("k3", {})
    for kind, harts in cfg.core_types:
        assert tuple(h for c in reg.by_kind[kind] for h in c.harts) == harts
        # the registry's per-core backend must agree with the board default
        assert {c.backend for c in reg.by_kind[kind]} == {
            bc.backend_for_core_type(cfg, kind)}


def test_shell_exports_roundtrip():
    out = bc.shell_exports(bc.resolve("k3", {}))
    assert "BOARD_HOST=spacemitk3" in out and "BOARD_LAUNCH_CPUS=0-7" in out
