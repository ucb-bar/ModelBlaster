"""kb_graph_opt passes the beam-search flags to run_multislot only when search != independent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelblaster.pipeline import kb_graph_opt as G, kb_resume  # noqa: E402


def _cmd(**kw):
    return G.KBRunner(G.Config(kb=Path("/kb"), **kw), Path("/k")).command([Path("/p")], Path("/o"))


def test_flags_only_for_beam():
    assert not {"--search", "--beam-width", "--beam-expand"} & set(_cmd())
    c = _cmd(search="beam", beam_width=3, beam_expand=2)
    i = c.index("--search")
    assert c[i:i + 6] == ["--search", "beam", "--beam-width", "3", "--beam-expand", "2"]


def test_round_key_unchanged_for_independent():
    a = kb_resume.round_key(G.Config(kb=Path("/kb")), [])
    assert "search" not in a and "beam_width" not in a
    assert kb_resume.round_key(G.Config(kb=Path("/kb"), search="beam"), [])["search"] == "beam"


def test_refill_and_exact_arm_flags():
    c = _cmd(search="beam", beam_width=2, beam_expand=2)
    assert "--beam-refill" not in c                                  # default off: not passed
    i = c.index("--search")
    assert c[i:i + 6] == ["--search", "beam", "--beam-width", "2", "--beam-expand", "2"]
    j = _cmd(search="beam", beam_refill="on").index("--beam-refill")
    assert _cmd(search="beam", beam_refill="on")[j:j + 2] == ["--beam-refill", "on"]
    assert "--beam-refill" not in _cmd(beam_refill="on")             # independent ignores beam flags
    assert "beam_refill" not in kb_resume.round_key(G.Config(kb=Path("/kb"), search="beam"), [])
    assert kb_resume.round_key(G.Config(kb=Path("/kb"), search="beam", beam_refill="on"), [])["beam_refill"] == "on"


def test_cli_flags_parse(monkeypatch):
    """--search beam --beam-width 2 --beam-expand 2 [--steps 2] reach Config through the generated CLI flags."""
    from dataclasses import asdict
    names = {"--" + k.replace("_", "-") for k in asdict(G.Config(kb=Path("."))) if k not in G._MANUAL_FIELDS}
    assert {"--search", "--beam-width", "--beam-expand", "--beam-refill", "--steps"} <= names


def test_steps_with_beam():
    c = _cmd(search="beam", beam_width=2, beam_expand=2, steps=2)
    assert c[c.index("--steps") + 1] == "2" and c[c.index("--search") + 1] == "beam"
