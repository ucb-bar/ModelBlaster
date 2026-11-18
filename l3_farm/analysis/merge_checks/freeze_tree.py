#!/usr/bin/env python3
"""Build a frozen tarball for the merge checks (read-only on the live trees; writes only under merge_checks/frozen/).

  freeze_tree.py pre-snapshot            rebuild the PRE tree from merge_prep (snapshot 20261025-1703): `git archive` of ModelBlaster HEAD 56f34bc
                                         and KernelBlaster HEAD c453100 + the *_tracked_*.patch + *_untracked_*.tgz. Output
                                         frozen/mb_pre_20261025-1703.tar.gz (+ .sha256). Already built; this documents how.
  freeze_tree.py live LABEL              the live ModelBlaster tree as farm.stage does (tar of ModelBlaster/, no .git / __pycache__ / runs),
                                         e.g. `live post` after the integration pass has been committed; stamp = now.
  freeze_tree.py live LABEL --git REV    `git archive REV` of ModelBlaster (+ the submodule at its recorded commit) instead of the working tree
                                         (a clean, committed state: preferred for the post-merge tree).

The label becomes the run-pack directory (runs/<label>/...) and the tree directory on the nodes (/tmp/krisd/mc/<label>).
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mc_common as C  # noqa: E402

MB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster"
PREP = C.L3 / "analysis" / "merge_prep"
OUT = HERE / "frozen"
ENV = {"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}


def sh(*a, **k):
    subprocess.run(list(map(str, a)), check=True, env=ENV, **k)


def tar(workdir: Path, label: str, stamp: str):
    OUT.mkdir(exist_ok=True)
    dst = OUT / f"mb_{label}_{stamp}.tar.gz"
    sh("tar", "czf", dst, "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=ModelBlaster/gpu_eval/runs",
       "--exclude=ModelBlaster/gpu_eval/ablations", "-C", workdir, "ModelBlaster")
    sha = C.sha256_file(dst)
    (OUT / f"mb_{label}_{stamp}.sha256").write_text(f"{sha}  {dst.name}\n")
    print(dst, sha, dst.stat().st_size // 2**20, "MB")


def pre_snapshot():
    stamp = (PREP / "STAMP").read_text().strip()
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "ModelBlaster"
        w.mkdir()
        kb = w / "third_party" / "KernelBlaster"
        kb.mkdir(parents=True)
        for repo, rev, dst, pre in ((MB, "56f34bc", w, "mb"), (MB / "third_party" / "KernelBlaster", "c453100", kb, "kb")):
            p = subprocess.Popen(["git", "-C", str(repo), "archive", rev], stdout=subprocess.PIPE, env=ENV)
            sh("tar", "x", "-C", dst, stdin=p.stdout)
            sh("git", "apply", "--whitespace=nowarn", PREP / f"{pre}_tracked_{stamp}.patch", cwd=dst)
            sh("tar", "xzf", PREP / f"{pre}_untracked_{stamp}.tgz", "-C", dst)
        tar(Path(td), "pre", stamp)


def live(label, rev=None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if not rev:
        tar(MB.parent, label, stamp)
        return
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "ModelBlaster"
        w.mkdir()
        p = subprocess.Popen(["git", "-C", str(MB), "archive", rev], stdout=subprocess.PIPE, env=ENV)
        sh("tar", "x", "-C", w, stdin=p.stdout)
        sub = subprocess.run(["git", "-C", str(MB), "rev-parse", f"{rev}:third_party/KernelBlaster"], capture_output=True, text=True, env=ENV)
        kb = w / "third_party" / "KernelBlaster"
        kb.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(["git", "-C", str(MB / "third_party" / "KernelBlaster"), "archive", sub.stdout.strip()], stdout=subprocess.PIPE, env=ENV)
        sh("tar", "x", "-C", kb, stdin=p.stdout)
        tar(Path(td), label, stamp)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "pre-snapshot":
        pre_snapshot()
    elif c == "live":
        live(sys.argv[2], sys.argv[sys.argv.index("--git") + 1] if "--git" in sys.argv else None)
