#!/usr/bin/env python3
"""Anonymize a checked-out tree in place (text files only; file names with a contributor prefix are renamed with git mv).

  python3 anonymize.py TREE [--check]       --check: only scan and report what would change / what remains, change nothing

Rules (documented in analysis/anonymize/README.md):
  - contributor names/emails/usernames in content and in paths -> neutral placeholders
  - 'Copyright (c) YEAR Name <email>' -> 'Copyright (c) YEAR The Authors'
  - the repository's own paper reference (README.md block) removed
  - personal GitHub URLs -> neutral/relative; third-party public links (e.g. chipyard docs, arXiv papers cited for a technique) are kept
Third-party license headers (SPDX NVIDIA notices, IBM, Linux Foundation, ...) are not authors of this work and are kept.
"""
import os, re, subprocess, sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
CHECK = "--check" in sys.argv

# ordered (pattern, replacement, flags) applied to text content
RULES = [
    (r"(Copyright \(c\) \d{4}) [^<\n]*<[^>\n]+@[^>\n]+>", r"\1 The Authors", 0),
    (r"chrisdong@firesim2", "user@host", 0),
    (r"git@github\.com:kris-dong/KernelBlaster\.git", "../KernelBlaster.git", 0),
    (r"(?:`)?kris-dong/KernelBlaster(?:`)?", "the KernelBlaster repository", 0),
    (r"NVlabs/KernelBlaster", "KernelBlaster", 0),
    (r"/Users/krisd", "/home/user", 0),
    (r"/tmp/krisd", "/tmp/user", 0),
    (r"/nscratch/krisd", "/nscratch/user", 0),
    (r"/scratch2?/(?:agustin|dima)", "/path/to", re.I),
    (r"/home/agustin", "/home/user", 0),
    (r"\b[\w.+-]+@(?:berkeley\.edu|gmail\.com|nvidia\.com|users\.noreply\.github\.com)\b", "anonymous@example.com", 0),
    (r"\bucb-bar fork\b", "project fork", 0),
    (r"\bucb-bar\b(?!/chipyard)", "the lab", 0),
    (r"(?m)^(\s*[-*#>]*\s*)Dima's\b", r"\1A collaborator's", 0),
    (r"\bDima Nikiforov\b", "a collaborator", 0),
    (r"\bDima's\b", "a collaborator's", 0),
    (r"\bDima-format\b", "collaborator-format", 0),
    (r"\bthe full Dima workload\b", "the full collaborator workload", 0),
    (r"\bDima\b", "a collaborator", 0),
    (r"\bdima's\b", "a collaborator's", 0),
    (r"\bdima\b", "a collaborator", 0),
    (r"\bdima_", "collab_", 0),
    (r"\bAgustin(?: N\.| Nahuel)? Coppari Hollmann\b", "a collaborator", 0),
    (r"\b(?:Agustin|Coppari|Hollmann|Nikiforov|copparihollmann|vnikiforov)\b", "a collaborator", 0),
    (r"\bagustin\b", "user", re.I),
    (r"/projects/DIMA/", "/projects/proj/", 0),
    (r"\bKris Dong\b", "a collaborator", 0),
    (r"\bkris-dong\b", "user", 0),
    (r"\b(?:chrisdong|krisdong)\b", "user", 0),
    (r"krisd", "user", 0),
]
README_PAPER = re.compile(
    r"read the KernelBlaster paper:\n\n- \*\*Paper:\*\* \*.*?\*\s+—\s+\n?\s*https://arxiv\.org/abs/2602\.14293\n- \*\*Source:\*\* vendored here as a git submodule under `third_party/KernelBlaster/`\n  \(NVlabs/KernelBlaster\)\.",
    re.S)
README_PAPER_NEW = "read the KernelBlaster documentation under `third_party/KernelBlaster/docs/`:\n\n- **Source:** vendored here as a git submodule under `third_party/KernelBlaster/`."
SCAN = re.compile(r"Dima|Nikiforov|Agustin|Coppari|Hollmann|Siva Hari|Kris Dong|kris-dong|chrisdong|krisdong|\bkrisd\b|@berkeley|@nvidia\.com|@gmail|users\.noreply|2602\.14293|/scratch2/agustin|/home/agustin|dima_", re.I)


def files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=TREE, capture_output=True).stdout.split(b"\0")
    return [f.decode() for f in out if f]


def is_text(p):
    try:
        b = (TREE / p).read_bytes()
    except OSError:
        return None
    if b"\0" in b[:4096]:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


changed, renamed, leftover = [], [], []
for p in files():
    if p == "uv.lock" or not (TREE / p).is_file() or (TREE / p).is_symlink():
        continue
    t = is_text(p)
    if t is None:
        continue
    n = t
    if p == "README.md":
        n = README_PAPER.sub(README_PAPER_NEW, n)
    if p == ".gitmodules":
        n = re.sub(r"(url = ).*KernelBlaster\.git", r"\1../KernelBlaster.git", n)
    for pat, rep, fl in RULES:
        n = re.sub(pat, rep, n, flags=fl)
    for m in SCAN.finditer(n):
        leftover.append((p, n[max(0, m.start() - 30):m.end() + 30].replace("\n", " ")))
        break
    if n != t:
        changed.append(p)
        if not CHECK:
            (TREE / p).write_text(n, encoding="utf-8")
if not CHECK:
    for p in files():
        base = os.path.basename(p)
        new = re.sub(r"dima", "collab", base, flags=re.I)
        if new != base:
            dst = os.path.join(os.path.dirname(p), new)
            subprocess.run(["git", "mv", p, dst], cwd=TREE, check=True)
            renamed.append((p, dst))
        # directory components named after a person
    for p in files():
        parts = p.split("/")
        if any(re.search(r"dima|kris|agustin|coppari|nikiforov", x, re.I) for x in parts[:-1]):
            new = "/".join(re.sub(r"dima", "collab", x, flags=re.I) for x in parts)
            if new != p:
                os.makedirs(os.path.dirname(TREE / new), exist_ok=True)
                subprocess.run(["git", "mv", p, new], cwd=TREE, check=True)
                renamed.append((p, new))

print(f"{'would change' if CHECK else 'changed'} {len(changed)} files; renamed {len(renamed)}")
left = list(leftover)
for p in files():
    if SCAN.search(p):
        left.append((p, "<path>"))
print(f"remaining matches after rules: {len(left)}")
for p, c in left[:40]:
    print("  ", p, "|", c)
