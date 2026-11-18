#!/usr/bin/env python3
"""Hunk splitter. hs.py list REPO FILE | hs.py patch REPO FILE i,j,k [--sub IDX=FILE]  (patch to stdout)
--sub IDX=FILE replaces hunk IDX by the text in FILE (a complete hunk incl. its @@ line)."""
import re, subprocess, sys
repo, f = sys.argv[2], sys.argv[3]
d = subprocess.run(["git", "-C", repo, "diff", "-U3", "--", f], capture_output=True, text=True, check=True).stdout
parts = re.split(r"(?m)^(?=@@ )", d)
head, hunks = parts[0], parts[1:]
if sys.argv[1] == "list":
    for i, h in enumerate(hunks):
        ch = [l for l in h.splitlines()[1:] if l[:1] in "+-"]
        print(i, h.splitlines()[0][:70], len(ch), "|", (ch[0][:90] if ch else ""))
else:
    sel = [int(x) for x in sys.argv[4].split(",") if x]
    sub = {}
    for a in sys.argv[5:]:
        if a.startswith("--sub"):
            continue
        k, v = a.split("=", 1); sub[int(k)] = open(v).read()
    sys.stdout.write(head + "".join(sub.get(i, hunks[i]) for i in sel))
