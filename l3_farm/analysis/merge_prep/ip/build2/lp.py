#!/usr/bin/env python3
"""Line-level feature patch builder.  lp.py SNAPSHOT.diff FILE K  -> patch (stdout) of FILE's hunks for feature K.
Features: 1 pool, 2 clock, 3 recording, 4 halfway, 9 = barrier / never committed.
Each changed line of the snapshot diff is tagged; for commit K: tag<K '+' lines become context, tag==K stay, tag>K or 9 dropped;
'-' lines: tag<K dropped, ==K kept, >K become context."""
import re, sys
snap, fname, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
txt = open(snap).read()
files = re.split(r"(?m)^(?=diff --git )", txt)
sec = [s for s in files if s.startswith(f"diff --git a/{fname} ")][0]
head = sec.split("\n@@")[0].split("\n")
hdr = "\n".join(l for l in head if not l.startswith("@@")) + "\n"
hunks = re.split(r"(?m)^(?=@@ )", sec)[1:]

def tag(fname, hi, kind, k, text):
    """k = index of this changed line among the changed lines of the hunk (of that kind)"""
    if fname == "pipeline/kb_export.py": return 1
    if fname == "pipeline/kb_global.py": return 4
    assert fname == "pipeline/kb_graph_opt.py"
    if hi == 0: return 9 if "kb_barrier" in text else 4
    if hi == 1: return 4 if k < 5 else 9
    if hi in (2, 3, 4, 7, 10): return 4
    if hi == 5: return 9 if "bar = kb_barrier" in text else 4
    if hi in (6, 8, 9, 13): return 9
    if hi == 11: return 1 if k < 6 else 2
    if hi == 12: return 1 if k < 4 else 2
    raise SystemExit(f"unclassified hunk {hi}")

out = []
delta = 0
for hi, h in enumerate(hunks):
    L = h.split("\n")
    m = re.match(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)", L[0])
    os_, rest = int(m.group(1)), m.group(5)
    body, k, dl = [], 0, 0
    for l in L[1:]:
        if l == "" : continue
        c = l[0]
        if c == "+":
            t = tag(fname, hi, "+", k, l); k += 1
            if t < K: dl += 1
            if t < K: body.append(" " + l[1:])
            elif t == K: body.append(l)
        elif c == "-":
            t = tag(fname, hi, "-", k, l)
            if t < K: dl -= 1
            if t < K: pass
            elif t == K: body.append(l)
            else: body.append(" " + l[1:])
        else:
            body.append(l)
    start = os_ + delta; delta += dl
    if not any(b[0] in "+-" for b in body): continue
    o = sum(1 for b in body if b[0] in " -"); n = sum(1 for b in body if b[0] in " +")
    out.append(f"@@ -{start},{o} +{start},{n} @@{rest}\n" + "\n".join(body) + "\n")
sys.stdout.write(hdr + "".join(out) if out else "")
