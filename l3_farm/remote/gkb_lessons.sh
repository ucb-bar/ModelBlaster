# Args: <runs dir> <name>. Prints the distinct "Lessons from earlier optimisation runs" sections of the codegen
# prompts in the run's trace stores (read-only), as JSON lines {"blob": ..., "lessons": ...}.
PY=/tmp/krisd/venv/bin/python
$PY - "$1/$2/work" <<"PYEOF"
import gzip, json, sys, pathlib
seen = set()
for b in sorted(pathlib.Path(sys.argv[1]).glob("r*/**/trace/blobs/*/*.gz")):
    try:
        t = gzip.decompress(b.read_bytes()).decode("utf-8", "replace")
    except Exception:
        continue
    i = t.find("Lessons from earlier optimisation runs:")
    if i < 0:
        continue
    ends = [t.find(m, i) for m in ("\n\nAlready tried on this slot", "\n\n## Technique to apply")]
    end = min([x for x in ends if x > i] or [len(t)])
    sec = t[i:end]
    if sec in seen:
        continue
    seen.add(sec)
    m = t.find("## Technique to apply: ")
    tech = t[m + 23:t.find("\n", m)] if m >= 0 else None
    print(json.dumps({"blob": str(b.relative_to(sys.argv[1])), "technique": tech, "lessons": sec}))
PYEOF
exit 0
