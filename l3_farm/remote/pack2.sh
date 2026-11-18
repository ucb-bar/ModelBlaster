# Args: <name>. pack.sh + the knowledge store: streams the small, resumable/result files of a problem (state,
# summaries, logs, winning sources, the knowledge dir's review.jsonl / impl index) and a CONSISTENT copy of
# work/knowledge/knowledge.db of any size (an online sqlite backup into /tmp/krisd/gk_pack/<name> while the run
# is live, a plain copy once it has closed it -- the problem's own dir is only read, so this is safe on a running problem).
# Build trees, caches and binaries (several GB) stay on the node.
# Args: <name> [farm dir, default /tmp/krisd/farm] [scratch dir, default /tmp/krisd/gk_pack]
cd ${2:-/tmp/krisd/farm}/$1 || exit 1
P=${3:-/tmp/krisd/gk_pack}/$1; rm -rf $P; mkdir -p $P/work/knowledge
EXTRA=""
if [ -f work/knowledge/knowledge.db ]; then
  PY=/tmp/krisd/venv/bin/python; [ -x $PY ] || PY=python3
  $PY - $PWD/work/knowledge/knowledge.db $P/work/knowledge/knowledge.db <<"PYEOF" >/dev/null 2>&1 && EXTRA="-C $P work/knowledge/knowledge.db"
import os, shutil, sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
ok = False
if os.path.exists(src + "-wal"):      # live (WAL): an online backup from a read-only connection = one snapshot
    try:
        s = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
        d = sqlite3.connect(dst)
        s.backup(d); d.close(); s.close(); ok = True
    except sqlite3.Error:
        pass
if not ok:                            # closed: the file alone is consistent (or copy db + wal and recover)
    shutil.copyfile(src, dst)
    if os.path.exists(src + "-wal"):
        shutil.copyfile(src + "-wal", dst + "-wal")
c = sqlite3.connect(dst, isolation_level=None)
c.execute("PRAGMA wal_checkpoint(TRUNCATE)"); c.execute("PRAGMA journal_mode=DELETE")
assert c.execute("PRAGMA quick_check").fetchone()[0] == "ok"
c.close()
PYEOF
fi
{ ls status job.env extract.log run.log worker.out 2>/dev/null
  ls work/knowledge/review.jsonl work/knowledge/impl/index.jsonl 2>/dev/null
  find work -type f -size -3M \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' -o -name '*.cu' -o -name '*.md' -o -name '*.db' -o -name '*.txt' \) \
    -not -path '*/cache/*' -not -path '*/run/work/*' -not -path '*/problems/*' -not -path 'work/gk_seed/*' \
    -not -name 'knowledge.db' -not -path 'work/knowledge/review.jsonl' -not -path 'work/knowledge/impl/index.jsonl' 2>/dev/null
  # the winners cache (work/cache: index + winner sources) is what keeps a resumed run's earlier rounds' kernels
  # (pack.sh had this; pack2.sh lost it -- found by the X-128 restart drill)
  find work/cache -type f -size -20M 2>/dev/null
} > $P/list
tar czf - -T $P/list $EXTRA 2>/dev/null
rm -rf $P
exit 0
