#!/usr/bin/env python3
"""Create the NEW technique-only headline knowledge store (path from config_headline.json global_knowledge.path; never
knowledge_global) and print an audit of the knowledge.db a run would be seeded with (row counts of every table).
The store is created EMPTY (no snapshot): the first seed materializes the repo's technique definitions only because the
farm runs the merge CLI with KERNELBLASTER_SEED_MINED=0 (global_knowledge.env). Zero LLM, no node."""
import json, os, sqlite3, subprocess, sys, tempfile
from pathlib import Path
L3 = Path(__file__).resolve().parents[2]
cfg = json.loads((Path(__file__).parent / "config_headline.json").read_text())
gk = cfg["global_knowledge"]
root = Path(os.path.expanduser(gk["path"]))
assert root.name != "knowledge_global", "must not be the live store"
KB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src"
sys.path.insert(0, str(KB))
from kernelblaster.knowledge.v2.merge import GlobalStore
GlobalStore(root).ensure()
env = dict(os.environ, PYTHONPATH=str(KB), **gk["env"])
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "seed"
    r = subprocess.run([sys.executable, "-m", "kernelblaster.knowledge.v2.merge", "--store", str(root), "seed",
                        "--part", f"cuda:sm_{cfg['kb']['arch']}|{gk['dtype']}", "--out", str(out), "--problem", "1_MLP"],
                       env=env, capture_output=True, text=True)
    print("seed rc", r.returncode, r.stdout.strip()[-400:], r.stderr.strip()[-300:])
    dbs = list(out.rglob("knowledge.db"))
    assert dbs, "no knowledge.db seeded"
    c = sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True)
    tabs = [t for (t,) in c.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name")]
    print("store:", root, "| snapshots:", sorted(p.name for p in (root / "snapshots").glob("*")), "| current:", (root / "current").exists())
    print("seeded knowledge.db tables (rows):")
    for t in tabs:
        print(f"  {t:28s} {c.execute(f'select count(*) from [{t}]').fetchone()[0]}")
    print("meta:", dict(c.execute("select key, value from meta").fetchall()) if "meta" in tabs else None)
