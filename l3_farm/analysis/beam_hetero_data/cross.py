#!/usr/bin/env python3
"""Cross re-measurement: copy each arm's final (cache + hints + IR) to every node as btest_X<tag>, then time all
three finals + torch on each node, sequentially per node (quiet). python3 cross.py"""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import nodes
farm = nodes.farm
BT = Path(__file__).resolve().parent
JOB = {"C": 5297008, "B": 5297013, "H": 5297016}
tags = sys.argv[1:] or ["C", "B", "H"]
HOME = {"C": "C", "B": "B", "H": "H", "C2": "B", "C3": "H"}   # node each run lived on
for t in tags:
    rc, data, err = nodes.run_local(JOB[HOME[t]], BT / "packfinal.sh", [t], timeout=1800)
    (BT / f"final_{t}.tar.gz").write_bytes(data)
    print(t, "packed", len(data) // 2**20, "MB")


def node(nj):
    n, j = nj
    out = []
    for t in tags:
        if HOME[t] == n:
            continue
        x = f"X{t}"   # btest_X<t> on node n, code root = node n's own arm code root (btcode_<n>) via symlink
        farm.put(j, BT / f"final_{t}.tar.gz", f"/tmp/krisd/btest_{x}/final.tar.gz")
        farm.ssh(f"srun --jobid={j} --overlap bash -c 'cd /tmp/krisd/btest_{x} && mkdir -p 43_MinGPTCausalAttention && "
                 f"tar xzf final.tar.gz -C 43_MinGPTCausalAttention && ln -sfn /tmp/krisd/btcode_{n} /tmp/krisd/btcode_{x}'", timeout=900)
    for t in tags:
        x = t if HOME[t] == n else f"X{t}"
        rc, o, e = nodes.run_local(j, BT / "remeasure.sh", [x, f"{t}_on_{n}"], timeout=3600)
        line = next((l for l in o.decode().splitlines() if l.startswith("@@RM@@")), None)
        out.append((t, n, json.loads(line[6:]) if line else {"error": o.decode()[-500:] + e.decode()[-500:]}))
    return out


res = []
with ThreadPoolExecutor(3) as ex:
    for r in ex.map(node, JOB.items()):
        res += r
(BT / f"cross_{'_'.join(tags)}.json").write_text(json.dumps(res, indent=1))
for t, n, d in res:
    print(f"final {t} on node {n}: model {d.get('model_e2e_us')} seed {d.get('seed_e2e_us')} torch graph {d.get('torch_graph_us')} "
          f"compile {d.get('torch_compile_us')} host {d.get('host')}")
