#!/usr/bin/env python3
"""Cross re-measurement on ONE quiet node (my S3 node): every final model (my 4 arms + the X-037 controls' saved finals) is
copied there and timed sequentially with remeasure.sh (kb_model_run reps 5 + torch bench). python3 cross.py"""
import json
from pathlib import Path
import st
BT = Path("/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad/bt")
ON = st.arm_job("S3")
fin = {}
for a in ("S5", "S0", "H5"):
    rc, data, err = st.run_local(st.arm_job(a), st.HERE / "packfinal.sh", [a], timeout=1800)
    p = st.DATA / "res" / f"final_{a}.tar.gz"; p.write_bytes(data); fin[a] = p
    print(a, "packed", len(data) >> 20, "MB", flush=True)
for c in ("C", "C2", "C3", "B", "H"):
    fin[c] = BT / f"final_{c}.tar.gz"
res = {}
for a, p in fin.items():
    tag = f"X{a}"
    st.farm.put(ON, p, f"/tmp/krisd/sonnet_{tag}/final.tar.gz")
    st.farm.ssh(f"srun --jobid={ON} --overlap bash -c 'cd /tmp/krisd/sonnet_{tag} && mkdir -p 43_MinGPTCausalAttention && "
                f"tar xzf final.tar.gz -C 43_MinGPTCausalAttention 2>/dev/null; true'", timeout=900)
for a in list(fin) + ["S3"]:
    tag = f"X{a}" if a != "S3" else "S3"
    rc, o, e = st.run_local(ON, st.HERE / "remeasure.sh", [tag, f"cross_{a}"], timeout=3600)
    line = next((l for l in o.decode().splitlines() if l.startswith("@@RM@@")), None)
    res[a] = json.loads(line[6:]) if line else {"error": o.decode()[-500:]}
    print(a, res[a].get("model_e2e_us"), res[a].get("torch_graph_us"), flush=True)
(st.DATA / "res" / "cross_one_node.json").write_text(json.dumps({"node_job": ON, "results": res}, indent=1))
