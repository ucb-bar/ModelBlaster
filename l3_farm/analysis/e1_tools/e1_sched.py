#!/usr/bin/env python3
"""E1 queue runner (local, long-running): one problem at a time per node, next problem when the node's worker is done.

  e1_sched.py JOB[,JOB...] P [P ...]     (P omitted: all 48 usable problems, largest graph first)

State in e1_data/sched.json (done / running per node / failures). After each problem its small result files are
fetched into e1_data/runs/<P>/. Only jobs from e1_data/my_jobs.json are used (nodes.run refuses others).
"""
import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e1_run, e1_prep, nodes  # noqa: E402

DATA = e1_run.DATA
import os
ST = DATA / os.environ.get("E1_STATE", "sched.json")


def order(ps):
    def size(p):
        try:
            g = json.loads((e1_prep.RES / p / "work" / "final" / p / "ir" / "graph.json").read_text())
            return len(g["ops"])
        except Exception:
            return 0
    return sorted(ps, key=lambda p: -size(p))


def worker_state(job):
    """-> (running?, last worker log name, text tail)"""
    sh = (f"cd {e1_run.R}; f=$(ls -t logs/worker_*.log 2>/dev/null | head -1); [ -z \"$f\" ] && {{ echo NOWORKER; exit; }}; "
          f"tail -3 $f; pgrep -f 'bin/worker.sh' >/dev/null && echo RUNNING || echo IDLE")
    rc, out, err = e1_run.run(job, sh, 120)
    return out


def main():
    jobs = [int(j) for j in sys.argv[1].split(",")]
    probs = sys.argv[2:] or order(e1_prep.problems())
    st = json.loads(ST.read_text()) if ST.exists() else {"done": [], "running": {}, "log": []}
    queue = [p for p in probs if p not in st["done"] and p not in st["running"].values()]
    print("queue", len(queue), flush=True)
    while (queue or st["running"]) and jobs:
        for j in list(jobs):
            try:
                txt = worker_state(j)
            except Exception as e:  # noqa: BLE001 - ssh hiccup: try again next round
                print("state error", j, repr(e)[:100], flush=True)
                continue
            cur = st["running"].get(str(j))
            if "RUNNING" in txt.splitlines()[-1:]:
                continue
            if cur:                                  # worker finished
                try:
                    ok = e1_run.fetch(j, [cur])
                except Exception as e:  # noqa: BLE001
                    print("fetch error", cur, repr(e)[:200], flush=True)
                    continue
                if not ok:                           # node gone / nothing to fetch: requeue, retire the node
                    print(time.strftime("%H:%M:%S"), "FETCH FAILED", cur, "on", j, "-> requeue, node retired", flush=True)
                    queue.insert(0, cur)
                    del st["running"][str(j)]
                    jobs.remove(j)
                    ST.write_text(json.dumps(st, indent=1))
                    continue
                st["done"].append(cur)
                st["log"].append({"p": cur, "job": j, "t": time.strftime("%H:%M:%S"), "tail": txt[-300:]})
                del st["running"][str(j)]
                print(time.strftime("%H:%M:%S"), "done", cur, "on", j, flush=True)
            if queue:
                p = queue.pop(0)
                try:
                    e1_run.start(j, [p])
                    st["running"][str(j)] = p
                    print(time.strftime("%H:%M:%S"), "start", p, "on", j, flush=True)
                except Exception as e:  # noqa: BLE001
                    queue.insert(0, p)
                    print("start error", repr(e)[:200], flush=True)
            ST.write_text(json.dumps(st, indent=1))
        time.sleep(75)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
