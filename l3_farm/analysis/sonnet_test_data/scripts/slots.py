#!/usr/bin/env python3
"""Per slot-problem final speedups (baseline_us / final_us of the slot) of an arm's work dir: slots.py DIR -> dict."""
import glob, json, sys
def slots(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/work/r0*/run/*/result.json")):
        r = json.load(open(f))
        s = next(iter(r["slots"].values()))
        key = f.split("/")[-4] + "/" + f.split("/")[-2].replace("kb_problem__", "").replace("__r00", "").replace("__r01", "").replace("__r02", "")
        out[key] = {"base_us": s["baseline_us"], "final_us": s["final_us"], "speedup": s["baseline_us"] / s["final_us"],
                    "e2e_speedup": r.get("e2e_speedup"), "steps": len(r.get("steps", [])), "techniques": s.get("techniques")}
    return out
if __name__ == "__main__":
    print(json.dumps(slots(sys.argv[1]), indent=1))
