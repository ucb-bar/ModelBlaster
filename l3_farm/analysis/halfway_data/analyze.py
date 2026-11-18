#!/usr/bin/env python3
"""Pause overhead of a halfway run from its results dir: analyze.py <results/<problem>/work> -> per round: first-half seconds,
exit -> done (the wait: barrier + controller + import), process restart, restored-state verification, whole round."""
import glob, json, sys
from pathlib import Path

def main(work):
    work = Path(work)
    out = []
    for hf in sorted(work.glob("r[0-9][0-9]/halfway.json")):
        h = json.loads(hf.read_text())
        rd = hf.parent
        run = json.loads((rd / "run" / "run.json").read_text())
        segs = run["segments"]
        s0, s1 = segs[0], segs[1] if len(segs) > 1 else {}
        tel = [json.loads(l) for l in open(rd / "run" / "telemetry.jsonl") if l.strip()]
        ver = [e for e in tel if e.get("t") == "span" and str((e.get("attrs") or {}).get("tag", "")).startswith("resume_verify")
               and (e.get("attrs") or {}).get("segment") == 1]
        vwall = (max(e["ts"] + e["dur"] for e in ver) - min(e["ts"] for e in ver)) if ver else 0.0
        rj = json.loads((rd / "round.json").read_text()) if (rd / "round.json").exists() else {}
        out.append({"round": h["round"], "first_half_s": round(s0["end"] - s0["start"], 1),
                    "exit_to_done_s": round(h["done_ts"] - s0["end"], 1), "wait_s": h.get("wait_s"),
                    "restart_s": round(s1.get("start", 0) - h["done_ts"], 1) if s1 else None,
                    "verify_s": round(vwall, 1), "outcome": h.get("outcome"), "imported_rows": h.get("imported_rows"),
                    "counts": h.get("counts"), "round_s": rj.get("seconds"), "steps_done": h.get("steps_done")})
    return out

if __name__ == "__main__":
    for r in main(sys.argv[1]):
        print(json.dumps(r))
