"""Recorded real flow prompts (E0 trace blobs) for the relay tests."""
import glob, gzip, json, os, random
E0 = "/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e0_runs"


def blob(idx, sha):
    return gzip.open(f"{os.path.dirname(idx)}/blobs/{sha[:2]}/{sha}.gz").read().decode()


def load(roles=("discovery", "reflect", "hypothesis"), lo=3000, hi=16000, seed=3):
    out = []
    for idx in sorted(glob.glob(f"{E0}/*/work/r*/run/trace/index.jsonl")):
        for l in open(idx):
            r = json.loads(l)
            if r.get("t") == "llm" and r.get("ok") and r.get("role") in roles and lo <= r.get("prompt_chars", 0) <= hi:
                try:
                    msgs = [{"role": m["role"], "content": blob(idx, m["content"]["sha"]) if isinstance(m["content"], dict) else m["content"]}
                            for m in r["messages"]]
                except FileNotFoundError:
                    continue
                out.append({"role": r["role"], "chars": r["prompt_chars"], "messages": msgs,
                            "src": idx.split("/e0_runs/")[1][:30]})
    random.Random(seed).shuffle(out)
    return out
