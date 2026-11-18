import json, statistics as st
from pathlib import Path
H = Path(__file__).parent / "results"
print("| case (kernel us) | timer | nvcc load (nice 19) | n | median us | bias vs idle | CV % | p95/median |")
print("|---|---|---|---|---|---|---|---|")
for c in ("small", "mid", "real"):
    d = json.loads((H / f"e2_{c}.json").read_text())
    for m in ("graph", "event"):
        base = st.median(d["raw"]["0"][m])
        for n in ("0", "3", "8", "32"):
            x = sorted(d["raw"][n][m]); med = st.median(x)
            print(f"| {c} ({base:.1f}) | {m} | {'idle' if n=='0' else n+' concurrent'} | {len(x)} | {med:.2f} | {100*(med/base-1):+.2f}% | {100*st.pstdev(x)/st.mean(x):.2f} | {x[int(.95*len(x))-1]/med:.4f} |")
