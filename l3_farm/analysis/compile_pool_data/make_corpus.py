#!/usr/bin/env python3
"""Build the replay corpus for the compile-pool load test from RECORDED real slot sources (no LLM, no API).

Every manifest-bearing problem dir under l3_farm/results/<P>/work/{r*,final}/{profile,variants,...} is one recorded
"candidate set" (manifest + walker head/tail + slots/*.cu) of problem P; the corpus keeps them in recorded order
(round, stage) per problem with a stub driver.cpp / empty data.bin (the load test only compiles slot units + links).
Output: corpus/<P>/<NNNN>_<tag>/{manifest.json,walker_head.cu,walker_tail.cu,slots/,driver.cpp,data.bin} and
corpus/index.json; prints distinct-vs-total unit statistics (the real duplicate rate of the recorded sources).
"""
import hashlib, json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
OUT = HERE / "corpus"
sys.path.insert(0, str(HERE.parents[2] / "ModelBlaster/third_party/KernelBlaster/src"))
from kernelblaster.knowledge.slots import SlotProblem  # noqa: E402


def order_key(p: Path):
    rel = p.relative_to(RES / p.relative_to(RES).parts[0] / "work").parts
    rnd = rel[0]
    r = 99 if rnd == "final" else int(re.match(r"r(\d+)", rnd).group(1)) if re.match(r"r\d+$", rnd) else 98
    stage = {"profile": 0, "variants": 1}.get(rel[1], 2) if len(rel) > 1 else 3
    return (r, stage, "/".join(rel))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    index = {}
    for pdir in sorted(RES.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("_") or not (pdir / "work").is_dir():
            continue
        mans = [m.parent for m in pdir.rglob("manifest.json")
                if "/ir/" not in str(m) and "stale" not in str(m) and "slots" in [c.name for c in m.parent.iterdir()]]
        mans.sort(key=order_key)
        jobs = []
        for i, d in enumerate(mans):
            tag = re.sub(r"[^A-Za-z0-9]+", "_", "/".join(d.relative_to(pdir / "work").parts[:3]))[:60]
            dst = OUT / pdir.name / f"{i:04d}_{tag}"
            dst.mkdir(parents=True)
            shutil.copy(d / "manifest.json", dst / "manifest.json")
            for n in ("walker_head.cu", "walker_tail.cu"):
                shutil.copy(d / n, dst / n)
            shutil.copytree(d / "slots", dst / "slots")
            (dst / "driver.cpp").write_text("// stub driver for the build-load replay\n")
            (dst / "data.bin").write_bytes(b"")
            m = json.loads((dst / "manifest.json").read_text())
            m["driver"], m["data"] = "driver.cpp", "data.bin"
            (dst / "manifest.json").write_text(json.dumps(m))
            jobs.append(dst.name)
        if jobs:
            index[pdir.name] = jobs
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    # statistics: unit texts (head + wrapped slot) = what one nvcc -c compiles
    tot, per = 0, {}
    seen_global = {}
    for p, jobs in index.items():
        seen = set()
        for j in jobs:
            prob = SlotProblem(OUT / p / j)
            for name, text in prob.units(prob.seed_sources()):
                h = hashlib.sha256(text.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    seen_global[h] = seen_global.get(h, 0) + 1
        per[p] = len(seen)
    n_per = sum(per.values())
    print(f"problems {len(index)}, candidate sets {sum(len(j) for j in index.values())}")
    print(f"distinct units per problem summed (= per-problem caches): {n_per}; distinct overall: {len(seen_global)}; "
          f"cross-problem duplicate rate of first-time builds: {1 - len(seen_global) / n_per:.3f}")
    (HERE / "corpus_stats.json").write_text(json.dumps({"problems": len(index), "sets": sum(len(j) for j in index.values()),
                                                         "units_per_problem_sum": n_per, "units_distinct": len(seen_global),
                                                         "cross_problem_dup_rate": round(1 - len(seen_global) / n_per, 4),
                                                         "per_problem_units": per}, indent=1))


main()
