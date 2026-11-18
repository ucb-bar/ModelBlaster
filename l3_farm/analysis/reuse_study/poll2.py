import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
import rs, reuse_run
H = reuse_run.HERE / "data" / "runs"
for j in rs.JOBS:
    for rd in ("runs", "runs2"):
        rc, o, e = rs.run(j, f"cd {rs.R}/{rd} 2>/dev/null && for d in *; do [ -f $d/torch_3.txt ] && grep -q '\"t1\"' $d/ru.json && grep -q median_us $d/ru.json && echo done $d; done")
        for p in re.findall(r"done (\S+)", o):
            if not (H / p / "torch_3.txt").exists() or (rd == "runs2" and not (H / p / "ru.json").exists()):
                reuse_run.fetch(j, [p], rd)
        print(j, rd, re.findall(r"done (\S+)", o))
