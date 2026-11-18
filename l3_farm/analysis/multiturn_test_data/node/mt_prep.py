# the ir_final part of A2's prune41.py, unchanged
import os, shutil, subprocess, sys
from pathlib import Path
bundle, work = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
src = work / "ir_src"
if not (src / ".done").exists():
    subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(bundle / "problem.py"),
                    "--bench-target-mb", "0", "--quant", "fp16", "--out-dir", str(src)], check=True, capture_output=True)
    (src / ".done").touch()
fin = work / "ir_final"; fin.mkdir(exist_ok=True)
shutil.copyfile(bundle / "ir" / "graph.json", fin / "graph.json")
for f in ("weights.npz", "io.npz"):
    if not (fin / f).exists():
        os.symlink(src / f, fin / f)
print("ir_final ok")
