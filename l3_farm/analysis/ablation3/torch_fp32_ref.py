#!/usr/bin/env python3
"""fp32 PyTorch reference timing (TF32 OFF, default fp32 math) of the ablation's 8 KernelBench problems, so the precision effect (flow runs fp16)
can be separated from the flow's own effect. Derived from torch_posthoc.py's BENCH (same method: native sizes, eval/no_grad, median of
per-iteration cudaEvent pairs, eager + CUDA graph + torch.compile) with the module and inputs left in float32. One run per ORIGINAL problem
(not per arm) on one node, over farm.ssh (srun --overlap, script on stdin). Output: analysis/ablation3/torch_fp32/<problem>.json.
  python3 torch_fp32_ref.py JOBID [problem ...]      (default: the 8 problems)"""
import json, shlex, sys, time
from pathlib import Path
FARM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FARM))
import farm, torch_posthoc as tp  # noqa: E402  (config.json read by farm: harmless, no side effects)

PROBLEMS = ["34_VanillaRNNHidden", "18_SqueezeNet", "14_DenseNet121DenseBlock", "46_NetVladWithGhostClusters",
            "47_NetVladNoGhostClusters", "8_ResNetBasicBlock", "15_DenseNet121", "7_GoogleNetInceptionV1"]
B = (tp.BENCH.replace('"dtype": "float16"', '"dtype": "float32 (TF32 off)"').replace("torch.set_default_dtype(torch.float16)", "torch.set_default_dtype(torch.float32)")
     .replace(".cuda().half().eval()", ".cuda().float().eval()").replace("v.cuda().half()", "v.cuda().float()")
     .replace("x.cuda().half() if", "x.cuda() if"))
assert "cuda().half()" not in B and "float16)" not in B, "bench not fully converted"
B = B.replace("import torch\n", "import torch\ntorch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False\n", 1)


def main():
    job = int(sys.argv[1]); names = sys.argv[2:] or PROBLEMS
    out = Path(__file__).parent / "torch_fp32"; out.mkdir(exist_ok=True)
    for n in names:
        sh = tp.SH.format(P="/tmp/krisd/abl3_torch32", BENCH=tp.b64(B.encode()), PROB=tp.b64((FARM / "problems" / f"{n}.py").read_bytes()), TO=900)
        cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(x)) for x in [n, 50, 10, 1])
        rc, o, e = farm.ssh(cmd, sh.encode(), timeout=1100)
        line = next((l for l in o.decode(errors="replace").splitlines() if l.startswith("@@POSTHOC@@")), None)
        if not line:
            print(n, "FAILED", o.decode()[-800:]); continue
        d = json.loads(line[len("@@POSTHOC@@"):]); d["when"] = time.strftime("%F %T"); d["job"] = job
        (out / f"{n}.json").write_text(json.dumps(d, indent=1))
        print(n, "fp32 eager", d.get("eager_us"), "graph", d.get("graph_us"), "compile", d.get("compile_us"))


if __name__ == "__main__":
    main()
