#!/usr/bin/env python3
"""Controller side of the torch-timing comparison (see torch_compare.md). usage: torch_compare_run.py JOB stage|start|fetch
stage: upload problems/scripts + start the node script detached; fetch: pull $W/out into analysis/ablation3/torch_compare_data/. Nodes only via farm.ssh."""
import io, sys, tarfile, time, base64
from pathlib import Path
FARM = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FARM))
import farm, torch_posthoc as tp  # noqa: E402
HERE = Path(__file__).resolve().parent
KB = Path.home() / "Documents/ModelBlaster-clone/mb_pilot_sol/third_party/KernelBlaster/scripts"
PROBS = ["34_VanillaRNNHidden", "18_SqueezeNet", "14_DenseNet121DenseBlock", "46_NetVladWithGhostClusters", "47_NetVladNoGhostClusters", "8_ResNetBasicBlock",
         "15_DenseNet121", "7_GoogleNetInceptionV1", "1_MLP", "4_LeNet5", "9_ResNet18", "20_MobileNetV2", "33_VanillaRNN", "39_GRU", "43_MinGPTCausalAttention", "45_UNetSoftmax"]
W = "/tmp/krisd/abl3cmp"


def main():
    job, cmd = int(sys.argv[1]), sys.argv[2]
    if cmd == "stage":
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            def add(name, data):
                ti = tarfile.TarInfo(name); ti.size = len(data); ti.mode = 0o755; t.addfile(ti, io.BytesIO(data))
            for p in PROBS:
                add(f"probs/{p}.py", (FARM / "problems" / f"{p}.py").read_bytes())
            add("bench.py", tp.BENCH.encode())
            add("run_baselines.py", (KB / "run_baselines.py").read_bytes())
            add("run_baselines_compile.py", (KB / "run_baselines_compile.py").read_bytes())
            add("cmp_node.sh", (HERE / "cmp_node.sh").read_bytes())
        farm.put(job, buf.getvalue(), f"{W}/stage.tgz")
        print(farm.ssh(f"srun --jobid={job} --overlap bash -c 'cd {W} && tar xzf stage.tgz && ls'")[1].decode())
    elif cmd == "start":
        rc, o, e = farm.ssh(f"nohup setsid srun --jobid={job} --overlap bash {W}/cmp_node.sh </dev/null >/dev/null 2>&1 & sleep 5; echo started")
        print(o.decode())
    elif cmd == "status":
        print(farm.ssh(f"srun --jobid={job} --overlap bash -c 'ls {W}/out | wc -l; ls {W}/DONE; tail -3 {W}/node.log'")[1].decode())
    elif cmd == "fetch":
        rc, o, e = farm.ssh(f"srun --jobid={job} --overlap bash -c 'cd {W} && tar czf - out | base64'", timeout=300)
        d = HERE / "torch_compare_data"; d.mkdir(exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(base64.b64decode(o)), mode="r:gz").extractall(d)
        print("fetched to", d)


if __name__ == "__main__":
    main()
