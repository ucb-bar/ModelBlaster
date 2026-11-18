#!/usr/bin/env python3
"""Generate flow_bench.cu: the flow's final 41_GRUBidirectional program (build/kernels.cu, unchanged) plus a driver
that times several schedules of the same slot kernels in CUDA graphs:

  S0 flow_final      launch_gpu_implementation() exactly as the flow ships it (v3/v2 for l0, v0 fwd / v1 rev for l1-5)
  S1 all_v0_seq      every gru_seq op uses the forward winner v0 (reverse=1 for reverse ops), still sequential
  S2 v0_bidir_conc   v0 for both directions, forward and reverse of each layer on two streams (fork/join in the graph)
  S3 S2_no_dead_hn   S2 without the three h_n concat ops (their output is never read by the model)
  per-op singles     each gru_seq variant alone, for both directions and both input widths

v0a/v0b are textual copies of the v0 namespace (the slot keeps static scratch -- gi buffer, flags, epoch -- so two
concurrent instances need separate copies) whose launches go to a selectable stream.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "kernels.cu").read_text()
man = json.loads((HERE / "manifest.json").read_text())

m = re.search(r"namespace mbslot_gru_seq_f16__v0 \{\n(.*?)\n\}  // namespace mbslot_gru_seq_f16__v0", src, re.S)
body = m.group(1)


def restream(b):
    def fix(mm):
        args = mm.group(1)
        n = args.count(",") + 1 - args.count("(")  # crude, refined below
        parts = [p for p in re.split(r",(?![^(]*\))", args)]
        if len(parts) == 2:
            return f"<<<{args}, 0, g_s>>>"
        if len(parts) == 3:
            return f"<<<{args}, g_s>>>"
        raise SystemExit("unexpected launch " + args)
    return re.sub(r"<<<(.*?)>>>", fix, b)


copies = ""
for nm in ("v0a", "v0b"):
    copies += f"namespace mbslot_gru_seq_f16__{nm} {{\nstatic cudaStream_t g_s = cudaStreamPerThread;\n" + restream(body) + \
              f"\n}}  // namespace mbslot_gru_seq_f16__{nm}\n"

bufs = man["buffers"]
ops = man["ops"]
nb = len(bufs)
alloc = []
for i, b in enumerate(bufs):
    if b["alias"]:
        alloc.append(f"  b[{i}] = (void*)(((__half*)b[{b['alias']['base']}]) + {b['alias']['offset']});")
for i, b in enumerate(bufs):
    if not b["alias"]:
        kind = {"input": 1, "const": 2}.get(b["kind"], 0)
        alloc.insert(0, f"  b[{i}] = dalloc({b['numel']}, {kind}, {i});")

tail = (HERE / "kernels.cu").read_text()
calls = {}
for ln in tail.splitlines():
    mm = re.match(r"\s*case (\d+): (mbslot_\w+)::(\w+)\((.*)\); break;", ln)
    if mm:
        calls[int(mm.group(1))] = (mm.group(2), mm.group(3), mm.group(4))

gru_ops = [i for i, (ns, fn, a) in calls.items() if fn == "kernel_gru_seq_f16"]


def call(i, ns=None):
    ns0, fn, a = calls[i]
    return f"{ns or ns0}::{fn}({a});"


s1 = "\n".join("    " + (call(i, "mbslot_gru_seq_f16__v0") if i in gru_ops else f"mb_run_op({i}, b);") for i in sorted(calls))
# S2: pairs (fwd, rev) are consecutive gru ops
s2 = []
i = 0
order = sorted(calls)
while i < len(order):
    k = order[i]
    if k in gru_ops and order[i + 1] in gru_ops:
        r = order[i + 1]
        s2.append("    fork();")
        s2.append("    mbslot_gru_seq_f16__v0a::g_s = cudaStreamPerThread; " + call(k, "mbslot_gru_seq_f16__v0a"))
        s2.append("    mbslot_gru_seq_f16__v0b::g_s = s2; " + call(r, "mbslot_gru_seq_f16__v0b"))
        s2.append("    join();")
        i += 2
    else:
        s2.append(f"    if (!(skip_dead && {k} >= 20)) mb_run_op({k}, b);")
        i += 1
s2 = "\n".join(s2)

# singles: (label, namespace, op index to borrow args from, reverse override)
singles = []
for i in gru_ops:
    ns, fn, a = calls[i]
    singles.append((f"op{i}_{ns.split('__')[-1]}_as_shipped", ns, i))
    if ns != "mbslot_gru_seq_f16__v0":
        singles.append((f"op{i}_v0_instead", "mbslot_gru_seq_f16__v0", i))
sing = "\n".join(f'    single("{lab}", [&]{{ {call(i, ns)} }});' for lab, ns, i in singles)

drv = r"""
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
static void **b;
static std::vector<void*> owned;
static cudaStream_t s2;
static cudaEvent_t evf, evj;
static unsigned long long rng = 88172645463325252ull;
static float urand() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 11) * (1.0 / 9007199254740992.0); }
static void *dalloc(long n, int kind, int idx) {
    void *p; cudaMalloc(&p, n * 2 + 256);
    std::vector<__half> h(n);
    for (long i = 0; i < n; i++) h[i] = __float2half(kind == 1 ? urand() : kind == 2 ? (urand() * 2 - 1) / 16.f : 0.f);
    cudaMemcpy(p, h.data(), n * 2, cudaMemcpyHostToDevice);
    owned.push_back(p);
    return p;
}
static void fork() { cudaEventRecord(evf, cudaStreamPerThread); cudaStreamWaitEvent(s2, evf, 0); }
static void join() { cudaEventRecord(evj, s2); cudaStreamWaitEvent(cudaStreamPerThread, evj, 0); }
static bool skip_dead = false;
static void S0() { launch_gpu_implementation(b); }
static void S1() {
@@S1@@
}
static void S2() {
@@S2@@
}
static double timed(std::function<void()> f, const char *name, int warm = 10, int iters = 50) {
    cudaStream_t s = cudaStreamPerThread;
    f(); f(); cudaDeviceSynchronize();   // eager warm-up (slots allocate their scratch on first call)
    cudaError_t e0 = cudaGetLastError();
    cudaGraph_t g; cudaGraphExec_t ge;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    f();
    cudaError_t e = cudaStreamEndCapture(s, &g);
    if (e != cudaSuccess || e0 != cudaSuccess) { printf("%s capture failed %s %s\n", name, cudaGetErrorString(e), cudaGetErrorString(e0)); exit(1); }
    size_t nodes = 0; cudaGraphGetNodes(g, nullptr, &nodes);
    cudaGraphInstantiate(&ge, g, 0);
    cudaEvent_t a, c; cudaEventCreate(&a); cudaEventCreate(&c);
    for (int i = 0; i < warm; i++) cudaGraphLaunch(ge, s);
    std::vector<float> t;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(a, s); cudaGraphLaunch(ge, s); cudaEventRecord(c, s); cudaEventSynchronize(c);
        float ms; cudaEventElapsedTime(&ms, a, c); t.push_back(ms * 1000.f);
    }
    std::sort(t.begin(), t.end());
    printf("TIME %-28s nodes=%zu med_us=%.1f min_us=%.1f err=%s\n", name, nodes, t[t.size() / 2], t[0], cudaGetErrorString(cudaGetLastError()));
    fflush(stdout);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    return t[t.size() / 2];
}
static std::vector<float> outh() {
    std::vector<__half> h(2621440); cudaDeviceSynchronize();
    cudaMemcpy(h.data(), b[81], h.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> f(h.size()); for (size_t i = 0; i < h.size(); i++) f[i] = __half2float(h[i]); return f;
}
static void single(const char *name, std::function<void()> f) { timed(f, name, 5, 20); }
int main(int argc, char **argv) {
    b = (void**)calloc(@@NB@@, sizeof(void*));
@@ALLOC@@
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&evf, cudaEventDisableTiming); cudaEventCreateWithFlags(&evj, cudaEventDisableTiming);
    std::string which = argc > 1 ? argv[1] : "all";
    S0(); auto r0 = outh();
    S1(); auto r1 = outh();
    S2(); auto r2 = outh();
    double d1 = 0, d2 = 0, mx = 0;
    for (size_t i = 0; i < r0.size(); i++) { d1 = std::max(d1, (double)fabsf(r0[i] - r1[i])); d2 = std::max(d2, (double)fabsf(r0[i] - r2[i])); mx = std::max(mx, (double)fabsf(r0[i])); }
    printf("CHECK max|out| %.4f  max|S1-S0| %.3g  max|S2-S0| %.3g\n", mx, d1, d2);
    for (int rep = 0; rep < 2; rep++) {
        timed(S0, "S0_flow_final");
        timed(S1, "S1_all_v0_sequential");
        skip_dead = false; timed(S2, "S2_v0_bidir_concurrent");
        skip_dead = true;  timed(S2, "S3_S2_minus_dead_hn_concat"); skip_dead = false;
    }
    if (which == "all") {
@@SINGLES@@
    }
    return 0;
}
"""
drv = drv.replace("@@S1@@", s1).replace("@@S2@@", s2).replace("@@NB@@", str(nb)).replace("@@ALLOC@@", "\n".join(alloc)) \
         .replace("@@SINGLES@@", sing)
(HERE / "flow_bench.cu").write_text(src + "\n" + copies + "\n" + drv)
print("wrote flow_bench.cu", len(gru_ops), "gru ops", len(singles), "singles")
