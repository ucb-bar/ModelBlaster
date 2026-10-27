"""OpenCL (Qualcomm Adreno) target for multi-slot KernelBlaster problems.

``kb_export.export_problem(..., target="opencl")`` writes the same
``mb-kb-problem/1`` directory as the CUDA target (partition, buffers, ops,
data.bin, tolerances), with OpenCL slots, runtime and driver:

  slots/<id>.cpp     C++11: the slot's OpenCL C programs as raw strings
                     (``static mb_program cl_x = MB_CL(R"CL(...)CL");``) and
                     the host launcher with exactly the op's KernelSpec
                     signature; pointers are coarse-grained SVM pointers
  walker_head.cpp    the OpenCL runtime (gpu_eval/harness/opencl/mb_opencl_runtime.h):
                     context/queue, program build + kernel cache, mb_launch,
                     event recording, scratch, mb_copy / mb_zero
  walker_tail.cpp    mb_run_op(i) dispatch + launch_gpu_implementation()
  driver.cpp         checks + times the partition; reads data.bin at start-up
                     (MB_DATA_PATH, argv[1], ./data.bin, then the export path)

Assembly is the CUDA rule (head, slots in ``namespace mbslot_<id>``, tail);
the kernels TU and the driver are compiled together with ``g++ -std=c++11
... -lOpenCL`` -- on the host against any OpenCL 2.0 device with
coarse-grained SVM, or natively on the Adreno board (KernelBlaster's Adreno
flow compiles on the RB5 with gcc; no NDK needed). Concatenating the kernels
TU and the driver gives one self-contained main.cpp for single-file paths.

The driver prints the CUDA driver's lines (MB_CHECK, MB_SLOT_CHECK,
MB_PARTITION_US, MB_SLOT_US, MB_OP_US, passed|failed) plus

  MB_DEVICE <name> | <OpenCL version>
  MB_DEVICE_INFO max_wg=<n> compute_units=<n> local_mem=<bytes> global_mem=<bytes> fp16=<0|1>
  MB_KERNEL_US <slot> <kernel> <us> n=<launches>   per-kernel event time, summed per slot
  MB_KINFO <slot> <kernel> wg_max=<n> pref_mult=<n> local_mem=<bytes> private_mem=<bytes>

All times are cl_event COMMAND_START..END spans (as the Adreno RL flow
measures): an op is the span from its first command's start to its last
command's end, the partition the span over all ops.

    python -m modelblaster.pipeline.kb_export_opencl PROBLEM_DIR      # build + run here
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from .kb_export import NS_PREFIX, output_cond_term, wrap_slot

MB_ROOT = Path(__file__).resolve().parents[1]
OPENCL_DIR = MB_ROOT / "gpu_eval" / "harness" / "opencl"
RUNTIME_H = OPENCL_DIR / "mb_opencl_runtime.h"
HAND_LIB = OPENCL_DIR / "mb_opencl_kernels.cpp"
# element sizes of the manifest's buffer dtypes
ELEM_BYTES = {"f32": 4, "i32": 4, "i64": 8, "f16": 2}
CXX = os.environ.get("MB_OPENCL_CXX", "g++")
CXXFLAGS = ["-O2", "-std=c++11", "-w"]
LDFLAGS = ["-lOpenCL", "-lm"]


class OpenCLTarget:
    """``kb_export`` hooks for the OpenCL target."""

    name = "opencl"
    slot_ext = ".cpp"
    head_name, tail_name, driver_name = "walker_head.cpp", "walker_tail.cpp", "driver.cpp"

    def library(self, hand=None, dtype: str = "fp32"):          # fp16 seeds: OpenCLSeeds.slot_source(<entry>_f16)
        if dtype == "bf16":
            raise NotImplementedError("bf16 partitions are CUDA-only (docs/bf16_cuda.md); "
                                      "OpenCL has no bfloat16 storage type")
        if dtype == "mixed":
            raise NotImplementedError("mixed-dtype partitions (--quant native, docs/tensor_dtypes.md) "
                                      "are CUDA-only; the OpenCL seeds have one storage dtype")
        from .opencl_seeds import OpenCLSeeds
        return OpenCLSeeds(Path(hand) if hand else HAND_LIB)

    def seed(self, lib, entry: str, spec) -> tuple[str, str]:
        if hasattr(spec, "sub_ops"):
            return lib.fused_seed(spec), "fused-seed"
        return lib.slot_source(entry), lib.origin(entry)

    def head_source(self, lib) -> str:
        return "// OpenCL runtime shared by every slot (gpu_eval/harness/opencl/mb_opencl_runtime.h).\n" \
            + RUNTIME_H.read_text()

    def driver_source(self, data_path: Path, bufs, op_recs, slots, tol) -> str:
        return driver_source(data_path, bufs, op_recs, slots, tol)


# ---------------------------------------------------------------- driver

def driver_source(data_path: Path, bufs, op_recs, slots, tol) -> str:
    slot_ix = {s["id"]: k for k, s in enumerate(slots)}

    def arr(name, ctype, vals):
        vals = list(vals) or [0]
        return f"static const {ctype} {name}[] = {{{', '.join(str(v) for v in vals)}}};"

    op_in = [i for r in op_recs for i in r["inputs"]]
    op_out = [i for r in op_recs for i in r["outputs"]]
    in_ofs, out_ofs, a, c = [], [], 0, 0
    for r in op_recs:
        in_ofs.append(a); a += len(r["inputs"])
        out_ofs.append(c); c += len(r["outputs"])
    in_ofs.append(a); out_ofs.append(c)
    kinds = {"input": 0, "const": 1, "weight": 2, "intermediate": 3, "output": 4}
    dts = {"f32": 0, "i32": 1, "i64": 2, "f16": 3}
    tables = "\n".join([
        f"static const int NB = {len(bufs)}, NOPS = {len(op_recs)}, NSLOTS = {len(slots)};",
        arr("B_KIND", "int", (kinds[b["kind"]] for b in bufs)),
        arr("B_DT", "int", (dts[b["dtype"]] for b in bufs)),
        arr("B_NUMEL", "long", (b["numel"] for b in bufs)),
        arr("B_BASE", "int", (b["alias"]["base"] if b["alias"] else -1 for b in bufs)),
        arr("B_AOFF", "long", (b["alias"]["offset"] if b["alias"] else 0 for b in bufs)),
        arr("B_REF", "long", (b["ref"][0] if b["ref"] else -1 for b in bufs)),
        arr("B_COND", "double", (output_cond_term(b, tol) for b in bufs)),
        arr("OP_IN_OFS", "int", in_ofs), arr("OP_IN", "int", op_in),
        arr("OP_OUT_OFS", "int", out_ofs), arr("OP_OUT", "int", op_out),
        arr("OP_SLOT", "int", (slot_ix[r["slot"]] for r in op_recs)),
        "static const char *B_NAME[] = {" + ", ".join(json.dumps(b["name"]) for b in bufs) + "};",
        "static const char *SLOT_NAME[] = {" + ", ".join(json.dumps(s["id"]) for s in slots) + "};",
        "static const char *OP_NAME[] = {" + ", ".join(json.dumps(r["name"]) for r in op_recs) + "};",
    ])
    return (_DRIVER_TEMPLATE.replace("@TABLES@", tables).replace("@DATA_PATH@", json.dumps(str(data_path)))
            .replace("@OUT_REL@", repr(tol["output_rel"])).replace("@OUT_ABS@", repr(tol["output_abs"]))
            .replace("@SLOT_REL@", repr(tol["slot_rel"])).replace("@SLOT_ABS@", repr(tol.get("slot_abs", 0.0))))


_DRIVER_TEMPLATE = r'''// Generated by modelblaster.pipeline.kb_export (target opencl) -- checks and times one partition.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#include <CL/cl.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

void launch_gpu_implementation(void **bufs);
void mb_run_op(int i, void **bufs);
int mb_num_ops();
// the OpenCL runtime (walker head, in the kernels TU)
cl_device_id mb_device();
cl_context mb_context();
cl_command_queue mb_queue();
int mb_cl_init();
const char *mb_take_error();
void mb_set_op(int i);
void mb_set_record(int on);
int mb_num_events();
void mb_event_at(int i, int *op, const char **kernel, cl_ulong *t0, cl_ulong *t1);
void mb_clear_events();
int mb_num_kernels();
void mb_kernel_at(int i, cl_kernel *k, const char **name, int *op);

@TABLES@
static const double OUT_REL = @OUT_REL@, OUT_ABS = @OUT_ABS@, SLOT_REL = @SLOT_REL@, SLOT_ABS = @SLOT_ABS@;
enum { K_INPUT, K_CONST, K_WEIGHT, K_INTER, K_OUTPUT };
static const int ESZ[] = {4, 4, 8, 2};
// binary16 -> float, exact (subnormals, inf, NaN)
static float half_to_float(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}

static unsigned char *blob;
static void *dev[NB];

static size_t nbytes(int b) { return (size_t)B_NUMEL[b] * ESZ[B_DT[b]]; }
static bool report(const char *what) {   // print + clear the runtime's first error
    const char *err = mb_take_error();
    cl_int e = clFinish(mb_queue());
    if (!err && e == CL_SUCCESS) return true;
    if (!err) { printf("MB_ERROR %s: clFinish error %d\n", what, e); return false; }
    std::string m = err;
    for (char &ch : m) if (ch == '\n' || ch == '\r') ch = '|';
    if (m.size() > 3000) m = m.substr(0, 3000) + "...";
    printf("MB_ERROR %s: %s\n", what, m.c_str());
    return false;
}
static const unsigned char *ref_bytes(int b) {
    int base = B_BASE[b] >= 0 ? B_BASE[b] : b;
    if (B_REF[base] < 0) return nullptr;
    return blob + B_REF[base] + (B_BASE[b] >= 0 ? B_AOFF[b] * ESZ[B_DT[b]] : 0);
}
static std::vector<double> widen(const unsigned char *p, int b) {
    std::vector<double> v(B_NUMEL[b]);
    for (long i = 0; i < B_NUMEL[b]; ++i) {
        if (B_DT[b] == 0) { float f; memcpy(&f, p + 4 * i, 4); v[i] = f; }
        else if (B_DT[b] == 1) { int x; memcpy(&x, p + 4 * i, 4); v[i] = x; }
        else if (B_DT[b] == 2) { long long x; memcpy(&x, p + 8 * i, 8); v[i] = (double)x; }
        else { unsigned short x; memcpy(&x, p + 2 * i, 2); v[i] = half_to_float(x); }
    }
    return v;
}
static std::vector<double> ref_vec(int b) { return widen(ref_bytes(b), b); }
static std::vector<double> dev_vec(int b) {
    std::vector<unsigned char> raw(std::max(nbytes(b), (size_t)1));
    clEnqueueSVMMemcpy(mb_queue(), CL_TRUE, raw.data(), dev[b], nbytes(b), 0, NULL, NULL);
    return widen(raw.data(), b);
}
static void put(int b, const unsigned char *src) {
    if (src && nbytes(b)) clEnqueueSVMMemcpy(mb_queue(), CL_TRUE, dev[b], src, nbytes(b), 0, NULL, NULL);
}
static double max_finite(const std::vector<double> &v) {
    double m = 0; for (double x : v) if (std::isfinite(x)) m = std::max(m, std::fabs(x)); return m;
}
static double max_err(const std::vector<double> &a, const std::vector<double> &r) {
    double e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double x = a[i], y = r[i];
        if (x == y || (std::isnan(x) && std::isnan(y))) continue;
        if (!std::isfinite(x) || !std::isfinite(y)) return INFINITY;
        e = std::max(e, std::fabs(x - y));
    }
    return e;
}
static void upload_inputs() {
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0 && (B_KIND[b] == K_INPUT || B_KIND[b] == K_CONST || B_KIND[b] == K_WEIGHT)) put(b, ref_bytes(b));
}
static bool load_blob(int argc, char **argv) {
    const char *cands[] = {getenv("MB_DATA_PATH"), argc > 1 ? argv[1] : nullptr, "data.bin", @DATA_PATH@};
    for (const char *path : cands) {
        if (!path) continue;
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        blob = (unsigned char *)malloc(n > 0 ? n : 1);
        bool ok = fread(blob, 1, n, f) == (size_t)n;
        fclose(f);
        if (!ok) { printf("MB_ERROR short read of %s\n", path); return false; }
        return true;
    }
    printf("MB_ERROR cannot open data.bin (set MB_DATA_PATH)\n");
    return false;
}

struct span { double t0 = 1e300, t1 = -1e300; void add(double a, double b) { t0 = std::min(t0, a); t1 = std::max(t1, b); }
              double us() const { return t1 > t0 ? (t1 - t0) / 1000.0 : 0.0; } };

static void kinfo_lines() {
    for (int i = 0; i < mb_num_kernels(); ++i) {
        cl_kernel k; const char *name; int op;
        mb_kernel_at(i, &k, &name, &op);
        size_t wg = 0, mult = 0; cl_ulong lm = 0, pm = 0;
        clGetKernelWorkGroupInfo(k, mb_device(), CL_KERNEL_WORK_GROUP_SIZE, sizeof wg, &wg, NULL);
        clGetKernelWorkGroupInfo(k, mb_device(), CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof mult, &mult, NULL);
        clGetKernelWorkGroupInfo(k, mb_device(), CL_KERNEL_LOCAL_MEM_SIZE, sizeof lm, &lm, NULL);
        clGetKernelWorkGroupInfo(k, mb_device(), CL_KERNEL_PRIVATE_MEM_SIZE, sizeof pm, &pm, NULL);
        printf("MB_KINFO %s %s wg_max=%zu pref_mult=%zu local_mem=%llu private_mem=%llu\n",
               op >= 0 ? SLOT_NAME[OP_SLOT[op]] : "?", name, wg, mult, (unsigned long long)lm, (unsigned long long)pm);
    }
}
// events recorded since the last call -> partition span, per-op spans, per-(slot, kernel) busy time
static void take_events(span *part, std::vector<span> *ops, std::map<std::pair<int, std::string>, std::pair<double, int> > *kus) {
    for (int i = 0; i < mb_num_events(); ++i) {
        int op; const char *kn; cl_ulong s = 0, t = 0;
        mb_event_at(i, &op, &kn, &s, &t);
        if (part) part->add((double)s, (double)t);
        if (op < 0) continue;
        if (ops) (*ops)[op].add((double)s, (double)t);
        auto &k = (*kus)[std::make_pair(OP_SLOT[op], std::string(kn))];
        k.first += (double)(t - s) / 1000.0; k.second += 1;
    }
    mb_clear_events();
}

int main(int argc, char **argv) {
    bool all_ok = true;
    if (!load_blob(argc, argv)) { printf("failed\n"); return 1; }
    if (mb_cl_init()) { printf("MB_ERROR %s\nfailed\n", mb_take_error()); return 1; }
    {
        char name[256] = {0}, ver[256] = {0};
        clGetDeviceInfo(mb_device(), CL_DEVICE_NAME, sizeof name, name, NULL);
        clGetDeviceInfo(mb_device(), CL_DEVICE_VERSION, sizeof ver, ver, NULL);
        size_t mwg = 0; cl_uint cus = 0; cl_ulong lmem = 0, gmem = 0; char ext[8192] = {0};
        clGetDeviceInfo(mb_device(), CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof mwg, &mwg, NULL);
        clGetDeviceInfo(mb_device(), CL_DEVICE_MAX_COMPUTE_UNITS, sizeof cus, &cus, NULL);
        clGetDeviceInfo(mb_device(), CL_DEVICE_LOCAL_MEM_SIZE, sizeof lmem, &lmem, NULL);
        clGetDeviceInfo(mb_device(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof gmem, &gmem, NULL);
        clGetDeviceInfo(mb_device(), CL_DEVICE_EXTENSIONS, sizeof ext - 1, ext, NULL);
        printf("MB_DEVICE %s | %s\n", name, ver);
        printf("MB_DEVICE_INFO max_wg=%zu compute_units=%u local_mem=%llu global_mem=%llu fp16=%d\n", mwg, cus,
               (unsigned long long)lmem, (unsigned long long)gmem, strstr(ext, "cl_khr_fp16") != NULL);
    }
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] < 0) {
            size_t n = std::max(nbytes(b), (size_t)ESZ[B_DT[b]]);
            dev[b] = clSVMAlloc(mb_context(), CL_MEM_READ_WRITE, n, 0);
            if (!dev[b]) { printf("MB_ERROR alloc %s (%zu bytes)\nfailed\n", B_NAME[b], n); return 1; }
            cl_uchar z = 0;
            clEnqueueSVMMemFill(mb_queue(), dev[b], &z, 1, n, 0, NULL, NULL);
        }
    for (int b = 0; b < NB; ++b)
        if (B_BASE[b] >= 0) dev[b] = (unsigned char *)dev[B_BASE[b]] + B_AOFF[b] * ESZ[B_DT[b]];
    clFinish(mb_queue());
    if (mb_num_ops() != NOPS) { printf("MB_ERROR walker has %d ops, driver %d\nfailed\n", mb_num_ops(), NOPS); return 1; }
    mb_set_record(0);

    // MB_MODE=profile (or argv[2] = profile): every op exactly once on the reference inputs, then
    // per-kernel event times and kernel resource info (for external profilers too)
    const char *mode = getenv("MB_MODE");
    if (!mode && argc > 2) mode = argv[2];      // exec paths that cannot pass environment variables
    if (mode && !strcmp(mode, "profile")) {
        upload_inputs();
        mb_set_record(1);
        for (int i = 0; i < NOPS; ++i) { mb_set_op(i); mb_run_op(i, dev); }
        bool ok = report("profile");
        std::map<std::pair<int, std::string>, std::pair<double, int> > kus;
        take_events(nullptr, nullptr, &kus);
        for (auto &kv : kus) printf("MB_KERNEL_US %s %s %.3f n=%d\n", SLOT_NAME[kv.first.first], kv.first.second.c_str(),
                                    kv.second.first, kv.second.second);
        kinfo_lines();
        printf(ok ? "MB_PROFILE_DONE\n" : "failed\n");
        return ok ? 0 : 1;
    }

    // 1. whole partition vs reference
    upload_inputs();
    for (int i = 0; i < NOPS; ++i) { mb_set_op(i); mb_run_op(i, dev); }
    if (!report("partition")) { printf("failed\n"); return 1; }
    for (int b = 0; b < NB; ++b) {
        if (B_KIND[b] != K_OUTPUT) continue;
        std::vector<double> r = ref_vec(b), a = dev_vec(b);
        double e = max_err(a, r), bound = OUT_REL * max_finite(r) + OUT_ABS + B_COND[b];
        bool ok = e <= bound;
        all_ok &= ok;
        printf("MB_CHECK output %s max_abs_err=%.6g bound=%.6g ok=%d\n", B_NAME[b], e, bound, (int)ok);
    }

    // 2. every op alone on the reference's inputs
    std::vector<double> slot_err(NSLOTS, 0.0);
    std::vector<int> slot_bad(NSLOTS, 0);
    for (int i = 0; i < NOPS; ++i) {
        upload_inputs();
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k) {
            int b = OP_IN[k];
            if (B_KIND[b] == K_CONST || B_KIND[b] == K_WEIGHT) continue;
            put(b, ref_bytes(b));
        }
        mb_set_op(i);
        mb_run_op(i, dev);
        if (!report(OP_NAME[i])) { slot_bad[OP_SLOT[i]] = 1; slot_err[OP_SLOT[i]] = INFINITY; continue; }
        double scale = 0;
        for (int k = OP_IN_OFS[i]; k < OP_IN_OFS[i + 1]; ++k)
            if (B_DT[OP_IN[k]] == 0 || B_DT[OP_IN[k]] == 3) scale = std::max(scale, max_finite(ref_vec(OP_IN[k])));
        for (int k = OP_OUT_OFS[i]; k < OP_OUT_OFS[i + 1]; ++k) {
            int b = OP_OUT[k];
            std::vector<double> r = ref_vec(b), a = dev_vec(b);
            double s = std::max(scale, max_finite(r)), e = max_err(a, r);
            double rel = e / std::max(s, 1e-12);
            slot_err[OP_SLOT[i]] = std::max(slot_err[OP_SLOT[i]], rel);
            if (!(e <= SLOT_REL * s + SLOT_ABS)) slot_bad[OP_SLOT[i]] = 1;
        }
    }
    for (int s = 0; s < NSLOTS; ++s) {
        all_ok &= !slot_bad[s];
        printf("MB_SLOT_CHECK %s rel_err=%.6g ok=%d\n", SLOT_NAME[s], slot_err[s], (int)!slot_bad[s]);
    }

    // 3. timing from cl_events: whole partition, every op, every kernel
    if (all_ok) {
        upload_inputs();
        for (int w = 0; w < 2; ++w) for (int i = 0; i < NOPS; ++i) { mb_set_op(i); mb_run_op(i, dev); }
        clFinish(mb_queue());
        const char *env = getenv("MB_TIME_MS");
        double budget_us = 1000.0 * (env ? atof(env) : 200.0), total = 0;
        std::vector<double> op_us(NOPS, 0.0);
        std::vector<int> op_seen(NOPS, 0);
        std::map<std::pair<int, std::string>, std::pair<double, int> > kus;
        int reps = 0;
        mb_set_record(1);
        while (reps < 3 || (total < budget_us && reps < 200)) {
            for (int i = 0; i < NOPS; ++i) { mb_set_op(i); mb_run_op(i, dev); }
            if (!report("timing")) { all_ok = false; break; }
            span part;
            std::vector<span> ops(NOPS);
            take_events(&part, &ops, &kus);
            total += part.us();
            for (int i = 0; i < NOPS; ++i) { op_us[i] += ops[i].us(); op_seen[i] |= ops[i].t1 >= ops[i].t0; }
            ++reps;
        }
        mb_set_record(0);
        if (all_ok) {
            for (int i = 0; i < NOPS; ++i)
                if (!op_seen[i]) { printf("MB_ERROR op %s enqueued no command through mb_launch/mb_copy/mb_zero\n", OP_NAME[i]); all_ok = false; }
            printf("MB_PARTITION_US %.3f reps=%d\n", total / reps, reps);
            std::vector<double> slot_us(NSLOTS, 0.0);
            for (int i = 0; i < NOPS; ++i) slot_us[OP_SLOT[i]] += op_us[i] / reps;
            for (int s = 0; s < NSLOTS; ++s) printf("MB_SLOT_US %s %.3f\n", SLOT_NAME[s], slot_us[s]);
            for (int i = 0; i < NOPS; ++i) printf("MB_OP_US %s %.3f\n", OP_NAME[i], op_us[i] / reps);
            for (auto &kv : kus) printf("MB_KERNEL_US %s %s %.3f n=%d\n", SLOT_NAME[kv.first.first], kv.first.second.c_str(),
                                        kv.second.first / reps, kv.second.second / reps);
            kinfo_lines();
        }
    }
    printf(all_ok ? "passed\n" : "failed\n");
    return all_ok ? 0 : 1;
}
'''


# ---------------------------------------------------------------- assembly + local build/run

def assemble(problem_dir: Path, sources: dict[str, str] | None = None) -> str:
    """The kernels TU: head + wrapped slots + tail (the CUDA assembly rule)."""
    problem_dir = Path(problem_dir)
    man = json.loads((problem_dir / "manifest.json").read_text())
    asm = man["assembly"]
    parts = [(problem_dir / asm["head"]).read_text()]
    for s in man["slots"]:
        code = (sources or {}).get(s["id"]) or (problem_dir / s["file"]).read_text()
        parts.append(wrap_slot(s["id"], code))
    parts.append((problem_dir / asm["tail"]).read_text())
    return "\n".join(parts)


def single_tu(kernels: str, driver: str) -> str:
    """kernels TU + driver as one main.cpp (single-file compile paths)."""
    return kernels + "\n#define MB_SINGLE_TU 1\n" + driver


def build_and_run(problem_dir, *, sources: dict[str, str] | None = None, work: Path | None = None,
                  timeout: int = 1800, env: dict | None = None) -> dict:
    """Assemble + g++-build + run the problem on this host's OpenCL device."""
    from .kb_export import parse_driver_output

    problem_dir = Path(problem_dir)
    man = json.loads((problem_dir / "manifest.json").read_text())
    work = Path(work or problem_dir / "build")
    work.mkdir(parents=True, exist_ok=True)
    (work / "kernels.cpp").write_text(assemble(problem_dir, sources))
    exe = work / "problem"
    cp = subprocess.run([CXX, *CXXFLAGS, "-o", str(exe), str(work / "kernels.cpp"),
                         str(problem_dir / man["driver"]), *LDFLAGS],
                        capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        return {"passed": False, "stage": "compile", "log": (cp.stdout + cp.stderr)[-4000:]}
    try:
        rp = subprocess.run([str(exe), str(problem_dir / man["data"])], capture_output=True, text=True,
                            timeout=timeout, env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired:
        return {"passed": False, "stage": "run", "log": f"timeout after {timeout}s"}
    res = parse_driver_output(rp.stdout)
    res["kernel_us"] = {f"{m.group(1)}/{m.group(2)}": float(m.group(3))
                        for m in re.finditer(r"^MB_KERNEL_US (\S+) (\S+) (\S+)", rp.stdout, re.M)}
    m = re.search(r"^MB_DEVICE (.*)$", rp.stdout, re.M)
    res["device"] = m.group(1) if m else None
    res.update(stage="run", log=(rp.stdout + rp.stderr)[-4000:])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem_dir")
    a = ap.parse_args()
    r = build_and_run(Path(a.problem_dir))
    print(json.dumps({k: r.get(k) for k in ("passed", "stage", "device", "partition_us", "slot_us", "slots",
                                            "errors")}, indent=1))
    if not r.get("passed"):
        print(r.get("log", "")[-3000:])


if __name__ == "__main__":
    main()
