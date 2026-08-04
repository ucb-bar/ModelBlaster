// weight_rebind_test.cpp
//
// Test: can a Conv1x1 ctx binary built with weight-as-graph-input have its
// weight tensor's clientBuf swapped between successive graphExecute calls,
// such that the output changes accordingly?
//
// Procedure:
//   1. Load the context binary (built from variant_b_input_weight.onnx).
//   2. Discover input tensors: "x" (activation) and "weight".
//   3. Allocate per-tensor buffers (x_buf, w1_buf, w2_buf, y1_buf, y2_buf).
//   4. Fill w1_buf with random pattern A, w2_buf with random pattern B.
//   5. Bind x→x_buf, weight→w1_buf, y→y1_buf; graphExecute. Capture y1.
//   6. Bind weight→w2_buf (different bytes), graphExecute. Capture y2.
//   7. Compare y1 vs y2:
//        - If outputs differ: weight rebinding WORKS on this backend.
//        - If outputs identical: weight is being cached/snapshotted; rebinding NOT effective.
//   8. Also compute expected y_ref from numpy-style conv on CPU (or just
//      verify that BOTH y1 != zero and y2 != zero and y1 != y2 — that's
//      enough to confirm the runtime is actually using each weight buf).
//
// Usage:
//   ./weight_rebind_test <ctx.bin> <backend_lib.so>
// Example:
//   ./weight_rebind_test variant_b_input_weight_q__Dsp.bin libQnnDsp.so

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

#define CHECK(expr) do { Qnn_ErrorHandle_t _e=(expr); \
    if (_e != QNN_SUCCESS) { std::fprintf(stderr,"QNN err 0x%llx %s:%d %s\n", \
        (unsigned long long)_e, __FILE__, __LINE__, #expr); std::exit(1);} } while(0)

static double now_us() {
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}
static void log_cb(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
    std::vfprintf(stderr, fmt, ap); std::fprintf(stderr, "\n");
}
static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr,"open %s failed\n",p.c_str()); std::exit(1); }
    auto n = f.tellg(); f.seekg(0); std::vector<uint8_t> v(n);
    f.read((char*)v.data(),n); return v;
}

static uint32_t trank(const Qnn_Tensor_t& t)        { return t.version==QNN_TENSOR_VERSION_2 ? t.v2.rank      : t.v1.rank; }
static const uint32_t* tdims(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions; }
static Qnn_DataType_t tdtype(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2 ? t.v2.dataType   : t.v1.dataType; }
static const char* tname(const Qnn_Tensor_t& t)     { return t.version==QNN_TENSOR_VERSION_2 ? t.v2.name       : t.v1.name; }

static size_t bpe(Qnn_DataType_t dt) {
    switch(dt) {
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8: return 1;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16: case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16: return 2;
        case QNN_DATATYPE_FLOAT_32: case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32: return 4;
        default: return 4;
    }
}
static size_t tbytes(const Qnn_Tensor_t& t) {
    size_t n=1; uint32_t r=trank(t); auto* d=tdims(t);
    for (uint32_t i=0;i<r;++i) n*=d[i];
    return n*bpe(tdtype(t));
}
static void rebind(Qnn_Tensor_t& t, std::vector<uint32_t>& d, std::string& nm) {
    uint32_t r=trank(t); auto* dp=tdims(t); const char* np=tname(t);
    d.assign(dp, dp+r); nm = np?np:"";
    if (t.version==QNN_TENSOR_VERSION_2) { t.v2.dimensions=d.data(); t.v2.name=nm.c_str(); }
    else                                  { t.v1.dimensions=d.data(); t.v1.name=nm.c_str(); }
}
static void set_buf(Qnn_Tensor_t& t, void* data, uint32_t bytes) {
    if (t.version==QNN_TENSOR_VERSION_2) {
        t.v2.memType = QNN_TENSORMEMTYPE_RAW;
        t.v2.clientBuf.data=data; t.v2.clientBuf.dataSize=bytes;
    } else {
        t.v1.memType = QNN_TENSORMEMTYPE_RAW;
        t.v1.clientBuf.data=data; t.v1.clientBuf.dataSize=bytes;
    }
}

// Compute a content hash so we can compare outputs across runs.
static uint64_t hash_bytes(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 1469598103934665603ull;   // FNV-1a 64
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static double compute_max_abs_diff(const std::vector<uint8_t>& a,
                                     const std::vector<uint8_t>& b,
                                     Qnn_DataType_t dt) {
    // Bytewise int8 diff is a reasonable signal for quantized outputs.
    size_t n = std::min(a.size(), b.size());
    if (dt == QNN_DATATYPE_FLOAT_32) {
        size_t nf = n / 4;
        const float* fa = (const float*)a.data();
        const float* fb = (const float*)b.data();
        double mx = 0;
        for (size_t i = 0; i < nf; ++i) {
            double d = std::fabs((double)fa[i] - (double)fb[i]);
            if (d > mx) mx = d;
        }
        return mx;
    } else {
        int mx = 0;
        for (size_t i = 0; i < n; ++i) {
            int d = std::abs((int)a[i] - (int)b[i]);
            if (d > mx) mx = d;
        }
        return mx;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <ctx.bin> <backend_lib.so>\n", argv[0]);
        return 1;
    }
    std::string bin_path = argv[1];
    std::string lib_path = argv[2];

    // 1. dlopen libQnnSystem.so for graph introspection.
    void* slib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!slib) { std::fprintf(stderr,"dlopen libQnnSystem.so: %s\n", dlerror()); return 1; }
    auto sgp = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
                dlsym(slib, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sprov=nullptr; uint32_t sn=0;
    CHECK(sgp(&sprov, &sn));
    auto& siface = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    auto bin = slurp(bin_path);
    QnnSystemContext_Handle_t sh=nullptr;
    CHECK(siface.systemContextCreate(&sh));
    const QnnSystemContext_BinaryInfo_t* bi=nullptr;
    Qnn_ContextBinarySize_t bisz=0;
    CHECK(siface.systemContextGetBinaryInfo(sh, bin.data(), bin.size(), &bi, &bisz));

    const QnnSystemContext_GraphInfo_t* gs=nullptr;
    if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)      gs=bi->contextBinaryInfoV3.graphs;
    else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) gs=bi->contextBinaryInfoV2.graphs;
    else                                                            gs=bi->contextBinaryInfoV1.graphs;

    const auto& g = gs[0];
    const char* gname=nullptr; uint32_t nIn=0,nOut=0;
    const Qnn_Tensor_t *inT=nullptr,*outT=nullptr;
    if (g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        gname=g.graphInfoV3.graphName;
        nIn=g.graphInfoV3.numGraphInputs;  inT=g.graphInfoV3.graphInputs;
        nOut=g.graphInfoV3.numGraphOutputs; outT=g.graphInfoV3.graphOutputs;
    } else if (g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        gname=g.graphInfoV2.graphName;
        nIn=g.graphInfoV2.numGraphInputs;  inT=g.graphInfoV2.graphInputs;
        nOut=g.graphInfoV2.numGraphOutputs; outT=g.graphInfoV2.graphOutputs;
    } else {
        gname=g.graphInfoV1.graphName;
        nIn=g.graphInfoV1.numGraphInputs;  inT=g.graphInfoV1.graphInputs;
        nOut=g.graphInfoV1.numGraphOutputs; outT=g.graphInfoV1.graphOutputs;
    }

    std::vector<Qnn_Tensor_t> inputs(inT, inT+nIn), outputs(outT, outT+nOut);
    std::vector<std::vector<uint32_t>> dimStorage(nIn+nOut);
    std::vector<std::string>           nameStorage(nIn+nOut);
    for (size_t i=0;i<inputs.size();++i)  rebind(inputs[i],  dimStorage[i],     nameStorage[i]);
    for (size_t i=0;i<outputs.size();++i) rebind(outputs[i], dimStorage[nIn+i], nameStorage[nIn+i]);

    std::string graph_name = gname ? gname : "";
    siface.systemContextFree(sh);

    std::printf("Graph '%s' has %u input(s), %u output(s):\n",
                 graph_name.c_str(), nIn, nOut);
    for (uint32_t i = 0; i < nIn; ++i) {
        auto& t = inputs[i];
        size_t n = tbytes(t);
        std::printf("  in[%u] '%s' rank=%u dtype=%d bytes=%zu\n",
                     i, nameStorage[i].c_str(), trank(t), (int)tdtype(t), n);
    }
    for (uint32_t i = 0; i < nOut; ++i) {
        auto& t = outputs[i];
        size_t n = tbytes(t);
        std::printf("  out[%u] '%s' rank=%u dtype=%d bytes=%zu\n",
                     i, nameStorage[nIn+i].c_str(), trank(t), (int)tdtype(t), n);
    }

    // Find the "weight" input index.
    int weight_idx = -1, x_idx = -1;
    for (uint32_t i = 0; i < nIn; ++i) {
        if (nameStorage[i] == "weight") weight_idx = (int)i;
        else if (nameStorage[i] == "x")  x_idx = (int)i;
    }
    if (weight_idx < 0) {
        std::printf("ERROR: this ctx binary has no 'weight' graph input — it's the "
                     "variant-A baked-in build. Use the variant-B binary.\n");
        return 1;
    }
    if (x_idx < 0) x_idx = 0;
    std::printf("Identified weight input at index %d, activation input at index %d.\n",
                 weight_idx, x_idx);

    // 2. dlopen backend lib + create context from binary.
    void* blib = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!blib) { std::fprintf(stderr,"dlopen %s: %s\n", lib_path.c_str(), dlerror()); return 1; }
    auto bgp = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
                dlsym(blib, "QnnInterface_getProviders");
    const QnnInterface_t** bprov=nullptr; uint32_t bn=0;
    CHECK(bgp(&bprov, &bn));
    auto biface = bprov[0]->QNN_INTERFACE_VER_NAME;

    Qnn_LogHandle_t logH=nullptr;
    CHECK(biface.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &logH));
    Qnn_BackendHandle_t backH=nullptr;
    CHECK(biface.backendCreate(logH, nullptr, &backH));

    Qnn_ContextHandle_t ctx=nullptr;
    CHECK(biface.contextCreateFromBinary(backH, nullptr, nullptr,
                                          bin.data(), bin.size(), &ctx, nullptr));
    Qnn_GraphHandle_t graph=nullptr;
    CHECK(biface.graphRetrieve(ctx, graph_name.c_str(), &graph));

    // 3. Allocate buffers.
    std::vector<std::vector<uint8_t>> inBufs(nIn), outBufs(nOut);
    for (uint32_t i = 0; i < nIn; ++i)  inBufs[i].assign(tbytes(inputs[i]),  0);
    for (uint32_t i = 0; i < nOut; ++i) outBufs[i].assign(tbytes(outputs[i]), 0);

    // 4. Fill 'x' with a stable pattern, and prep two distinct weight buffers.
    std::mt19937 rng(42);
    auto fill_random_bytes = [&](std::vector<uint8_t>& buf, uint32_t seed){
        std::mt19937 r(seed);
        for (auto& b : buf) b = (uint8_t)(r() & 0xff);
    };
    fill_random_bytes(inBufs[x_idx], 100);
    std::vector<uint8_t> w_buf_A = inBufs[weight_idx];
    std::vector<uint8_t> w_buf_B = inBufs[weight_idx];
    fill_random_bytes(w_buf_A, 200);
    fill_random_bytes(w_buf_B, 300);

    auto bind_all = [&](void* wbuf){
        for (uint32_t i = 0; i < nIn; ++i) {
            void* data = (i == (uint32_t)weight_idx) ? wbuf : inBufs[i].data();
            set_buf(inputs[i], data, (uint32_t)inBufs[i].size());
        }
        for (uint32_t i = 0; i < nOut; ++i)
            set_buf(outputs[i], outBufs[i].data(), (uint32_t)outBufs[i].size());
    };

    auto run_once = [&](){
        double t0 = now_us();
        CHECK(biface.graphExecute(graph,
            inputs.data(),  (uint32_t)inputs.size(),
            outputs.data(), (uint32_t)outputs.size(),
            nullptr, nullptr));
        return now_us() - t0;
    };

    // Warmup (some QNN backends pay a cold-start cost on the first call).
    bind_all(w_buf_A.data());
    run_once();

    // Run 1: weight = pattern A
    bind_all(w_buf_A.data());
    double dt_A = run_once();
    std::vector<uint8_t> y_A = outBufs[0];
    uint64_t hash_y_A = hash_bytes(y_A.data(), y_A.size());
    bool y_A_allzero = true;
    for (auto b : y_A) if (b != 0) { y_A_allzero = false; break; }

    // Run 2: weight = pattern B (different bytes, same shape/dtype)
    std::fill(outBufs[0].begin(), outBufs[0].end(), 0);
    bind_all(w_buf_B.data());
    double dt_B = run_once();
    std::vector<uint8_t> y_B = outBufs[0];
    uint64_t hash_y_B = hash_bytes(y_B.data(), y_B.size());
    bool y_B_allzero = true;
    for (auto b : y_B) if (b != 0) { y_B_allzero = false; break; }

    // Run 3: re-run with weight A. Expect y_A2 == y_A (determinism check).
    std::fill(outBufs[0].begin(), outBufs[0].end(), 0);
    bind_all(w_buf_A.data());
    run_once();
    std::vector<uint8_t> y_A2 = outBufs[0];
    uint64_t hash_y_A2 = hash_bytes(y_A2.data(), y_A2.size());

    double diff_AB = compute_max_abs_diff(y_A, y_B, tdtype(outputs[0]));
    double diff_AA = compute_max_abs_diff(y_A, y_A2, tdtype(outputs[0]));

    std::printf("\n=== Weight rebind result (%s on %s) ===\n",
                 bin_path.c_str(), lib_path.c_str());
    std::printf("  run A:  exec %.1f us, output hash %016lx  (all-zero=%s)\n",
                 dt_A, (unsigned long)hash_y_A, y_A_allzero ? "YES" : "no");
    std::printf("  run B:  exec %.1f us, output hash %016lx  (all-zero=%s)\n",
                 dt_B, (unsigned long)hash_y_B, y_B_allzero ? "YES" : "no");
    std::printf("  run A2: output hash %016lx  (should equal A)\n",
                 (unsigned long)hash_y_A2);
    std::printf("  max_abs(y_A - y_B)  = %.4f  (>0 means weight rebind has effect)\n", diff_AB);
    std::printf("  max_abs(y_A - y_A2) = %.4f  (==0 confirms determinism)\n", diff_AA);
    std::printf("\n");
    if (hash_y_A == hash_y_B && !y_A_allzero) {
        std::printf("VERDICT: REBIND HAS NO EFFECT — backend cached weights at "
                     "context-create time.\n");
        return 2;
    } else if (y_A_allzero && y_B_allzero) {
        std::printf("VERDICT: UNCLEAR — both outputs zero. Maybe quant range "
                     "collapsed input data; rerun with non-byte-randomized data.\n");
        return 3;
    } else if (hash_y_A2 != hash_y_A) {
        std::printf("VERDICT: NON-DETERMINISTIC — re-running with weight A gave "
                     "a different result than the first run. Suspicious.\n");
        return 4;
    } else {
        std::printf("VERDICT: WEIGHT REBIND WORKS — different weight bytes "
                     "produce different outputs, and re-running with the same "
                     "weight is deterministic.\n");
        return 0;
    }
}
