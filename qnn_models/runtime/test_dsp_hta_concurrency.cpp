// Empirical test: do HTA and DSP run concurrently in one process?
//
// Loads dronet-on-HTA + yolov8n-backbone-on-DSP and runs them in two
// configurations:
//   1. SERIAL — main thread fires HTA call, waits, fires DSP call,
//      waits. Repeat N times.
//   2. PARALLEL — two std::thread workers each loop their own
//      graphExecute calls N times. Wait for both to finish.
//
// If the accelerators are truly independent, parallel-elapsed ≈
// max(HTA_only, DSP_only). If a process-global lock or shared
// FastRPC session serialises them, parallel-elapsed ≈ HTA_only +
// DSP_only. Speedup = serial / parallel; >1.5× ⇒ meaningful
// concurrency, ~1× ⇒ effectively serialised.
//
// Build (on QRB5165):
//   g++ -std=c++2a -O2 -pthread -I/root/qairt/include \
//       test_dsp_hta_concurrency.cpp -o test_concurrency -ldl

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/System/QnnSystemInterface.h"
#include "QNN/System/QnnSystemContext.h"

#define CHECK(expr) do {                                                     \
    Qnn_ErrorHandle_t _e = (expr);                                           \
    if (_e != QNN_SUCCESS) {                                                 \
        std::fprintf(stderr, "QNN err 0x%llx %s:%d %s\n",                    \
                      (unsigned long long)_e, __FILE__, __LINE__, #expr);    \
        std::exit(1);                                                         \
    } } while (0)

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
static void log_cb(const char* fmt, QnnLog_Level_t, uint64_t, va_list ap) {
    std::vfprintf(stderr, fmt, ap); std::fprintf(stderr, "\n");
}
static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(1); }
    auto n = f.tellg(); f.seekg(0); std::vector<uint8_t> v(n);
    f.read((char*)v.data(), n); return v;
}

static uint32_t trank(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}
static const uint32_t* tdims(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}
static Qnn_DataType_t tdtype(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dataType : t.v1.dataType;
}
static const char* tname(const Qnn_Tensor_t& t) {
    return t.version == QNN_TENSOR_VERSION_2 ? t.v2.name : t.v1.name;
}
static size_t bpe(Qnn_DataType_t dt) {
    switch (dt) {
        case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_BOOL_8: return 1;
        case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_INT_16:
        case QNN_DATATYPE_UINT_16: return 2;
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

struct Ctx {
    std::string name;
    void* lib=nullptr;
    QNN_INTERFACE_VER_TYPE iface{};
    Qnn_LogHandle_t log=nullptr;
    Qnn_BackendHandle_t backend=nullptr;
    Qnn_ContextHandle_t ctx=nullptr;
    Qnn_GraphHandle_t   graph=nullptr;
    std::string graphName;
    std::vector<Qnn_Tensor_t> inputs, outputs;
    std::vector<std::vector<uint8_t>> inBufs, outBufs;
    std::vector<std::vector<uint32_t>> dimStorage;
    std::vector<std::string>           nameStorage;
};

struct SysFns {
    QnnSystemContext_CreateFn_t create=nullptr;
    QnnSystemContext_GetBinaryInfoFn_t getInfo=nullptr;
    QnnSystemContext_FreeFn_t  free=nullptr;
};
static SysFns load_sys() {
    SysFns f{};
    void* l = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!l) { std::fprintf(stderr, "dlopen libQnnSystem.so: %s\n", dlerror()); std::exit(1); }
    auto fn = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
                dlsym(l, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** prov=nullptr; uint32_t n=0;
    CHECK(fn(&prov, &n));
    auto& s = prov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    f.create=s.systemContextCreate; f.getInfo=s.systemContextGetBinaryInfo;
    f.free=s.systemContextFree;
    return f;
}

static void load_ctx(Ctx& bc, const std::string& libpath, const std::string& binpath,
                      const std::string& tag, const SysFns& sys) {
    bc.name = tag;
    bc.lib = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!bc.lib) { std::fprintf(stderr, "dlopen %s: %s\n", libpath.c_str(), dlerror()); std::exit(1); }
    auto getProv = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
                    dlsym(bc.lib, "QnnInterface_getProviders");
    const QnnInterface_t** prov=nullptr; uint32_t np=0;
    CHECK(getProv(&prov, &np));
    bc.iface = prov[0]->QNN_INTERFACE_VER_NAME;
    CHECK(bc.iface.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &bc.log));
    CHECK(bc.iface.backendCreate(bc.log, nullptr, &bc.backend));

    auto bin = slurp(binpath);
    QnnSystemContext_Handle_t sh=nullptr;
    CHECK(sys.create(&sh));
    const QnnSystemContext_BinaryInfo_t* bi=nullptr;
    Qnn_ContextBinarySize_t bisz=0;
    CHECK(sys.getInfo(sh, bin.data(), bin.size(), &bi, &bisz));

    const QnnSystemContext_GraphInfo_t* gs=nullptr;
    if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) gs=bi->contextBinaryInfoV3.graphs;
    else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) gs=bi->contextBinaryInfoV2.graphs;
    else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) gs=bi->contextBinaryInfoV1.graphs;
    const auto& g = gs[0];
    const char* nm=nullptr; uint32_t nIn=0, nOut=0;
    const Qnn_Tensor_t* inT=nullptr,* outT=nullptr;
    if (g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
        nm=g.graphInfoV3.graphName;
        nIn=g.graphInfoV3.numGraphInputs;  inT=g.graphInfoV3.graphInputs;
        nOut=g.graphInfoV3.numGraphOutputs; outT=g.graphInfoV3.graphOutputs;
    } else if (g.version==QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
        nm=g.graphInfoV2.graphName;
        nIn=g.graphInfoV2.numGraphInputs;  inT=g.graphInfoV2.graphInputs;
        nOut=g.graphInfoV2.numGraphOutputs; outT=g.graphInfoV2.graphOutputs;
    } else {
        nm=g.graphInfoV1.graphName;
        nIn=g.graphInfoV1.numGraphInputs;  inT=g.graphInfoV1.graphInputs;
        nOut=g.graphInfoV1.numGraphOutputs; outT=g.graphInfoV1.graphOutputs;
    }
    bc.graphName = nm?nm:"";
    bc.inputs.assign(inT, inT+nIn);
    bc.outputs.assign(outT, outT+nOut);
    bc.dimStorage.resize(nIn+nOut);
    bc.nameStorage.resize(nIn+nOut);
    for (size_t i=0;i<bc.inputs.size();++i)
        rebind(bc.inputs[i], bc.dimStorage[i], bc.nameStorage[i]);
    for (size_t i=0;i<bc.outputs.size();++i)
        rebind(bc.outputs[i], bc.dimStorage[nIn+i], bc.nameStorage[nIn+i]);
    sys.free(sh);

    CHECK(bc.iface.contextCreateFromBinary(bc.backend, nullptr, nullptr,
        bin.data(), bin.size(), &bc.ctx, nullptr));
    CHECK(bc.iface.graphRetrieve(bc.ctx, bc.graphName.c_str(), &bc.graph));

    bc.inBufs.resize(bc.inputs.size());
    for (size_t i=0;i<bc.inputs.size();++i) {
        size_t sz=tbytes(bc.inputs[i]);
        bc.inBufs[i].assign(sz, 0);
        set_buf(bc.inputs[i], bc.inBufs[i].data(), (uint32_t)sz);
    }
    bc.outBufs.resize(bc.outputs.size());
    for (size_t i=0;i<bc.outputs.size();++i) {
        size_t sz=tbytes(bc.outputs[i]);
        bc.outBufs[i].assign(sz, 0);
        set_buf(bc.outputs[i], bc.outBufs[i].data(), (uint32_t)sz);
    }
    std::printf("[load] %s graph=%s in=%zu out=%zu\n",
                tag.c_str(), bc.graphName.c_str(),
                bc.inputs.size(), bc.outputs.size());
}

static void exec_loop(Ctx& bc, int iters,
                       std::vector<double>* per_call_us = nullptr) {
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        CHECK(bc.iface.graphExecute(bc.graph,
            bc.inputs.data(),  (uint32_t)bc.inputs.size(),
            bc.outputs.data(), (uint32_t)bc.outputs.size(),
            nullptr, nullptr));
        if (per_call_us) per_call_us->push_back((now_ms() - t0) * 1000.0);
    }
}

int main(int argc, char** argv) {
    int iters    = (argc > 1) ? std::atoi(argv[1]) : 30;
    int warmup_n = 3;

    SysFns sys = load_sys();
    Ctx hta, dsp;
    load_ctx(hta, "libQnnHta.so",
              "/root/qnn_runtime_ctx/ctx_dronet_HTA_split_seg200.bin",
              "dronet/HTA",  sys);
    load_ctx(dsp, "libQnnDsp.so",
              "/root/qnn_runtime_ctx/ctx_yolov8n_HTA_split_seg100.bin",
              "yolov8n/DSP", sys);

    std::printf("\n[warmup] %d iters each\n", warmup_n);
    exec_loop(hta, warmup_n);
    exec_loop(dsp, warmup_n);

    // ---- A) Single-backend baselines (each alone, no other backend running).
    std::vector<double> hta_only_us, dsp_only_us;
    double t0 = now_ms();
    exec_loop(hta, iters, &hta_only_us);
    double hta_only = now_ms() - t0;

    t0 = now_ms();
    exec_loop(dsp, iters, &dsp_only_us);
    double dsp_only = now_ms() - t0;

    auto avg = [](const std::vector<double>& v) {
        double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size();
    };
    std::printf("\n[baseline]\n");
    std::printf("  HTA-only %d iters: %8.2f ms total, per-call avg %.2f us\n",
                iters, hta_only, avg(hta_only_us));
    std::printf("  DSP-only %d iters: %8.2f ms total, per-call avg %.2f us\n",
                iters, dsp_only, avg(dsp_only_us));

    // ---- B) Serial — same thread fires HTA call, waits, then DSP call.
    std::vector<double> ser_hta_us, ser_dsp_us;
    t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        double s = now_ms();
        CHECK(hta.iface.graphExecute(hta.graph,
            hta.inputs.data(),  (uint32_t)hta.inputs.size(),
            hta.outputs.data(), (uint32_t)hta.outputs.size(),
            nullptr, nullptr));
        ser_hta_us.push_back((now_ms() - s) * 1000.0);
        s = now_ms();
        CHECK(dsp.iface.graphExecute(dsp.graph,
            dsp.inputs.data(),  (uint32_t)dsp.inputs.size(),
            dsp.outputs.data(), (uint32_t)dsp.outputs.size(),
            nullptr, nullptr));
        ser_dsp_us.push_back((now_ms() - s) * 1000.0);
    }
    double serial_total = now_ms() - t0;

    // ---- C) Parallel — two threads, each loops its own backend.
    std::vector<double> par_hta_us, par_dsp_us;
    t0 = now_ms();
    std::thread th_hta([&] { exec_loop(hta, iters, &par_hta_us); });
    std::thread th_dsp([&] { exec_loop(dsp, iters, &par_dsp_us); });
    th_hta.join();
    th_dsp.join();
    double parallel_total = now_ms() - t0;

    std::printf("\n[experiment] %d iters per backend\n", iters);
    std::printf("  serial   (HTA;DSP) total: %8.2f ms  "
                "(HTA-call avg %.2f us, DSP-call avg %.2f us)\n",
                serial_total, avg(ser_hta_us), avg(ser_dsp_us));
    std::printf("  parallel (HTA||DSP) total: %8.2f ms  "
                "(HTA-call avg %.2f us, DSP-call avg %.2f us)\n",
                parallel_total, avg(par_hta_us), avg(par_dsp_us));

    double speedup = serial_total / parallel_total;
    std::printf("\n[result] serial/parallel ratio = %.2fx\n", speedup);
    if (speedup > 1.5) {
        std::printf("  → HTA and DSP run TRULY CONCURRENTLY in this process.\n");
    } else if (speedup > 1.1) {
        std::printf("  → Partial concurrency (some overlap, some serialisation).\n");
    } else {
        std::printf("  → Effectively SERIALISED — process-global lock or shared FastRPC session.\n");
    }
    std::printf("  (a perfectly-overlapped run would give ratio ≈ "
                "(serial_total) / max(HTA-only, DSP-only) = %.2fx)\n",
                serial_total / std::max(hta_only, dsp_only));
    return 0;
}
