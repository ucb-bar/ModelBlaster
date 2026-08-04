// Per-segment per-backend profiling harness. For one sub-DLC, time
// every QnnGraph_execute call's full wallclock (the same measurement
// the trace's actual_start_ms..actual_end_ms uses) so the scheduler's
// cost model captures launch + RPC + dispatch + compute together —
// which is what the runtime actually pays per segment.
//
// Usage:
//   ./profile_seg <dlc_path> <backend_lib> <iters> [--csv <path>]
// Example:
//   ./profile_seg sub_dlc/dronet_HTA_split_seg0_quantized.dlc \
//                 libQnnHta.so 100 --csv /tmp/seg0_hta.csv
//
// Build (on QRB5165):
//   g++ -std=c++2a -O2 -pthread -I/root/qairt/include \
//       profile_segments.cpp -o profile_seg -ldl

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

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

static uint32_t trank(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2?t.v2.rank:t.v1.rank; }
static const uint32_t* tdims(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2?t.v2.dimensions:t.v1.dimensions; }
static Qnn_DataType_t tdtype(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2?t.v2.dataType:t.v1.dataType; }
static const char* tname(const Qnn_Tensor_t& t) { return t.version==QNN_TENSOR_VERSION_2?t.v2.name:t.v1.name; }
static size_t bpe(Qnn_DataType_t dt) {
    switch(dt) {
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <dlc_path> <backend_lib> <iters> [--csv <path>]\n", argv[0]);
        return 1;
    }
    std::string dlc = argv[1];
    std::string lib = argv[2];
    int iters = std::atoi(argv[3]);
    std::string csv;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    }

    // 1. dlopen + introspect via libQnnSystem.
    void* slib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!slib) { std::fprintf(stderr,"dlopen libQnnSystem.so: %s\n", dlerror()); return 1; }
    auto sgp = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
                dlsym(slib, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sprov=nullptr; uint32_t sn=0;
    CHECK(sgp(&sprov, &sn));
    auto& siface = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

    // 2. Build a context binary from the DLC at runtime is heavy; the
    //    profile harness instead loads via libQnnModelDlc.so which is
    //    what qnn-net-run does for --dlc_path. But that requires an
    //    additional `--model` argument and BackendExtensions hooks
    //    that complicate this stand-alone tool. Instead, this harness
    //    expects the caller to have already built per-backend context
    //    binaries via qnn-context-binary-generator (e.g.
    //    `ctx_<base>__<backend>.bin`). We accept the .bin path
    //    directly via the first argument.
    auto bin = slurp(dlc);

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
    const Qnn_Tensor_t* inT=nullptr,* outT=nullptr;
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
    for (size_t i=0;i<inputs.size();++i)  rebind(inputs[i],  dimStorage[i], nameStorage[i]);
    for (size_t i=0;i<outputs.size();++i) rebind(outputs[i], dimStorage[nIn+i], nameStorage[nIn+i]);
    // Deep-copy the graph name BEFORE freeing the system handle —
    // otherwise systemContextFree reclaims the memory `gname` points
    // into and graphRetrieve gets garbage.
    std::string graph_name = gname ? gname : "";
    siface.systemContextFree(sh);

    // 3. Load the backend lib + create context from binary.
    void* blib = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!blib) { std::fprintf(stderr,"dlopen %s: %s\n", lib.c_str(), dlerror()); return 1; }
    auto bgp = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
                dlsym(blib, "QnnInterface_getProviders");
    const QnnInterface_t** bprov=nullptr; uint32_t bn=0;
    CHECK(bgp(&bprov, &bn));
    auto biface = bprov[0]->QNN_INTERFACE_VER_NAME;

    Qnn_LogHandle_t logH=nullptr;
    CHECK(biface.logCreate(log_cb, QNN_LOG_LEVEL_ERROR, &logH));
    Qnn_BackendHandle_t backH=nullptr;
    CHECK(biface.backendCreate(logH, nullptr, &backH));

    double t_init = now_us();
    Qnn_ContextHandle_t ctx=nullptr;
    Qnn_ErrorHandle_t cerr = biface.contextCreateFromBinary(backH, nullptr, nullptr,
                                                              bin.data(), bin.size(),
                                                              &ctx, nullptr);
    double init_us = now_us() - t_init;
    if (cerr != QNN_SUCCESS) {
        std::fprintf(stderr, "{\"backend\":\"%s\",\"status\":\"compose_fail\",\"err\":\"0x%llx\"}\n",
                      lib.c_str(), (unsigned long long)cerr);
        return 2;
    }
    Qnn_GraphHandle_t graph=nullptr;
    CHECK(biface.graphRetrieve(ctx, graph_name.c_str(), &graph));

    // 4. Allocate I/O buffers (zero-init — we're measuring time, not
    //    correctness). The end_ms-start_ms we capture is the same
    //    measurement the runtime emits in its trace, so the scheduler's
    //    cost model gets the launch + RPC + sync + compute total.
    std::vector<std::vector<uint8_t>> inBufs(inputs.size()), outBufs(outputs.size());
    for (size_t i=0;i<inputs.size();++i) {
        size_t sz = tbytes(inputs[i]);
        inBufs[i].assign(sz, 0);
        set_buf(inputs[i], inBufs[i].data(), (uint32_t)sz);
    }
    for (size_t i=0;i<outputs.size();++i) {
        size_t sz = tbytes(outputs[i]);
        outBufs[i].assign(sz, 0);
        set_buf(outputs[i], outBufs[i].data(), (uint32_t)sz);
    }

    // 5. Warmup + timed runs.
    int warmup = std::min(5, iters/4 + 1);
    for (int i=0;i<warmup;++i) {
        CHECK(biface.graphExecute(graph,
            inputs.data(),  (uint32_t)inputs.size(),
            outputs.data(), (uint32_t)outputs.size(),
            nullptr, nullptr));
    }
    std::vector<double> per_call;
    per_call.reserve(iters);
    for (int i=0;i<iters;++i) {
        double t0 = now_us();
        CHECK(biface.graphExecute(graph,
            inputs.data(),  (uint32_t)inputs.size(),
            outputs.data(), (uint32_t)outputs.size(),
            nullptr, nullptr));
        per_call.push_back(now_us() - t0);
    }
    std::sort(per_call.begin(), per_call.end());
    double mean = std::accumulate(per_call.begin(), per_call.end(), 0.0) / per_call.size();
    double mn = per_call.front();
    double mx = per_call.back();
    double p50 = per_call[per_call.size()/2];
    double p99 = per_call[std::min(per_call.size()-1, per_call.size()*99/100)];
    double sumsq = 0;
    for (double x : per_call) sumsq += (x-mean)*(x-mean);
    double std_us = std::sqrt(sumsq / per_call.size());

    // 6. Emit one CSV row + a JSON-ish status line for the host parser.
    std::printf("{\"dlc\":\"%s\",\"backend\":\"%s\",\"status\":\"ok\","
                "\"graph\":\"%s\",\"iters\":%d,\"init_us\":%.1f,"
                "\"mean_us\":%.2f,\"median_us\":%.2f,\"min_us\":%.2f,"
                "\"max_us\":%.2f,\"std_us\":%.2f,\"p99_us\":%.2f}\n",
                dlc.c_str(), lib.c_str(), graph_name.c_str(), iters, init_us,
                mean, p50, mn, mx, std_us, p99);
    if (!csv.empty()) {
        std::ofstream f(csv);
        f << "iter,call_us\n";
        for (size_t i=0;i<per_call.size();++i)
            f << i << "," << per_call[i] << "\n";
    }
    biface.contextFree(ctx, nullptr);
    biface.backendFree(backH);
    biface.logFree(logH);
    return 0;
}
