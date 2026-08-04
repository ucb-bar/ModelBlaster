// Per-op profiler — uses QNN_PROFILE_LEVEL_DETAILED to attribute
// graphExecute time to individual ops. Dumps a CSV {op_id, identifier,
// type, value_us, value_cyc} per op so we can compare per-conv-layer
// timings between backends (e.g. yolov8 backbone DSP vs HTA).
//
// Usage:
//   ./profile_per_op <ctx.bin> <backend.so> <iters> [--csv <out.csv>]
//
// The .bin must be a pre-built QNN context binary (from
// qnn-context-binary-generator) for the same backend lib that's being
// profiled. Iters > 1 averages per-op times across runs.
//
// Build (board):
//   g++ -std=c++2a -O2 -pthread -I/root/qairt/include \
//       profile_per_op.cpp -o profile_per_op -ldl

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnProfile.h"
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
    Qnn_ClientBuffer_t cb{data, bytes};
    if (t.version==QNN_TENSOR_VERSION_2) { t.v2.memType=QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf=cb; }
    else                                  { t.v1.memType=QNN_TENSORMEMTYPE_RAW; t.v1.clientBuf=cb; }
}


// Walk profile sub-events recursively. Each leaf has a non-zero
// MICROSECOND value attributable to a specific op. Accumulate into the
// `accum` map keyed by identifier.
static void walk_events(const QNN_INTERFACE_VER_TYPE& iface,
                         const QnnProfile_EventId_t* eids,
                         uint32_t n,
                         std::map<std::string, std::pair<uint64_t,int>>& accum,
                         int depth = 0) {
    for (uint32_t i = 0; i < n; ++i) {
        QnnProfile_EventData_t ed = QNN_PROFILE_EVENT_DATA_INIT;
        if (iface.profileGetEventData(eids[i], &ed) != QNN_SUCCESS) continue;
        std::string id = ed.identifier ? ed.identifier : "<noname>";
        // Only accumulate MICROSEC events with a value (the op's compute
        // time). Type 0 = init / generic; non-zero values per op show up
        // as {unit=MICROSEC, value=N}.
        if (ed.unit == QNN_PROFILE_EVENTUNIT_MICROSEC && ed.value > 0) {
            auto& entry = accum[id];
            entry.first  += ed.value;
            entry.second += 1;
        }
        // Recurse into sub-events (HTP/DSP backends nest nodes one
        // level under the top-level execute event).
        const QnnProfile_EventId_t* sub = nullptr; uint32_t ns = 0;
        if (iface.profileGetSubEvents(eids[i], &sub, &ns) == QNN_SUCCESS && ns > 0) {
            walk_events(iface, sub, ns, accum, depth + 1);
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <ctx.bin> <backend.so> <iters> [--csv <out>]\n", argv[0]);
        return 2;
    }
    std::string dlc = argv[1];
    std::string lib = argv[2];
    int iters = std::atoi(argv[3]);
    std::string csv;
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv = argv[++i];
    }

    void* slib = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!slib) { std::fprintf(stderr,"dlopen libQnnSystem.so: %s\n", dlerror()); return 1; }
    auto sgp = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
                dlsym(slib, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** sprov=nullptr; uint32_t sn=0;
    CHECK(sgp(&sprov, &sn));
    auto& siface = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;

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
    std::string graph_name = gname ? gname : "";
    siface.systemContextFree(sh);

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

    Qnn_ContextHandle_t ctx=nullptr;
    Qnn_ErrorHandle_t cerr = biface.contextCreateFromBinary(backH, nullptr, nullptr,
                                                              bin.data(), bin.size(),
                                                              &ctx, nullptr);
    if (cerr != QNN_SUCCESS) {
        std::fprintf(stderr, "{\"backend\":\"%s\",\"status\":\"compose_fail\"}\n", lib.c_str());
        return 2;
    }
    Qnn_GraphHandle_t graph=nullptr;
    CHECK(biface.graphRetrieve(ctx, graph_name.c_str(), &graph));

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

    // Warmup runs (no profiling).
    int warmup = std::min(3, iters/4 + 1);
    for (int i=0;i<warmup;++i) {
        CHECK(biface.graphExecute(graph,
            inputs.data(),  (uint32_t)inputs.size(),
            outputs.data(), (uint32_t)outputs.size(),
            nullptr, nullptr));
    }

    // Per-op accumulator: identifier -> (sum_us, n_samples).
    std::map<std::string, std::pair<uint64_t,int>> per_op;
    double total_wall_us = 0;
    int events_collected = 0;

    for (int i=0;i<iters;++i) {
        Qnn_ProfileHandle_t prof = nullptr;
        CHECK(biface.profileCreate(backH, QNN_PROFILE_LEVEL_DETAILED, &prof));
        double t0 = now_us();
        CHECK(biface.graphExecute(graph,
            inputs.data(),  (uint32_t)inputs.size(),
            outputs.data(), (uint32_t)outputs.size(),
            prof, nullptr));
        total_wall_us += (now_us() - t0);
        // Pull events.
        const QnnProfile_EventId_t* eids = nullptr; uint32_t ne = 0;
        CHECK(biface.profileGetEvents(prof, &eids, &ne));
        events_collected += ne;
        walk_events(biface, eids, ne, per_op);
        biface.profileFree(prof);
    }

    // Emit summary header (JSON-ish for the host parser).
    double mean_wall_us = total_wall_us / iters;
    std::printf("{\"dlc\":\"%s\",\"backend\":\"%s\",\"status\":\"ok\","
                "\"graph\":\"%s\",\"iters\":%d,\"mean_wall_us\":%.2f,"
                "\"unique_op_events\":%zu,\"top_events_per_iter\":%d}\n",
                dlc.c_str(), lib.c_str(), graph_name.c_str(), iters,
                mean_wall_us, per_op.size(), events_collected/std::max(1,iters));

    // Sort by mean per-op time descending and print to stderr (human-readable).
    std::vector<std::tuple<double,std::string,int>> sorted;
    for (auto& kv : per_op) {
        double mean_us = (double)kv.second.first / std::max(1, kv.second.second);
        sorted.emplace_back(mean_us, kv.first, kv.second.second);
    }
    std::sort(sorted.rbegin(), sorted.rend());
    std::fprintf(stderr, "\n=== top ops by mean per-call time ===\n");
    std::fprintf(stderr, "%-60s %12s %8s\n", "identifier", "mean_us", "samples");
    int shown = 0;
    for (auto& [mean_us, id, n] : sorted) {
        std::fprintf(stderr, "%-60s %12.2f %8d\n", id.c_str(), mean_us, n);
        if (++shown >= 30) break;
    }

    if (!csv.empty()) {
        std::ofstream f(csv);
        f << "identifier,mean_us,samples\n";
        for (auto& [mean_us, id, n] : sorted) {
            // Quote the identifier if it has commas.
            f << '"' << id << "\"," << mean_us << "," << n << "\n";
        }
    }

    biface.contextFree(ctx, nullptr);
    biface.backendFree(backH);
    biface.logFree(logH);
    return 0;
}
