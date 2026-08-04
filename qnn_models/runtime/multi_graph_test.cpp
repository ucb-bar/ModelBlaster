// multi_graph_test.cpp
//
// Verify two things about QNN multi-graph context binaries:
//   1. contextCreateFromBinary on a multi-graph .bin succeeds and exposes
//      ALL graphs in the binary (via QnnSystemContext_GraphInfo_t list).
//   2. graphRetrieve by name returns each graph and graphExecute works on
//      each, individually.
//
// (3) is the architectural question: does the firmware count the multi-
// graph .bin as ONE context or as N? We can't measure firmware count
// directly, but we can chain N multi-graph loads (e.g. 5 binaries with 5
// graphs each = 25 graphs total) and see if we exceed the simul-cap of
// ~30 contexts WITHOUT failing. If 25 graphs from 5 multi-graph loads
// fits where 25 single-graph loads would have hit the cap, we win.
//
// Usage:
//   ./multi_graph_test <multi.bin> <backend.so>

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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <multi.bin> <backend.so>\n", argv[0]);
        return 1;
    }
    std::string bin_path = argv[1];
    std::string lib_path = argv[2];

    // 1. Load system lib + introspect binary
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

    uint32_t nG = 0;
    const QnnSystemContext_GraphInfo_t* gs=nullptr;
    if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        nG = bi->contextBinaryInfoV3.numGraphs;
        gs = bi->contextBinaryInfoV3.graphs;
    } else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        nG = bi->contextBinaryInfoV2.numGraphs;
        gs = bi->contextBinaryInfoV2.graphs;
    } else {
        nG = bi->contextBinaryInfoV1.numGraphs;
        gs = bi->contextBinaryInfoV1.graphs;
    }
    std::printf("Binary '%s' (%zu B) holds %u graph(s):\n",
                 bin_path.c_str(), bin.size(), nG);

    struct GraphMeta {
        std::string name;
        std::vector<Qnn_Tensor_t> inputs, outputs;
        std::vector<std::vector<uint32_t>> dimStorage;
        std::vector<std::string> nameStorage;
    };
    std::vector<GraphMeta> metas(nG);
    for (uint32_t gi = 0; gi < nG; ++gi) {
        const auto& g = gs[gi];
        const char* nm=nullptr; uint32_t nIn=0,nOut=0;
        const Qnn_Tensor_t *inT=nullptr,*outT=nullptr;
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
        metas[gi].name = nm ? nm : "";
        metas[gi].inputs.assign(inT, inT+nIn);
        metas[gi].outputs.assign(outT, outT+nOut);
        metas[gi].dimStorage.resize(nIn+nOut);
        metas[gi].nameStorage.resize(nIn+nOut);
        for (size_t i=0;i<metas[gi].inputs.size();++i)
            rebind(metas[gi].inputs[i],  metas[gi].dimStorage[i],     metas[gi].nameStorage[i]);
        for (size_t i=0;i<metas[gi].outputs.size();++i)
            rebind(metas[gi].outputs[i], metas[gi].dimStorage[nIn+i], metas[gi].nameStorage[nIn+i]);
        std::printf("  [%u] %s  in=%u out=%u\n", gi, metas[gi].name.c_str(), nIn, nOut);
    }
    siface.systemContextFree(sh);

    // 2. dlopen backend, create ONE context from this multi-graph binary
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

    double t_create0 = now_us();
    Qnn_ContextHandle_t ctx=nullptr;
    CHECK(biface.contextCreateFromBinary(backH, nullptr, nullptr,
                                          bin.data(), bin.size(), &ctx, nullptr));
    double t_create_us = now_us() - t_create0;
    std::printf("\ncontextCreateFromBinary OK in %.1f us — ONE context handle holds %u graphs\n",
                 t_create_us, nG);

    // 3. Retrieve each graph by name, allocate buffers, run a single execute
    std::vector<Qnn_GraphHandle_t> graphs(nG);
    std::vector<std::vector<std::vector<uint8_t>>> inBufs(nG), outBufs(nG);
    for (uint32_t gi = 0; gi < nG; ++gi) {
        CHECK(biface.graphRetrieve(ctx, metas[gi].name.c_str(), &graphs[gi]));
        inBufs[gi].resize(metas[gi].inputs.size());
        for (size_t i=0;i<metas[gi].inputs.size();++i) {
            size_t sz = tbytes(metas[gi].inputs[i]);
            inBufs[gi][i].assign(sz, 0);
            set_buf(metas[gi].inputs[i], inBufs[gi][i].data(), (uint32_t)sz);
        }
        outBufs[gi].resize(metas[gi].outputs.size());
        for (size_t i=0;i<metas[gi].outputs.size();++i) {
            size_t sz = tbytes(metas[gi].outputs[i]);
            outBufs[gi][i].assign(sz, 0);
            set_buf(metas[gi].outputs[i], outBufs[gi][i].data(), (uint32_t)sz);
        }
    }

    // Warmup all then time each
    for (uint32_t gi = 0; gi < nG; ++gi)
        for (int w = 0; w < 2; ++w)
            CHECK(biface.graphExecute(graphs[gi],
                metas[gi].inputs.data(),  (uint32_t)metas[gi].inputs.size(),
                metas[gi].outputs.data(), (uint32_t)metas[gi].outputs.size(),
                nullptr, nullptr));

    std::printf("\nPer-graph execute timings (50 iters each):\n");
    for (uint32_t gi = 0; gi < nG; ++gi) {
        std::vector<double> samples(50);
        for (int it=0; it<50; ++it) {
            double t0 = now_us();
            CHECK(biface.graphExecute(graphs[gi],
                metas[gi].inputs.data(),  (uint32_t)metas[gi].inputs.size(),
                metas[gi].outputs.data(), (uint32_t)metas[gi].outputs.size(),
                nullptr, nullptr));
            samples[it] = now_us() - t0;
        }
        double sum = 0, mn = samples[0], mx = samples[0];
        for (double v : samples) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
        double mean = sum / samples.size();
        std::printf("  %-40s mean=%.1f us  min=%.1f  max=%.1f\n",
                     metas[gi].name.c_str(), mean, mn, mx);
    }

    biface.contextFree(ctx, nullptr);
    biface.backendFree(backH);
    biface.logFree(logH);
    return 0;
}
