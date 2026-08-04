// Minimal end-to-end test of the yolov8n partition slices.
//
// Loads the two pre-built per-segment context binaries
// (ctx_yolov8n_HTA_split_seg100.bin = backbone @ DSP, _seg101.bin = head
// @ DSP), wires the backbone's 3 outputs into the head's 3 inputs via
// host-memory memcpy, and times each call. Validates the
// slice_from_partition.py + build_subdlcs.sh pipeline end-to-end on
// real hardware without needing schedule integration.
//
// Build (on QRB5165): see build_test_yolov8n.sh.

#include <chrono>
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

    uint32_t nG=0; const QnnSystemContext_GraphInfo_t* gs=nullptr;
    if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        nG=bi->contextBinaryInfoV3.numGraphs; gs=bi->contextBinaryInfoV3.graphs;
    } else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        nG=bi->contextBinaryInfoV2.numGraphs; gs=bi->contextBinaryInfoV2.graphs;
    } else if (bi->version==QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        nG=bi->contextBinaryInfoV1.numGraphs; gs=bi->contextBinaryInfoV1.graphs;
    }
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
                tag.c_str(), bc.graphName.c_str(), bc.inputs.size(), bc.outputs.size());
    for (size_t i=0;i<bc.inputs.size();++i)
        std::printf("    in[%zu]  %s  bytes=%zu\n", i, tname(bc.inputs[i]), tbytes(bc.inputs[i]));
    for (size_t i=0;i<bc.outputs.size();++i)
        std::printf("    out[%zu] %s  bytes=%zu\n", i, tname(bc.outputs[i]), tbytes(bc.outputs[i]));
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 10;
    SysFns sys = load_sys();
    Ctx backbone, head;
    load_ctx(backbone, "libQnnDsp.so",
              "/root/qnn_runtime_ctx/ctx_yolov8n_HTA_split_seg100.bin",
              "yolov8n/backbone(seg100)", sys);
    load_ctx(head, "libQnnDsp.so",
              "/root/qnn_runtime_ctx/ctx_yolov8n_HTA_split_seg101.bin",
              "yolov8n/head(seg101)",     sys);

    std::printf("\n[run] %d iterations of (backbone → host-memcpy → head)\n", iters);
    double sum_back=0, sum_handoff=0, sum_head=0, sum_total=0;
    for (int it=0; it<iters; ++it) {
        // Backbone input: zero-initialized 1×3×640×640 (the test isn't
        // about correctness of output values, just per-segment timing
        // and successful execution chain).
        double t0 = now_ms();
        CHECK(backbone.iface.graphExecute(backbone.graph,
            backbone.inputs.data(),  (uint32_t)backbone.inputs.size(),
            backbone.outputs.data(), (uint32_t)backbone.outputs.size(),
            nullptr, nullptr));
        double t1 = now_ms();

        // Cross-segment handoff: copy each backbone output to the
        // matching head input (matched by tensor name — both DLCs
        // use the same names because the slicer used the same source
        // ONNX). For this v1 we do n^2 search; with 3 outputs that's
        // fine. The DSP/HTP backend internally handles the int8
        // requantize at the boundary if scales differ between the
        // backbone's output quant and the head's input quant — see
        // PARTITIONING_GUIDE §6 on the cross-backend handoff cost
        // model (~3-5 ms for an 80×80×64 i8 buffer).
        for (auto& bo : backbone.outputs) {
            const char* bn = tname(bo);
            for (size_t hi = 0; hi < head.inputs.size(); ++hi) {
                if (std::strcmp(tname(head.inputs[hi]), bn) == 0) {
                    size_t n = std::min(tbytes(bo), tbytes(head.inputs[hi]));
                    std::memcpy(head.inBufs[hi].data(),
                                 head.inputs[hi].v2.clientBuf.data ? bo.v2.clientBuf.data
                                                                    : bo.v1.clientBuf.data,
                                 n);
                    // Guard: prefer v2 path; v2 exposes the buffer
                    // pointer at the same offset on both versions.
                    std::memcpy(head.inBufs[hi].data(),
                                 backbone.outBufs[(&bo - backbone.outputs.data())].data(), n);
                    break;
                }
            }
        }
        double t2 = now_ms();

        CHECK(head.iface.graphExecute(head.graph,
            head.inputs.data(),  (uint32_t)head.inputs.size(),
            head.outputs.data(), (uint32_t)head.outputs.size(),
            nullptr, nullptr));
        double t3 = now_ms();

        double back_ms = t1 - t0;
        double hand_ms = t2 - t1;
        double head_ms = t3 - t2;
        double tot_ms  = t3 - t0;
        sum_back += back_ms; sum_handoff += hand_ms; sum_head += head_ms; sum_total += tot_ms;
        if (it < 3 || it == iters-1)
            std::printf("  iter %d: backbone=%.2f ms  handoff=%.2f ms  head=%.2f ms  total=%.2f ms\n",
                         it, back_ms, hand_ms, head_ms, tot_ms);
    }

    std::printf("\n[summary] over %d iters (averages):\n", iters);
    std::printf("  backbone (DSP, 103-op slice): %.2f ms\n", sum_back / iters);
    std::printf("  handoff  (host memcpy 3×):    %.2f ms\n", sum_handoff / iters);
    std::printf("  head     (DSP, 138-op slice): %.2f ms\n", sum_head / iters);
    std::printf("  total per inference:           %.2f ms\n", sum_total / iters);

    return 0;
}
