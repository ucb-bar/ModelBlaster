#!/bin/bash
# host_verify.sh IR_DIR WORK_DIR  -> builds the IR's C model (reference kernels, scalar
# backend) as a host x86 binary via MB's harness_linux and prints MODELBLASTER_VERIFY.
set -e
IR=$(realpath "$1"); W=$(realpath -m "$2")
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"; MB=$MB_ROOT
rm -rf "$W"; mkdir -p "$W/gen"; cp "$IR/graph.json" "$W/gen/graph.json"
# fp16 IRs (`--quant fp16`) build the `_f16` (_Float16) reference kernels
Q=$(python -c "import json, sys; print(json.load(open(sys.argv[1])).get('quant', 'fp32'))" "$IR/graph.json")
# bf16 IRs are CUDA-only (docs/bf16_cuda.md): no host C build
[ "$Q" = bf16 ] && { echo "SKIP bf16 IR: the C codegen is fp32/fp16/int8 (bf16 is CUDA-only)"; exit 2; }
python -m modelblaster.pipeline.generate_skeleton --ir "$W/gen/graph.json" --weights "$IR/weights.npz" \
  --io "$IR/io.npz" --out-dir "$W/gen" --backend scalar --platform host > "$W/skeleton.log" 2>&1
python -m modelblaster.pipeline.generate_kernels --ir "$W/gen/graph.json" --out-dir "$W/gen" \
  --target scalar --backend reference --quant "$Q" --global-curated-dir "$MB/kernels" > "$W/kernels.log" 2>&1
make -s -C "$MB/harness_linux" MODEL_DIR="$W/gen" OUT="$W/harness" CFLAGS="-O2 -mcmodel=medium" > "$W/make.log" 2>&1 \
  || { [ -n "$KEEP_BUILD" ] || rm -rf "$W/gen" "$W/harness.objs"; echo "BUILD FAILED"; tail -3 "$W/make.log"; exit 1; }
timeout ${HOST_TIMEOUT:-900} "$W/harness" > "$W/run.out" 2>&1 || true
# generated weights.c / objects / binary run to GBs for the big nets
[ -n "$KEEP_BUILD" ] || rm -rf "$W/gen" "$W/harness.objs" "$W/harness"
grep -m1 "MODELBLASTER_VERIFY" "$W/run.out" || { echo "NO VERIFY LINE"; tail -3 "$W/run.out"; exit 1; }
