#!/bin/bash
# spike_gate.sh NAME PLANPREFIX FEED HOSTDUMP REFNPZ
# Spike + pk at VLEN 1024 (a100 kernels) and 256 (x100 kernels): (1) default flags with --check, byte comparison
# of the exact kinds against the host reference dump (dumpdiff.py), isolation gate against the walk;
# (2) --attn 1 --attn-check (fused fp32 attention; the IME attention modes need the board).
# Outputs in $K3IR_OUT/spike/NAME. Needs `make spike` (and `make host` for HOSTDUMP via host_gate.sh).
source "$(dirname "$0")/env.sh"
n=$1; p=$2; fd=$3; hd=$4; ref=$5; o=$K3IR_OUT/spike/$n; mkdir -p "$o"
EX=linear,matmul,bmm_strided,conv2d,conv_transpose2d,add,mul,relu,ewise2,mul_scalar,rope_half,quantize_rows_s8,linear_dq_s8,linear3_dq_s8,linear_dq_add_s8,cast,strided_copy,maxpool2d,upsample_nearest,catn,index_select
ir=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['ir'])" "$p.names.json")
[ "${ir:0:1}" = / ] || ir=$(dirname "$p")/$ir
for v in 1024 256; do
  be=$([ $v = 1024 ] && echo a100 || echo x100)
  timeout 6000 "$SPIKE" --isa=${SPIKE_ISA}_zvl${v}b "$PK" "$R/build/k3ir_spike" feed "$p.plan" "$p.wblob" "$p.cblob" \
    --feed "$fd" --dump "$o/dump$v" --backend $be --check > "$o/spike$v.log" 2>&1
  echo "$n VLEN $v: $(grep RESULT "$o/spike$v.log") $(grep -c CHECK_MISMATCH "$o/spike$v.log") mismatch lines"
  $PY "$R/ref_walk.py" compare "$ir" "$p.names.json" "$ref" "$o/dump$v" --json "$o/gate$v.json" > /dev/null 2>&1
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('  $n $v gate vs walk: ok=',d['ok'],'over_bound=',d['n_fp32_ops_over_slot_bound'],'worst=%.2e'%d['worst_fp32_rel_err'],'s8=',sum(d['s8_mismatch_bytes_by_kind'].values()))" "$o/gate$v.json"
  $PY "$R/dumpdiff.py" "$p.names.json" "$hd" "$o/dump$v" --exact-kinds $EX --json "$o/diff$v.json" 2>/dev/null | \
    $PY -c "import json,sys; d=json.load(sys.stdin); print('  $n $v vs host ref C: exact kinds identical=',d['ok'], {k:(v['differ'],v['max_ulps']) for k,v in d['by_kind'].items() if v['differ']})"
  timeout 6000 "$SPIKE" --isa=${SPIKE_ISA}_zvl${v}b "$PK" "$R/build/k3ir_spike" feed "$p.plan" "$p.wblob" "$p.cblob" \
    --feed "$fd" --dump "$o/dumpattn$v" --backend $be --check --attn 1 --attn-check > "$o/spike_attn$v.log" 2>&1
  echo "  $n $v attn1: $(grep -E '^ATTN .*fused_groups' "$o/spike_attn$v.log" | tr '\n' ' ') $(grep RESULT "$o/spike_attn$v.log") $(grep ATTN_CHECK "$o/spike_attn$v.log")"
done
