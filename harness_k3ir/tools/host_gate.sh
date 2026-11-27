#!/bin/bash
# host_gate.sh NAME PLANPREFIX FEED [OLDDUMP|/dev/null] [REFNPZ]
# Host k3ir feed run (reference kernels) on an isolation feed (ref_walk.py feed); optional byte comparison
# against an earlier host dump; isolation gate against the reference walk (ref_walk.py compare).
# Outputs in $K3IR_OUT/gate/NAME. Needs `make host`.
source "$(dirname "$0")/env.sh"
n=$1; p=$2; fd=$3; old=${4:-/dev/null}; ref=${5:-${fd%.feed}.ref.npz}
o=$K3IR_OUT/gate/$n; mkdir -p "$o"
"$R/build/k3ir_host" feed "$p.plan" "$p.wblob" "$p.cblob" --feed "$fd" --dump "$o/hostdump" > "$o/host.log" 2>&1
same=-; [ "$old" != /dev/null ] && same=$(cmp -s "$o/hostdump" "$old" && echo identical || echo DIFFERENT)
ir=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['ir'])" "$p.names.json")
[ "${ir:0:1}" = / ] || ir=$(dirname "$p")/$ir
gate=-
if [ -f "$ref" ]; then
  $PY "$R/ref_walk.py" compare "$ir" "$p.names.json" "$ref" "$o/hostdump" --json "$o/gate_host.json" > /dev/null 2>&1
  gate=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('ok=%s over=%s worst=%.2e s8=%d'%(d['ok'],d['n_fp32_ops_over_slot_bound'],d['worst_fp32_rel_err'],sum(d['s8_mismatch_bytes_by_kind'].values())))" "$o/gate_host.json" 2>&1 | tail -1)
fi
echo "$n $(tail -1 "$o/host.log") vs_old_hostdump=$same gate: $gate"
