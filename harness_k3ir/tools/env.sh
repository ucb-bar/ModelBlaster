# Source me before the gate scripts. Every value can be overridden from the environment.
#   PY      python with torch + the ModelBlaster dev dependencies
#   SPIKE, PK, SPIKE_ISA   Spike + proxy kernel for the spike_gate.sh functional runs
#   K3IR_OUT  where gate outputs go
_here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export R=${R:-$_here}
export MB=${MB:-$(cd "$R/.." && pwd)}
export PYTHONPATH=$MB/src:$MB/gpu_eval/harness:$R${PYTHONPATH:+:$PYTHONPATH}
export PY=${PY:-python3}
export SPIKE=${SPIKE:-spike}
export PK=${PK:-pk}
export SPIKE_ISA=${SPIKE_ISA:-rv64gcv_zicntr_zicbop_zicboz_zfh_zvfh_zba_zbb_zbs_zicond}
export K3IR_OUT=${K3IR_OUT:-$PWD/k3ir_out}
