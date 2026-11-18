. /tmp/krisd/x142/env.sh
cd /tmp/krisd/x142
python walker.py 49_Mamba2ReturnFinalState --dev cpu --no-build 2>&1 | tail -5
python - <<'PY'
import json; r=json.load(open('/tmp/krisd/x142/out/49_Mamba2ReturnFinalState.json')); print(r['bit_identical'], r['n_inf_nan'], r['nan_aware_mismatch'], r['golden_nan'])
PY
