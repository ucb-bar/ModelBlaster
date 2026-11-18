. /tmp/krisd/pa/env.sh
cd /tmp/krisd/pa; mkdir -p irs logs
for p in 8_ResNetBasicBlock 19_MobileNetV1; do
  if [ ! -f irs/$p/.done ]; then
    ( cd /tmp/krisd/pa/code && python -m modelblaster.pipeline.extract_graph --bench-file /tmp/krisd/pa/problems/$p.py --bench-target-mb 0 --quant fp16 --out-dir /tmp/krisd/pa/irs/$p > /tmp/krisd/pa/logs/extract_$p.log 2>&1 ) && touch irs/$p/.done
    echo "$p extract rc=$? $(ls irs/$p | tr '\n' ' ')"
  fi
done
python - <<'PY'
import json
for p in ("8_ResNetBasicBlock","19_MobileNetV1"):
    g=json.load(open(f"/tmp/krisd/pa/irs/{p}/graph.json"))
    print(p, len(g["ops"]), [ (o["name"],o["op"]) for o in g["ops"]][:40])
PY
