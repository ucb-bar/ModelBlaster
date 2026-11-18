. /tmp/krisd/ng/env.sh
cd /tmp/krisd/ng; mkdir -p irs logs
for p in 8_ResNetBasicBlock 19_MobileNetV1; do
  if [ ! -f irs/$p/.done ]; then
    ( cd /tmp/krisd/ng/code && python -m modelblaster.pipeline.extract_graph --bench-file /tmp/krisd/ng/problems/$p.py --bench-target-mb 0 --quant fp16 --out-dir /tmp/krisd/ng/irs/$p > /tmp/krisd/ng/logs/extract_$p.log 2>&1 ) && touch irs/$p/.done
    echo "$p extract rc=$? $(ls irs/$p | tr '\n' ' ')"
  fi
done
