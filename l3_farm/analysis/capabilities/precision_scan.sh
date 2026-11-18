#!/bin/bash
# A2 item 3: run the precision-policy overflow detector (policy off: the IR is a plain seeded fp16 extract) over every
# Level 3 problem; record the detector per problem; delete the IR afterwards.
S=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
OUT=${1:-$HOME/Documents/ModelBlaster-clone/l3_farm/analysis/capabilities/precision_scan}
mkdir -p "$OUT"
cd $S/pp
for f in $HOME/Documents/ModelBlaster-clone/l3_farm/problems/*.py; do
  p=$(basename "$f" .py)
  [ -f "$OUT/$p.json" ] && continue
  d=$S/a2/scan/$p; rm -rf "$d"; mkdir -p "$d"
  PYTHONPATH=$S/pp $S/a2t/bin/python -m modelblaster.pipeline.precision_policy extract --policy off \
      --bench-file "$f" --out-dir "$d" -- --bench-target-mb 0 > "$OUT/$p.log" 2>&1
  if [ -f "$d/precision_policy.json" ]; then cp "$d/precision_policy.json" "$OUT/$p.json"; else echo '{"error": true}' > "$OUT/$p.json"; fi
  rm -rf "$d"
done
echo SCAN_DONE
