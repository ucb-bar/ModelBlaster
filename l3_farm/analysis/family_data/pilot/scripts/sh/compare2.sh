. /tmp/krisd/family/sh/env.sh; cd $W
until grep -q "^rc=" logs/compare.log; do sleep 20; done
cp pilot/compare.json pilot/compare_split.json
B=$W/pilot/B/r00; BS=$(ls $B/run/kb_*/best/slots/conv2d_f16.cu); TD=$(ls -d $B/family_tune/*/ | head -1)
python pyn/prepA.py --rep-only > logs/prepA_rep.log 2>&1
python pyn/compare.py 3 A_rep=$W/pilot/A_rep.json > logs/compare_rep.log 2>&1; echo rc=$? >> logs/compare_rep.log
python pyn/compare1.py 3 B=$BS C=$TD/tuned.cu C_pool=$TD/pool/pool.cu A_pool4=$W/pilot/A_pool4.cu > logs/compare1.log 2>&1; echo rc=$? >> logs/compare1.log
