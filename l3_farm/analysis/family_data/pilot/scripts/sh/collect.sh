cd /tmp/krisd/family/pilot; B=B/r00; TD=$(ls -d $B/family_tune/*/ | head -1)
mkdir -p /tmp/krisd/family/collect; C=/tmp/krisd/family/collect
cp compare.json compare_split.json compare1.json A_family.json A_model.json matrix.json $C/ 2>/dev/null
cp $B/run/kb_*/result.json $C/B_result.json; cp $B/round.json $C/B_round.json; cp $B/plan.json $C/B_plan.json
cp $B/problems/*/family.json $C/B_family.json
cp $TD/tune.json $C/C_tune.json; cp $TD/pool/pool.json $C/Cpool_pool.json; cp A_pool4/pool.json $C/A_pool4_pool.json
cp $B/run/kb_*/best/slots/conv2d_f16.cu $C/B.cu; cp $TD/tuned.cu $C/C.cu; cp $TD/pool/pool.cu $C/C_pool.cu; cp A_pool4.cu $C/A_pool4.cu
cp /tmp/krisd/farm/llm_ledger.jsonl $C/llm_ledger.jsonl
cd $C && tar czf /tmp/krisd/family/collect.tgz . && ls
