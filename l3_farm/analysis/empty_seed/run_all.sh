#!/bin/zsh
# Offline: copy the live KernelBlaster tree (read-only), apply patches/kb_empty_seed.patch to the COPY, run the
# verification script in 3 modes and the existing knowledge tests with the empty seed. Nothing in the live tree is touched.
# usage: run_all.sh [PY]   (PY: python with pytest numpy pandas loguru python-dotenv pydantic pyyaml tenacity aiohttp openai requests psutil)
set -u
E=${0:A:h}; KBLIVE=${KBLIVE:-$E/../../../ModelBlaster/third_party/KernelBlaster}; PY=${1:-/tmp/mbvenv_es/bin/python}
W=$(mktemp -d /tmp/es_work.XXXX); mkdir -p $W/orig $W/patched/third_party/KernelBlaster
for d in src data tests scripts; do cp -R $KBLIVE/$d $W/orig/; done
cp -R $W/orig/{src,data,tests,scripts} $W/patched/third_party/KernelBlaster/
find $W -name __pycache__ -prune -exec rm -rf {} +
(cd $W/patched && patch -p1 < $E/patches/kb_empty_seed.patch) || exit 1
P=$W/patched/third_party/KernelBlaster
mkdir -p $E/results
$PY $E/verify_empty_seed.py $W/orig/src                > $E/results/verify_unpatched_seeddir_only.txt 2>&1
$PY $E/verify_empty_seed.py $P/src                     > $E/results/verify_patched_noflag.txt 2>&1
$PY $E/verify_empty_seed.py $P/src --flag              > $E/results/verify_patched_flag.txt 2>&1
T=(tests/test_knowledge_global.py tests/test_knowledge_global_a5.py tests/test_knowledge_refresh.py tests/test_knowledge_seed.py tests/test_knowledge_v2.py tests/test_knowledge_v2_followups.py tests/test_knowledge_v2_open.py tests/test_seed_mined_env.py tests/test_database_knowledge_rollout.py tests/test_multislot_agent.py tests/test_multislot_units.py)
(cd $W/orig && env -u KB_KNOWLEDGE_SEED_DIR $PY -m pytest $T -q -p no:cacheprovider 2>&1 | tail -3) > $E/results/pytest_baseline_real_seed.txt
(cd $W/orig && KB_KNOWLEDGE_SEED_DIR=$E/seed KERNELBLASTER_SEED_MINED=0 $PY -m pytest $T -q -p no:cacheprovider -rf 2>&1 | grep -E "^FAILED|passed|failed") > $E/results/pytest_unpatched_seeddir.txt
(cd $P && KB_KNOWLEDGE_SEED_DIR=$E/seed KB_EMPTY_SEED=1 KERNELBLASTER_SEED_MINED=0 $PY -m pytest $T -q -p no:cacheprovider -rf 2>&1 | grep -E "^FAILED|passed|failed") > $E/results/pytest_patched_flag.txt
(cd $P && $PY -m pytest $T -q -p no:cacheprovider 2>&1 | tail -3) > $E/results/pytest_patched_noenv_default_off.txt
echo "done; work copy $W; results in $E/results"
