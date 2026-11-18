. /tmp/krisd/x142/env.sh
cd /tmp/krisd/x142/code/ModelBlaster
python -m pytest -q -p no:cacheprovider pipeline/tests/test_bmm_strided_seed.py pipeline/tests/test_layout_fold.py pipeline/tests/test_kb_export_fp16.py pipeline/tests/test_kb_export_opencl.py pipeline/tests/test_kb_export.py 2>&1 | tail -30
