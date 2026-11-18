. /tmp/krisd/family/sh/env.sh; cd $W/py
python -m unittest modelblaster.pipeline.tests.test_kb_plan_family 2>&1 | grep -B2 -A12 ERROR | head -30
cd $KB && PYTHONPATH=src:. python -m unittest tests/test_ncu.py 2>&1 | tail -12
