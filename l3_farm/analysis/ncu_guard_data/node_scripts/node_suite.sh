. /tmp/krisd/ng/env.sh
cd /tmp/krisd/ng; mkdir -p out kb_base
KBN=/tmp/krisd/ng/code/ModelBlaster/third_party/KernelBlaster
echo "== KB new"; (cd $KBN && PYTHONPATH=src:tests python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors > /tmp/krisd/ng/out/suite_kb_new.log 2>&1); tail -3 /tmp/krisd/ng/out/suite_kb_new.log
echo "== KB base"; rm -rf kb_base/*; tar xzf /tmp/krisd/ng/kb_base.tar.gz -C kb_base 2>/dev/null; find kb_base -name '._*' -delete
(cd kb_base/KernelBlaster && PYTHONPATH=src:tests python -m pytest tests -q -p no:cacheprovider --continue-on-collection-errors > /tmp/krisd/ng/out/suite_kb_base.log 2>&1); tail -3 /tmp/krisd/ng/out/suite_kb_base.log
echo "== MB pipeline new"; (cd /tmp/krisd/ng/code && python -m pytest ModelBlaster/pipeline/tests -q -p no:cacheprovider --continue-on-collection-errors > /tmp/krisd/ng/out/suite_mb_new.log 2>&1); tail -3 /tmp/krisd/ng/out/suite_mb_new.log
echo done
