d=/tmp/krisd/farm/15_DenseNet121/work
ls -d $d/r0* $d/r0*.stale* 2>/dev/null
mkdir -p /tmp/krisd/evidence15
for r in $d/r02 $d/r02.stale*; do [ -d "$r" ] && cp -a "$r" /tmp/krisd/evidence15/ 2>/dev/null; done
ls /tmp/krisd/evidence15; du -sh /tmp/krisd/evidence15
