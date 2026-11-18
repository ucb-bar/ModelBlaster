R=/tmp/krisd/mt_$1
tail -n 1 $R/cost.log 2>/dev/null; tail -n 3 $R/run.log 2>/dev/null | cut -c1-260; ls $R/out/*/result.json 2>/dev/null; tail -n 2 $R/arm.log
