R=/tmp/krisd/mt_$1
pkill -f "mt_run.sh $1 " ; pkill -f "run_multislot.py $R/prob"; sleep 3
pgrep -af "mt_$1" | grep -v pgrep | head; echo killed
mv $R/out $R/out_aborted_$(date +%s) 2>/dev/null; mv $R/arm.log $R/arm_aborted.log; mv $R/cost.log $R/cost_aborted.log; ls $R
