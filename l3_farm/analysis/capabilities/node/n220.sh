cd /tmp/krisd/a2 && . /tmp/krisd/a2/env.sh
python /tmp/krisd/a2/node/regfinal.py 5 > logs/regfinal.log 2>&1; grep @@ logs/regfinal.log
VIEW31_LLM=1 bash /tmp/krisd/a2/node/view31.sh > logs/view31.log 2>&1; grep "@@\|===" logs/view31.log
echo N220_DONE
