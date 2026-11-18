set -u
cd /Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/merge_prep/ip
until python3 ip_run.py sh 5300606 'grep -q "#### cmb3" /tmp/krisd/mc/ip/chain1.log && echo y' 2>/dev/null | grep -q y; do sleep 20; done
python3 ip_run.py sh 5300606 'pkill -f chain1.sh; pkill -f pytest; sleep 1; echo killed'
python3 ip_run.py bg 5300606 chain1b "$(cat mk_chain4.sh)"
echo relayed
