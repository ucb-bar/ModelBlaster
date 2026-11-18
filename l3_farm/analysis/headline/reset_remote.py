"""Kill the headline instance's problem workers on its nodes and remove their dirs (only /tmp/krisd/farm_headline/<problem>/).
Run with FARM_CONFIG=analysis/headline/config_headline.json from l3_farm; the dispatcher must be stopped. Usage: reset_remote.py JOB..."""
import sys
sys.path.insert(0, ".")
import farm
for job in sys.argv[1:]:
    rc = farm.ssh(f"srun --jobid={job} --overlap bash -c 'pkill -f [/]tmp/krisd/farm_headline/ ; sleep 3; "
                  f"find /tmp/krisd/farm_headline -maxdepth 1 -mindepth 1 -name \"[0-9]*_*\" -exec rm -rf {{}} + ; "
                  f"ls /tmp/krisd/farm_headline | head -20'")
    print(job, "rc", rc)
