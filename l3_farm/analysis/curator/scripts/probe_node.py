import sys, os
sys.path.insert(0,'.')
os.environ.setdefault('FARM_CONFIG','analysis/headline/config_headline.json')
import farm
cmd = ("srun --jobid=5321587 --overlap bash -c 'hostname; sed -E \"s/(TOKEN[A-Z_]*=).*/\\1<redacted>/\" /tmp/krisd/farm/relay.env | head; "
       "ls /tmp/krisd/venv/bin | head -3; /tmp/krisd/venv/bin/python -c \"import aiohttp;print(aiohttp.__version__)\"; mkdir -p /tmp/krisd/curator; ls /tmp/krisd/curator | head; "
       "ls /tmp/krisd/headline_code/ModelBlaster/third_party/KernelBlaster/src | head -3'")
rc,out,err=farm.ssh(cmd,b'',timeout=120)
print(rc,out.decode(),err.decode()[-500:])
