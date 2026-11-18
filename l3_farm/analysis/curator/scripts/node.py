"""Helper: push files to the node and run a script there. usage: node.py push local remote | run script [args]"""
import sys, os
sys.path.insert(0,'/Users/krisd/Documents/ModelBlaster-clone/l3_farm')
os.environ.setdefault('FARM_CONFIG','analysis/headline/config_headline.json')
_cwd=os.getcwd()
os.chdir('/Users/krisd/Documents/ModelBlaster-clone/l3_farm')
SCRIPTDIR=os.path.dirname(os.path.abspath(__file__))
import farm, shlex
JOB='5321587'; R='/tmp/krisd/curator'
def push(local, name):
    rc,o,e=farm.put(JOB, local, f'{R}/{name}'); assert rc==0, e[-300:]
def run(script, args='', timeout=3000):
    cmd=(f"srun --jobid={JOB} --overlap bash -c "+shlex.quote(f"cd {R}; source /tmp/krisd/farm/relay.env; export PYTHONUNBUFFERED=1; /tmp/krisd/venv/bin/python {script} {args}"))
    rc,o,e=farm.ssh(cmd,b'',timeout=timeout); return rc,o.decode('utf8','ignore'),e.decode('utf8','ignore')
def pull(name, local):
    rc,o,e=farm.ssh(f"srun --jobid={JOB} --overlap cat {R}/{name}",b'',timeout=600)
    open(local,'wb').write(o); return rc
if __name__=='__main__':
    sys.argv=[sys.argv[0]]+[os.path.join(_cwd,a) if (i in (2,3) and sys.argv[1] in('push','pull') and not os.path.isabs(a) and ('/' in a)) else a for i,a in enumerate(sys.argv[1:],1)]
    if sys.argv[1]=='push': push(sys.argv[2], sys.argv[3])
    elif sys.argv[1]=='run':
        rc,o,e=run(sys.argv[2],' '.join(sys.argv[3:])); print(o); print(e[-1500:]); print('rc',rc)
    elif sys.argv[1]=='pull': print(pull(sys.argv[2], sys.argv[3]))
