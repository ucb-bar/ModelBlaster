import sys, base64, io, tarfile
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm')
import farm
S='/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad'
job, pat, dest = int(sys.argv[1]), sys.argv[2], sys.argv[3]
rc,out,err=farm.ssh(f"srun --jobid={job} --overlap bash -s", f"cd /tmp/krisd/a3 && tar czf - {pat} 2>/dev/null | base64\n".encode(), timeout=600)
b=base64.b64decode(out)
if b:
    tarfile.open(fileobj=io.BytesIO(b)).extractall(dest)
print('fetched', len(b))
