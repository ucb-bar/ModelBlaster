import subprocess, sys, time
S = "/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad"
job, arm, needle, mins = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
t_end = time.time() + 60 * mins
while time.time() < t_end:
    p = subprocess.run(["python3", f"{S}/a4/nodes.py", "run", job, f"{S}/a4/peek.sh", arm], capture_output=True,
                       text=True, cwd="/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
    if needle in p.stdout:
        print(p.stdout)
        break
    time.sleep(60)
else:
    print("timeout", p.stdout[-3000:])
