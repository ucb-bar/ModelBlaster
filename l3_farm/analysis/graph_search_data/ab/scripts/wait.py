import subprocess, sys, time
S = "/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad"
mins = float(sys.argv[1]) if len(sys.argv) > 1 else 30
t_end = time.time() + 60 * mins
while True:
    st = {}
    for job, arm in ((5299265, "R"), (5299305, "I")):
        p = subprocess.run(["python3", f"{S}/a4/nodes.py", "run", str(job), f"{S}/a4/status.sh", arm, "3"],
                           capture_output=True, text=True, cwd="/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
        line = next((l for l in p.stdout.splitlines() if l.startswith("status ")), "status ?")
        st[arm] = line.split(" ", 1)[1]
    print(time.strftime("%H:%M:%S"), st, flush=True)
    if all(v not in ("running", "?") for v in st.values()) or time.time() > t_end:
        break
    time.sleep(120)
