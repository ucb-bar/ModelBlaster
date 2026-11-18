#!/usr/bin/env python3
"""Poll the 4 arms every 5 min; stop an arm whose ledger cost exceeds the $90 cap or that runs past 3 h (kill.sh, own
arm only); exit when all are terminal."""
import re, sys, time
import st
CAP = 90.0
done = set()
while True:
    line = []
    for arm, job in st.mj()["arms"].items():
        if arm in done:
            continue
        try:
            rc, out, err = st.run_local(job, st.HERE / "status.sh", [arm], timeout=120)
        except Exception as e:
            line.append(f"{arm}:err"); continue
        t = out.decode()
        s = re.search(r"status=(\S*) start=(\d+) end=(\d*)", t)
        usd = re.search(r"ledger total usd ([\d.]+)", t)
        status = s.group(1) if s else "?"
        u = float(usd.group(1)) if usd else 0.0
        el = (time.time() - int(s.group(2))) / 60 if s else 0
        line.append(f"{arm}:{status}:${u:.1f}:{el:.0f}m")
        if status.startswith(("done", "failed")):
            done.add(arm)
        elif u > CAP or el > 180:
            st.run_local(job, st.HERE / "kill.sh", [arm], timeout=60)
            line.append(f"KILLED {arm} (cost cap/time)")
    print(time.strftime("%H:%M"), " ".join(line), flush=True)
    if len(done) == 4:
        print("ALL_DONE", flush=True)
        break
    time.sleep(300)
