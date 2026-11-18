import sys, time
import mtrun
from mtrun import sh, check


def done(a):
    job, R = check(a)
    return "ARM_DONE" in sh(job, f"tail -n 3 {R}/arm.log", 120)


pending = set(mtrun.ARMS)
while pending:
    for a in list(pending):
        try:
            if done(a):
                pending.discard(a)
                print(time.strftime("%H:%M:%S"), a, "done", flush=True)
        except Exception as e:
            print("poll err", a, repr(e)[:100], flush=True)
    if pending:
        time.sleep(240)
print("ALL DONE")
