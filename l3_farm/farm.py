#!/usr/bin/env python3
"""Level-3 farm: queue KernelBench problems onto however many L40S Computelab nodes we hold, grabbing more as
needed. State is local (state/farm.json); nodes are driven only through ssh -> srun --overlap scripts, and
every problem runs detached on its node, so this dispatcher can be stopped/restarted freely.

  farm.py init            fetch the level's problems, build the queue
  farm.py tick            one scheduling pass      farm.py run     loop (tick_s)
  farm.py status          queue + nodes            farm.py report  results table (results/report.md)
  farm.py requeue NAME    farm.py skip NAME        farm.py stage   rebuild the code tarball sent to new nodes
  farm.py gk status|show|merge NAME|pull NAME|off NAME|on NAME|rollback VID|rebuild [VID] [--exclude A,B]|prune|bootstrap
                          global knowledge store (config.json "global_knowledge"; analysis/global_knowledge.md)
  farm.py rerun NAME [--from-scratch]   stop the problem's worker and redo it: resume from its last pack (default; a finished
                          problem is resumed from its FINAL pack) or from scratch (old results archived to results/_prev/)
  farm.py dupcheck        read-only: is any problem running on two nodes / on a node the state does not know?
  farm.py sched [est|plan]  scheduling: estimates and the staged order (config.json "scheduling";
                          analysis/scheduling.md); with scheduling.barrier: the phase 1 / phase 2 orders
  farm.py barrier [status|open]   barrier scheduling (scheduling.barrier.enabled; analysis/barrier_scheduling.md): phase, parked /
                          barrier state and decision log; `open` forces the barrier open (applied at the dispatcher's next tick)
FARM_CONFIG=other.json farm.py ...   a second, independent farm instance (its "instance" block: own state,
                          results, remote dir, code root, jobs) -- e.g. the scheduling smoke test.
The API key never touches argv or a node's disk: it is read from ../set_key.sh and sent over stdin to relay.sh.

Global knowledge (on via config.json): before a problem starts, its seed is materialized from the store's current
snapshot (partition cuda:sm_<arch>|<dtype>) into results/<name>/gk/seed and uploaded; worker4.sh starts
kb_graph_opt with --knowledge-global read; pack2.sh brings back a consistent knowledge.db; when the problem
reaches a final state the controller merges it (delta against its seed, ledger-idempotent, under the store lock).
"""
import argparse, fcntl, hashlib, json, os, re, shlex, shutil, subprocess, sys, threading, time, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import farm_sched as SCH  # noqa: E402  (l3_farm/farm_sched.py: scheduling policies, pure functions)
import farm_defaults as FD  # noqa: E402  (allocation / relay defaults, pure functions; analysis/defaults_fix/README.md)

CFG = json.loads((ROOT / os.environ.get("FARM_CONFIG", "config.json")).read_text())
# "instance" (absent in config.json = the production farm, unchanged): a second farm with its own state, results,
# remote dir and code tree on nodes that the production dispatcher has set up (venv, relay are shared)
INST = CFG.get("instance") or {}
STATE = ROOT / INST.get("state", "state/farm.json")
RES = ROOT / INST.get("results", "results")
MB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster"
KEYFILE = Path.home() / "Documents/ModelBlaster-clone/set_key.sh"
F = INST.get("remote_dir", "/tmp/krisd/farm")
SHARED = "/tmp/krisd/farm"                 # setup.status / relay.env of a node (production farm's)
CODE_ROOT = INST.get("code_root")          # None: /tmp/krisd (the production tree)
GK = CFG.get("global_knowledge") or {}
SC = SCH.conf(CFG)
KBSRC = MB / "third_party" / "KernelBlaster" / "src"
# the new behaviours (staged / pooled scheduling, round merges, mid-run refresh, a second instance) use worker5.sh
# and probe2.sh; the default configuration keeps worker4.sh / probe.sh exactly as before
# halfway sync (global_knowledge.halfway: true; analysis/halfway_sync.md): default off = nothing below changes
HALFWAY = bool(GK.get("halfway")) and bool(GK.get("enabled")) and GK.get("mode", "off") != "off"
# characterization sync (global_knowledge.char_sync: true; X-151): after A4's characterization pass the run waits for a package (wait
# mode; with scheduling.barrier the run parks instead). Default off = nothing changes
CHARSYNC = bool(GK.get("char_sync")) and bool(GK.get("enabled")) and GK.get("mode", "off") != "off"
# barrier scheduling (scheduling.barrier.enabled: true; analysis/barrier_scheduling.md): default off = nothing below changes
BAR = SCH.barrier_enabled(SC)
NEW = (SC["mode"] != "independent" or GK.get("merge_on") == "round" or GK.get("refresh", "off") != "off"
       or bool(INST) or HALFWAY or BAR or CHARSYNC)
WORKER, PACK = ("worker5.sh" if NEW else "worker4.sh"), "pack2.sh"
PROBE = "probe2.sh" if NEW else "probe.sh"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ssh(cmd, inp=None, timeout=300):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", CFG["host"], cmd],
                       input=inp, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def srun(job, script, args=(), inp=b"", key=False, timeout=300):
    """Run remote/<script> on the job's node. The script arrives on stdin (after the key line when key=True)."""
    body = (ROOT / "remote" / script).read_bytes()
    pre = b""
    if key:
        k = re.search(r'^[ \t]*(?:export[ \t]+)?API_KEY="?([^"\n]+)', KEYFILE.read_text(), re.M).group(1)   # skips commented-out old keys
        # the script's own `read -r KEY` consumes the first stdin line after the script is fed through bash -s,
        # so deliver the key as a heredoc ahead of it instead (same scheme as b200/run_remote.sh)
        body = body.replace(b"read -r KEY\n", b"")
        pre = b"read -r KEY <<'__K__'\n" + k.encode() + b"\n__K__\n"
    cmd = f"srun --jobid={job} --overlap bash -s -- " + " ".join(shlex.quote(str(a)) for a in args)
    return ssh(cmd, pre + body + b"\n" + inp, timeout)


def detach(job, script, args=(), inp=b"", key=False):
    """Start remote/<script> as a long-lived srun step owned by a nohup'd process on the frontend (a process
    started *inside* a short step is killed when the step ends). stdin (script + optional key) is handed to it
    before the ssh session closes."""
    a = " ".join(shlex.quote(str(x)) for x in args)
    if not key:  # run from a file so child processes cannot swallow the script from stdin
        put(job, ROOT / "remote" / script, f"{F}/{script}")
        cmd = f"nohup setsid srun --jobid={job} --overlap bash {F}/{script} {a} </dev/null >/dev/null 2>&1 & sleep 8; echo started"
        rc, out, err = ssh(cmd, b"", timeout=120)
        return rc == 0 and b"started" in out
    body = (ROOT / "remote" / script).read_bytes()
    k = re.search(r'^[ \t]*(?:export[ \t]+)?API_KEY="?([^"\n]+)', KEYFILE.read_text(), re.M).group(1)   # skips commented-out old keys
    body = body.replace(b"read -r KEY\n", b"")
    pre = b"read -r KEY <<'__K__'\n" + k.encode() + b"\n__K__\n"
    cmd = f"nohup setsid srun --jobid={job} --overlap bash -s -- {a} <&0 >/dev/null 2>&1 & sleep 12; echo started"
    rc, out, err = ssh(cmd, pre + body + b"\n" + inp, timeout=120)
    return rc == 0 and b"started" in out


def put(job, local, remote):
    cmd = f"srun --jobid={job} --overlap bash -c {shlex.quote(f'mkdir -p $(dirname {remote}) && cat > {remote}')}"
    return ssh(cmd, Path(local).read_bytes() if not isinstance(local, bytes) else local, timeout=900)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"problems": {}, "nodes": {}, "last_alloc": 0, "tree": ""}


def mode_line() -> str:
    """What `farm.py run` / `status` print: the active scheduling and sharing configuration (ablation row)."""
    gk = (f"global knowledge {'on' if gk_enabled() else 'OFF'} (mode {GK.get('mode', 'off')}, merge_on "
          f"{GK.get('merge_on', 'final')}, refresh {GK.get('refresh', 'off')})")
    sc = SC["mode"]
    if sc == "staged":
        sc += (f" (seed {SC['seed'] if isinstance(SC['seed'], list) else 'auto x' + str(SC['seed_n'])}, W={SC['gate_w']}, "
               f"max_wait {SC['max_wait_min']} min, idle_max {SC['idle_max_min']} min)")
    elif sc == "pooled":
        sc += f" ({json.dumps(SC.get('pool') or {})})"
    if BAR:
        bc = SCH.bconf(SC)
        sc += (f" + BARRIER after {bc['after']} (min_frac {bc['min_frac']}, max_wait {bc['max_wait_min']} min, phase 1 "
               f"{bc['phase1_order']}, phase 2 {bc['phase2_order']})")
    inst = f" | instance {INST.get('name', '?')} state={STATE.relative_to(ROOT)} remote={F}" if INST else ""
    return f"scheduling: {sc} | {gk} | worker {WORKER} probe {PROBE}{inst}"


def save(s):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def left_min(s):
    m = re.match(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$", s.strip())
    if not m:
        return 0
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 1440 + h * 60 + mi


def jobs():
    rc, out, err = ssh("squeue --me -h -o '%i|%T|%N|%L|%P'")
    if rc:
        raise RuntimeError(err.decode()[:200])
    res = []
    for ln in out.decode().splitlines():
        p = ln.split("|")
        if len(p) == 5 and CFG["partition_match"] in p[4].lower() and int(p[0]) not in CFG["ignore_jobs"]:
            if INST.get("only_jobs") is not None and int(p[0]) not in INST["only_jobs"]:
                continue
            raw = left_min(p[3])             # left: conservative (minus left_safety_min, default 15); left_raw: squeue's value
            res.append(dict(job=int(p[0]), state=p[1], host=p[2], left=FD.effective_left(raw, CFG), left_raw=raw))
    return res


def probe(job):
    rc, out, _ = srun(job, PROBE, [F] if PROBE != "probe.sh" else [], timeout=60)
    d = {"jobs": {}, "rounds": {}, "halfway": {}, "charsync": {}}
    for ln in out.decode().splitlines():
        if ln.startswith("charsync "):                 # probe2.sh: a run waiting after its characterization pass (X-151)
            _, n, k = ln.split(" ", 2)
            d["charsync"][n] = int(k.split()[0]) if k.split()[0].isdigit() else 0
        elif ln.startswith("halfway "):                  # probe2.sh: a run parked at its halfway point, waiting for us
            _, n, k = ln.split(" ", 2)
            d["halfway"][n] = int(k.split()[0]) if k.split()[0].isdigit() else 0
        elif ln.startswith("job "):
            _, n, st = ln.split(" ", 2)
            d["jobs"][n] = st
        elif ln.startswith("rounds "):
            _, n, k = ln.split(" ", 2)
            d["rounds"][n] = int(k) if k.strip().isdigit() else 0
        elif "=" in ln:
            k, v = ln.split("=", 1)
            d[k] = v
    d["ok"] = rc == 0 and "setup" in d
    return d


def stage(force=False):
    s = load()
    tb = STATE.parent / "mb.tar.gz"
    if not force and tb.exists() and s.get("tree"):
        return s["tree"]
    STATE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "czf", str(tb), "-C", str(MB.parent), "--exclude=.git", "--exclude=__pycache__",
                    "--exclude=*.pyc", "--exclude=ModelBlaster/gpu_eval/runs", "--exclude=ModelBlaster/gpu_eval/ablations",
                    "ModelBlaster"], check=True, env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
    s["tree"] = time.strftime("%Y%m%d-%H%M%S")
    save(s)
    log("staged tree", s["tree"], tb.stat().st_size // 2**20, "MB")
    return s["tree"]


def fetch_problems():
    d = ROOT / "problems"
    d.mkdir(exist_ok=True)
    rc = subprocess.run(["gh", "api", f"repos/ScalingIntelligence/KernelBench/contents/KernelBench/level{CFG['level']}",
                         "--jq", ".[] | select(.name|endswith(\".py\")) | .name + \" \" + .download_url"],
                        capture_output=True, text=True)
    if rc.returncode:
        sys.exit(rc.stderr)
    for ln in rc.stdout.splitlines():
        name, url = ln.split()
        if not (d / name).exists():
            (d / name).write_bytes(subprocess.run(["curl", "-sfL", url], capture_output=True, check=True).stdout)
    return sorted(d.glob("*.py"), key=lambda p: int(p.name.split("_")[0]))


def cmd_init(a):
    s = load()
    for p in fetch_problems():
        n = p.stem
        num = n.split("_")[0]
        if CFG["include"] and num not in map(str, CFG["include"]):
            continue
        if num in map(str, CFG["exclude"]):
            continue
        s["problems"].setdefault(n, dict(status="pending", attempts=0, node=None))
    save(s)
    log(len(s["problems"]), "problems queued")


def node_ready(nd, left):
    return (nd.get("ok") and "L40S" in nd.get("gpu", "") and nd["setup"] == "done" and nd["relay"] == "ok"
            and nd["tree"] == load()["tree"] and not nd["jobs"].get("__running__") and nd["foreign"] == "0"
            and left >= CFG["min_left_min"])


def bring_up(job, nd, s):
    """Advance one node one step toward ready (all steps non-blocking on the node side)."""
    if "L40S" not in nd.get("gpu", ""):
        return "not-l40s"
    if nd["tree"] != s["tree"] and not any(v == "running" for v in nd["jobs"].values()):
        if CODE_ROOT:           # an instance: its own tree under its code root (the production tree is not touched)
            put(job, STATE.parent / "mb.tar.gz", f"{F}/mb.tar.gz")
            srun(job, "install_tree2.sh", [s["tree"], F, CODE_ROOT])
            return "tree"
        put(job, ROOT / "state" / "mb.tar.gz", f"{F}/mb.tar.gz")
        srun(job, "install_tree.sh", [s["tree"]])
        return "tree"
    if INST and not INST.get("bring_up", False):
        return "wait (setup/relay are the production dispatcher's)"
    if nd["setup"] == "none" or nd["setup"] == "failed":
        detach(job, "setup.sh")
        return "setup"
    if nd["setup"] == "done" and nd["relay"] != "ok":
        rc_ = FD.relay_conf(CFG)       # config "relay" block, else 96 / 1024 / 10000 (analysis/defaults_fix/README.md)
        detach(job, "relay.sh", [rc_["conc"], rc_["budget"], rc_["queue"]], key=True)
        return "relay"
    return "wait"


def relay_guard(s, job, nd, busy):
    """Verify the running relay on a node against the configured max-concurrency / queue / budget (rate-limited read-only
    remote/relay_info.sh) and restart it with remote/relay_restart.sh (key over stdin, same port/token) when it differs.
    Returns True when a restart was launched (the node is not ready this tick). Never raises."""
    try:
        want = FD.relay_conf(CFG)
        if not want["enforce"] or nd.get("relay") != "ok" or (INST and not INST.get("bring_up", False)):
            return False
        st = s.setdefault("relay_check", {}).setdefault(str(job), {})
        now = time.time()
        if not FD.check_due(st, now, want["check_min"]):
            return False
        st["t"] = now
        st["restarted_pending"] = False
        rc, out, _ = srun(job, "relay_info.sh", timeout=60)
        have = FD.parse_relay_info(out.decode()) if rc == 0 else {}
        act, diffs = FD.relay_decision(want, have, st, busy)
        st.update(have=have, diff=bool(diffs))
        if act == "restart":
            log("relay", job, "differs from config:", ", ".join(diffs), "-> restarting via relay_restart.sh (port", have["port"], ")")
            ok = detach(job, "relay_restart.sh", [want["conc"], want["queue"], want["budget"], have["port"]], key=True)
            st.update(tries=st.get("tries", 0) + 1, restarted=now, restarted_pending=True)
            log("relay", job, "restart", "launched" if ok else "FAILED to launch", "| try", st["tries"], "of", want["max_tries"])
            return bool(ok)
        if act == "busy":
            if not st.get("busy_logged"):
                log("relay", job, "differs from config (", ", ".join(diffs), ") but problems are running on it: not restarted (relay.restart_busy=false)")
                st["busy_logged"] = True
        elif act == "giveup":
            if not st.get("giveup_logged"):
                log("WARNING relay", job, "still differs after", st.get("tries"), "restarts:", ", ".join(diffs), "- manual action needed")
                st["giveup_logged"] = True
        elif act == "noport":
            log("relay", job, "differs (", ", ".join(diffs), ") but its port is unknown: not restarted")
        elif act == "ok" and st.get("tries"):
            log("relay", job, "now matches config", {k: have.get(k) for k in ("conc", "queue", "budget")})
            st["tries"] = 0
    except Exception as e:                       # the health check must never stop the dispatcher
        log("relay check error on", job, repr(e)[:200])
    return False


# ---------------------------------------------------------------- global knowledge (controller side)

def gk_enabled(p=None):
    return bool(GK.get("enabled")) and GK.get("mode", "off") != "off" and not (p or {}).get("gk_off")


def gk_root():
    return Path(os.path.expanduser(GK.get("path") or "~/Documents/ModelBlaster-clone/knowledge_global"))


def gk_part():
    a = str(CFG["kb"]["arch"])
    return f"cuda:sm_{a}|{GK.get('dtype', 'fp16')}" if a.isdigit() else f"cuda:{a}|{GK.get('dtype', 'fp16')}"


def gk_cli(args, timeout=1800):
    """kernelblaster.knowledge.v2.merge as a subprocess (the farm's python needs nothing but the stdlib)."""
    env = dict(os.environ, PYTHONPATH=str(KBSRC), **{k: str(v) for k, v in (GK.get("env") or {}).items()})   # e.g. KERNELBLASTER_SEED_MINED=0
    cmd = [GK.get("python") or sys.executable, "-m", "kernelblaster.knowledge.v2.merge", "--store", str(gk_root())] + args
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(KBSRC.parent), timeout=timeout)
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"error": f"exit {p.returncode}: {(p.stderr or p.stdout)[-600:]}"}


def gk_policy():
    return json.dumps(GK.get("policy") or {})


def gk_seed(name, p):
    """Materialize (or reuse, for a retry of the same run) the problem's seed. -> tar.gz bytes or None."""
    gd = RES / name / "gk"
    seed = gd / "seed"
    g = p.get("gk") or {}
    if not (g.get("run_uid") and (seed / "knowledge.db").exists()):
        if seed.exists():
            shutil.rmtree(seed)
        gd.mkdir(parents=True, exist_ok=True)
        uid = uuid.uuid4().hex
        res = gk_cli(["seed", "--part", gk_part(), "--out", str(seed), "--problem", name, "--run-uid", uid,
                      "--policy", gk_policy()])
        if "error" in res:
            log(name, "global knowledge seed failed:", res["error"][:300])
            p["gk"] = {"mode": "off", "error": res["error"][:300]}
            return None
        p["gk"] = {"mode": GK.get("mode", "readwrite"), "applied": True, "run_uid": uid, "snapshot": res.get("snapshot"),
                   "part": res.get("part"), "seed_counts": res.get("counts"), "seeded": time.time()}
    tb = gd / "seed.tar.gz"
    subprocess.run(["tar", "czf", str(tb), "-C", str(seed), "."], check=True,
                   env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
    return tb


def gk_merge(name, p, kdir=None, why="final"):
    """Merge a collected run into the store (delta against its seed, or the copy merged last). Never raises."""
    if not gk_enabled(p) or GK.get("mode") == "read":
        return None
    out = RES / name
    kdir = Path(kdir or out / "work" / "knowledge")
    gd = out / "gk"
    g = p.setdefault("gk", {})
    if not (kdir / "knowledge.db").exists():
        g["merge_error"] = f"{why}: no knowledge.db collected"
        log(name, "global knowledge:", g["merge_error"])
        return None
    merged = gd / "merged"
    sj = kdir.parent / "state.json"                # did the run actually start from its seed? (old tree: no)
    try:
        rec = json.loads(sj.read_text()).get("knowledge_global") or {} if sj.exists() else {"applied": True}
    except (OSError, ValueError):
        rec = {"applied": True}
    if g.get("applied") and not rec.get("applied") and not (merged / "knowledge.db").exists():
        g.update(applied=False, run_uid=None, note="the run did not apply its seed; merged against the repo seed")
    args = ["merge", "--run", str(kdir), "--problem", name, "--attempt", str(p.get("attempts", 0)),
            "--model", CFG["kb"]["model"], "--policy", gk_policy(), "--label", f"{why} {name}"]
    if (merged / "knowledge.db").exists():
        args += ["--base", str(merged)]
    elif g.get("applied") and (gd / "seed" / "knowledge.db").exists():
        args += ["--base", str(gd / "seed")]
    if not g.get("run_uid"):                       # a run started without a seed (before global knowledge)
        g["run_uid"] = f"farm:{name}"
        args += ["--part", gk_part()]
    args += ["--run-uid", g["run_uid"]]
    res = gk_cli(args)
    if "error" in res:
        g["merge_error"] = res["error"][:400]
        log(name, "global knowledge merge failed:", g["merge_error"])
        return res
    g.pop("merge_error", None)
    g.update(merged_version=res.get("version"), merge_status=res.get("status"), merged_at=time.time(), merged_on=why)
    g.setdefault("first_merged_at", time.time())
    g["merges"] = (g.get("merges", []) + [{"why": why, "status": res.get("status"), "version": res.get("version"),
                                           "t": time.time(), "outcomes": res.get("outcomes"),
                                           "claims": res.get("claims"),
                                           "adjust": res.get("refresh_base_adjust")}])[-50:]
    if res.get("status") == "merged":            # the incremental base of a later merge of this run
        tmp = gd / "merged.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "impl").mkdir(parents=True)
        shutil.copyfile(kdir / "knowledge.db", tmp / "knowledge.db")
        if (kdir / "impl" / "index.jsonl").exists():
            shutil.copyfile(kdir / "impl" / "index.jsonl", tmp / "impl" / "index.jsonl")
        shutil.rmtree(merged, ignore_errors=True)
        tmp.rename(merged)
        keep = GK.get("snapshot_keep")
        if keep:
            gk_cli(["prune", "--keep", str(keep)])
    log(name, "global knowledge:", res.get("status"), res.get("version"), "| claims", json.dumps(res.get("claims")),
        "outcomes", res.get("outcomes"))
    return res


def efficiency_flags(k, resumed):
    """config.json kb.efficiency_rules / stop_patience / plan_min_share_late -> kb_graph_opt flags (X-129;
    analysis/efficiency_rules.md), worker5.sh only (KB_EXTRA). None for a resumed work dir: it keeps the rules it
    was started with (kb_graph_opt records them; a work dir from before X-129 continues with the rules off)."""
    if resumed:
        return ""
    fl = [f"--{n.replace('_', '-')} {k[n]}" for n in ("efficiency_rules", "stop_patience", "plan_min_share_late")
          if k.get(n) is not None]
    return " ".join(fl)


# config.json kb.problem_flags {problem: "--flag value ..."}: extra kb_graph_opt flags for ONE problem (KB_EXTRA, last, so they
# override the headline flags; 2026-10-26 live-sweep fix runs, e.g. 27_RegNet --seed-nearest on --graph-survival off).

def headline_flags(k):
    """config.json kb.headline_config (path to a headline.json, relative to the farm dir or absolute; analysis/headline/) ->
    kb_graph_opt flags, worker5.sh only (KB_EXTRA). Absent / empty = no flags (defaults unchanged). Also applied to a
    resumed work dir: the flags are part of the round key, so a resume must pass the same ones. Entries with
    "enabled": false are skipped; only the "flags" list is read (env-only items under "env_pending" are NOT applied)."""
    path = k.get("headline_config")
    if not path:
        return ""
    pth = Path(path).expanduser()
    pth = pth if pth.is_absolute() else ROOT / pth
    fl = []
    for e in json.loads(pth.read_text())["flags"]:
        if e.get("enabled", True):
            fl.append(f"{e['flag']} {e['value']}".strip())
    return " ".join(fl)


def headline_env(k):
    """config.json kb.headline_config -> the "env" block of that headline.json ({NAME: value}): exported in job.env (worker5.sh
    sources it), so the node-side KernelBlaster processes see it (e.g. KERNELBLASTER_KIND_MATCH=base). Absent = {}."""
    path = k.get("headline_config")
    if not path:
        return {}
    pth = Path(path).expanduser()
    pth = pth if pth.is_absolute() else ROOT / pth
    return {str(a): str(b) for a, b in (json.loads(pth.read_text()).get("env") or {}).items()}


def routing_flags(k, resumed):
    """config.json kb.model_simple / model_hard / fix_model / reflect_model / upkeep_model / hetero_policy -> kb_graph_opt
    model-routing flags (analysis/beam_hetero_test.md), worker5.sh only (KB_EXTRA; they come after the worker's own
    --reflect-model/--upkeep-model, so they override). None of them set = every role uses kb.model (as in the sweep).
    Not applied to a resumed work dir: one run keeps one routing."""
    if resumed:
        return ""
    flag = {"model_simple": "--model-simple", "model_hard": "--model-hard", "reflect_model": "--reflect-model",
            "upkeep_model": "--upkeep-model", "hetero_policy": "--hetero-policy"}
    fl = [f"{f} {k[n]}" for n, f in flag.items() if k.get(n)]
    if k.get("fix_model"):                        # kb_graph_opt has no flag of its own: pass through to run_multislot
        fl.append(f"--kb-arg=--fix-model --kb-arg={k['fix_model']}")
    return " ".join(fl)


def start(job, name, p=None, pool_hosts=None, park=False):
    p = p if p is not None else {}
    k = CFG["kb"]
    extra = ""
    if _S is not None:        # crash window: "about to launch on job" is on disk before the worker exists (dup_guard)
        p["launching"] = {"node": job, "t": time.time()}
        save(_S)
    env = (f"MODEL={k['model']}\nARCH={k['arch']}\nMIN_GAIN={k['min_gain_pct']}\nROUNDS={k['rounds']}\n"
           f"ROLLOUTS={k['rollouts']}\nSTEPS={k['steps']}\nVSTEPS={k['variant_steps']}\nMAXVAR={k['max_variants']}\n"
           f"TIMEOUT_H={CFG['problem_timeout_h']}\nPLAN_TOPK={k.get('plan_top_k', 0)}\nPLAN_MINSHARE={k.get('plan_min_share', 0.0)}\n")
    d = f"{F}/{name}"
    put(job, ROOT / "problems" / f"{name}.py", f"{d}/problem.py")
    resume = RES / name / "resume.tar.gz"
    if resume.exists():   # the resume pack carries the old job.env/status, so unpack it before writing the current ones
        put(job, resume, f"{d}/resume.tar.gz")
        ssh(f"srun --jobid={job} --overlap bash -c 'cd {d} && tar xzf resume.tar.gz'")
    if gk_enabled(p):
        if resume.exists() and not p.get("gk"):
            # resuming a run that started without global knowledge: keep it that way (same round key);
            # it is still merged (against the repo seed) when it finishes
            p["gk"] = {"mode": "off", "reason": "resumed a run started without global knowledge"}
        elif (p.get("gk") or {}).get("mode", "x") != "off" or not p.get("gk"):
            tb = gk_seed(name, p)
            if tb is not None:
                put(job, tb, f"{d}/gk_seed.tar.gz")
                env += f"GK_MODE=read\nGK_PART='{p['gk']['part'] or gk_part()}'\n"     # quoted: '|' in a sourced file
    if NEW:                   # worker5.sh: code root, relay, mid-run refresh, pooled GPUs
        env += f"CODE_ROOT={CODE_ROOT or '/tmp/krisd'}\nRELAY_ENV={SHARED}/relay.env\n"
        if (GK.get("refresh", "off") == "round" or HALFWAY or CHARSYNC) and "GK_MODE=read" in env:
            env += "GK_REFRESH=1\n"
            if CHARSYNC and not park:                          # (a parking run does not wait: the park is the sync)
                env += f"GK_CHAR_SYNC=1\nGK_CHAR_SYNC_WAIT_S={int(GK.get('char_sync_wait_s', GK.get('halfway_wait_s', 180)))}\n"
            g_ = p.setdefault("gk", {})
            if "halfway_on" not in g_:                        # a resumed run keeps what it started with (round key)
                g_["halfway_on"] = HALFWAY and not resume.exists()
            if HALFWAY and g_["halfway_on"]:
                env += f"GK_HALFWAY=1\nGK_HALFWAY_WAIT_S={int(GK.get('halfway_wait_s', 180))}\n"
            g_.pop("halfway_done", None)                      # (re)started: a parked run is served again
            (p.get("gk") or {}).pop("refresh_for", None)      # a (re)start: the node's inbox is empty
        if pool_hosts:
            pool = SC.get("pool") or {}
            pdir = pool.get('remote_dir', F + '_pool')
            if pool.get('per_problem_dir'):      # one pool dir per problem: identical kernels of different problems (same content-hash names) never share files
                pdir = f"{pdir}/{name}"
            env += (f"POOL_HOSTS='{' '.join(pool_hosts)}'\nPOOL_DIR={pdir}\n"
                    f"POOL_CVD={pool.get('cuda_visible_devices', 'all')}\n")
            p["pool_hosts"] = list(pool_hosts)
        he = p.get("headline_env")                           # a resumed work dir keeps the env it started with
        if he is None:
            he = {} if resume.exists() else headline_env(k)
            if he:
                p["headline_env"] = he
        for ek, ev in he.items():
            env += f"export {ek}='{ev}'\n"
        for ek, ev in ((k.get("problem_env") or {}).get(name) or {}).items():      # config kb.problem_env {problem: {NAME: value}}
            env += f"export {ek}='{ev}'\n"
        rs = resume.exists() and not (BAR and p.get("phase"))   # a barrier problem keeps the flags of its first start in every phase
        extra = " ".join(x for x in (k.get("extra") or "", headline_flags(k), efficiency_flags(k, rs),
                                              routing_flags(k, rs), (k.get("problem_flags") or {}).get(name, "")) if x)
        if extra:
            env += f"KB_EXTRA='{extra}'\n"
    if BAR and park:          # barrier phase 1: park after the characterization pass (worker5.sh: --park-after-char on)
        env += "PARK_AFTER_CHAR=1\n"
    put(job, env.encode(), f"{d}/job.env")
    if BAR and not park and p.get("phase") == "parked" and "GK_REFRESH=1" in env:
        barrier_package(job, name, p)        # the post-barrier refresh package is in the node's inbox before the worker starts
    ok = detach(job, WORKER, [name, F] if NEW else [name])
    try:
        meta_event(job, name, p, "start", resumed=resume.exists(), extra=extra, env=env, pool_hosts=pool_hosts, launched=ok)
    except Exception as e:                                  # metadata never stops a launch
        log(name, "run_meta start failed:", repr(e)[:200])
    return ok


def collect(job, name, final):
    out = RES / name
    out.mkdir(parents=True, exist_ok=True)
    rc, data, _ = srun(job, PACK, [name, F, F + "_pack"] if INST else [name], timeout=1800)
    if data:
        (out / ("final.tar.gz" if final else "resume.tar.gz")).write_bytes(data)
        if final:
            subprocess.run(["tar", "xzf", str(out / "final.tar.gz"), "-C", str(out)], check=False)
    return bool(data)


def alloc_role(s, js):
    """'driver' when the next allocation should become the pooled driver (pooled mode only; see FD.need_driver), else 'pool'."""
    if SC["mode"] != "pooled":
        return "pool"
    sch = s.get("sched") or {}
    pool = SC.get("pool") or {}
    left = {j["job"]: (j.get("left_raw", j["left"]) if j["state"] == "RUNNING" else None) for j in (js or [])}
    return "driver" if FD.need_driver(pool.get("driver") or sch.get("driver"), s.get("alloc_roles") or {}, left,
                                      FD.alloc_conf(CFG)["successor_min"]) else "pool"


_JS = []                                    # the jobs() of the current tick (alloc role decision)


def alloc(s, js=None):
    """New L40S allocation. alloc_mode "crun" (default, unchanged: crun --no-shell) or "batch" (sbatch holder, remote/holder.sh via
    batch_nodes.py), per role (alloc.driver_mode / driver_time, alloc.pool_*); the time is alloc_time (24:00:00 -> 1-00:00:00)."""
    if INST and not INST.get("alloc", False):
        return
    js = _JS if js is None else js
    role = alloc_role(s, js)
    plan = FD.alloc_plan(CFG, role)
    if plan["note"]:
        log("alloc:", plan["note"], "->", plan["time"])
    ac = FD.alloc_conf(CFG)
    s["last_alloc"] = time.time()
    if plan["mode"] == "batch":
        import batch_nodes as B
        try:
            job = B.submit_holder(time=plan["time"], partitions=ac.get("partitions"), account=ac["account"], warn_s=ac["warn_s"],
                                  name=ac["name"], ssh=ssh)
        except Exception as e:
            log("batch holder submit failed:", repr(e)[:300])
            return
        s.setdefault("alloc_roles", {})[str(job)] = role
        log("submitted batch holder", job, "| role", role, "| time", plan["time"], "| USR1 warning", ac["warn_s"], "s before the limit")
        return
    q = CFG["alloc_query"]
    cmd = f"crun -q {shlex.quote(q)} --no-shell --gpus 1 -t {plan['time']}"
    lf = open(ROOT / "state" / "alloc.log", "ab")
    subprocess.Popen(["ssh", "-o", "BatchMode=yes", CFG["host"], cmd], stdout=lf, stderr=lf,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    if role == "driver" and (ac.get("driver_mode") or ac.get("driver_time")):
        s["alloc_pending"] = {"role": "driver", "t": time.time(), "known": [j["job"] for j in (js or [])]}
    log("requested a new L40S allocation | crun | role", role, "| time", plan["time"])


def tag_new_alloc(s, js):
    """A crun allocation has no id at request time: the first job that appears after a driver-role request gets the role."""
    pend = s.get("alloc_pending")
    if not pend:
        return
    new = [j["job"] for j in js if j["job"] not in pend["known"]]
    if new:
        s.setdefault("alloc_roles", {})[str(min(new))] = pend["role"]
        s.pop("alloc_pending")
    elif time.time() - pend["t"] > 3600:
        s.pop("alloc_pending")


# ---------------------------------------------------------------- hardening (X-128; analysis/restart_drill.md)

_S = None                                   # the state tick() is working on (start() persists a "launching" marker)
LOCK = STATE.with_suffix(".lock")
CMDS = STATE.with_suffix(".cmds")
EXPECT_MEM_MHZ = int(CFG.get("expect_mem_mhz", 9001))     # a slower L40S variant (job 5298298) has a lower max memory clock
SMI_KEYS = ["gpu", "gpu_uuid", "driver", "vbios", "mem_clock_max_mhz", "mem_clock_mhz", "sm_clock_max_mhz", "sm_clock_mhz",
            "pstate", "mem_total_mib"]


def take_lock():
    """One dispatcher per state file (flock, released by the OS when the process dies): a restart while the old
    dispatcher is still alive would otherwise start the same problem twice. -> the open lock file (keep it)."""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.seek(0)
        sys.exit(f"another dispatcher holds {LOCK} (pid {f.read().strip() or '?'}): refusing to run a second one on this state")
    f.seek(0)
    f.truncate()
    f.write(str(os.getpid()))
    f.flush()
    return f


def dispatcher_alive():
    f = open(LOCK, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        f.close()


def tree_info(s):
    """tree stamp + sha256 of the tarball new nodes unpack (cached in farm.json)."""
    tb = STATE.parent / "mb.tar.gz"
    ti = s.get("tree_info") or {}
    if tb.exists() and (ti.get("stamp") != s.get("tree") or ti.get("size") != tb.stat().st_size):
        h = hashlib.sha256()
        with open(tb, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""):
                h.update(ch)
        ti = {"stamp": s.get("tree"), "sha256": h.hexdigest(), "size": tb.stat().st_size}
        s["tree_info"] = ti
    return ti


def node_meta(job):
    """nvidia-smi facts of a node (remote/nodemeta.sh): GPU, driver, VBIOS, memory / SM clocks. mem_clock_ok False
    = the max memory clock differs from the expected one (config expect_mem_mhz, default 9001)."""
    try:
        rc, out, _ = srun(job, "nodemeta.sh", timeout=60)
    except Exception as e:
        return {"job": job, "error": repr(e)[:200]}
    d = {"job": job}
    for ln in out.decode(errors="replace").splitlines():
        if ln.startswith("smi="):
            d.update(zip(SMI_KEYS, [x.strip() for x in ln[4:].split(",")]))
        elif "=" in ln:
            k, v = ln.split("=", 1)
            d[k] = v
    try:
        d["mem_clock_ok"] = int(float(d.get("mem_clock_max_mhz"))) == EXPECT_MEM_MHZ
    except (TypeError, ValueError):
        d["mem_clock_ok"] = None
    return d


def meta_event(job, name, p, event, resumed=False, extra="", env="", pool_hosts=None, launched=True, status=None):
    """Append a start / finish record to p["meta"] (farm.json) and results/<name>/run_meta.json."""
    s = _S if _S is not None else load()
    now = time.time()
    nm = node_meta(job)
    ev = {"event": event, "t": now, "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "node_job": job,
          "node": nm, "instance": INST.get("name"), "sched": SC["mode"], "worker": WORKER}
    if nm.get("mem_clock_ok") is False:
        ev["flag"] = f"slow memory clock {nm.get('mem_clock_max_mhz')} MHz (expected {EXPECT_MEM_MHZ})"
        s.setdefault("slow_nodes", {})[str(job)] = {"mem_clock_max_mhz": nm.get("mem_clock_max_mhz"), "host": nm.get("host"),
                                                    "vbios": nm.get("vbios"), "seen": ev["iso"], "problem": name}
        log(name, "WARNING node", job, ev["flag"], "| vbios", nm.get("vbios"), "driver", nm.get("driver"))
    if event == "start":
        k = CFG["kb"]
        ti = tree_info(s)
        sm = re.search(r"--seed\s+(\d+)", extra or "")
        g = p.get("gk") or {}
        ev.update(attempt=p.get("attempts", 0) + 1, resumed=resumed, launched=launched, tree=ti,
                  kb={x: k.get(x) for x in ("model", "rounds", "rollouts", "steps", "variant_steps", "max_variants", "plan_top_k",
                                            "plan_min_share", "min_gain_pct", "arch")},
                  kb_extra=extra, seeds={"kb_seed": int(sm.group(1)) if sm else 0, "kb_seed_source": "flag" if sm else "default",
                                         "gk_snapshot": g.get("snapshot"), "gk_run_uid": g.get("run_uid"), "gk_mode": g.get("mode")},
                  job_env_sha256=hashlib.sha256(env.encode()).hexdigest() if env else None)
        if pool_hosts:
            ev["pool_hosts"] = list(pool_hosts)
            ev["pool_nodes"] = [node_meta(int(j)) for j, nd in (s.get("nodes") or {}).items()
                                if nd.get("host") in pool_hosts and int(j) != job]
    else:
        ev.update(status=status or p.get("status"), detail=p.get("detail"), attempts=p.get("attempts"),
                  wall_min=round((now - p.get("started", now)) / 60, 1))
    p.setdefault("meta", {})[event] = ev
    (p["meta"].setdefault("history", [])).append({"event": event, "iso": ev["iso"], "node_job": job, "status": ev.get("status"),
                                                  "mem_clock_ok": nm.get("mem_clock_ok")})
    out = RES / name
    out.mkdir(parents=True, exist_ok=True)
    rm = out / "run_meta.json"
    try:
        doc = json.loads(rm.read_text())
    except (OSError, ValueError):
        doc = {"problem": name, "events": []}
    doc["events"].append(ev)
    rm.write_text(json.dumps(doc, indent=1, default=str))
    return ev


def kill_worker(job, name):
    """Stop one problem's worker (and its kb_graph_opt tree) on its node; the problem dir is left alone."""
    pat = "[/]" + F.lstrip("/")
    return ssh(f"srun --jobid={job} --overlap bash -c 'pkill -f {pat}/{name}/; pkill -f \"worker[0-9]*.sh {name}\"; sleep 3; true'")


def dup_guard(s, probes, live, dry=False):
    """dupcheck.py folded into the tick. From this tick's probes: which problems run where.
    * running on a node although the state says pending / failed (dispatcher died between launch and save, or `requeue`
      of a live problem): ADOPT (state -> running on that node), so no second worker is started;
    * running on two nodes: keep the recorded one (else the one with more completed rounds) and kill the other worker;
    * a pending problem whose launch marker is on disk but whose node cannot be probed: HELD this tick;
    * state running on a live, probed node that has no trace of it for 5 min: requeue (the launch never happened).
    config "dup_guard": repair (default) | warn | off. Returns the names to hold (not to be started this tick)."""
    mode = CFG.get("dup_guard", "repair")
    if mode == "off":
        return []
    where = {}
    for j, nd in probes.items():
        if nd.get("ok"):
            for n, st in nd["jobs"].items():
                if st == "running" and n in s["problems"]:
                    where.setdefault(n, []).append(j)
    held, now, fix = [], time.time(), (mode == "repair" and not dry)
    for n, p in s["problems"].items():
        nodes = where.get(n, [])
        if len(nodes) > 1:
            keep = p["node"] if p["node"] in nodes else max(nodes, key=lambda j: (probes[j].get("rounds", {}).get(n, 0), -j))
            log(n, "DUPLICATE: running on", nodes, "-> keep", keep, "" if fix else "(not repaired: " + mode + ")")
            for j in nodes:
                if j != keep and fix:
                    kill_worker(j, n)
                    log(n, "killed the duplicate worker on", j)
            nodes = [keep]
            if fix:
                s.setdefault("dup_events", []).append({"t": now, "problem": n, "what": "duplicate", "nodes": where[n], "kept": keep})
        if nodes and p["status"] in ("pending", "failed"):
            log(n, "ORPHAN: state", p["status"], "but running on", nodes[0], "-> adopt" if fix else "(not adopted: " + mode + ")")
            if fix:
                p.update(status="running", node=nodes[0], started=p.get("started") or now, synced=now, adopted=now)
                p.pop("launching", None)
                s.setdefault("dup_events", []).append({"t": now, "problem": n, "what": "adopted", "node": nodes[0]})
        elif nodes and p["status"] == "running" and p["node"] != nodes[0]:
            log(n, "MOVED: state says", p["node"], "but running on", nodes[0], "-> follow" if fix else "")
            if fix:
                p["node"] = nodes[0]
        elif nodes and p["status"] not in ("running",):
            log(n, "WARNING: worker running on", nodes[0], "although state is", p["status"])
        ln = p.get("launching")
        if ln and p["status"] == "pending" and not nodes:
            nd = probes.get(ln["node"])
            if nd and nd.get("ok") and now - ln["t"] > 90:
                p.pop("launching")                 # probed after the launch window: nothing started there
            elif ln["node"] not in live:
                p.pop("launching")
            else:
                held.append(n)
                log(n, "HELD: launch marker on", ln["node"], "but that node cannot be checked yet")
        if (p["status"] == "running" and p["node"] in probes and probes[p["node"]].get("ok") and not nodes
                and n not in probes[p["node"]]["jobs"] and now - p.get("started", now) > 300 and fix):
            log(n, "VANISHED: no trace on", p["node"], "-> requeue")
            p.update(status="pending", node=None)
    return held


def cmd_dupcheck(a=None):
    """farm.py dupcheck: read-only report (nothing is changed or saved)."""
    s = load()
    js = jobs()
    live = {j["job"]: j for j in js if j["state"] == "RUNNING"}
    with ThreadPoolExecutor(8) as ex:
        probes = dict(zip(live, ex.map(probe, live)))
    held = dup_guard(s, probes, live, dry=True)
    where = {}
    for j, nd in probes.items():
        for n, st in (nd.get("jobs") or {}).items():
            if st == "running":
                where.setdefault(n, []).append(j)
    print("running on nodes:", json.dumps(where), "| held:", held)
    print("clean" if all(len(v) == 1 and s["problems"].get(n, {}).get("node") == v[0] for n, v in where.items()) else "NOT CLEAN")


def keep_resume(name):
    """A worker that ended without finishing leaves its newest state in final.tar.gz: make it the resume pack, so a retry
    (same or other node) continues from there instead of from the last periodic sync (or from scratch)."""
    f = RES / name / "final.tar.gz"
    if f.exists() and f.stat().st_size > 0:
        shutil.copyfile(f, RES / name / "resume.tar.gz")


def rerun(s, name, p, from_scratch=False):
    """Stop the problem's worker, clear its node dir and queue it again. Default: resume from the pack kept in results
    (a finished problem: its FINAL pack, so it returns its complete state again). from_scratch: results/<name> is moved to
    results/_prev/ and the problem starts as new (a fresh global-knowledge seed, no resume)."""
    job = p.get("node")
    if job:
        try:
            kill_worker(job, name)
            ssh(f"srun --jobid={job} --overlap bash -c 'rm -rf {F}/{name}'")
        except Exception as e:
            log(name, "rerun: node cleanup failed:", repr(e)[:200])
    out = RES / name
    if from_scratch:
        if out.exists():
            dst = RES / "_prev" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(dst))
            log(name, "rerun from scratch: results archived to", dst)
        keep = {"gk_off": p["gk_off"]} if p.get("gk_off") else {}
        p.clear()
        p.update(status="pending", attempts=0, node=None, rerun=time.time(), **keep)
        s.setdefault("dup_events", []).append({"t": time.time(), "problem": name, "what": "rerun-from-scratch"})
    else:
        if p.get("status") in ("done", "failed") and (out / "final.tar.gz").exists():
            keep_resume(name)
        p.update(status="pending", attempts=0, node=None, detail="")
        p.pop("launching", None)
        p["rerun"] = time.time()
    log(name, "rerun queued", "(from scratch)" if from_scratch else "(resume)")


def apply_cmd(s, cmd, name, args=()):
    if cmd == "barrier-open":                   # manual override: open the barrier now (persisted like an automatic decision)
        b = s.setdefault("sched", {}).setdefault("barrier", {})
        if not b.get("open"):
            b.update(open=True, opened=time.time(), why="manual: farm.py barrier open")
            barrier_log(s, "open", why=b["why"])
            log("BARRIER OPEN:", b["why"])
        return
    p = s["problems"].get(name)
    if p is not None and BAR and SCH.is_parked(p) and cmd in ("requeue", "rerun") and "--from-scratch" not in args:
        log(name, cmd, "ignored: the problem is parked (its pack is kept; it resumes after the barrier); use --from-scratch to redo it")
        return
    if p is None:
        log("command", cmd, name, "ignored: unknown problem")
    elif cmd == "skip":
        p.update(status="skipped", attempts=0, node=None)
    elif cmd == "requeue":                      # state only, as before (the worker is NOT stopped: dup_guard re-adopts a live one)
        p.update(status="pending", attempts=0, node=None)
    elif cmd == "rerun":
        rerun(s, name, p, "--from-scratch" in args)


def apply_cmds(s):
    """Commands queued by the CLI while this dispatcher owns farm.json (requeue / skip / rerun)."""
    if not CMDS.exists():
        return
    lines = CMDS.read_text().splitlines()
    CMDS.unlink()
    for ln in lines:
        try:
            c = json.loads(ln)
            log("command:", c["cmd"], c["name"], c.get("args") or "")
            apply_cmd(s, c["cmd"], c["name"], c.get("args") or [])
        except Exception as e:
            log("bad command", ln[:100], repr(e)[:200])


def queue_cmd(cmd, name, args=()):
    if dispatcher_alive():       # the dispatcher owns farm.json: it applies the command at its next tick
        with open(CMDS, "a") as f:
            f.write(json.dumps({"cmd": cmd, "name": name, "args": list(args), "t": time.time()}) + "\n")
        print(f"{cmd} {name}: queued for the running dispatcher ({CMDS.name}), applied at its next tick")
        return
    lk = take_lock()
    s = load()
    apply_cmd(s, cmd, name, args)
    save(s)
    print(f"{cmd} {name}: done (no dispatcher running)")
    lk.close()


def tick():
    global _S
    s = load()
    tree = stage()
    s = load()
    _S = s
    apply_cmds(s)
    js = jobs()
    live = {j["job"]: j for j in js if j["state"] == "RUNNING"}
    pending_alloc = [j for j in js if j["state"] == "PENDING"]
    tag_new_alloc(s, js)
    with ThreadPoolExecutor(8) as ex:
        probes = dict(zip(live, ex.map(probe, live)))
    if not BAR:                                      # barrier switched off while problems are parked: they just resume
        for n_, p_ in s["problems"].items():
            if p_["status"] == "parked":
                log(n_, "parked but barrier scheduling is off -> pending (resumes from its pack)")
                p_["status"] = "pending"
    held = dup_guard(s, probes, live)                # adopt / repair a double or orphan launch before anything is started
    for n in held:
        s["problems"][n]["status"] = "pending_hold"  # not startable this tick (restored before the save)
    # nodes that disappeared: requeue their running problems (resume from last sync if any)
    for name, p in s["problems"].items():
        if p["status"] == "running" and p["node"] not in live:
            log(name, "lost node", p["node"], "-> requeue")
            p.update(status="pending", node=None)
    idle = 0
    free = []                                       # staged / pooled: ready nodes, decided after the loop
    _mid = []                                       # (job, name, p, nd) of running problems: their mid-run sharing, parallel after the loop
    for job, nd in probes.items():
        left = live[job]["left"]
        if not nd["ok"]:
            log("node", job, "probe failed")
            continue
        mine = {n: st for n, st in nd["jobs"].items()}
        nd["jobs"]["__running__"] = any(st == "running" for st in mine.values())
        for name, st in mine.items():
            p = s["problems"].get(name)
            if not p or p["status"] != "running" or p["node"] != job:
                continue
            if st == "running":
                if GK.get("merge_on") == "round" or GK.get("refresh", "off") != "off" or HALFWAY or CHARSYNC:
                    _mid.append((job, name, p, nd))   # run in a pool after the node loop (serial, a 64-problem pass took ~20 min of refresh uploads)
                elif time.time() - p.get("synced", p["started"]) > CFG["sync_min"] * 60:
                    collect(job, name, False)
                    p["synced"] = time.time()
                    if GK.get("merge_on") == "sync":
                        gk_merge(name, p, gk_unpack_sync(name), why="sync")
            elif BAR and st.startswith("parked:") and p.get("phase") == "p1":
                try:
                    barrier_park(job, name, p, nd)
                except Exception as e:   # never let the barrier stop the dispatcher
                    log(name, "barrier park error:", repr(e)[:300])
            elif BAR and st.startswith("parked:"):
                log(name, "stale parked status on", job, "(phase", p.get("phase"), "): ignored")
            else:
                collect(job, name, True)
                if st == "done":
                    p["status"] = "done"
                elif st == "failed:extract":
                    p["status"] = "unsupported"
                else:
                    p["attempts"] += 1
                    p["status"] = "pending" if p["attempts"] < CFG["max_attempts"] else "failed"
                p["detail"] = st
                p["finished"] = time.time()
                log(name, "->", p["status"], st, "on", job)
                if p["status"] in ("pending", "failed"):
                    keep_resume(name)
                try:
                    meta_event(job, name, p, "finish", status=p["status"])
                except Exception as e:
                    log(name, "run_meta finish failed:", repr(e)[:200])
                # a final state: what the run learned goes into the global store (a failed run's knowledge is
                # still validated evidence -- outcomes are measured, claims cite them; see global_knowledge.md)
                if p["status"] == "done" or (p["status"] == "failed" and GK.get("merge_failed", True)):
                    try:
                        gk_merge(name, p)
                    except Exception as e:   # never let knowledge stop the dispatcher
                        log(name, "global knowledge merge error:", repr(e)[:300])
                nd["jobs"]["__running__"] = False
        if SC["mode"] != "pooled" and nd.get("ok") and relay_guard(s, job, nd, nd["jobs"]["__running__"]):
            continue                                # relay restarted to the configured settings; ready again after the next probe
        if SC["mode"] != "independent" or BAR:
            if nd.get("ok") and nd["tree"] != s["tree"] and not nd["jobs"]["__running__"] and nd["foreign"] == "0":
                log("node", job, "bring-up:", bring_up(job, nd, s), "| tree", nd["tree"])
            elif node_ready(nd, left) or (SC["mode"] == "pooled" and pool_member(nd, left)):
                free.append(job)
                if SC["mode"] == "pooled" and job in sorted(live, key=int)[:2] and (nd["setup"] != "done" or nd["relay"] != "ok") \
                        and not nd["jobs"]["__running__"]:
                    # pooled: pool members never get setup / relay, so a fresh instance had no node that could become the driver;
                    # the two oldest allocations are brought up as driver candidates (venv + relay), the rest stay timing-only
                    log("node", job, "driver candidate bring-up:", bring_up(job, nd, s), "| setup", nd["setup"], "relay", nd["relay"])
            elif not nd["jobs"]["__running__"] and nd["foreign"] == "0":
                step = bring_up(job, nd, s)
                log("node", job, "bring-up:", step, "| setup", nd["setup"], "relay", nd["relay"], "tree", nd["tree"],
                    "foreign", nd["foreign"])
            continue
        if node_ready(nd, left):
            nxt = next((n for n, p in s["problems"].items() if p["status"] == "pending"), None)
            if nxt and start(job, nxt, s["problems"][nxt]):
                s["problems"][nxt].update(status="running", node=job, started=time.time(), synced=time.time())
                log("started", nxt, "on", job)
            elif not nxt:
                idle += 1
        elif not nd["jobs"]["__running__"] and nd["foreign"] == "0":
            step = bring_up(job, nd, s)
            log("node", job, "bring-up:", step, "| setup", nd["setup"], "relay", nd["relay"], "tree", nd["tree"], "foreign", nd["foreign"])
    if _mid:
        def _one(t):
            try:
                mid_run(*t)
            except Exception as e:   # never let knowledge stop the dispatcher
                log(t[1], "mid-run sharing error:", repr(e)[:300])
        with ThreadPoolExecutor(int(CFG.get("mid_run_threads", 8))) as ex:
            list(ex.map(_one, _mid))
    if BAR:
        barrier_tick(s, probes, live, free)
    elif SC["mode"] == "staged":
        staged_tick(s, probes, live, free)
    elif SC["mode"] == "pooled":
        pooled_tick(s, probes, live, free)
    # scale out: more waiting problems than nodes that are or will be free
    n_pending = sum(p["status"] == "pending" for p in s["problems"].values())
    if BAR:                 # parked problems need a node only once the barrier is open (a closed barrier holds them)
        bopen = ((s.get("sched") or {}).get("barrier") or {}).get("open")
        n_pending = sum(bool((p["status"] == "pending" and (bopen or not SCH.is_parked(p))) or (bopen and p["status"] == "parked"))
                        for p in s["problems"].values())
    n_free = sum(1 for j, nd in probes.items() if nd["ok"] and not nd["jobs"].get("__running__")) + len(pending_alloc)
    if (n_pending > n_free and len(js) < CFG["max_nodes"] and time.time() - s["last_alloc"] > CFG.get("alloc_gap_s", 180)):
        _JS[:] = js                    # alloc(s) reads the job list from here (keeps alloc's one-argument signature)
        alloc(s)
    s["nodes"] = {str(j): dict(left=live[j]["left"], gpu=nd.get("gpu"), setup=nd.get("setup"), relay=nd.get("relay"),
                               busy=bool(nd["jobs"].get("__running__")), host=live[j]["host"],
                               **({"idle_since": (s.get("sched") or {}).get("idle_since", {}).get(str(j))}
                                  if SC["mode"] != "independent" or BAR else {}))
                  for j, nd in probes.items()}
    if SC["mode"] != "independent" or BAR:
        acct_idle(s, probes, live)
    for n in held:
        if s["problems"][n]["status"] == "pending_hold":
            s["problems"][n]["status"] = "pending"
    for p in s["problems"].values():
        if p["status"] == "running":
            p.pop("launching", None)
    save(s)


# ---------------------------------------------------------------- scheduling (farm_sched.py; analysis/scheduling.md)

def estimates(s):
    """Per-problem estimates (cached in farm.json; `farm.py sched est` recomputes)."""
    c = s.setdefault("sched", {}).setdefault("est", {})
    for n in s["problems"]:
        if n not in c:
            c[n] = SCH.estimate(n, ROOT / "problems" / f"{n}.py", {**SC, "history_dirs": [str(ROOT / d) for d in
                                                                                         SC["history_dirs"]]},
                                CFG["kb"].get("plan_top_k", 0))
    return c


def file_order(s):
    return {n: i for i, n in enumerate(s["problems"])}


def staged_tick(s, probes, live, free):
    now = time.time()
    sch = s.setdefault("sched", {})
    idle_since = sch.setdefault("idle_since", {})
    for j in list(idle_since):
        if int(j) not in free:
            idle_since.pop(j)
    for j in free:
        idle_since.setdefault(str(j), now)
    picks = SCH.staged_pick(s["problems"], estimates(s), sch, SC, now, free, {int(k): v for k, v in idle_since.items()},
                            file_order(s))
    for job, name, why in picks:
        p = s["problems"][name]
        if start(job, name, p):
            p.update(status="running", node=job, started=now, synced=now, sched_why=why)
            idle_since.pop(str(job), None)
            log("started", name, "on", job, "| sched:", why)
        else:
            log("start of", name, "on", job, "failed | sched:", why)


def pool_member(nd, left):
    """A pooled-mode GPU node: an L40S with at least min_left; it only runs timed candidates (no relay or tree
    needed unless it is also the driver)."""
    return nd.get("ok") and "L40S" in nd.get("gpu", "") and left >= CFG["min_left_min"] and nd["foreign"] == "0"


def pooled_tick(s, probes, live, free):
    """Pooled mode, first version: problem controllers (kb_graph_opt + its KernelBlaster runs: LLM calls, nvcc
    builds) run on ONE driver node; every candidate / profile / verification is timed on a pool GPU over ssh
    (KernelBlaster SSHEvaluator, --ssh-host per pool node; the per-GPU flock on the host serialises timed runs
    of all problems). Lease queues are static per start (farm_sched.lease_queues)."""
    pool = SC.get("pool") or {}
    sch = s.setdefault("sched", {})
    now = time.time()
    members = [j for j in free if (not pool.get("gpu_jobs") or j in pool["gpu_jobs"])]
    drv = pool.get("driver") or sch.get("driver")
    ready_driver = [j for j in members if probes[j].get("relay") == "ok" and probes[j].get("setup") == "done"
                    and probes[j].get("tree") == s["tree"]]
    if drv not in live or drv not in ready_driver and not any(p["status"] == "running" and p["node"] == drv
                                                             for p in s["problems"].values()):
        drv = FD.pick_driver(ready_driver, s.get("alloc_roles") or {})
    if drv is None:
        return
    if sch.get("driver") != drv:
        sch["driver"] = drv
        log("pooled: driver node", drv, live[drv]["host"])
    # pooled-driver health: expiry warning (save the ledger / plan failover) and relay settings vs config
    warned = sch.setdefault("expiry_warned", {})
    if FD.expiry_due(live[drv].get("left_raw", live[drv]["left"]), warned, drv, CFG):
        warned[str(drv)] = now
        msg = (f"driver {drv} ({live[drv]['host']}) expires in ~{live[drv]['left_raw']} min: save the ledger, pack the running problems "
               f"and plan failover (a new driver needs a relay + tree; set pool.driver)")
        log("WARNING", msg)
        sch.setdefault("decisions", []).append({"t": now, "event": "driver_expiry_warning", "node": drv, "left_min": live[drv].get("left_raw", live[drv]["left"])})
        s.setdefault("warnings", []).append({"t": now, "what": "driver_expiry", "job": drv, "msg": msg})
        del s["warnings"][:-50]
    busy = any(p["status"] == "running" and p["node"] == drv for p in s["problems"].values())
    if relay_guard(s, drv, probes[drv], busy):
        return                                       # restarting: start problems once the relay is back (next tick)
    hosts = sorted(live[j]["host"] for j in members)
    sch["pool_hosts"] = hosts
    running = [n for n, p in s["problems"].items() if p["status"] == "running"]
    cap = int(pool.get("max_problems") or (len(hosts) + 1))
    order = pool.get("order", "file")
    ests = estimates(s)
    lim, n_started = int(pool.get("max_starts_per_tick") or 0), 0    # 0 = unlimited; a long start burst delays the next pass's collection / sharing
    while len(running) < cap:
        if lim and n_started >= lim:
            break
        pending = [n for n, p in s["problems"].items() if p["status"] == "pending"]
        if not pending and not BAR:
            break
        if BAR:                 # barrier scheduling: phase 1 cheap first (park flag), after the barrier the longest first
            pick = SCH.barrier_next(s["problems"], ests, sch.setdefault("barrier", {}), SC, file_order(s))
            if pick is None:
                break
            name, bwhy, bpark = pick
        elif order == "staged":
            name = SCH.rank(ests, pending, SCH.merged_names(s["problems"]), SC, file_order(s), list(ests))[0][0]
        else:
            name = pending[0]
        q = SCH.lease_queues(running + [name], hosts, ests, pool.get("policy", "share"),
                             current={n: s["problems"][n].get("pool_hosts") for n in running})[name]
        p = s["problems"][name]
        if not start(drv, name, p, pool_hosts=q, park=bpark if BAR else False):
            log("pooled: start of", name, "failed")
            break
        was_parked = BAR and SCH.is_parked(p)
        p.update(status="running", node=drv, started=now, synced=now,
                 sched_why=f"pooled on driver {drv}, lease queue {q} ({pool.get('policy', 'share')})")
        if BAR:
            barrier_mark(sch, name, p, bwhy, bpark, was_parked, now)
        log("started", name, "on driver", drv, "| pool hosts", q)
        running.append(name)
        n_started += 1
        sch.setdefault("decisions", []).append({"t": now, "event": "start", "name": name, "node": drv, "hosts": q})


def acct_idle(s, probes, live):
    """Idle accounting per node (seconds a ready node had nothing to run) for the report."""
    now = time.time()
    a = s.setdefault("sched", {}).setdefault("idle_acct", {})
    for j, nd in probes.items():
        r = a.setdefault(str(j), {"idle_s": 0.0, "busy_s": 0.0, "last": now, "first": now})
        dt = max(0.0, now - r["last"])
        if nd.get("ok") and nd["jobs"].get("__running__"):
            r["busy_s"] += dt
        elif nd.get("ok") and nd.get("relay") == "ok":
            r["idle_s"] += dt
        r["last"] = now


# ---------------------------------------------------------------- barrier scheduling (X-151; analysis/barrier_scheduling.md)
# Phase 1: every problem runs to the end of its characterization pass and PARKS (worker5.sh exit 78 -> status parked:char); the
# controller packs it, merges its knowledge (why="round0c") and frees its slot. A global soft barrier opens when (nearly) all have
# parked (farm_sched.barrier_update); phase 2 resumes the parked problems, longest first, each importing a package built just
# before its start. State: problem phase p1|parked|p2, parked_at, park_node; farm.json sched.barrier {open, why, log}.

def pack_ok(f):
    """The resume pack is a readable, non-empty tar.gz (a dropped ssh stream leaves a truncated one)."""
    try:
        return Path(f).stat().st_size > 0 and subprocess.run(["tar", "tzf", str(f)], capture_output=True).returncode == 0
    except OSError:
        return False


def barrier_log(s_or_sch, event, **kw):
    sch = s_or_sch if "barrier" in s_or_sch or "problems" not in s_or_sch else s_or_sch.setdefault("sched", {})
    b = sch.setdefault("barrier", {})
    b.setdefault("log", []).append({"t": time.time(), "event": event, **kw})
    del b["log"][:-60]


def barrier_mark(sch, name, p, why, park, was_parked, now):
    """After a successful start under the barrier: phase, the wait of a resumed problem, the decision log."""
    p["phase"] = "p1" if park else "p2"
    p["sched_why"] = f"{p.get('sched_why', '')} | {why}".strip(" |")
    p.pop("park_tries", None)
    if was_parked:
        p["resumed_at"] = now
        p["barrier_wait_s"] = round(now - (p.get("parked_at") or now), 1)
    barrier_log(sch, "start", name=name, phase=p["phase"], resumed=bool(was_parked), why=why)
    log("started", name, "| barrier:", why, "| phase", p["phase"], "(resumed from park)" if was_parked else "")


def barrier_package(job, name, p):
    """Phase 2 start of a parked problem: a fresh refresh package (many runs merged since this problem's own park-time merge)
    into the node's inbox, imported by kb_graph_opt at resume start (kb_barrier, why=barrier). Never raises."""
    try:
        (p.get("gk") or {}).pop("refresh_for", None)
        res = gk_refresh_upload(job, name, p, force=True)
        p["barrier_pkg"] = {"t": time.time(), "status": (res or {}).get("status"), "rid": (res or {}).get("rid"),
                            "counts": (res or {}).get("counts"), "to": (res or {}).get("to")}
        log(name, "barrier: refresh package", p["barrier_pkg"]["status"], p["barrier_pkg"]["rid"],
            json.dumps(p["barrier_pkg"]["counts"]))
    except Exception as e:  # noqa: BLE001 -- knowledge never stops a start
        p["barrier_pkg"] = {"t": time.time(), "status": "error", "error": repr(e)[:300]}
        log(name, "barrier: refresh package failed:", repr(e)[:300])


def barrier_park(job, name, p, nd):
    """The worker exited 78 (status parked:char): pull a consistent pack (kept as the resume pack), merge the knowledge
    (why=round0c, incremental, ledger-idempotent), release the slot. A failed pull / corrupt pack is retried at the next tick (the
    node still holds the run); after 3 failures the problem goes back to phase 1 (it re-parks from the node dir / pack)."""
    now = time.time()
    p["park_tries"] = p.get("park_tries", 0) + 1
    if not (collect(job, name, False) and pack_ok(RES / name / "resume.tar.gz")):
        log(name, f"barrier: park pack failed ({p['park_tries']}/3)", "-> retry next tick" if p["park_tries"] < 3 else "-> phase 1 again")
        if p["park_tries"] >= 3:
            p.update(status="pending", node=None, phase="p1", park_tries=0)
        return
    merge = None
    if gk_enabled(p):
        kd = gk_unpack_sync(name)
        if GK.get("mode") != "read":
            res = gk_merge(name, p, kd, why="round0c")
            merge = (res or {}).get("status") or ((res or {}).get("error") or "none")[:100]
    p.update(status="parked", phase="parked", parked_at=now, park_node=job, node=None, synced=now, detail="parked:char")
    p.pop("park_tries", None)
    p.pop("launching", None)
    barrier_log(_S or load(), "park", name=name, node=job, merge=merge)
    log(name, "PARKED after the characterization pass on", job, "| merge", merge, "| pack",
        (RES / name / "resume.tar.gz").stat().st_size // 1024, "KB")
    try:
        meta_event(job, name, p, "park", status="parked")
    except Exception as e:
        log(name, "run_meta park failed:", repr(e)[:200])


def barrier_housekeeping(s):
    """A parked problem whose pack is gone cannot resume: rerun its phase 1 (it parks again; if the barrier is open by then it
    simply starts as a fresh run without the park flag)."""
    for n, p in s["problems"].items():
        f = RES / n / "resume.tar.gz"
        if SCH.is_parked(p) and not (f.exists() and f.stat().st_size > 0):
            log(n, "barrier: parked but its pack is missing -> phase 1 again")
            p.update(status="pending", phase="p1", node=None)
            p.pop("parked_at", None)
            p.pop("park_node", None)
            barrier_log(s, "pack-missing", name=n, why="resume pack missing: phase 1 rerun")


def barrier_tick(s, probes, live, free):
    """The barrier's scheduling pass: decide whether it opens (persisted, idempotent), then start problems on the free nodes
    (independent) or on the driver (pooled: pooled_tick asks farm_sched.barrier_next)."""
    now = time.time()
    sch = s.setdefault("sched", {})
    bst = sch.setdefault("barrier", {})
    barrier_housekeeping(s)
    why = SCH.barrier_update(s["problems"], bst, SC, now)
    if why:
        log("BARRIER OPEN:", why)
    if SC["mode"] == "pooled":
        return pooled_tick(s, probes, live, free)
    ests, order = estimates(s), file_order(s)
    for job in free:
        pick = SCH.barrier_next(s["problems"], ests, bst, SC, order)
        if pick is None:
            break
        name, bwhy, park = pick
        p = s["problems"][name]
        was_parked = SCH.is_parked(p)
        if start(job, name, p, park=park):
            p.update(status="running", node=job, started=now, synced=now, sched_why=bwhy)
            barrier_mark(sch, name, p, bwhy, park, was_parked, now)
        else:
            log("start of", name, "on", job, "failed | barrier:", bwhy)


def barrier_status(s):
    sch = s.get("sched") or {}
    b = sch.get("barrier") or {}
    c = SCH.census(s["problems"])
    bc = SCH.bconf(SC)
    ph = {}
    for n, p in s["problems"].items():
        ph.setdefault(f"{p['status']}/{p.get('phase', '-')}", []).append(n)
    n, a = len(c["considered"]), len(c["arrived"])
    need = SCH.barrier_plan(s["problems"], {}, b, SC, {})["need"]
    if b.get("open"):
        print(f" BARRIER OPEN since {time.strftime('%H:%M:%S', time.localtime(b['opened']))}: {b.get('why')}")
    else:
        la = b.get("last_arrival")
        print(f" BARRIER CLOSED: {a}/{n} arrived (need {need} = {float(bc['min_frac']):.0%}), last arrival "
              + (f"{(time.time() - la) / 60:.0f} min ago (stall limit {bc['max_wait_min']} min)" if la else "none yet"))
    print("  phases:", {k: len(v) for k, v in sorted(ph.items())}, "| parked:", sorted(c["parked"]))
    for d in (b.get("log") or [])[-8:]:
        print("  ", time.strftime("%H:%M:%S", time.localtime(d["t"])), d.get("event"), d.get("name", ""), d.get("why", "") or
              d.get("merge", "") or "")


_GK_LOCK = threading.Lock()


def mid_run(job, name, p, nd):
    """merge_on round: when the run's state.json shows a newly completed round, pull it and merge it
    (incremental, ledger-idempotent); refresh round: upload a package of what OTHER runs merged since this run's
    last sync, imported by kb_graph_opt at its next round start. Plus the periodic sync as before."""
    now = time.time()
    g = p.get("gk") or {}
    rounds = nd.get("rounds", {}).get(name)
    if HALFWAY and (g.get("halfway_on", True)) and gk_enabled(p):
        halfway_sync(job, name, p, nd)                       # a run parked at its halfway point: merge + package now
    if CHARSYNC and gk_enabled(p):
        halfway_sync(job, name, p, nd, tag="char")           # a run waiting after its characterization pass (X-151)
    sync_due = now - p.get("synced", p["started"]) > CFG["sync_min"] * 60
    round_due = (GK.get("merge_on") == "round" and rounds is not None and rounds > g.get("rounds_merged", 0))
    if sync_due or round_due:
        collect(job, name, False)
        p["synced"] = now
        if gk_enabled(p) and GK.get("mode") != "read" and (round_due or GK.get("merge_on") == "sync"):
            with _GK_LOCK:                                   # one merge at a time (mid_run runs in a thread pool)
                res = gk_merge(name, p, gk_unpack_sync(name), why=f"round{rounds}" if round_due else "sync")
            if round_due and res and "error" not in res:
                p.setdefault("gk", {})["rounds_merged"] = rounds
        elif round_due:
            p.setdefault("gk", {})["rounds_merged"] = rounds
    if GK.get("refresh", "off") == "round" and gk_enabled(p):
        gk_refresh_upload(job, name, p)


def gk_refresh_upload(job, name, p, force=False):
    g = p.get("gk") or {}
    if not g.get("applied") or not g.get("run_uid"):
        return None
    curf = gk_root() / "current"
    cur = curf.read_text().strip() if curf.exists() else None
    if not cur or (cur == g.get("refresh_for") and not force):    # force: the halfway sync builds one whatever was sent
        return None
    out = RES / name
    db = out / "gk" / "sync" / "work" / "knowledge" / "knowledge.db"
    if not db.exists():
        db = out / "gk" / "seed" / "knowledge.db"
    pk = out / "gk" / "refresh" / f"{time.strftime('%H%M%S')}-{cur}"
    shutil.rmtree(pk, ignore_errors=True)
    res = gk_cli(["refresh", "--run-db", str(db), "--out", str(pk), "--run-uid", g["run_uid"], "--problem", name,
                  "--part", g.get("part") or gk_part(), "--policy", gk_policy()])
    st = res.get("status") if "error" not in res else "error"
    if st == "ready":
        tb = pk.with_suffix(".tar.gz")
        subprocess.run(["tar", "czf", str(tb), "-C", str(pk), "."], check=True,
                       env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
        d, rid = f"{F}/{name}/gk_refresh", res["rid"]
        put(job, tb, f"{d}/.{rid}.part")
        mv = f"mv {d}/.{rid}.part {d}/{rid}.tar.gz"
        ssh(f"srun --jobid={job} --overlap bash -c {shlex.quote(mv)}")
        g["refresh_uploads"] = (g.get("refresh_uploads", []) + [{"t": time.time(), "rid": res["rid"], "to": res.get("to"),
                                                                 "from": res.get("from"), "counts": res.get("counts")}])[-30:]
        log(name, "global knowledge refresh uploaded", res["rid"], (res.get("from") or {}).get("snapshot"), "->", cur,
            json.dumps(res.get("counts")))
    elif st != "up_to_date":
        g["refresh_error"] = (res.get("reason") or res.get("error") or "")[:300]
        log(name, "global knowledge refresh", st, g["refresh_error"])
    g["refresh_for"] = cur
    p["gk"] = g
    return res


def halfway_sync(job, name, p, nd, tag="half"):
    """Halfway sync (analysis/halfway_sync.md). The node parked the round's plan run at half its steps and waits for us
    (probe2.sh `halfway <name> <round>`): pull the run (pack2.sh: online backup of knowledge.db), merge it
    (``why="round<n>h"``: incremental against the copy merged last, so the round-end merge counts nothing twice), build and
    upload the refresh package of what OTHER runs merged, then write the ack the node waits for. Never raises; a failed
    pull is retried at the next poll (the node waits ``halfway_wait_s``)."""
    # tag "half" = the base halfway sync; "char" = the characterization sync (X-151: after A4's characterization pass, wait mode):
    # the same pull / merge / package / ack, keyed per (round, tag): own marker list, merge label (round<n>h / round<n>c), ack file
    ch = tag == "char"
    r = (nd.get("charsync" if ch else "halfway") or {}).get(name)
    g = p.setdefault("gk", {})
    dk = "charsync_done" if ch else "halfway_done"
    if r is None or r in g.get(dk, []):
        return None
    t0 = time.time()
    rec = {"round": r, "t": t0}
    try:
        if not collect(job, name, False):
            g["halfway_error"] = "pull failed"
            log(name, "halfway: pull failed (retry)")
            return None
        rec["pull_s"] = round(time.time() - t0, 1)
        p["synced"] = time.time()
        kd = gk_unpack_sync(name)
        if GK.get("mode") != "read":
            res = gk_merge(name, p, kd, why=f"round{r}c" if ch else f"round{r + 1}h")
            rec["merge"] = (res or {}).get("status") or ((res or {}).get("error") or "none")[:120]
        up = gk_refresh_upload(job, name, p, force=True)
        rec["package"] = (up or {}).get("rid") if (up or {}).get("status") == "ready" else None
        rec["status"] = (up or {}).get("status") or ("error" if up and "error" in up else "none")
        ack = json.dumps({"round": r, "package": rec["package"], "status": rec["status"], "merge": rec.get("merge"),
                          "counts": (up or {}).get("counts"), "ts": time.time()}).encode()
        d = f"{F}/{name}/gk_refresh"
        an = f"{'char' if ch else 'halfway'}-r{r:02d}"
        put(job, ack, f"{d}/.{an}.part")
        ssh(f"srun --jobid={job} --overlap bash -c " + shlex.quote(f"mv {d}/.{an}.part {d}/{an}.ack.json"))
        rec["total_s"] = round(time.time() - t0, 1)
        g[dk] = (g.get(dk, []) + [r])[-20:]
        g.pop("halfway_error", None)
        g["halfway_log"] = (g.get("halfway_log", []) + [rec])[-30:]
        log(name, f"{'char' if ch else 'halfway'} sync round {r}: pulled {rec['pull_s']}s, merge {rec.get('merge')}, package {rec['package']}, "
                  f"acked after {rec['total_s']}s")
    except Exception as e:  # noqa: BLE001 -- knowledge never stops the dispatcher
        g["halfway_error"] = repr(e)[:300]
        log(name, "halfway sync error:", g["halfway_error"])
    return rec


def halfway_poll():
    """Between ticks (halfway on): probe the nodes that run a problem and serve any run parked at its halfway point, so a
    node does not wait for the next 90 s tick (config global_knowledge.halfway_poll_s, default 30)."""
    global _S
    s = load()
    _S = s
    nodes = sorted({p["node"] for p in s["problems"].values() if p["status"] == "running" and p.get("node")})
    if not nodes:
        return
    with ThreadPoolExecutor(8) as ex:
        probes = dict(zip(nodes, ex.map(probe, nodes)))
    changed = False
    for job, nd in probes.items():
        for name in (nd.get("halfway") or {}) if nd.get("ok") else ():
            p = s["problems"].get(name)
            if p and p["status"] == "running" and p.get("node") == job and gk_enabled(p) \
                    and (p.get("gk") or {}).get("halfway_on", True):
                changed = bool(halfway_sync(job, name, p, nd)) or changed
        for name in (nd.get("charsync") or {}) if nd.get("ok") and CHARSYNC else ():
            p = s["problems"].get(name)
            if p and p["status"] == "running" and p.get("node") == job and gk_enabled(p):
                changed = bool(halfway_sync(job, name, p, nd, tag="char")) or changed
    if changed:
        save(s)


def wait_next_tick():
    """The dispatcher's sleep between ticks; with the halfway sync it is cut into polls that serve parked runs."""
    if not (HALFWAY or CHARSYNC):
        time.sleep(CFG["tick_s"])
        return
    end = time.time() + CFG["tick_s"]
    poll = max(5, int(GK.get("halfway_poll_s", 30)))
    while time.time() < end:
        time.sleep(max(0.0, min(poll, end - time.time())))
        if time.time() < end:
            try:
                halfway_poll()
            except Exception as e:  # noqa: BLE001
                log("halfway poll error:", repr(e)[:300])


def gk_unpack_sync(name):
    """The knowledge dir of the last sync pack (merge_on: sync)."""
    out = RES / name
    d = out / "gk" / "sync"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    subprocess.run(["tar", "xzf", str(out / "resume.tar.gz"), "-C", str(d), "work/knowledge"], check=False,
                   capture_output=True)
    return d / "work" / "knowledge"


def gk_brief():
    if not GK:
        return "global knowledge: not configured"
    cur = (gk_root() / "current").read_text().strip() if (gk_root() / "current").exists() else None
    m = {}
    if cur and (gk_root() / "snapshots" / f"{cur}.json").exists():
        m = json.loads((gk_root() / "snapshots" / f"{cur}.json").read_text())
    st = "on" if gk_enabled() else "OFF"
    return (f"global knowledge: {st} ({GK.get('mode')}, merge_on {GK.get('merge_on', 'final')}) {gk_root()} "
            f"current={cur} counts={json.dumps(m.get('counts', {}))}")


def cmd_gk(a):
    s = load()
    sub = a.rest[0] if a.rest else "status"
    arg = a.rest[1] if len(a.rest) > 1 else None
    if sub == "status":
        res = gk_cli(["status"])
        if "error" in res:
            sys.exit(res["error"])
        print(gk_brief())
        print("versions:", " ".join(res.get("versions", [])) or "(none)")
        print("counts:", json.dumps(res.get("counts", {})))
        for part, v in sorted((res.get("parts") or {}).items()):
            print(" ", part, json.dumps(v))
        print("ledger (runs, merges):", res.get("ledger"), "| bytes", res.get("bytes"))
        for n, p in s["problems"].items():
            g = p.get("gk") or {}
            if g.get("merge_error") or p.get("gk_off"):
                print(" ", n, "gk_off" if p.get("gk_off") else "", g.get("merge_error", ""))
    elif sub == "show":
        print(json.dumps(gk_cli(["show", arg or "claims", "--part", gk_part(), "--top", "25"]), indent=1))
    elif sub in ("off", "on"):
        s["problems"][arg]["gk_off"] = sub == "off"
        save(s)
        print(arg, "global knowledge", sub)
    elif sub == "merge":                      # (re)merge a collected problem by hand
        p = s["problems"][arg]
        print(json.dumps(gk_merge(arg, p, why="manual"), indent=1, default=str))
        save(s)
    elif sub == "pull":                       # fetch the knowledge of a finished problem from its (live) node
        p = s["problems"][arg]
        if not p.get("node"):
            sys.exit(f"{arg}: no node recorded")
        tmp = RES / arg / "gk" / "pull"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        rc, data, _ = srun(p["node"], PACK, [arg], timeout=1800)
        (tmp / "pull.tar.gz").write_bytes(data)
        subprocess.run(["tar", "xzf", str(tmp / "pull.tar.gz"), "-C", str(tmp)], check=False)
        print(json.dumps(gk_merge(arg, p, tmp / "work" / "knowledge", why="pull"), indent=1, default=str))
        save(s)
    elif sub == "rollback":
        print(json.dumps(gk_cli(["rollback", arg, "--reason", "farm.py gk rollback"]), indent=1))
    elif sub == "rebuild":
        # roll back to VID (default v0001: the bootstrap) and merge every collected problem again, in the order they
        # were merged (minus --exclude): the ledger skips what VID already holds, seeds keep their deltas exact
        base = arg if arg and not arg.startswith("--") else "v0001"
        excl = set()
        if "--exclude" in a.rest:
            excl = set(a.rest[a.rest.index("--exclude") + 1].split(","))
        print(json.dumps(gk_cli(["rollback", base, "--reason", f"rebuild excluding {sorted(excl)}"])))
        order = sorted((p.get("gk", {}).get("merged_at", 0), n) for n, p in s["problems"].items()
                       if (p.get("gk") or {}).get("merged_at") and n not in excl)
        for _, n in order:
            p = s["problems"][n]
            shutil.rmtree(RES / n / "gk" / "merged", ignore_errors=True)
            r = gk_merge(n, p, why="rebuild") or {}
            print(n, r.get("status"), r.get("version"), r.get("error", ""))
        save(s)
    elif sub == "prune":
        print(json.dumps(gk_cli(["prune", "--keep", str(arg or GK.get("snapshot_keep", 30))])))
    elif sub == "bootstrap":
        subprocess.run([GK.get("python") or sys.executable, str(ROOT / "gk_bootstrap.py")] + a.rest[1:], check=False)
    else:
        sys.exit(f"gk {sub}: unknown")


def cmd_status(a):
    s = load()
    c = {}
    for n, p in s["problems"].items():
        c[p["status"]] = c.get(p["status"], 0) + 1
    print("problems:", c, "| tree", s.get("tree"))
    print(mode_line())
    print("dispatcher:", "running (lock held)" if dispatcher_alive() else "no lock holder (stopped, or an older farm.py without the lock)")
    for j, v in (s.get("slow_nodes") or {}).items():
        print(f" SLOW NODE {j}: max memory clock {v.get('mem_clock_max_mhz')} MHz (expected {EXPECT_MEM_MHZ}), vbios {v.get('vbios')}, problem {v.get('problem')}")
    for e in (s.get("dup_events") or [])[-5:]:
        print(" dup event:", time.strftime("%H:%M:%S", time.localtime(e["t"])), e)
    print(gk_brief())
    if BAR:
        barrier_status(s)
    if SC["mode"] != "independent":
        sched_status(s)
    for j, nd in s["nodes"].items():
        print(" node", j, nd)
    for n, p in s["problems"].items():
        if p["status"] in ("running", "failed", "unsupported", "parked"):
            print(" ", n, p["status"], p.get("node"), p.get("detail", ""), f"attempts={p['attempts']}")


def sched_status(s, n_dec=12):
    sch = s.get("sched") or {}
    for w in sch.get("waves", []):
        print(f" wave {w['k']}: members {w['members']} size {w['size']} opened "
              f"{time.strftime('%H:%M', time.localtime(w['opened']))} | {w.get('why', '')}")
    if sch.get("waiting"):
        print(" ", sch["waiting"])
    if sch.get("driver"):
        print(" pooled driver", sch["driver"], "pool hosts", sch.get("pool_hosts"))
    pend = [n for n, p in s["problems"].items() if p["status"] == "pending"]
    if pend and SC["mode"] == "staged":
        ests = estimates(s)
        r = SCH.rank(ests, pend, SCH.merged_names(s["problems"], SC.get("gate_on", "any")), SC, file_order(s), list(ests))
        print(" next:", ", ".join(f"{n} (sim {i['sim']}, cost {i['cost']}{', expensive' if i['expensive'] else ''})"
                                  for n, i in r[:6]))
    for d in sch.get("decisions", [])[-n_dec:]:
        print("  ", time.strftime("%H:%M:%S", time.localtime(d["t"])), d.get("event"), d.get("name", ""),
              d.get("node", ""), d.get("why", "") or d.get("hosts", ""))
    for j, r in sorted((sch.get("idle_acct") or {}).items()):
        print(f"  node {j}: idle {r['idle_s'] / 60:.0f} min, busy {r['busy_s'] / 60:.0f} min")


def cmd_sched(a):
    """farm.py sched est | plan: the estimator's view and the order staged mode would start problems in now."""
    s = load()
    sub = a.rest[0] if a.rest else "plan"
    if sub == "est":
        s.setdefault("sched", {}).pop("est", None)
        ests = estimates(s)                  # not saved: the running dispatcher owns farm.json
        for n, e in ests.items():
            print(f"{n:40s} pairs {e['pairs']:4d} cost {e['cost']:7.1f} ({e['cost_src']}) kinds {','.join(e['kinds'])}")
        return
    ests = estimates(s)
    names = list(ests)
    order = file_order(s)
    print(mode_line())
    if BAR:
        bp = SCH.barrier_plan(s["problems"], ests, (s.get("sched") or {}).get("barrier") or {}, SC, order)
        print(f"barrier: {'OPEN' if bp['open'] else 'closed'} ({bp['arrived']}/{bp['considered']} arrived, need {bp['need']})")
        print("phase 1 order (admitted with the park flag):", ", ".join(f"{n}({ests[n]['cost']:.0f})" for n in bp["phase1"]))
        print("phase 2 order (after the barrier,", SCH.bconf(SC)["phase2_order"], "):",
              ", ".join(f"{n}({ests[n]['cost']:.0f})" for n in bp["phase2"]))
        return
    print("seed wave:", SCH.seed_wave(ests, names, SC, order))
    print("expensive (last):", sorted(SCH.expensive_set(ests, names, float(SC["expensive_frac"]))))
    merged = SCH.merged_names(s["problems"], SC.get("gate_on", "any"))
    pend = [n for n, p in s["problems"].items() if p["status"] == "pending"] or names
    for i, (n, info) in enumerate(SCH.rank(ests, pend, merged, SC, order, names)):
        print(f" {i + 1:3d} {n:40s} sim {info['sim']:.2f} cov {info['cov']:.2f} cost {info['cost']:7.1f}"
              f"{'  expensive' if info['expensive'] else ''}")


def cmd_report(a):
    s = load()
    rows = ["| problem | status | seed (us) | final (us) | speedup | torch (us) | vs torch |", "|---|---|---|---|---|---|---|"]
    for n, p in s["problems"].items():
        if p["status"] == "skipped":
            continue
        f0 = f1 = tt = None
        sj = RES / n / "work" / "summary.json"
        if sj.exists():
            irs = json.loads(sj.read_text()).get("irs", {})
            for v in irs.values():
                f0 = v.get("seed_e2e_us") or v.get("first_us")
                f1 = v.get("model_e2e_us") or v.get("final_us")
                tt = v.get("torch_us")
        sp = f"{f0 / f1:.1f}x" if f0 and f1 else ""
        vt = f"{tt / f1:.2f}x" if tt and f1 else ""
        rows.append(f"| {n} | {p['status']} | {f0 or ''} | {f1 or ''} | {sp} | {tt or ''} | {vt} |")
    (RES / "report.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def cmd_alloc_test(a):
    """The one real check: `sbatch --test-only` of the batch holder with the configured account/partitions (default 24 h).
    Nothing is submitted. `--print` only prints the command and the allocation plans (offline)."""
    args = [x for x in a.rest if x != "--print"]
    cmd = FD.sbatch_test_cmd(CFG, args[0] if args else "24:00:00")
    for role in ("driver", "pool"):
        print("plan", role, FD.alloc_plan(CFG, role))
    print("command:", cmd)
    if "--print" in a.rest:
        return
    rc, out, err = ssh(cmd, b"", 120)
    print("rc", rc, "\n", (out + err).decode().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "tick", "run", "status", "report", "requeue", "skip", "stage", "gk", "sched", "rerun", "dupcheck",
                                    "barrier", "alloc-test"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    a.name = a.rest[0] if a.rest else None
    if a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "stage":
        stage(force=True)
    elif a.cmd == "tick":
        lk = take_lock()
        tick()
    elif a.cmd == "run":
        lk = take_lock()                       # one dispatcher per state file (kept open for the process lifetime)
        log("farm.py run |", mode_line(), "| pid", os.getpid())
        rc_ = FD.relay_conf(CFG)
        log("relay defaults:", {k: rc_[k] for k in ("conc", "queue", "budget", "enforce")}, "| alloc:", FD.alloc_plan(CFG, "driver"),
            FD.alloc_plan(CFG, "pool"), "| left_safety_min", CFG.get("left_safety_min", 15))
        if FD.legacy_relay_differs(CFG):
            log("NOTE: legacy", ", ".join(FD.legacy_relay_differs(CFG)), "are ignored; set the \"relay\" block instead")
        while True:
            try:
                tick()
            except Exception as e:
                log("tick error:", repr(e)[:300])
            wait_next_tick()
    elif a.cmd == "status":
        cmd_status(a)
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "gk":
        cmd_gk(a)
    elif a.cmd == "sched":
        cmd_sched(a)
    elif a.cmd == "dupcheck":
        cmd_dupcheck(a)
    elif a.cmd in ("requeue", "skip", "rerun"):
        queue_cmd(a.cmd, a.name, a.rest[1:])
    elif a.cmd == "alloc-test":                   # farm.py alloc-test [TIME|--print]: sbatch --test-only, never submits
        cmd_alloc_test(a)
    elif a.cmd == "barrier":                      # farm.py barrier [status|open]
        if a.rest[:1] == ["open"]:
            queue_cmd("barrier-open", "-")
        else:
            barrier_status(load())


if __name__ == "__main__":
    main()
