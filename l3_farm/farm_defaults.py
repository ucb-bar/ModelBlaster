"""Pure helpers for the allocation / relay defaults of farm.py (no I/O, no network, no side effects on import).
Documented in analysis/defaults_fix/README.md; tested in analysis/defaults_fix/tests/.

Config (all optional; absent = the previous behaviour, except the relay defaults which are now 96 / 1024 / 10000):
  "relay":  {"concurrency": 96, "queue": 1024, "budget_usd": 10000, "enforce": true, "check_min": 10,
             "restart_busy": false, "max_tries": 3}
  "alloc_mode": "crun" | "batch", "alloc_time": "10:00:00"
  "alloc":  {"driver_mode": .., "driver_time": .., "pool_mode": .., "pool_time": .., "account": "cuopt_fallback",
             "partitions": [..], "warn_s": 240, "max_h": 24, "name": "mb-holder", "driver_successor_min": 120}
  "left_safety_min": 15, "driver_warn_min": 30
"""
import re

RELAY_DEFAULTS = {"concurrency": 96, "queue": 1024, "budget_usd": 10000}
L40S_PARTITIONS = ["l40s@cr+mp/romed8nl/1gpu-32cpu-128gb", "l40s@qs1/romed8nl/1gpu-32cpu-128gb"]   # as batch_nodes.py
MODES = ("crun", "batch")


# ------------------------------------------------------------------ relay
def relay_conf(cfg):
    """Wanted relay settings. Source: cfg["relay"], else the 96/1024/10000 defaults. The legacy kb.relay_concurrency /
    kb.relay_budget_usd (stale 12 / 150 in config.json) are deliberately NOT used any more."""
    r = cfg.get("relay") or {}
    return dict(conc=int(r.get("concurrency", RELAY_DEFAULTS["concurrency"])),
                queue=int(r.get("queue", RELAY_DEFAULTS["queue"])),
                budget=r.get("budget_usd", RELAY_DEFAULTS["budget_usd"]),
                enforce=bool(r.get("enforce", True)), check_min=float(r.get("check_min", 10)),
                restart_busy=bool(r.get("restart_busy", False)), max_tries=int(r.get("max_tries", 3)))


def legacy_relay_differs(cfg):
    """Names of kb.relay_* keys whose value differs from the effective relay settings (for a one-time log line)."""
    k, w = cfg.get("kb") or {}, relay_conf(cfg)
    out = []
    if "relay_concurrency" in k and int(k["relay_concurrency"]) != w["conc"]:
        out.append(f"kb.relay_concurrency={k['relay_concurrency']}")
    if "relay_budget_usd" in k and float(k["relay_budget_usd"]) != float(w["budget"]):
        out.append(f"kb.relay_budget_usd={k['relay_budget_usd']}")
    return out


def parse_relay_info(text):
    """remote/relay_info.sh output: 'conc=96 queue=1024 budget=10000 port=23456' (missing field = not on the cmdline)."""
    d = {}
    for m in re.finditer(r"\b(conc|queue|budget|port)=([0-9.]+)", text or ""):
        d[m.group(1)] = float(m.group(2)) if m.group(1) == "budget" else int(float(m.group(2)))
    return d


def relay_diffs(want, have):
    """What differs between the wanted settings and the running relay's command line. Empty have -> [] (unknown: no action).
    conc must be present on a live relay (the server's default is not the config's); queue/budget only compared if shown."""
    if not have or "conc" not in have:
        return []
    d = []
    if have["conc"] != want["conc"]:
        d.append(f"max-concurrency {have['conc']}->{want['conc']}")
    if "queue" in have and have["queue"] != want["queue"]:
        d.append(f"max-queue {have['queue']}->{want['queue']}")
    if "budget" in have and abs(have["budget"] - float(want["budget"])) > 1e-9:
        d.append(f"budget-usd {have['budget']:g}->{float(want['budget']):g}")
    return d


def check_due(st, now, check_min):
    """Rate limit of the relay probe per node: every check_min minutes; every 60 s while a mismatch / restart is pending."""
    t = st.get("t")
    if t is None:
        return True
    gap = 60 if (st.get("diff") or st.get("restarted_pending")) else check_min * 60
    return now - t >= gap


def relay_decision(want, have, st, busy):
    """-> ('ok'|'unknown'|'busy'|'giveup'|'noport'|'restart', diffs). Pure; the caller performs the restart."""
    diffs = relay_diffs(want, have)
    if not have or "conc" not in have:
        return "unknown", []
    if not diffs:
        return "ok", []
    if busy and not want["restart_busy"]:
        return "busy", diffs
    if st.get("tries", 0) >= want["max_tries"]:
        return "giveup", diffs
    if "port" not in have:
        return "noport", diffs
    return "restart", diffs


# ------------------------------------------------------------------ time handling
def time_minutes(t):
    """'10:00:00' / '1-00:00:00' / '24:00:00' / '90' (minutes) -> minutes; None if unparseable."""
    t = str(t).strip()
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", t)
    if m:
        d, h, mi, s = (int(x or 0) for x in m.groups())
        return d * 1440 + h * 60 + mi + (1 if s else 0)
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+)", t)          # D-H:M is not HH:MM; only accept H:M without a day part
    if m and m.group(1) is None:
        return int(m.group(2)) * 60 + int(m.group(3))
    m = re.fullmatch(r"(\d+)", t)
    return int(m.group(1)) if m else None


def fmt_time(minutes):
    """Slurm form: H:MM:SS below 24 h (unchanged for the old 10:00:00), D-HH:MM:SS from 24 h (what sbatch --test-only accepted)."""
    d, rem = divmod(int(minutes), 1440)
    h, mi = divmod(rem, 60)
    return f"{d}-{h:02d}:{mi:02d}:00" if d else f"{h:02d}:{mi:02d}:00"


def norm_time(t, max_h=24):
    """-> (slurm time string, note|None). Unparseable -> ValueError; above max_h (the batch QOS cap) -> clamped with a note."""
    m = time_minutes(t)
    if m is None or m <= 0:
        raise ValueError(f"bad alloc time {t!r}")
    cap = int(float(max_h) * 60)
    note = None
    if m > cap:
        note = f"alloc time {t} exceeds the {max_h} h QOS cap -> clamped"
        m = cap
    s = str(t).strip()
    return (fmt_time(m) if (m >= 1440 or note) else (s if re.fullmatch(r"\d+:\d\d:\d\d", s) else fmt_time(m))), note


# ------------------------------------------------------------------ allocation plan
def alloc_conf(cfg):
    a = cfg.get("alloc") or {}
    return dict(a, account=a.get("account", "cuopt_fallback"), warn_s=int(a.get("warn_s", 240)), max_h=a.get("max_h", 24),
               name=a.get("name", "mb-holder"), successor_min=float(a.get("driver_successor_min", 120)))


def alloc_plan(cfg, role="pool"):
    """-> dict(mode, time, note, role). role 'driver' uses alloc.driver_mode/driver_time, 'pool' alloc.pool_*; each falls back
    to the global alloc_mode (default 'crun') / alloc_time. Default config = crun at config alloc_time, as before."""
    a = alloc_conf(cfg)
    mode = a.get(f"{role}_mode") or cfg.get("alloc_mode") or "crun"
    if mode not in MODES:
        raise ValueError(f"alloc mode {mode!r} not in {MODES}")
    t = a.get(f"{role}_time") or cfg.get("alloc_time", "10:00:00")
    tstr, note = norm_time(t, a["max_h"])
    return dict(mode=mode, time=tstr, note=note, role=role)


def need_driver(sch_driver, roles, left_raw, successor_min):
    """Should the next allocation be a driver-role one? Only in pooled mode (caller decides). left_raw: {job: raw minutes left,
    None for PENDING}. False while a driver-role job exists that is pending or has > successor_min left, or the current
    (untagged) driver does."""
    for j, r in left_raw.items():
        if roles.get(str(j)) == "driver" and (r is None or r > successor_min):
            return False
    cur = left_raw.get(sch_driver) if sch_driver is not None else None
    return not (cur is not None and cur > successor_min)


def pick_driver(ready, roles, fallback=None):
    """Driver among ready nodes: prefer driver-role (batch holder) jobs, else the old rule min(ready)."""
    pref = [j for j in ready if roles.get(str(j)) == "driver"]
    pool = pref or list(ready)
    return min(pool) if pool else fallback


def sbatch_test_cmd(cfg, time_s=None):
    """Remote shell command of the one real check: sbatch --test-only with the batch settings (never submits)."""
    import shlex
    a = alloc_conf(cfg)
    t = norm_time(time_s or a.get("driver_time") or cfg.get("alloc_time", "24:00:00"), a["max_h"])[0]
    parts = a.get("partitions") or L40S_PARTITIONS
    args = ["sbatch", "--test-only", "-J", a["name"] + "-test", "-p", ",".join(parts), "--gpus=1", "-t", t,
            "-A", a["account"], "--no-requeue", "--wrap", "sleep 1"]
    return " ".join(shlex.quote(x) for x in args)


# ------------------------------------------------------------------ remaining time / expiry warning
def effective_left(raw_min, cfg):
    """Conservative remaining minutes: squeue's left minus left_safety_min (default 15; nodes end ~10 min before the nominal)."""
    return max(0, int(raw_min) - int(cfg.get("left_safety_min", 15)))


def expiry_due(raw_left, warned, job, cfg):
    """True once per job when the driver has <= driver_warn_min (default 30) minutes left."""
    return raw_left is not None and raw_left <= float(cfg.get("driver_warn_min", 30)) and str(job) not in warned
