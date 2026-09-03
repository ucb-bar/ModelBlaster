#!/usr/bin/env python3
"""Join ViNT uartlog per-dispatch cycles with the IR + kernel_picks."""
import csv, json, re, sys, collections, os

GEN = "/scratch/dima/rose-infra/RoSE/soc/sw/xpu-rt/zephyr-chipyard-sw/modelblaster/examples/vint/int8/generated"
uart = sys.argv[1]
picks_path = sys.argv[2]
csv_out = sys.argv[3] if len(sys.argv) > 3 else None

ir = json.load(open(f"{GEN}/graph.json"))
picks = json.load(open(picks_path))["picks"]
irops = {o["dispatch_id"]: o for o in ir["ops"]}

def region(d):
    if d <= 295:  return "A goal-enc EffNet (fp16)"
    if d <= 538:  return "B obs-enc EffNet (int8)"
    if d == 539:  return "cast boundary"
    if d <= 590:  return "C transformer (int8)"
    if d == 591:  return "cast boundary"
    if d <= 600:  return "D head MLP (fp16)"
    if d == 601:  return "cast boundary"
    return "E tail (int8)"

def work(op, s):
    g = lambda k, d=1: int(s.get(k, d))
    if op.startswith("conv2d"):
        OH=(g("IH")+2*g("PH")-g("KH"))//g("SH")+1; OW=(g("IW")+2*g("PW")-g("KW"))//g("SW")+1
        return g("N")*g("OC")*OH*OW*g("IC")*g("KH")*g("KW"), g("N")*g("OC")*OH*OW
    if op.startswith("depthwise_conv2d"):
        OH=(g("IH")+2*g("PH")-g("KH"))//g("SH")+1; OW=(g("IW")+2*g("PW")-g("KW"))//g("SW")+1
        return g("N")*g("OC")*OH*OW*g("KH")*g("KW"), g("N")*g("OC")*OH*OW
    if op.startswith("linear") or op.startswith("matmul"): return g("M")*g("K")*g("N"), g("M")*g("N")
    if op.startswith("adaptive_avg_pool2d"): return 0, g("N")*g("C")*g("IH")*g("IW")
    if op.startswith("batchnorm2d"): return 0, g("N")*g("C")*g("H")*g("W")
    if op.startswith("mul_c1"): return 0, g("N")*g("C")*g("HW")
    if op.startswith("pad"): return 0, g("N")*g("C")*g("IH")*g("IW")
    if op.startswith("cat"): return 0, g("N")*g("H")*g("W")*g("C_total")
    if op.startswith("softmax") or op.startswith("layer_norm"): return 0, g("M")*g("K")
    if "n" in s: return 0, g("n")
    return 0, 0

rows=[]
for line in open(uart, errors="ignore"):
    m = re.match(r"^(\d+),([^,]*),([a-z0-9_]+),([^,]*),(\d+)\s*$", line.strip())
    if m: rows.append(dict(did=int(m.group(1)), name=m.group(2), op=m.group(3),
                           shape=m.group(4), cycles=int(m.group(5))))
if not rows:
    print("NO PROFILE ROWS"); sys.exit(1)
total=sum(r["cycles"] for r in rows)
print(f"dispatches={len(rows)}  IR ops={len(irops)}  TOTAL CYCLES={total:,}  "
      f"({total/60e6:.1f}s @60MHz, {total/1e9:.2f}s @1GHz)\n")

agg=collections.defaultdict(lambda: dict(n=0,cyc=0,macs=0,elems=0))
for r in rows:
    o=irops.get(r["did"],{}); m,e=work(r["op"], o.get("shape") or {})
    a=agg[r["op"]]; a["n"]+=1; a["cyc"]+=r["cycles"]; a["macs"]+=m; a["elems"]+=e
print(f"{'op':26s} {'src':10s} {'algorithm':16s} {'n':>4s} {'cycles':>15s} {'%':>7s} {'MMAC':>9s} {'Melem':>8s} {'cyc/MAC':>8s} {'cyc/elem':>9s}")
print("-"*135)
ref=0
for op,a in sorted(agg.items(), key=lambda kv:-kv[1]["cyc"]):
    p=picks.get(op,{}); src=p.get("source","?")
    if src!="curated": ref+=a["cyc"]
    cpm=f"{a['cyc']/a['macs']:.2f}" if a["macs"] else "-"
    cpe=f"{a['cyc']/a['elems']:.1f}" if a["elems"] else "-"
    print(f"{op:26s} {src:10s} {str(p.get('algorithm') or '-')[:16]:16s} {a['n']:4d} {a['cyc']:15,d} "
          f"{100*a['cyc']/total:6.2f}% {a['macs']/1e6:9.2f} {a['elems']/1e6:8.3f} {cpm:>8s} {cpe:>9s}")
print("-"*135)
print(f"REFERENCE (scalar) share: {ref:,} = {100*ref/total:.2f}%   CURATED share: {total-ref:,} = {100*(total-ref)/total:.2f}%\n")

rg=collections.defaultdict(lambda: dict(n=0,cyc=0,ref=0,cur=0))
for r in rows:
    a=rg[region(r["did"])]; a["n"]+=1; a["cyc"]+=r["cycles"]
    if picks.get(r["op"],{}).get("source")=="curated": a["cur"]+=r["cycles"]
    else: a["ref"]+=r["cycles"]
print(f"{'region':28s} {'ops':>5s} {'cycles':>15s} {'%':>7s} {'curated':>15s} {'reference':>15s} {'ref%':>7s}")
for k in sorted(rg):
    a=rg[k]; print(f"{k:28s} {a['n']:5d} {a['cyc']:15,d} {100*a['cyc']/total:6.2f}% {a['cur']:15,d} {a['ref']:15,d} {100*a['ref']/a['cyc']:6.2f}%")
print()
ps=collections.defaultdict(lambda: dict(n=0,cyc=0))
for r in rows:
    op=r["op"]
    k="cast" if op.startswith("cast_") else ("fp16" if op.endswith("_f16") else ("int8" if op.endswith("_s8") or op.endswith("_s8_pc") else "app"))
    ps[k]["n"]+=1; ps[k]["cyc"]+=r["cycles"]
print(f"{'precision':10s} {'ops':>5s} {'cycles':>15s} {'%':>7s}")
for k,a in sorted(ps.items(), key=lambda kv:-kv[1]["cyc"]):
    print(f"{k:10s} {a['n']:5d} {a['cyc']:15,d} {100*a['cyc']/total:6.2f}%")
print("\nMISSING CURATED KERNELS ranked by cycles they cost today")
print(f"{'op':26s} {'n':>4s} {'cycles':>15s} {'% model':>9s} {'Melem':>8s} {'cyc/elem':>9s}")
for op,a in sorted(agg.items(), key=lambda kv:-kv[1]["cyc"]):
    if picks.get(op,{}).get("source")=="curated": continue
    cpe=f"{a['cyc']/a['elems']:.1f}" if a["elems"] else "-"
    print(f"{op:26s} {a['n']:4d} {a['cyc']:15,d} {100*a['cyc']/total:8.2f}% {a['elems']/1e6:8.3f} {cpe:>9s}")
if csv_out:
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["dispatch_id","name","op","shape","cycles"])
        for r in rows: w.writerow([r["did"],r["name"],r["op"],r["shape"],r["cycles"]])
    print(f"\nwrote {csv_out}")
