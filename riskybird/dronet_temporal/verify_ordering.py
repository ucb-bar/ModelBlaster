#!/usr/bin/env python3
"""Verify frame ordering / contiguity within each acquisition of the PULP-DroNet v3
HM01B0 dataset. Reports: are frames sequential by int(filename)? What is the
inter-frame spacing distribution? Are there partition changes within acquisitions?
"""
import os, csv, glob, numpy as np
ROOT="/scratch2/dima/misc_sw/XPU-RT/datasets/pulp_dronet_himax/raw/Dataset_PULP_Dronet_v3"
csvs=sorted(glob.glob(os.path.join(ROOT,"**","labels_partitioned.csv"),recursive=True))
print(f"[csvs] {len(csvs)} acquisitions")

n_acq=0
frames_total=0
all_diffs=[]                # inter-frame filename spacing (csv order)
sorted_matches_csv=0        # acquisitions whose csv order already == int-sorted order
csv_out_of_order=0
mixed_partition_acq=0       # acquisitions with >1 partition value
part_counts={}
per_acq_nframes=[]
example_spacings=None
dup_frames=0
none_only=0

for c in csvs:
    d=os.path.dirname(c); imgd=os.path.join(d,"images")
    rows=list(csv.DictReader(open(c)))
    if not rows: continue
    n_acq+=1
    names=[r["filename"] for r in rows]
    parts=[r["partition"] for r in rows]
    for p in parts: part_counts[p]=part_counts.get(p,0)+1
    # integer keys
    ints=[]
    ok=True
    for nm in names:
        base=nm.rsplit(".",1)[0]
        try: ints.append(int(base))
        except: ok=False; break
    if not ok:
        print(f"[WARN] non-integer filename in {c}: {names[:3]}")
        continue
    ints=np.array(ints)
    frames_total+=len(ints)
    per_acq_nframes.append(len(ints))
    # is csv order == sorted order?
    srt=np.sort(ints)
    if np.array_equal(ints,srt): sorted_matches_csv+=1
    else: csv_out_of_order+=1
    # spacing on sorted order
    if len(srt)>1:
        diffs=np.diff(srt)
        all_diffs.append(diffs)
        dup_frames+=int((diffs==0).sum())
        if example_spacings is None:
            example_spacings=(os.path.relpath(d,ROOT), srt[:8].tolist(), diffs[:8].tolist())
    # partition mixing within acquisition
    up=set(parts)
    if len(up)>1: mixed_partition_acq+=1
    if up=={"None"} or up==set(["None"]): none_only+=1

print(f"[acquisitions] total={n_acq} frames_total={frames_total}")
print(f"[order] csv_already_int_sorted={sorted_matches_csv}  csv_out_of_order={csv_out_of_order}")
print(f"[nframes/acq] min={min(per_acq_nframes)} median={int(np.median(per_acq_nframes))} max={max(per_acq_nframes)} mean={np.mean(per_acq_nframes):.1f}")
if all_diffs:
    D=np.concatenate(all_diffs)
    print(f"[spacing] n_gaps={len(D)} min={D.min()} p05={np.percentile(D,5):.0f} median={np.median(D):.0f} mean={D.mean():.1f} p95={np.percentile(D,95):.0f} max={D.max()}")
    # spacing is a timestamp-in-ms-ish filename; report the mode + how tight
    vals,cnts=np.unique(D,return_counts=True); top=np.argsort(-cnts)[:8]
    print(f"[spacing top values (val:count)] "+"  ".join(f"{vals[i]}:{cnts[i]}" for i in top))
    print(f"[spacing] frac in [30,80]={np.mean((D>=30)&(D<=80)):.3f}  frac==dup(0)={np.mean(D==0):.4f}  frac>500(gap)={np.mean(D>500):.4f}")
    print(f"[dups] duplicate-timestamp frames (spacing==0)={dup_frames}")
print(f"[partition mixing] acquisitions_with_multiple_partitions={mixed_partition_acq}")
print(f"[partition counts] {part_counts}")
if example_spacings:
    print(f"[example acq] {example_spacings[0]}\n  first8 int names={example_spacings[1]}\n  first8 spacings={example_spacings[2]}")
print("VERIFY_ORDER_DONE")
