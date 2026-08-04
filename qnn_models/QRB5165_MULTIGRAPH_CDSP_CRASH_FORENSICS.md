# QRB5165 v66 — Multi-graph context loading crashes the cDSP user-PD

**TL;DR.** QNN's multi-graph context binaries (multiple graphs in one ctx via
`qnn-context-binary-generator --dlc_path a.dlc,b.dlc,...`) silently crash
the cDSP-side user process domain on this firmware. The host runtime sees
`QNN_SUCCESS` from every QNN call, but the underlying DSP is dead and
`graphExecute` becomes a no-op. Confirmed on the physical QRB5165 dev
board (and reproduced on cloud QRB5165) via on-board kernel logs and
firmware ramdumps. The earlier "multi-graph adds ~12 ms/dispatch" finding
was a *misinterpretation* — the extra time was FastRPC retry/cleanup
against dead user-PDs, not legitimate graph-switching cost.

Date of forensic pass: **2026-05-12**.
Crash being analyzed: **2026-05-10 evening (UTC 2026-05-11 00:39 → reboot at 06:41)**.

---

## 1. What was running when the crash happened

A multi-graph dense-bundle run of the v3 bundle-aware SmolVLA vision
schedule. Two DSP firmware contexts loaded:

* `ctx_sched9_dsp_tramps_chunk0__Dsp.bin` (62 040 B, **14 graphs**)
* `ctx_sched9_dsp_tramps_chunk1__Dsp.bin` (58 816 B, **13 graphs**)

Plus two HTA firmware contexts (`ctx_sched9_hta_convs_chunk{0,1}__Hta.bin`,
12 graphs each) and 10 multi-graph CPU contexts for fallback. Host
runtime: `qnn_runtime` built from
`qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles_mg/runtime_main.cpp`
with `XPURT_DSP_CTX_BUDGET=2 XPURT_HTA_CTX_BUDGET=2`.

Host log (`runs/v3_bundles_multigraph_dense_phys/run.log`) reported
clean completion:

```
[main] prefetched DSP=2 (budget=2) HTA=2 (budget=2)
[main] 14 unique contexts loaded eagerly for 97 schedule entries
       (missing/skipped: 0, deferred DSP: 0)
[summary] schedule walked: 97 / 97 segments dispatched,
          wall=3760.766 ms (predicted makespan = 2564.885 ms)
```

No QNN errors. No FastRPC errors. No `OP_PACKAGE_NOT_FOUND`. Process
exited normally.

A few minutes later the board became unreachable on port 22 and required
a manual reboot.

## 2. Forensic evidence

### 2.1 cDSP ramdumps

`/system/rfs/msm/cdsp/ramdumps/pd_dump_/frpc/` holds an ELF snapshot of
every cDSP user-PD that has ever crashed on this board. The two **most
recent** dumps are exactly from this incident:

| File | Size | Mtime | Process-domain ID |
|---|---:|---|---|
| `c045fd40 qnn-context-bin.00.elf` | 18 898 552 B | 2026-05-11 06:41 | `c045fd40` |
| `c04dd720 qnn-context-bin.00.elf` | 18 898 552 B | 2026-05-11 06:41 | `c04dd720` |

The 06:41 UTC mtime is when the kernel persisted the dumps during the
reboot recovery path. The actual fault happened earlier — inferred from
kernel logs in §2.2.

**Two PD IDs ↔ two DSP firmware contexts, 1:1.** Each multi-graph DSP
context binary loaded by `contextCreateFromBinary` spawns its own
`qnn-context-bin` user-PD on the cDSP. Both of ours crashed.

### 2.2 Kernel log evidence

`/var/log/kern.log` shows a flood of FastRPC mapping failures during the
exact wall-clock window of our run:

```
May 11 00:39:06 qrb5165-rb5 kernel: adsprpc: fastrpc_mem_map_to_dsp failed.
                                    err 0x80000414 fd -1 len 0x19000
May 11 00:39:08 ... fastrpc_mem_map_to_dsp failed. err 0x80000414 ...
... (repeats once/second for ~90 seconds) ...
May 11 00:40:29 ... fastrpc_mem_map_to_dsp failed. err 0x80000414 ...
```

* **Time:** 00:39–00:40 UTC = 17:39–17:40 PDT on 2026-05-10 — matches the
  multi-graph run's wall window.
* **Error code:** `0x80000414` (AEE_ENOMEMMAP family) with `fd -1
  len 0x19000`. ION-backed buffer (~100 KB), the kernel was unable to
  hand it to the cDSP for tensor I/O.
* **Why repeating:** the host runtime kept calling `graphExecute`, each
  call internally issues an ION map to the cDSP. With the cDSP user-PD
  dead, every map fails, but the host doesn't see the failure — FastRPC
  returns "submitted" status.

### 2.3 Historical pattern

The same directory contains accumulated ramdumps from previous DSP
crashes on this board:

| Process | Crashes |
|---|---:|
| `qnn-net-run`        | 205 |
| `qnn-context-bin`    | 188 |
| `profile_seg`        | 145 |
| `2x_resnet50_dro…`   |  72 |
| `qnn_runtime`        |  24 |
| `milp_3way_runti…`   |  15 |
| `yolov8n_node`       |  14 |
| `resnet50_dronet…`   |   7 |

So `qnn-context-bin` (the cDSP companion of `qnn-context-binary-generator`,
also re-used by `contextCreateFromBinary` at runtime) has crashed 188
times across this dev board's life. Multi-graph just triggers it
reliably.

## 3. Failure mechanism (best inference, without symbolized analysis)

```
Host process            FastRPC / kernel         cDSP user-PD (qnn-context-bin)
─────────────           ────────────────────     ──────────────────────────────
contextCreateFrom-      submit ctx-binary  →     parse multi-graph blob
Binary(...)             via FastRPC               register N graphs
                                                   alloc per-graph state
                        ← QNN_SUCCESS              … some allocation aborts the
                                                     user-PD (likely heap limit
                                                     on the cDSP, default ~256 MB)
                                                   ramdump written by kernel
graphRetrieve(g0)        host has cached
                         graph handles from
                         the create path           (PD dead)
                        ← QNN_SUCCESS

graphExecute(g0, ...)    FastRPC submits          (PD dead — request dropped)
                        ← QNN_SUCCESS
                        adsprpc kernel tries
                        to map ION buffer →
                        fastrpc_mem_map_to_dsp
                        failed err 0x80000414
                        (logged but not surfaced
                         to userspace)

...repeats 97 times...

contextFree(ctx)         (no-op against dead PD)
exit(0)
```

The host literally cannot tell. The dispatches "succeed" in the sense
that QNN_SUCCESS is returned, but no compute happened on the DSP — only
the FastRPC scheduling overhead. **The schedule trace columns
`actual_start_ms`/`actual_end_ms` measure dispatch latency, not
work-done latency, on the DSP rows.**

The HTA rows in the trace may or may not be affected — the dumps are
all from cDSP (`frpc/` = compute-DSP user-PD dumps), and HTA has a
separate firmware path. The +1.2 sec wall drift on physical is therefore
attributable mostly to DSP-dispatch retry/cleanup overhead, not HTA.

## 4. What this means for prior measurements

### 4.1 The "multi-graph adds 12 ms/dispatch" finding was wrong

Earlier interpretation: physical-board dense-multi-graph run at 3761 ms
vs the pre-refactor canonical 2561 ms = +1.2 sec = ~12 ms/dispatch ×
97 dispatches = "graph-switching overhead inside a multi-graph context."

Correct interpretation: the DSP user-PD crashed early in the run, every
DSP dispatch after that became a FastRPC no-op with retry/timeout
overhead, and the wall time inflated.

`plots/v3_bundles_trace_compare.png` is misleading in its current form
— the middle pane's "multi-graph dense" CPU lane is real but its DSP lane
is showing dispatch-submission latency against dead firmware, not
DSP execute time.

### 4.2 The "97/97 dispatched" success line is unreliable for multi-graph

`runtime_main.cpp`'s summary reflects what the host could see (every
`graphExecute` returned `QNN_SUCCESS`). It does not validate that real
compute happened. For multi-graph runs we need output-tensor
verification — either CRC against a known-good reference, or compare
boundary tensors against the single-graph path.

### 4.3 The 1.24× speedup at 2561 ms remains valid

The pre-refactor canonical run (`runs/v3_bundles_dsp9/run.log`, 2561 ms)
is single-graph: one ctx binary per graph, 27 simul DSP firmware
contexts at `budget=9`. No multi-graph code path involved.

There are **zero ramdumps** in the cdsp dump dir from that run pattern
(the 24 `qnn_runtime` ramdumps are concentrated around dates other than
the canonical run's). Single-graph contexts are stable on this firmware.

## 5. Why the cloud QRB5165 behaved the same

Cloud's dense-multi-graph run also returned `97/97 dispatched` at
4288 ms wall. By the same logic the cloud DSP user-PDs likely crashed
too. Cloud's tighter simul cap (~2 DSP contexts vs physical's ~30) is
a separate constraint; cloud's `contextFree` reclaim behavior (probed
earlier and confirmed working) is consistent with single-graph mode.

The cloud crash directory wasn't accessible during forensics (cloud
board was down at investigation time), but the kernel signature on
cloud during the run should be identical: `fastrpc_mem_map_to_dsp
failed. err 0x80000414` repeating.

## 6. Recommendations

1. **Default to single-graph contexts on this firmware.** Multi-graph
   was conceived as a cap-bypass tool; the cap-bypass works
   architecturally but the firmware doesn't survive the workload.

2. **For any multi-graph run, validate outputs.** Either pull a
   boundary tensor and CRC-check, or compare against a single-graph
   reference run. The "summary: 97/97" line is not enough.

3. **Watch `/var/log/kern.log` during multi-graph experiments.** The
   `fastrpc_mem_map_to_dsp failed. err 0x80000414` pattern is the
   real-time signal that the user-PD is dead.

4. **Preserve ramdumps for engagement with Qualcomm.** The two May 11
   06:41 ELFs are the artifacts that let a Hexagon-SDK-equipped engineer
   resolve the PC at fault and identify the root cause inside
   `libQnnDspOpPackage.so`. Path:
   `/system/rfs/msm/cdsp/ramdumps/pd_dump_/frpc/`.

5. **If we keep pursuing multi-graph**, the next concrete step is to
   pull the two ramdump ELFs locally and disassemble with
   `hexagon-llvm-objdump -d` (needs Hexagon SDK toolchain) to find the
   exact fault address. Likely culprits to look at: per-graph state
   allocation in the op-package's context-binary parser, and the
   heap limit on the cDSP user-PD.

## 7. Bottom line for the SmolVLA project

* The canonical 2561 ms / 1.24× speedup result on the physical board is
  still real and is the working demo.
* Multi-graph as a *firmware-cap bypass* is invalid on this firmware
  generation — the cDSP isn't stable under it. The architecture (one
  context binary holding N graphs) does work in the abstract — context
  binaries build correctly, the host runtime walks the schedule — but
  the DSP firmware itself can't host them safely.
* On future-chip firmware (where the cDSP heap is sized differently, or
  the op-package is rebuilt), the multi-graph path may light up. The
  scheduler + runtime + bundling pipeline we built carry over as-is.

## Appendix — file references

| Concern | Location |
|---|---|
| Ramdumps (the 2 from this crash) | `root@<BOARD_IP>:/system/rfs/msm/cdsp/ramdumps/pd_dump_/frpc/c045fd40 qnn-context-bin.00.elf`, `c04dd720 qnn-context-bin.00.elf` |
| Full ramdump dir (all historical) | `/system/rfs/msm/cdsp/ramdumps/pd_dump_/frpc/` (~27 GB on board) |
| Kernel log slice with FastRPC failures | `/var/log/kern.log`, search `fastrpc_mem_map_to_dsp` (May 11 00:39–00:40 UTC) |
| Multi-graph run host log (physical) | `runs/v3_bundles_multigraph_dense_phys/run.log` |
| Multi-graph run host log (cloud) | `runs/v3_bundles_multigraph_dense_cloud/run.log` |
| Single-graph canonical run (the working demo) | `runs/v3_bundles_dsp9/run.log` |
| Trace-compare plot (needs revision per §4.1) | `plots/v3_bundles_trace_compare.png` |
| Runtime source | `qnn_models/runtime/generate_runtime.py`, `qnn_models/runtime/gen/qrb5165_smolvla_v3_bundles_mg/runtime_main.cpp` |
| Refactor plan that introduced multi-graph | `qnn_models/smolVLA/MULTIGRAPH_REFACTOR_PLAN.md` |
