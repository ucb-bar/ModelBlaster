# Observability gaps in the current benchmark

Tracking what we *can't* measure today and what unblocks each item.

## Cache miss / branch mispredict / memory bandwidth

**State:** Not captured.

**Cause:** FireSim's autocounter is a build-time feature. Probes are
inserted when `WithAutoCounter` is added to the `PLATFORM_CONFIG` during
bitstream synthesis. The currently-deployed bitstream
(`alveo_u250_firesim_shuttle_gemmini_opu`, built 2026-05-06) was
synthesized **without** that config — verified by grepping the build
artifacts under
`deploy/results-build/2026-05-06--22-08-53-alveo_u250_firesim_shuttle_gemmini_opu/`:
zero `AutoCounter` references in `design/`, `driver/`, or `vivado.log`.

Setting `autocounter: { read_rate: 1000 }` in `config_runtime.yaml` at
runtime would therefore be a no-op — there's nothing in the FPGA to
sample.

**To unblock:** add `WithAutoCounter` to the `PLATFORM_CONFIG` of the
hwconfig (or its parent build recipe), and rebuild. The build recipe
is in `deploy/config_build_recipes.yaml`. The rebuild itself is a
4-6 hour Vivado run on the chipyard build host. Once the rebuilt
bitstream is in `config_hwdb.yaml`, runtime enabling is a few hours
of wiring:

1. `firesim_queue/bin/firesim_queue.py:_render_per_job_runtime_yaml`
   - extend to write `autocounter: { read_rate: 1000 }` into the
     per-job YAML.
2. `benchmarks/runners/firesim.py`
   - after `runworkload` exits, copy
     `results-workload/<...>/<workload>0/AUTOCOUNTERFILE0.csv`
     into the per-cell artifact dir as `autocounter.csv`.
3. New `benchmarks/ingest/autocounter.py`
   - extractors: `dcache_misses`, `icache_misses`, `branch_mispred`,
     `dram_read_bytes`, `dram_write_bytes`. Drop unknown counter
     columns silently — autocounter's set of probes can change
     between bitstream builds.
4. `benchmarks/config/metrics.yaml`
   - five new entries, all `phase: kernel_synthesis`, `arms: [A, B, C]`,
     `nullable_if: "runner != 'firesim'"`.

Until then, the "IHWOC layout fix unlocked 17× speedup on RVV" finding
is supported by cycle deltas only — no quantitative L1-miss-rate
attribution.

## Energy / power

**State:** Not captured.

**Cause:** FireSim is a cycle-accurate functional simulator on an FPGA;
it doesn't model dynamic power. Real power numbers need a separate
energy model (something like Accelergy + Timeloop fed by per-op activity
counts), or a post-hoc model based on op-count × per-op-energy
estimates from prior silicon characterizations.

**Decision:** Out of scope for this benchmark cycle. Cycle counts and
(eventually) cache metrics are sufficient for the comparisons we want
to make (RVV vs Gemmini vs OPU on the same workloads).

## TracerV waveforms

**State:** Not captured. Probably never want to be — file sizes are
enormous and we have cycle counts at higher levels of abstraction via
the existing `profile_firesim.csv`.

## Per-op cache attribution

**Want:** "L1 miss rate during the first conv2d was X%, fell to Y%
after layout fix."

**Need:** Either (a) interval-sampled autocounter values aligned to op
boundaries from the existing profile CSV's timestamps, or (b) per-op
probes triggered by START/END markers similar to how we already do
cycle attribution. Option (a) is easier once autocounter is wired
(post-rebuild); option (b) requires Chisel-side work to add `synth_print`
markers around each kernel call.
