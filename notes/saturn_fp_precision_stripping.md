# Saturn + Rocket FP precision stripping

Two non-default chipyard targets that remove FP hardware from the scalar
Rocket FPU **and** the Saturn vector unit, verified end-to-end against
the generated Verilog and a Vivado synth area report:

| Config | Scalar FPU | Vector FP | Trap behaviour |
| --- | --- | --- | --- |
| `REFV256D128RocketConfig` (baseline) | FP64 + FP32 + FP16 | FP64 + FP32 + FP16 | — |
| `REFV256D128RocketFP32OnlyConfig` | FP32 + FP16 | FP32 + FP16 | FP64 → illegal-instruction |
| `REFV256D128RocketFP16OnlyConfig` | **FP16 only** | **FP16 only** | FP32 *and* FP64 → illegal-instruction |

The FP16-only config is **non-spec** — RISC-V requires `F` (FP32) as a
prerequisite for `Zfh` (FP16) — but produces a genuinely FP16-only
SoC for the FPGAs where FP32 LUTs are too expensive to fit.  Toolchains
must use a custom `-march` (e.g. `rv64imac_zfh`) since GCC rejects
Zfh-without-F.

The same machinery composes at any Saturn VLEN/DLEN.  A V128D128
prototype matrix was synthesized to VCU118 for area comparison
(see `plots/saturn_fp_area_v128d128.png`).

## TL;DR

| Knob | Surface | What it does |
| --- | --- | --- |
| `chipyard.WithRocketFPU32` | Scalar Rocket FPU | `fpu.fLen = 32`; drops ISA "d", keeps "f" + "zfh". |
| `chipyard.WithRocketFPU16` | Scalar Rocket FPU | `fpu.fLen = 16, minFLen = 16`; drops ISA "f" + "d"; keeps "zfh". Requires patched FPU.scala (new `(16,16) => h` case + gated `sfma` module). |
| `chipyard.WithSaturnVfLen32` | Vector ISA decode | Post-mutates `RocketCoreVectorParams` to `vfLen = 32`. ISA reports `zve64f` instead of `zve64d`. Used by both FP32-only and FP16-only (no `vfLen=16` in spec, so the DTS still says `zve64f` — the runtime trap happens at Saturn's vector dispatch validation). |
| `VectorParams.noFP64 = true` | Saturn HW | Fork-local flag (default false). Filters SEW=3 ops from every Saturn FP factory and gates the FP64 module instantiations inside `FPFMAPipe`, `FPDivSqrt`, `FPCompPipe`, `FPConvBlock`. |
| `VectorParams.noFP32 = true` | Saturn HW | Sister flag to `noFP64`. Filters SEW=2 ops and gates every FP32 hardfloat instantiation. |

Composition (`chipyard.config.SaturnConfigs`):

```scala
class REFV256D128RocketFP32OnlyConfig extends Config(
  new chipyard.WithRocketFPU32 ++
  new chipyard.WithSaturnVfLen32 ++
  new saturn.rocket.WithRocketVectorUnit(256, 128,
    VectorParams.refParams.copy(noFP64 = true)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)

class REFV256D128RocketFP16OnlyConfig extends Config(
  new chipyard.WithRocketFPU16 ++
  new chipyard.WithSaturnVfLen32 ++
  new saturn.rocket.WithRocketVectorUnit(256, 128,
    VectorParams.refParams.copy(noFP64 = true, noFP32 = true)) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.AbstractConfig)
```

Generated DTS ISA strings:

| Config | `riscv,isa` |
| --- | --- |
| Baseline | `rv64imafdcb_..._zve64d_zvfh_zfh_...` |
| FP32-only | `rv64imafcb_..._zve64f_zvfh_zfh_...` |
| FP16-only | `rv64imacb_..._zve64f_zvfh_zfh_...` |

Both `f` and `d` letters absent for FP16-only.  `zfh` and `zvfh` are
the only FP advertised.

## Verified FP-precision elimination

Built `sims/verilator/generated-src/chipyard.harness.TestHarness.<config>/gen-collateral/`
for each and grepped hardfloat module widths:

| Width | FP type | Baseline | FP32-only | FP16-only |
| --- | --- | --- | --- | --- |
| `e11_s53` | FP64 | present (19 refs) | **0** | **0** |
| `e8_s24`  | FP32 | present            | present | **0** |
| `e5_s11`  | FP16 | present            | present | present (35 refs) |

The only `FPUFMAPipe_*.sv` file in the FP16-only build is
`FPUFMAPipe_l3_f16.sv` — no `_f32` or `_f64` siblings.  Every
`MulAddRecFNPipe`, `INToRecFN`, `RecFNToIN(Dynamic)`,
`DivSqrtRecFM_small`, `RoundRawFNToRecFN` instance is `_e5_s11`.

## VCU118 area comparison (V128D128 prototype)

Vivado 2023.1 hierarchical post-synth on `xcvu9p-flga2104-2L-e`, three
configs synthesized with the new `make synth-only-report` target:

| Component | Vanilla | No-FP64 | FP16-only | Δ vs vanilla |
| --- | ---: | ---: | ---: | ---: |
| DDR controller (MIG) | 18,531 | 18,531 | 18,531 | — |
| L2 cache | 9,652 | 9,652 | 9,652 | — |
| **Saturn vector unit** | **90,208** | **76,368** | **64,899** | **−25,309 (−28%)** |
| **Scalar FPU** | **14,223** | **7,229** | **3,357** | **−10,866 (−76%)** |
| Rocket core (integer) | 6,793 | 6,823 | 6,896 | ±1% noise |
| L1 D-cache | 11,110 | 11,042 | 10,882 | ±1% noise |
| L1 I-cache + frontend | 5,191 | 5,135 | 5,123 | ±1% noise |
| System bus & periph | 6,989 | 6,983 | 6,983 | — |
| Other | 2,621 | 2,631 | 2,565 | — |
| **Total** | **165,318** | **144,394** | **128,888** | **−36,430 (−22%)** |

Non-FP components are unchanged across all three designs — confirming
the patches isolate FP hardware cleanly.  Plot:
`FreshScheduler/plots/saturn_fp_area_v128d128.png`.  Per-component %
of total:

- **Removing FP64** saves ~13K LUTs (8%) on the Saturn unit + ~7K (4%) on
  the scalar FPU, ~13% of total design area.
- **Going further to FP16-only** saves another ~12K (7%) on Saturn + ~4K
  (2%) on scalar FPU, another ~10% of total.
- Saturn scales: per-tile savings will roughly double at V256D128
  (FPFMAPipe lane count doubles) and ~quadruple at V512D256.

## Why `WithRocketFPU32` and not `WithRV32`

`freechips.rocketchip.rocket.WithRV32` also forces RV32 base ISA
(`xLen=32`, sv32 paging), incompatible with our rv64 software stack.
`WithRocketFPU32` only touches `fpu.fLen`, leaving xLen=64.

Per `FPU.scala:167-172` the historically valid `(minFLen, fLen)` combos
were `(32,32)`, `(16,32)`, `(32,64)`, `(16,64)`.  Saturn's
`WithRocketVectorUnit` forces `minFLen=16` (needed for zvfh
half-precision); combined with `fLen=32` we get `(16,32) = h + f` — no
`d`, no FP64 hardware.

`WithRocketFPU16` adds a **new** `(16, 16)` combination.  This requires
a `case (16, 16) => h` arm in the FPU decoder match AND a structural
patch to `FPU.scala` because the historical Rocket FPU instantiates an
FP32 `sfma` *unconditionally*; we now gate it:

```scala
val sfma = if (fLen >= 32) Some(Module(new FPUFMAPipe(cfg.sfmaLatency, FType.S))) else None
```

The main rocket decoder already gates `FDecode` / `DDecode` correctly
(`if (fLen >= 32) new FDecode`, `if (fLen >= 64) new DDecode` in
`RocketCore.scala:226-228`), so FP32 and FP64 instructions naturally
raise illegal-instruction at scalar decode when `fLen=16`.

## What was wrong with `useElementwiseFP64`

We initially thought Saturn was "FP32-only by default" because
`VectorParams.useElementwiseFP64 = false`.  That's backwards: at
`FPFMAPipe.scala:177` the original code

```scala
val fma_pipe = Module(new TandemFMAPipe(depth, i == 0 || !elementwiseFP64))
```

passes `buildFP64 = (i == 0) || !elementwiseFP64`.  With
`elementwiseFP64=false`, `buildFP64 = true` for **every** lane — full
SIMD FP64.  `elementwiseFP64=true` would build FP64 only in lane 0
(elementwise execution).  Neither value disables FP64.  Independently,
`FPConvBlock` and `FPDivSqrt` had **no FP64 knob at all** — their
FP64 paths were unconditional.  Removing FP precision from Saturn
required a source patch.

## Saturn source changes (fork-local)

All changes in `chipyard-fsim/generators/saturn/src/main/scala`.  Add
two booleans (`noFP64`, `noFP32`) to `VectorParams` and gate every
matching surface on `!noFP64` / `!noFP32`.

| File | Change |
| --- | --- |
| `common/Parameters.scala` | New `noFP64`, `noFP32` fields in `VectorParams`. `sharedFPFMA`/`fpFMA`/`fpMisc`/`allFPFUs` helpers thread both flags. All `VectorIssueStructure` callers (Unified, Shared, Split, MultiFMA, MultiALU, MultiMAC) pass `params.noFP64, params.noFP32`. |
| `exu/fp/SharedFPFMA.scala` | `SharedScalarFPFMAFactory(depth, noFP64, noFP32)` overrides `sews` to drop SEW=3 / SEW=2 accordingly. |
| `exu/fp/FPFMAPipe.scala` | `FMAFactory.base_insns` uses an overridable `sews` / `widen_sews` member (widening ops need their own narrower sew set: only SEWs whose +1 step is also enabled). `SIMDFPFMAFactory(depth, elementWiseFP64, noFP64, noFP32)` and `FPFMAPipe(...)` take both flags. `TandemFMAPipe(depth, buildFP64, buildFP32)` — `buildFP64 = !noFP64 && (i == 0 || !elementwiseFP64)`, `buildFP32 = !noFP32`. The body conditionally emits `da/db/dc` (FP64 recoded) and `sa/sb/sc` (FP32 recoded) Seqs; the FP64-only and FP32-only `widen` modules are gated. |
| `exu/fp/FPDiv.scala` | `FPDivSqrtFactory` → `case class` with `noFP64, noFP32`; `sews` filter. `FPDivSqrt` constructor takes both; `fTypes = Seq(FType.D, FType.S, FType.H).filter(t => !(noFP64 && t == D) && !(noFP32 && t == S))`; `gen_vfclass` Mux1H drops the FType.D / FType.S entries similarly. |
| `exu/fp/FPComp.scala` | `FPCmpFactory` → `case class`. `FPCompPipe` replaces `for (eew <- 1 until maxEew)` with `for (eew <- activeEews)` where `activeEews = (1 until 4).filterNot(e => (noFP64 && e==3) || (noFP32 && e==2))`. Dropped indices in `minmax_results` / `exceptions` get default-zero so downstream Mux1H selectors don't read uninitialised wires. |
| `exu/fp/FPConv.scala` | `FPConvFactory` → `case class` with three SEW filters (`sglSews`, `nrwSews`, `widSews`) covering all FCVT widths. `FPConvPipe(noFP64, noFP32)` → `FPConvBlock(noFP64, noFP32)`. Inside `FPConvBlock`: `raw64`/`raw32as64`/`raw16as64`/`d2i`/`i2d`/`s2d`/`d2s` become `if (noFP64) Nil else Seq(...)`; `raw32`/`raw16as32`/`s2i`/`i2s`/`h2s`/`s2h` become `if (noFP32) Nil else Seq(...)`. The result mux is restructured into four branches (`!noFP64 && !noFP32`, `!noFP64 && noFP32`, `noFP64 && !noFP32`, `noFP64 && noFP32`); only the SEW combinations whose hardware is present are emitted. |

## Rocket-chip source changes (fork-local)

`chipyard-fsim/generators/rocket-chip/src/main/scala/tile/`:

| File | Change |
| --- | --- |
| `FPU.scala` | Adds `case (16, 16) => h` to the FPUDecoder `insns` match (line 173). Gates `sfma` instantiation behind `Option`: `val sfma = if (fLen >= 32) Some(Module(new FPUFMAPipe(cfg.sfmaLatency, FType.S))) else None`. Updates the `pipes` list to `sfma.map(f => Pipe(f, ...))` so the FP32 FMA pipe only appears when `fLen>=32`. Fixes a latent `Fill(0, ...)` zero-width crash in `FPToInt` (lines 483-488): `Fill(math.max(1, maxType.ieeeWidth / minXLen), ...)` so FP16 maxType doesn't produce a zero-width replica. |
| `BaseTile.scala` | Drops the `"f"` ISA letter from the DTS string when `fpu.fLen < 32`: `val f = if (fpu.nonEmpty && fpu.get.fLen >= 32) "f" else ""`. Previously emitted `f` unconditionally regardless of fLen. |

## Chipyard fragment + config additions

`chipyard-fsim/generators/chipyard/src/main/scala/config/SaturnConfigs.scala`:

- `WithRocketFPU32` — sets `fpu.fLen = 32`.
- `WithRocketFPU16` — sets `fpu.fLen = 16, minFLen = 16`.
- `WithSaturnVfLen32` — post-mutates RocketCoreVectorParams `vfLen = 32`.
- `REFV256D128RocketFP32OnlyConfig` — V256D128, no-FP64.
- `REFV256D128RocketFP16OnlyConfig` — V256D128, no-FP64 + no-FP32.
- `REFV128D128RocketNoFP64Config` — V128D128 area-comparison target.
- `REFV128D128RocketNoFP32Config` — V128D128 area-comparison target.

`chipyard-fsim/fpga/src/main/scala/vcu118/Configs.scala` adds VCU118
wrappers for the three V128D128 prototypes.

`chipyard-fsim/fpga/Makefile` + `fpga/scripts/run_synth_only_with_report.tcl`:
new `synth-only-report` make target that runs Vivado synth then writes
hierarchical utilization reports without place/route.

## Where every FP surface is closed (FP16-only)

| Surface | Default | After patch |
| --- | --- | --- |
| Scalar FPU `fLen` | 64 | **16** |
| Scalar FPU `minFLen` | 16 | 16 (unchanged) |
| `misa.F` | 1 | **0** (`CSR.scala`: F bit gated on `fLen >= 32`) |
| `misa.D` | 1 | **0** |
| DTS ISA "f" / "d" | yes / yes | **no / no** (BaseTile patched) |
| DTS ISA "zfh" | yes | yes |
| `FDecode` loaded in main decoder | yes | **no** (`RocketCore.scala:226`: `if (fLen >= 32)`) |
| `DDecode` loaded | yes | **no** |
| `HDecode` loaded | yes | yes |
| Scalar `FPUFMAPipe_l3_f64` | yes | **gone** |
| Scalar `FPUFMAPipe_l3_f32` (`sfma`) | yes (unconditional!) | **gone** (now `Option`-gated) |
| Scalar `FPUFMAPipe_l3_f16` (`hfma`) | yes (since `minFLen==16`) | yes |
| Vector `vfLen` | 64 | **32** (cosmetic — DTS says `zve64f`) |
| Vector ISA `zve${eLen}d/f/x` | `zve64d` | **`zve64f`** |
| Vector ISA "v" string | yes | no |
| Saturn `TandemFMAPipe` FP64 lanes | per-SIMD | **gone** |
| Saturn `TandemFMAPipe` FP32 lanes | per-SIMD | **gone** |
| Saturn `FPDivSqrt` FType.D / FType.S | unconditional | **gone** / **gone** |
| Saturn `FPCompPipe` SEW=3 / SEW=2 | built | **gone** / **gone** |
| Saturn `FPConvBlock` D-format paths | unconditional | **gone** |
| Saturn `FPConvBlock` S-format paths | unconditional | **gone** |
| Hardfloat `_e11_s53` (FP64) | many | **never generated** |
| Hardfloat `_e8_s24` (FP32) | many | **never generated** |
| Hardfloat `_e5_s11` (FP16) | sometimes | **all 35 FP modules** |

## Trap behaviour

**Scalar FP32 / FP64 ops** (e.g. `fadd.s`, `fld`):
1. Main rocket decoder doesn't have `FDecode` / `DDecode` loaded.
2. `misa.F` / `misa.D` = 0.
3. Decode falls through → illegal-instruction trap.

**Vector FP32 / FP64 ops** (e.g. `vfadd.vv` with SEW=32):
1. Saturn's `EarlyVectorDecode` checks the op against
   `supported_ex_insns` (built from every FU factory's `insns`).
2. Our `restrictSEW(sews:_*)` filtering removed all SEW=2 (when
   `noFP32`) and SEW=3 (when `noFP64`) ops from those insn lists.
3. No FU accepts the dispatch → Saturn signals "not mine" back to
   Rocket → illegal-instruction trap at the scalar pipeline.

## Verification recipe

```bash
cd /scratch2/dima/chipyard-fsim
source env.sh

# FP32-only
cd sims/verilator
make CONFIG=REFV256D128RocketFP32OnlyConfig
DIR=generated-src/chipyard.harness.TestHarness.REFV256D128RocketFP32OnlyConfig
grep -c "e11_s53" $DIR/gen-collateral/*.sv | awk -F: '{s+=$2} END {print s}'   # 0
grep -c "e8_s24"  $DIR/gen-collateral/*.sv | awk -F: '{s+=$2} END {print s}'   # >0
grep "riscv,isa\b" $DIR/*.dts   # rv64imafcb_..._zve64f_zvfh_zfh_...

# FP16-only
make CONFIG=REFV256D128RocketFP16OnlyConfig
DIR=generated-src/chipyard.harness.TestHarness.REFV256D128RocketFP16OnlyConfig
grep -c "e11_s53\|e8_s24" $DIR/gen-collateral/*.sv | awk -F: '{s+=$2} END {print s}'   # 0
ls $DIR/gen-collateral/FPUFMAPipe_*.sv   # only FPUFMAPipe_l3_f16.sv
grep "riscv,isa\b" $DIR/*.dts   # rv64imacb_..._zve64f_zvfh_zfh_...
```

VCU118 area sweep (re-runs all three syntheses in parallel):

```bash
cd /scratch2/dima/chipyard-fsim/fpga
source ../env.sh
for cfg in REFV128D128Rocket{,NoFP64,NoFP32}VCU118Config; do
  make SUB_PROJECT=vcu118 CONFIG=$cfg CONFIG_PACKAGE=chipyard.fpga.vcu118 synth-only-report &
done
wait
python3 /scratch2/dima/misc_sw/FreshScheduler/scripts/plot_fpga_lut_breakdown.py \
  --report 'Vanilla\n(FP64+FP32+FP16)' \
    generated-src/chipyard.fpga.vcu118.VCU118FPGATestHarness.REFV128D128RocketVCU118Config/obj/report/utilization.txt \
  --report 'No FP64\n(FP32+FP16)' \
    generated-src/chipyard.fpga.vcu118.VCU118FPGATestHarness.REFV128D128RocketNoFP64VCU118Config/obj/report/utilization.txt \
  --report 'FP16-only\n(no FP32/64)' \
    generated-src/chipyard.fpga.vcu118.VCU118FPGATestHarness.REFV128D128RocketNoFP32VCU118Config/obj/report/utilization.txt \
  --out /scratch2/dima/misc_sw/FreshScheduler/plots/saturn_fp_area_v128d128.png
```

## Toolchain notes

| Config | `-march` |
| --- | --- |
| FP32-only | `rv64gc_zve64f_zvfh_zfh` (or `rv64imafc_zve64f_zvfh_zfh`) |
| FP16-only | `rv64imac_zfh` for scalar Zephyr; vector code needs `_zvfh` but no spec'd `zve64h` exists — use `_zve64f_zvfh` and accept that the toolchain thinks FP32 vector exists; runtime trap catches misuse. |

Running an ELF that uses missing precision will trap on the first FP
op of that precision.  Don't mix incompatible ELFs.

## Trap on FP32 demonstration

A 5-line C kernel that does `float a = b + c;` compiled for
`rv64imafc_zfh` and run on an FP16-only build will trap at decode of
`fadd.s` with `mcause=2` (illegal instruction).  Compile the same
kernel as `__fp16 a = b + c;` (`-march=rv64imac_zfh`) and it runs to
completion — the FP16 hardware handles it.

## Extending to other targets

The same fragments compose with any Saturn-rocket config:

```scala
class REFV256D128DualRocketGemminiQ31FP16OnlyConfig extends Config(
  new chipyard.WithRocketFPU16 ++
  new chipyard.WithSaturnVfLen32 ++
  new gemmini.Q31GemminiConfig ++
  new saturn.rocket.WithRocketVectorUnit(256, 128,
    VectorParams.refParams.copy(noFP64 = true, noFP32 = true)) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new freechips.rocketchip.rocket.WithNHugeCores(2) ++
  new chipyard.config.AbstractConfig)
```

Gemmini Q31's systolic array is integer-only — unaffected.
`GemminiFP32DefaultConfig` would obviously conflict with FP16-only.

For a Shuttle target, write an equivalent `WithShuttleVfLen32` that
matches `case tp: ShuttleTileAttachParams` (`saturn/rocket/Configs.scala:30`
and `saturn/shuttle/Configs.scala:35` both hard-code `vfLen=64`).  The
`WithRocketFPU16` fragment uses `RocketCoreConfig` directly and applies
to any Rocket-based tile.

## Related notes

- `zephyr_rvv_fix_summary.md` — Saturn V CSR save/restore fix.
- `zephyr_v_decouple_design.md` — V vs F context-switch decoupling.

Both are independent of this change; the V CSR save/restore code in
`v.c` saves the four V CSRs and the v0..v31 register block as raw
bytes (`vse8.v` / `vle8.v`), all of which are width-independent — the
absence of FP precision from vector ops doesn't affect save/restore.
