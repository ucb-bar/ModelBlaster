# Adding Saturn OPU support to spike — scoping note

Goal: a functional spike that can execute Saturn OPU kernels (the
`bme.h` asm-macro programming model — `VOPACC`, `OPMVINBCAST`,
`VMV_VR`, `VMV_RV`) so we can iterate on `rvv_opu` curated kernels in
the agents flow without round-tripping through FireSim.

Integer-only (i8×i8 → i32 acc, `opuParams`) is the v1 target. The fp8
OPU (`opuMxParams`) layers on top once the int path works.

## Source tree

Spike source: `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/` —
this is the canonical riscv-software-src spike, currently at
`9c190a07` (master). chipyard builds it into
`.conda-env/riscv-tools/bin/spike`.

Extension mechanism: `customext/*.cc` — each file registers an
`extension_t` subclass via `REGISTER_EXTENSION(name, constructor)`.
The existing `dummy_rocc.cc` is the template, but it handles the
RoCC custom-0 opcode (0x0b), NOT what we need: OPU rides the V opcode
(0x57) with custom funct3/funct7 combos.

Custom instructions are pushed into `processor.custom_instructions`
which is searched **before** the base instruction table (see
`processor.cc::register_insn` and `register_extension`). So our
extension can claim encoding space that overlaps standard V
instructions, and our handler will win on dispatch.

## OPU instruction encodings

From `hw/chipyard/generators/saturn/benchmarks/common/bme.h` and
cross-referenced with `OPMFunct6` in
`generators/saturn/src/main/scala/common/Consts.scala`:

| op            | funct3 | funct7 | funct6 | semantic                          |
|---|---|---|---|---|
| `VOPACC`      | 0x2    | 0x51   | 0x28   | md += vs2 ⊗ vs1 (int outer-prod MAC) |
| `OPMVINBCAST` | 0x6    | 0x59   | 0x2C   | broadcast vector → all rows of md  |
| `VMV_VR`      | 0x6    | 0x5d   | 0x2E   | vd = row[rs1] of matrix ms2         |
| `VMV_RV`      | 0x6    | 0x55   | 0x2A   | row[rs1] of md = vs2                |
| `OPFMACC`     | 0x1    | 0x4b   | 0x25   | fp variant of VOPACC (skip for v1)  |

(funct7 = (funct6 << 1) | vm; OPU instructions all have vm=1.)

Encoding cheat-sheet for spike's MATCH/MASK pairs (opcode=0x57, rs1=rs2=rd=0):

```
MATCH_VOPACC      = (0x51 << 25) | (0x2 << 12) | 0x57 = 0xA2002057
MATCH_OPMVINBCAST = (0x59 << 25) | (0x6 << 12) | 0x57 = 0xB2006057
MATCH_VMV_VR      = (0x5D << 25) | (0x6 << 12) | 0x57 = 0xBA006057
MATCH_VMV_RV      = (0x55 << 25) | (0x6 << 12) | 0x57 = 0xAA006057

MASK              = 0xFE00707F   // funct7 + funct3 + opcode, all bits matter
```

**Encoding collision:** Saturn's OPU sits in encoding slots that
upstream RVV recently allocated for Zvqdotq:

- standard `VQDOT.VX` = funct6=0x2C funct3=0x6 (collides with OPMVINBCAST)
- standard `VQDOT.VV` = funct6=0x2C funct3=0x2 (no funct3 overlap with OPU)
- standard `VQDOTSU.VV` = funct6=0x2A funct3=0x2 (collides with VMV_RV's funct6 but not funct3)

Stock spike's masks for Zvqdotq are `0xfc00707f` — they DON'T include
bit 25 (vm). Our OPU mask `0xfe00707f` DOES include vm, and OPU always
sets vm=1. So:

- For vm=1: custom_instructions table hit first → OPU handler wins.
- For vm=0: our MATCH (which requires vm=1) doesn't fire → standard
  Zvqdotq handler fires through the base table.

This means OPU + Zvqdotq can coexist in the same spike build as long
as our extension is registered before the base instructions (which is
the default for `--extension=`).

## OPU architectural state to model

From `generators/saturn/src/main/scala/exu/OuterProductUnit.scala` and
`OPUParameters`:

```
case class OPUParameters(
  aWidth = 8, bWidth = 8, cWidth = 32, nMrfRegs = 4)

// derived (from HasOPUParams):
clusterXdim = cWidth / bWidth  = 4
clusterYdim = cWidth / aWidth  = 4
yDim        = (dLen / aWidth) / clusterYdim   // V256D128 → 4
xDim        = (dLen / bWidth) / clusterXdim   // V256D128 → 4
```

For the standard V256D128 opuParams config:
- 4 matrix registers m0..m3.
- Each matrix = 16 rows × 16 cols of i32 (yDim × clusterYdim × dLen/aWidth = 4·4·4 i8 elems × 4 cells).
- Actually closer-look: each "tile" is xDim × yDim cells, each cell is clusterXdim × clusterYdim PEs → tile is 16×16 of i32.

So the spike model needs: `int32_t mrf[nMrfRegs][rows][cols]` where
`rows = yDim * clusterYdim` and `cols = xDim * clusterXdim`. For the
common V256D128 opuParams that's `int32_t mrf[4][16][16]`.

**Dimensioning concern:** `dLen` and `vLen` (and therefore the matrix
size) are bitstream-configurable. The spike extension should be
parameterizable — either via CLI (`--opu-vlen=256 --opu-dlen=128`)
or auto-derived from the VLEN spike is built with. Cleanest is a
constructor arg + `--extension=saturn_opu:vlen=256,dlen=128` parsing.

## Per-instruction semantics

### VOPACC md, vs2, vs1   (i8 outer-product accumulate)

```c++
reg_t VOPACC::exec(processor_t *p, insn_t insn, reg_t pc) {
    int md  = insn.rd();   // 0..3
    int vs1 = insn.rs1();  // standard V reg, e8 elems
    int vs2 = insn.rs2();
    size_t vl = p->VU.vl;  // current vector length
    require_extension(EXT_V);
    require(md < 4 && p->VU.vsew == 8);  // OPU is i8-mode

    for (size_t r = 0; r < vl; r++) {
        int8_t a = (int8_t)p->VU.elt<int8_t>(vs1, r);
        for (size_t c = 0; c < vl; c++) {
            int8_t b = (int8_t)p->VU.elt<int8_t>(vs2, c);
            mrf[md][r][c] += (int32_t)a * (int32_t)b;
        }
    }
    return pc + insn_length(insn.bits());
}
```

Wraparound on signed i32 overflow (matches the HW which doesn't
saturate). LMUL handling: when `vsetvli`'s LMUL > 1, vs1/vs2 span
multiple V registers — the loop must walk `p->VU.elt<int8_t>(vs1 + r/VLMAX_E8, r%VLMAX_E8)` (or use VU's element accessor which already does this).

### OPMVINBCAST md, vs2   (broadcast vector into all rows)

```c++
reg_t OPMVINBCAST::exec(...) {
    int md = insn.rd();
    int vs2 = insn.rs2();
    size_t vl = p->VU.vl;
    require(md < 4);
    require(p->VU.vsew == 32);   // bcast is in e32 mode (per bme.h usage)

    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < vl; c++) {
            mrf[md][r][c] = (int32_t)p->VU.elt<int32_t>(vs2, c);
        }
    }
    ...
}
```

Note the asymmetry: vsew for OPMVINBCAST is e32 (it's loading a bias
vector of i32 into the accumulator). Saturn's HW also supports a
column-broadcast variant gated by bit 4 of rd (`mvin_col := dis_inst.rd(4)`);
since md ∈ {m0..m3} uses bit 0 only (rd field = 0..3), bit 4 is
always 0 in our usage — column broadcast is unreachable from the
public asm macros. Skip in v1.

### VMV_VR vd, rs1, ms2   (read row out of matrix)

```c++
reg_t VMV_VR::exec(...) {
    int vd  = insn.rd();
    int rs1 = (int)p->get_state()->XPR[insn.rs1()];  // row index (scalar)
    int ms2 = insn.rs2();
    size_t vl = p->VU.vl;
    require(ms2 < 4 && rs1 < rows);

    for (size_t c = 0; c < vl; c++) {
        p->VU.elt<int32_t>(vd, c, true) = mrf[ms2][rs1][c];
    }
    ...
}
```

The `true` flag on `elt(...)` is the V state's "this is a write" hint
(see spike's vector.cc for the API).

### VMV_RV md, rs1, vs2   (write row into matrix)

```c++
reg_t VMV_RV::exec(...) {
    int md  = insn.rd();
    int rs1 = (int)p->get_state()->XPR[insn.rs1()];
    int vs2 = insn.rs2();
    size_t vl = p->VU.vl;
    require(md < 4 && rs1 < rows);

    for (size_t c = 0; c < vl; c++) {
        mrf[md][rs1][c] = (int32_t)p->VU.elt<int32_t>(vs2, c);
    }
    ...
}
```

## Implementation outline

### File `customext/saturn_opu.cc`  (~300 LOC)

```c++
#include "extension.h"
#include "mmu.h"
#include "processor.h"
#include <vector>
#include <cstring>

class saturn_opu_t : public extension_t {
public:
    const char* name() const override { return "saturn_opu"; }
    saturn_opu_t() { reset_state(); }

    std::vector<insn_desc_t> get_instructions(const processor_t&) override;
    std::vector<disasm_insn_t*> get_disasms(const processor_t* p = nullptr) override;
    void reset(processor_t&) override { reset_state(); }

private:
    // Sized for V256D128 opuParams; make dynamic if dLen/vLen vary.
    static constexpr int N_MRF = 4;
    static constexpr int MAX_ROWS = 16;
    static constexpr int MAX_COLS = 16;
    int32_t mrf[N_MRF][MAX_ROWS][MAX_COLS];

    void reset_state() { std::memset(mrf, 0, sizeof(mrf)); }

    // One static dispatcher per insn, registered by get_instructions().
    static reg_t exec_vopacc      (processor_t*, insn_t, reg_t);
    static reg_t exec_opmvinbcast (processor_t*, insn_t, reg_t);
    static reg_t exec_vmv_vr      (processor_t*, insn_t, reg_t);
    static reg_t exec_vmv_rv      (processor_t*, insn_t, reg_t);
};

// MATCH/MASK constants
constexpr reg_t MATCH_VOPACC      = 0xA2002057;
constexpr reg_t MATCH_OPMVINBCAST = 0xB2006057;
constexpr reg_t MATCH_VMV_VR      = 0xBA006057;
constexpr reg_t MATCH_VMV_RV      = 0xAA006057;
constexpr reg_t MASK_OPU          = 0xFE00707F;

std::vector<insn_desc_t> saturn_opu_t::get_instructions(const processor_t&) {
    // insn_desc_t bundles fast_rv32i/fast_rv64i/logged_rv32i/logged_rv64i
    // function pointers. For a simple extension we point all eight to
    // the same handler — see dummy_rocc.cc for the macro trick.
    return {
        { MATCH_VOPACC,      MASK_OPU, exec_vopacc,      ... },
        { MATCH_OPMVINBCAST, MASK_OPU, exec_opmvinbcast, ... },
        { MATCH_VMV_VR,      MASK_OPU, exec_vmv_vr,      ... },
        { MATCH_VMV_RV,      MASK_OPU, exec_vmv_rv,      ... },
    };
}

REGISTER_EXTENSION(saturn_opu, []() { return new saturn_opu_t; })
```

(The full `insn_desc_t` boilerplate — fast/logged × rv32/rv64 — is
mechanical; see `processor.cc::register_base_instructions` for the
exact macro pattern. Most can be the same function pointer.)

### Wiring the MRF state to the extension instance

The static `exec_*` functions get a `processor_t*` from the dispatcher
but not the extension instance directly. Two clean ways:

1. **Look up by name.** `p->get_extension("saturn_opu")` returns
   `extension_t*` which we `static_cast<saturn_opu_t*>`. One extra
   indirection per OPU instruction (negligible for functional sim).

2. **Per-processor singleton.** Stash the extension pointer in a
   thread-local map keyed by `processor_t*`. Faster but more state.

Option 1 is what gemmini's spike extension does. Stick with that.

### Disassembly (`get_disasms`)

Optional. Without it, `spike --log` will print the raw .insn bytes for
OPU instructions, which is debuggable but ugly. Adding 4
`disasm_insn_t*` entries to print e.g. `vopacc m1,v18,v16` takes ~30
LOC and is worth doing alongside the executor.

### Build wiring

1. Add `saturn_opu.cc` to `customext/customext.mk.in`:
   ```
   customext_subproject_deps += saturn_opu
   $(eval $(call subproject_template,saturn_opu))
   saturn_opu_subproject_deps = ...
   ```
   (Follow the `dummy_rocc` pattern in that file.)

2. Rebuild spike:
   ```
   cd hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/build
   ../configure --prefix=$RISCV
   make -j$(nproc) install
   ```

3. Use:
   ```
   spike --extension=saturn_opu --isa=rv64gcv_zicntr <elf>
   ```

   (Same registration pattern as `--extension=gemmini`.)

## Estimated effort

- Core extension file + 4 executors + MATCH/MASK: ~1 day for someone
  familiar with spike's internals.
- Disassembly: ~half a day.
- LMUL / multi-VREG handling for VOPACC: another half-day; easy to
  miss edge cases.
- Build wiring + smoke test against upstream `opu-gemm` benchmark:
  ~half a day.
- Parameterization (`--extension=saturn_opu:vlen=256,dlen=128`):
  ~half a day if we go beyond a hardcoded V256D128 dimensioning.

**Total: 2–3 working days** for a correct, parameterized,
disassembly-equipped functional model. Could ship a hardcoded
V256D128-only proof-of-concept in **1 day** if we accept the
parameterization debt.

## What spike will NOT model

These would be true HW differences spike can't reproduce — agents-flow
correctness verify is still sufficient because the model output is
deterministic w.r.t. the asm sequence:

- **Pipeline latency / scoreboarding** (the entire
  `OuterProductSequencer.scala` is a microarchitectural detail; spike
  semantics are atomic per-instruction).
- **VAT / data hazards** between OPU and standard V instructions
  sharing the VRF.
- **FP latency stalls** (Saturn's `latency=2` MulAddRecFNPipe for
  fp8 — not relevant for the integer-only v1).

## Integration with agents flow

Once `spike-opu` is built, the agents flow needs one small change in
`agents/pipeline/backends.py::RVV_OPU`:

```python
spike_args=(
    "--extension=saturn_opu",
    "--isa=rv64gcv_zicntr",
),
```

and `spike_runner.py` picks up an `AGENTS_OPU_SPIKE` env var (mirror
of `AGENTS_GEMMINI_SPIKE`) pointing to the OPU-built spike binary.
The build wiring change in spike_runner is one-line, following the
gemmini precedent.

## References

- spike extension API: `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/riscv/extension.h`
- Template: `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/dummy_rocc.cc`
- VRF accessors: `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/riscv/vector_unit.h` (`VU.elt<T>(reg, idx, write?)`)
- Saturn OPU RTL: `hw/chipyard/generators/saturn/src/main/scala/exu/OuterProductUnit.scala` (branch `origin/opu-fp8`)
- OPMFunct6 enum: `hw/chipyard/generators/saturn/src/main/scala/common/Consts.scala` (branch `origin/opu-fp8`)
- Asm macros: `hw/chipyard/generators/saturn/benchmarks/common/bme.h` (branch `origin/opu-fp8`)
- Reference test: `hw/chipyard/generators/saturn/benchmarks/opu-gemm/main.c` (branch `origin/opu-fp8`)
