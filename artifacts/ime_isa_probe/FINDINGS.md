# Which IME instruction families this K1 implements

The extension spec (`github.com/spacemit-com/riscv-ime-extension-spec`, `src/instruction-list.adoc`,
`src/instruction-func.adoc`, `src/images/wavedrom/ma-format.adoc`) defines three instruction families,
not one:

| family | func7 | mnemonics | what it does |
|---|---|---|---|
| OPMMA | `111000` | `vmadot`, `vmadotu`, `vmadotsu`, `vmadotus` | `C += A x B`, int8 in, int32 accumulate |
| OPMMA-k | `111001` | `vmadot1`, `vmadot2`, `vmadot3`, `vmadotn` | the same dot with **A re-indexed by a slide offset**: `A[cp*M*K + slide*K + i*K + k]`; 1-3 are fixed slides, `vmadotn` takes the slide in `t0` |
| OPFMMA | `111010` | `vfmadot`, `vfmadot1/2/3/n` | fp16 / bf16 matrix dot |

`func3` selects signedness for OPMMA (uu / us / su / ss) and the slide for OPFMMA. `.insn`'s funct7
field is 7 bits — the spec's 6-bit func7 plus the vm/mode bit — so `111000|1` is `0x71`, which is what
`kernels/ime/ime_*_s8_ime_vmadot_4x4x8.c` emit.

## What was measured

`ime_family_probe.c` pins itself to a hart and executes each candidate under the **same** vector
configuration our working kernel uses, catching `SIGILL`. The spec raises illegal-instruction for an
unsupported MAC *configuration* as well as for an absent instruction, so a plain `vmadot` is probed
identically alongside as the control, and `ime_fp_probe_e16.c` re-probes the fp family under an `e16`
configuration (with an `e16` `vmadot` as that config's own control) so an fp result cannot be an
artefact of having asked at SEW=8.

`probe_output.txt` is the raw output, all eight harts, plus the e16 pass on one hart of each cluster.

| candidate | cluster 0 (`CPU_P#0-3`) | cluster 1 (`CPU_E#0-3`) |
|---|---|---|
| `vmadot` ss (`0x71`, f3=3) — what we emit | EXECUTES | SIGILL |
| `vmadot` uu / us / su (`0x71`, f3=0/1/2) | EXECUTES | SIGILL |
| `vmadot1` ss (`0x73`, f3=3) — sliding window | **EXECUTES** | SIGILL |
| `vfmadot` (`0x74`-`0x77`, f3=0/1) under e8 **and** e16 | SIGILL | SIGILL |

## What follows

* **The sliding-window family is present and unused.** Every IME kernel here emits only `vmadot`.
  `vmadot1/2/3/n` re-index A by a slide offset — `A[cp*M*K + slide*K + i*K + k]` — so the slide moves
  along **m in whole K strides**: it selects a different four-row window of the im2col matrix, letting
  overlapping windows reuse an A register group instead of reloading it. It does not shift along the
  tap axis k, so it does not change how a patch is gathered.

  That matters for where the gap actually is. The measured table
  (`artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv`) loses to RVV on 8 of 57 shapes, and grouping
  the 3x3 convolutions by output positions and output channels shows the loss is not simply "small M":

  | M | OC | n | median speedup |
  |---|---|---|---|
  | 6 | 64 | 4 | **0.815** |
  | 6 | 128 | 5 | 1.030 |
  | 6 | 256 | 1 | 0.885 |
  | 24 | 64 | 13 | 1.253 |
  | 96 | 64 | 5 | 1.638 |

  The four worst are the detect-head convolutions at M=6 with OC=64, and the same M=6 with OC=128 wins.
  A per-dispatch cost amortised over the MAC work does not separate those two: the B pack is K*OC bytes
  against M*OC*K MACs, so its ratio to useful work is 1/M for both, identical at M=6.

  At M=6 there are two m-tiles, `ceil(6/4)`, so there is almost no overlapping-window reuse for a slide
  to capture — the reuse the instruction offers grows with the number of m-tiles, and these are the
  shapes that have fewest. The slide form is therefore unlikely to be the lever for the shapes that
  lose, and what it would help is the shapes that already win. Both statements are inference from the
  spec's indexing and the table above; **no slide-form kernel has been written or timed**, so the
  instruction's cost on this part remains unmeasured, and so does whatever else separates OC=64 from
  OC=128 at M=6.
* **The fp16 matrix family is absent on this part.** `cores/spacemit_k1.json` and
  `pipeline/ime_cost.py` both state that fp16 reaches the engine only through accuracy-gated int8
  requantisation. That statement now has a probe behind it, at two vector configurations and six
  encodings, rather than standing alone.
* **Only the ss signedness variant is emitted**, though all four execute. Our int8 path is symmetric
  per-tensor with zero-point folded into the accumulator, so ss is the correct choice; the other three
  are recorded here as available rather than as something to switch to.

## Reproducing

```bash
scp artifacts/ime_isa_probe/ime_family_probe.c artifacts/ime_isa_probe/ime_fp_probe_e16.c k1:/root/imeprobe/
ssh k1 'cd /root/imeprobe && gcc -O1 -march=rv64gcv -o probe ime_family_probe.c \
        && gcc -O1 -march=rv64gcv -o probe2 ime_fp_probe_e16.c \
        && for h in 0 1 2 3 4 5 6 7; do ./probe $h; done && for h in 0 4; do ./probe2 $h; done'
```
