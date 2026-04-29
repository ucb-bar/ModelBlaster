"""Memory-model stanza for the optimize-phase LLM prompt.

When the firesim re-rank step is active, the inner optimize loop is no
longer scoring candidates on a flat-memory simulator — cache-locality
wins now show up in the cycle counts. To steer the LLM toward those
wins we add a structured stanza describing the target's memory hierarchy
to the optimize-phase system prompt.

The stanza is target-specific. Today we only ship one (the FireSim
alveo_u250 quad-rocket-saturn hwconfig). New stanzas can be added as
new entries in the registry below; the shape of each is a dict that
prompt-assembly code can render however it wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MemoryModel:
    """Cache hierarchy + DRAM latency for a target.

    Sizes are in bytes. Latencies are in core cycles (the unit that
    matters for the LLM's reasoning — `rdcycle` is at the core clock,
    not at DRAM clock). All numbers are documented at the source.
    """

    name: str
    description: str
    # L1 data cache (per hart).
    l1d_size_bytes: int
    l1d_line_bytes: int
    l1d_assoc: int
    l1d_sets: int
    # L2 / LLC. On the quad-rocket-saturn build the LLC is the only mid-
    # level cache (no per-core L2 between L1 and LLC). Modelled here as
    # one shared layer with the LLC's parameters.
    llc_size_bytes: int
    llc_line_bytes: int
    llc_assoc: int
    # Outer-memory miss penalty (LLC -> DRAM round trip), in core cycles.
    dram_miss_cycles: int
    # Notes the LLM should keep in mind.
    notes: tuple[str, ...] = field(default_factory=tuple)


# alveo_u250_firesim-quad-rocket-saturn-no-nic-l2-llc4mb-ddr3
#
# Source-of-truth references:
#  - L1D: WithNHugeCores in
#    chipyard-fsim/generators/rocket-chip/src/main/scala/rocket/Configs.scala
#    (DCacheParams: nSets=64, nWays=8, blockBytes=site(CacheBlockBytes))
#  - blockBytes default: rocket-chip subsystem default = 64 (verified
#    against `WithCacheBlockBytes`-free configs; the chipyard
#    AbstractConfig used here does not override it).
#  - LLC: hwconfig name "llc4mb" + WithInclusiveCache default
#    (capacityKB=4096, nWays=8 — see WithInclusiveCache class).
#  - dram_miss_cycles: WithInclusiveCache `outerLatencyCycles` default of
#    40, plus a small SBus/DDR queueing buffer. We use 60 as a working
#    estimate in the prompt; the LLM only needs an order-of-magnitude
#    figure to reason "cache miss is ~50-100x an L1 hit".
QUAD_ROCKET_SATURN_MEMORY_MODEL = MemoryModel(
    name="firesim_quad_rocket_saturn",
    description=(
        "FireSim alveo_u250 quad-rocket-saturn-no-nic-l2-llc4mb-ddr3. "
        "Four RVV-capable Rocket harts at 1 GHz target, sharing a 4 MB LLC."
    ),
    l1d_size_bytes=32 * 1024,    # 64 sets × 8 ways × 64 B = 32 KB
    l1d_line_bytes=64,
    l1d_assoc=8,
    l1d_sets=64,
    llc_size_bytes=4 * 1024 * 1024,   # 4 MB shared
    llc_line_bytes=64,
    llc_assoc=8,
    dram_miss_cycles=60,
    notes=(
        "L1D is per-hart (32 KB). LLC is shared across all 4 harts (4 MB). "
        "An L1 miss that hits in LLC costs ~20-30 cycles; an LLC miss to "
        "DRAM costs ~60-100 cycles.",
        "RVV vector loads warm the L1D one cache line per 64-byte chunk; "
        "the prefetcher does not auto-stride for you here. If your inner "
        "loop reads weights with a stride > 64 bytes (e.g. by output "
        "channel inside an OIHW-laid-out weight tensor), every iteration "
        "is an independent cache fill.",
        "When the loop's working set exceeds 32 KB the compiler will not "
        "save you — you must restructure the loop nest to keep the hot "
        "tile in L1D yourself. For conv2d this typically means blocking "
        "OC into tiles small enough that one output-channel slab of "
        "weights (TILE_OC × IC × KH × KW × 4 B) fits in ~24 KB, leaving "
        "~8 KB for the input tile and the output write set.",
    ),
)


def memory_model_stanza(
    *, model: Optional[MemoryModel] = None,
    include_worked_example: bool = True,
) -> str:
    """Render the memory-model section that goes into the optimize-phase
    system prompt. Returns a markdown-shaped block ready to splice
    into the prompt body.

    `model` defaults to the quad-rocket-saturn hwconfig. The worked
    example uses an OC=128, IC=128, K=3 conv2d (DroNet's heaviest 3x3
    conv) — its weight footprint is 576 KB so the LLM can see, end to
    end, why blocking OC matters.
    """
    if model is None:
        model = QUAD_ROCKET_SATURN_MEMORY_MODEL
    parts = [
        "## Target memory hierarchy",
        "",
        f"This kernel runs on the **{model.name}** target "
        f"({model.description}).",
        "",
        "Cache parameters the optimizer should respect:",
        "",
        f"  - L1D: {model.l1d_size_bytes // 1024} KB per hart, "
        f"{model.l1d_line_bytes}-byte lines, "
        f"{model.l1d_assoc}-way / {model.l1d_sets}-set",
        f"  - LLC: {model.llc_size_bytes // (1024*1024)} MB shared, "
        f"{model.llc_line_bytes}-byte lines, "
        f"{model.llc_assoc}-way",
        f"  - DRAM miss penalty: ~{model.dram_miss_cycles} core cycles",
        "",
    ]
    for n in model.notes:
        parts.append(f"- {n}")
    parts.append("")

    if include_worked_example:
        parts.extend([
            "### Worked example: OC-blocked 3x3 conv2d",
            "",
            "Take the heaviest dronet conv2d: OC=128, IC=128, OH=4, OW=4, "
            "KH=KW=3 (dispatch_23 in the IR). The weight tensor footprint is",
            "",
            "    OC * IC * KH * KW * sizeof(float)",
            "  = 128 * 128 * 3 * 3 * 4 = 589,824 bytes",
            "",
            "That's ~576 KB — 18x the L1D. A direct conv2d that walks the "
            "whole weight tensor for each (oh, ow) output position will pull "
            "every weight from LLC (or DRAM) on every reuse. To hold one "
            "OC-tile of weights resident in L1D, pick TILE_OC such that",
            "",
            "    TILE_OC * IC * KH * KW * 4   <=   ~24 KB",
            "    TILE_OC                       <=   24*1024 / (128*9*4) = 5",
            "",
            "Round down to a power of 2 (4) to keep RVV-friendly LMUL. "
            "Loop:",
            "",
            "    for (oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {",
            "        // load TILE_OC * IC * KH * KW weights into L1D once",
            "        for (oh = 0; oh < OH; oh++)",
            "          for (ow = 0; ow < OW; ow++) {",
            "             // read input tile (small for OH=OW=4) + reuse",
            "             for (oc_inner = 0; oc_inner < TILE_OC; oc_inner++)",
            "               compute output[oc_outer + oc_inner, oh, ow]",
            "          }",
            "    }",
            "",
            "The outer OC tile is loaded **once per output-spatial sweep** "
            "instead of once per output element. On a flat-memory simulator "
            "(spike) this rewrite has zero effect; on the firesim cache "
            "hierarchy it cuts traffic to LLC by a factor of OH*OW = 16x "
            "for this op.",
            "",
            "Apply the same shape of reasoning to your kernel: identify "
            "the largest tensor your inner loop touches, check its "
            "footprint against the L1D ceiling, and tile the dimension "
            "that gives the most reuse.",
            "",
        ])
    return "\n".join(parts)
