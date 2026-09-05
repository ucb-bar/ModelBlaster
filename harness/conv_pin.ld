/*
 * modelblaster conv0 I-cache placement pin.
 *
 * Wired in by harness/CMakeLists.txt when -DMB_CONV_PIN=1 is passed, as an
 * extra `-T` linker fragment. Because it contains an INSERT command it AUGMENTS
 * (does not replace) Zephyr's main linker script.
 *
 * Why: `kernel_conv2d_s8_dronet` is a single ~5.8 KB monolithic function (the
 * Gemmini tiled_conv / loop-conv issue path is inlined into it). It is 1.43x
 * the Rocket L1I way-size (64 sets x 64 B = 4096 B), so it wraps the sets once
 * and its "gemm-handoff" inner loop set-conflicts with its own requant/im2col
 * loops. WHICH lines collide depends only on conv's start address mod 4096, so
 * the layer's cycle count swings +/-~290K purely from where the function lands
 * (PROVEN: two 64-aligned builds, byte-identical conv, differed +292K, all of
 * it in the gemm-handoff phase). -falign-loops / -falign-functions do NOT fix
 * it -- they just move conv to another (often worse) mod-4096 offset.
 *
 * Fix: gather conv (and its one hot callee) into a section pinned to
 * ALIGN(4096) + mb_conv_pin_offset. ALIGN(4096) normalises out all upstream
 * code-size changes, so conv's (addr % 4096) == mb_conv_pin_offset regardless
 * of what sibling work adds to .text -> deterministic, and tunable via the
 * --defsym offset. INSERT BEFORE .text makes this rule win the input-section
 * match ahead of the generic *(.text.*) catch-all, so no source edit is needed
 * (survives kernels.c regeneration).
 *
 * mb_conv_pin_offset is supplied with -Wl,--defsym=mb_conv_pin_offset=<n>;
 * it defaults to 0 when absent.
 */
SECTIONS
{
    .mbconvpin : ALIGN(4096)
    {
        . = . + (DEFINED(mb_conv_pin_offset) ? mb_conv_pin_offset : 0);
        __mb_convpin_start = .;
        KEEP(*(.text.kernel_conv2d_s8_dronet))
        __mb_conv_body_end = .;
        KEEP(*(.text.tiled_conv_total_spad_rows*))
        __mb_convpin_end = .;
    }
} INSERT BEFORE text;
