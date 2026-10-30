/*
 * modelblaster large-io section placement.
 *
 * Included via zephyr_linker_sources(SECTIONS ...) ONLY under
 * CONFIG_RISCV_CMODEL_LARGE. This include point (snippets-sections.ld) sits
 * AFTER .rodata/.data/.bss/.noinit in the RISC-V linker script, so `.mb_bigio`
 * lands at the top of RAM.
 *
 * Why: the multi-GB baked test io (test_io.S, emitted into .mb_bigio when
 * MB_BIGIO_HIGH is defined) would otherwise sit in .rodata between .text and
 * .bss/.sdata, shoving __global_pointer$ / _kernel / z_interrupt_stacks past
 * the fixed +/-2GiB PC-relative reach of the hand-written startup assembly
 * (vector.S/reset.S/isr.S) -- which -mcmodel=large does NOT rewrite. Placing
 * the io last keeps .text/.data/.bss/gp inside a low <2GiB window the startup
 * asm can reach; the C code reaches the far io via the large model's
 * R_RISCV_64 constant-pool indirection.
 */
SECTION_PROLOGUE(.mb_bigio,,SUBALIGN(16))
{
	. = ALIGN(16);
	__mb_bigio_start = .;
	KEEP(*(.mb_bigio))
	KEEP(*(SORT_BY_NAME(.mb_bigio.*)))
	__mb_bigio_end = .;
} GROUP_LINK_IN(RAMABLE_REGION)
