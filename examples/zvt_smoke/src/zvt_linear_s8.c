/* The pipeline prepends Backend.kernel_includes to curated kernels. */
#include <stdint.h>
#include <stddef.h>
#include "zvt_insn.h"
#define kernel_linear_s8 zvt_linear_s8
#include "../../../kernels/zvt/zvt_linear_s8_zvt_tile_i8i32.c"
