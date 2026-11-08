#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "zvt_insn.h"
#define kernel_conv2d_s8 zvt_conv2d_s8
#include "../../../kernels/zvt/zvt_conv2d_s8_zvt_igemm_i8i32.c"
