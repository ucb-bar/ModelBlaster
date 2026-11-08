#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#define kernel_linear rvv_linear_f32
#include "../../../kernels/rvv/rvv_linear_direct.c"
