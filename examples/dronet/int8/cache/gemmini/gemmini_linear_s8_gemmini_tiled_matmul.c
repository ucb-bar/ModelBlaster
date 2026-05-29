void kernel_linear_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int M, int K, int N,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    enum { GEMMINI_LIN_ACC_MAX = 16 * 4096 };
    static int32_t ws_acc[GEMMINI_LIN_ACC_MAX] __attribute__((aligned(64)));

    int total_out = M * N;
    if (input_offset != 0 || filter_offset != 0 || output_offset != 0
            || output_shift < 0 || output_shift > 30
            || (size_t)(M * N) > GEMMINI_LIN_ACC_MAX
            || M <= 0 || K <= 0 || N <= 0
            || total_out * K < 256) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int32_t acc = bias ? bias[n] : 0;
                for (int k = 0; k < K; k++) {
                    int32_t in_v = (int32_t)input[m * K + k] + input_offset;
                    int32_t w_v  = (int32_t)weight[n * K + k] + filter_offset;
                    acc += in_v * w_v;
                }
                int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                prod = (prod + (1LL << 30)) >> 31;
                int32_t scaled = (int32_t)prod;
                if (output_shift > 0) {
                    int32_t round = (1 << (output_shift - 1));
                    scaled = (scaled + round) >> output_shift;
                } else if (output_shift < 0) {
                    scaled = scaled << (-output_shift);
                }
                scaled += output_offset;
                if (scaled < activation_min) scaled = activation_min;
                if (scaled > activation_max) scaled = activation_max;
                output[m * N + n] = (int8_t)scaled;
            }
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);
    asm volatile("fence" ::: "memory");

    tiled_matmul_auto(
        (size_t)M, (size_t)N, (size_t)K,
        input, weight,
        NULL, (void *)ws_acc,
        (size_t)K, (size_t)K, (size_t)N, (size_t)N,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
        false,
        false, true,
        true, false,
        0, WS
    );

    gemmini_fence();
    gemmini_flush(0);

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = ws_acc[m * N + n] + (bias ? bias[n] : 0);
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + ((int64_t)1 << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0) {
                scaled = (int32_t)(((int64_t)scaled
                    + ((int64_t)1 << (output_shift - 1))) >> output_shift);
            } else if (output_shift < 0) {
                scaled <<= (-output_shift);
            }
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            output[m * N + n] = (int8_t)scaled;
        }
    }
}