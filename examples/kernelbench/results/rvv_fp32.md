# KernelBench level1 on rvv/fp32 (reference, spike)

| bench | status | err | note |
|---|---|---|---|
| 19_ReLU | PASS | max_abs_err=0 | ok |
| 20_LeakyReLU | PASS | max_abs_err=0 | ok |
| 21_Sigmoid | PASS | max_abs_err=1.19e-07 | ok |
| 22_Tanh | PASS | max_abs_err=1.19e-07 | ok |
| 25_Swish | PASS | max_abs_err=1.19e-07 | ok |
| 26_GELU_ | PASS | max_abs_err=1.79e-07 | ok |
| 27_SELU_ | PASS | max_abs_err=0 | ok |
| 28_HardSigmoid | PASS | max_abs_err=5.96e-08 | ok |
| 29_Softplus | PASS | max_abs_err=1.19e-07 | ok |
| 30_Softsign | PASS | max_abs_err=0 | ok |
| 31_ELU | PASS | max_abs_err=0 | ok |
| 32_HardTanh | PASS | max_abs_err=0 | ok |
| 33_BatchNorm | PASS | max_abs_err=0 | ok |
| 37_FrobeniusNorm_ | PASS | max_abs_err=2.65e-08 | ok |
| 39_L2Norm_ | PASS | max_abs_err=5.22e-08 | ok |
| 42_Max_Pooling_2D | PASS | max_abs_err=0 | ok |
| 47_Sum_reduction_over_a_dimension | PASS | max_abs_err=5.72e-06 | ok |
| 48_Mean_reduction_over_a_dimension | PASS | max_abs_err=1.79e-07 | ok |
| 49_Max_reduction_over_a_dimension | PASS | max_abs_err=0 | ok |
| 50_conv_standard_2D__square_input__square_kernel | PASS | max_abs_err=1.55e-06 | ok |
| 51_Argmax_over_a_dimension | PASS | max_abs_err=0 | ok |
| 52_Argmin_over_a_dimension | PASS | max_abs_err=0 | ok |
| 53_Min_reduction_over_a_dimension | PASS | max_abs_err=0 | ok |
| 55_conv_standard_2D__asymmetric_input__square_kernel | PASS | max_abs_err=8.94e-07 | ok |
| 56_conv_standard_2D__asymmetric_input__asymmetric_kernel | PASS | max_abs_err=1.07e-06 | ok |
| 62_conv_standard_2D__square_input__asymmetric_kernel | PASS | max_abs_err=1.31e-06 | ok |
| 63_conv_standard_2D__square_input__square_kernel | PASS | max_abs_err=7.15e-07 | ok |
| 82_conv_depthwise_2D_square_input_square_kernel | PASS | max_abs_err=0 | ok |
| 83_conv_depthwise_2D_square_input_asymmetric_kernel | PASS | max_abs_err=0 | ok |
| 84_conv_depthwise_2D_asymmetric_input_square_kernel | PASS | max_abs_err=0 | ok |
| 85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel | PASS | max_abs_err=0 | ok |
| 86_conv_depthwise_separable_2D | PASS | max_abs_err=0 | ok |
| 87_conv_pointwise_2D | PASS | max_abs_err=0 | ok |
| 88_MinGPTNewGelu | PASS | max_abs_err=1.19e-07 | ok |

_34 PASS / 0 FAIL_
