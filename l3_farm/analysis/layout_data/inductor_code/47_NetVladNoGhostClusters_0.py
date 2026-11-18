# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/krisd/layout/cache/ind_47_NetVladNoGhostClusters_1792942126/lv/clvzjxjd6lkdbcn2tjffzkfehbgzsxtijnt2picg63glk4r4zi7b.py
# Topologically Sorted Source Nodes: [assignment_1, assignment_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   assignment_1 => add, add_1, convert_element_type_2, convert_element_type_3, convert_element_type_4, mul, mul_1, rsqrt, sub
#   assignment_2 => convert_element_type_5, convert_element_type_6, div, exp_default, sub_tensor
# Graph fragment:
#   %mm : Tensor "f16[204800, 32][32, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg2_1 : Tensor "f16[32][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f16[32][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f16[32][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f16[32][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %convert_element_type_5 : Tensor "f32[204800, 32][32, 1]cuda:0" = PlaceHolder[target=convert_element_type_5]
#   %getitem : Tensor "f32[204800, 1][1, 204800]cuda:0" = PlaceHolder[target=getitem]
#   %getitem_1 : Tensor "f32[204800, 1][1, 204800]cuda:0" = PlaceHolder[target=getitem_1]
#   %convert_element_type_2 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %sub : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm, %convert_element_type_2), kwargs = {})
#   %convert_element_type_3 : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.float32), kwargs = {})
#   %add : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[32][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg4_1), kwargs = {})
#   %add_1 : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg5_1), kwargs = {})
#   %convert_element_type_4 : Tensor "f16[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float16), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_4, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_5, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %convert_element_type_6 : Tensor "f16[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %convert_element_type_5,%getitem,%getitem_1,%convert_element_type_6
triton_per_fused__native_batch_norm_legit_no_training__softmax_prepare_softmax_online_0 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training__softmax_prepare_softmax_online_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'out_ptr3': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training__softmax_prepare_softmax_online_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 39321856}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_prepare_softmax_online_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 204800
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 32*x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1, 1], 1e-05, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp4 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp23 = triton_helpers.fmax2(tmp21, 1)[:, None].to(tl.float32)
    tmp24 = tmp19 - tmp23
    tmp25 = libdevice.exp(tmp24)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp29 = tmp18 - tmp23
    tmp30 = libdevice.exp(tmp29)
    tmp31 = (tmp30 / tmp28)
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 32*x0), tmp32, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_47_NetVladNoGhostClusters_1792942126/mi/cmimoxr3du6zfdfmenlxzx72lbpw6tsuadnshqyconhltw2yznyq.py
# Topologically Sorted Source Nodes: [assignment_2, assignment_4, a_sum], Original ATen: [aten._softmax, aten.view, aten.sum]
# Source node to ATen node mapping:
#   a_sum => sum_2
#   assignment_2 => convert_element_type_6, div, exp_default, sub_tensor
#   assignment_4 => view_1
# Graph fragment:
#   %convert_element_type_6 : Tensor "f16[204800, 32][32, 1]cuda:0" = PlaceHolder[target=convert_element_type_6]
#   %sub_tensor : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %getitem), kwargs = {})
#   %exp_default : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_1), kwargs = {})
#   %convert_element_type_6 : Tensor "f16[204800, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   %view_1 : Tensor "f16[2048, 100, 32][3200, 32, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_6, [-1, 100, 32]), kwargs = {})
#   %sum_2 : Tensor "f16[2048, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_1, [1], True), kwargs = {})
#   return %sum_2
triton_red_fused__softmax_sum_view_1 = async_compile.triton('triton_red_fused__softmax_sum_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__softmax_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 13369344, 'r0_': 0}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__softmax_sum_view_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 100
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 32)
    x1 = xindex // 32
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 32*r0_2 + 3200*x1), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_47_NetVladNoGhostClusters_1792942126/pz/cpzqh2vjzqlsbr7eadfqqayvmkszlb2i36aqoiekwbligzcgxp37.py
# Topologically Sorted Source Nodes: [vlad_1, a, vlad_2, vlad_3], Original ATen: [aten.transpose, aten.mul, aten.sub, aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   a => mul_2
#   vlad_1 => permute_1
#   vlad_2 => sub_2
#   vlad_3 => convert_element_type_9, pow_1, sum_3
# Graph fragment:
#   %bmm : Tensor "f16[2048, 32, 512][16384, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %sum_2 : Tensor "f16[2048, 1, 32][32, 65536, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %arg6_1 : Tensor "f16[1, 512, 32][16384, 32, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %permute_1 : Tensor "f16[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm, [0, 2, 1]), kwargs = {})
#   %mul_2 : Tensor "f16[2048, 512, 32][16384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %arg6_1), kwargs = {})
#   %sub_2 : Tensor "f16[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_1, %mul_2), kwargs = {})
#   %convert_element_type_9 : Tensor "f32[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_2, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_9, 2.0), kwargs = {})
#   %sum_3 : Tensor "f32[2048, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1], True), kwargs = {})
#   return %sum_3
triton_per_fused_linalg_vector_norm_mul_sub_transpose_2 = async_compile.triton('triton_per_fused_linalg_vector_norm_mul_sub_transpose_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_linalg_vector_norm_mul_sub_transpose_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 688128, 'r0_': 67108864}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_mul_sub_transpose_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 512
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0 + 32*r0_2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 - tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp9, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_47_NetVladNoGhostClusters_1792942126/7h/c7hc2zgwtxtn3mlifyqhptqeje7vcgkok7bxrb6whhudiswonxa3.py
# Topologically Sorted Source Nodes: [vlad_1, a, vlad_2, vlad_3, vlad_4, vlad_5], Original ATen: [aten.transpose, aten.mul, aten.sub, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   a => mul_2
#   vlad_1 => permute_1
#   vlad_2 => sub_2
#   vlad_3 => clamp_min, convert_element_type_10, div_1, expand_2, pow_2
#   vlad_4 => clone, view_6
#   vlad_5 => clamp_min_1, convert_element_type_11, convert_element_type_12, div_2, expand_3, pow_3, pow_4, sum_4
# Graph fragment:
#   %bmm : Tensor "f16[2048, 32, 512][16384, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %sum_2 : Tensor "f16[2048, 1, 32][32, 65536, 1]cuda:0" = PlaceHolder[target=sum_2]
#   %arg6_1 : Tensor "f16[1, 512, 32][16384, 32, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %sum_3 : Tensor "f32[2048, 1, 32][32, 65536, 1]cuda:0" = PlaceHolder[target=sum_3]
#   %sum_4 : Tensor "f32[2048, 1][1, 2048]cuda:0" = PlaceHolder[target=sum_4]
#   %permute_1 : Tensor "f16[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%bmm, [0, 2, 1]), kwargs = {})
#   %mul_2 : Tensor "f16[2048, 512, 32][16384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %arg6_1), kwargs = {})
#   %sub_2 : Tensor "f16[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_1, %mul_2), kwargs = {})
#   %pow_2 : Tensor "f32[2048, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %convert_element_type_10 : Tensor "f16[2048, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_2, torch.float16), kwargs = {})
#   %clamp_min : Tensor "f16[2048, 1, 32][32, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_10, 1e-12), kwargs = {})
#   %expand_2 : Tensor "f16[2048, 512, 32][32, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min, [2048, 512, 32]), kwargs = {})
#   %div_1 : Tensor "f16[2048, 512, 32][16384, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, %expand_2), kwargs = {})
#   %clone : Tensor "f16[2048, 512, 32][16384, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%div_1,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : Tensor "f16[2048, 16384][16384, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [2048, 16384]), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[2048, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[2048, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_11, 2.0), kwargs = {})
#   %sum_4 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1], True), kwargs = {})
#   %pow_4 : Tensor "f32[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %convert_element_type_12 : Tensor "f16[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%pow_4, torch.float16), kwargs = {})
#   %clamp_min_1 : Tensor "f16[2048, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_12, 1e-12), kwargs = {})
#   %expand_3 : Tensor "f16[2048, 16384][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%clamp_min_1, [2048, 16384]), kwargs = {})
#   %div_2 : Tensor "f16[2048, 16384][16384, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_6, %expand_3), kwargs = {})
#   return %sum_4,%div_2
triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_3 = async_compile.triton('triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 134643712}, 'add_persistent_rblock': True, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.full([1, 1], 1e-12, tl.float32)
        tmp9 = tl.maximum(tmp7, tmp8, tl.PropagateNan.ALL)
        tmp10 = (tmp4 / tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp16 - tmp19
        tmp22 = tl.sqrt_rn(tmp21)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.full([1, 1], 1e-12, tl.float32)
        tmp25 = tl.maximum(tmp23, tmp24, tl.PropagateNan.ALL)
        tmp26 = (tmp20 / tmp25)
        tmp27 = tl.sqrt_rn(tmp14)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tl.maximum(tmp28, tmp24, tl.PropagateNan.ALL)
        tmp30 = (tmp26 / tmp29)
        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp30, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
        args.clear()
        assert_size_stride_grouped((arg0_1, arg1_1), ((2048, 100, 512), (512, 32)), ((51200, 512, 1), (32, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((204800, 32), (32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [x, assignment], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg0_1, (204800, 512), (512, 1), 0), arg1_1, out=buf0)
            del arg1_1
            assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1), ((32, ), (32, ), (32, ), (32, )), ((1, ), (1, ), (1, ), (1, )), 'input')
            buf4 = empty_strided_cuda((204800, 32), (32, 1), torch.float16)
            # Topologically Sorted Source Nodes: [assignment_1, assignment_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__native_batch_norm_legit_no_training__softmax_prepare_softmax_online_0.run(buf0, arg2_1, arg3_1, arg4_1, arg5_1, buf4, 204800, 32, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            del buf0
            buf5 = empty_strided_cuda((2048, 32, 512), (16384, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [x, assignment_2, assignment_4, assignment_5, x_1, vlad], Original ATen: [aten.view, aten._softmax, aten.transpose, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf4, (2048, 32, 100), (3200, 1, 32), 0), arg0_1, out=buf5)
            del arg0_1
            buf6 = empty_strided_cuda((2048, 1, 32), (32, 65536, 1), torch.float16)
            # Topologically Sorted Source Nodes: [assignment_2, assignment_4, a_sum], Original ATen: [aten._softmax, aten.view, aten.sum]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__softmax_sum_view_1.run(buf4, buf6, 65536, 100, stream=raw_stream0)
            del buf4
            assert_size_stride(arg6_1, (1, 512, 32), (16384, 32, 1), 'input')
            buf7 = empty_strided_cuda((2048, 1, 32), (32, 65536, 1), torch.float32)
            # Topologically Sorted Source Nodes: [vlad_1, a, vlad_2, vlad_3], Original ATen: [aten.transpose, aten.mul, aten.sub, aten.linalg_vector_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_linalg_vector_norm_mul_sub_transpose_2.run(buf5, buf6, arg6_1, buf7, 65536, 512, stream=raw_stream0)
            buf9 = empty_strided_cuda((2048, 16384), (16384, 1), torch.float16)
            # Topologically Sorted Source Nodes: [vlad_1, a, vlad_2, vlad_3, vlad_4, vlad_5], Original ATen: [aten.transpose, aten.mul, aten.sub, aten.linalg_vector_norm, aten.clamp_min, aten.expand, aten.div, aten.clone, aten._unsafe_view]
            raw_stream0 = get_raw_stream(0)
            triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_3.run(buf5, buf6, arg6_1, buf7, buf9, 2048, 16384, stream=raw_stream0)
            del arg6_1
            del buf5
            del buf6
            del buf7
        return (buf9, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 100, 512), (51200, 512, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((1, 512, 32), (16384, 32, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
