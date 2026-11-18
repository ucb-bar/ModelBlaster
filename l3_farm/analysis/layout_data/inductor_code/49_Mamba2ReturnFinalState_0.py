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


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/pz/cpzqqknwfut753hgzr37bulhqq5g5w3wb273f2nkkyssb3gt46ns.py
# Topologically Sorted Source Nodes: [tensor_1, tensor_4, A_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
# Source node to ATen node mapping:
#   A_cumsum => cumsum
#   tensor_1 => view_1
#   tensor_4 => permute
# Graph fragment:
#   %arg0_1 : Tensor "f16[2048, 128, 8][1024, 8, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %view_1 : Tensor "f16[2048, 2, 64, 8][1024, 512, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg0_1, [2048, 2, 64, 8]), kwargs = {})
#   %permute : Tensor "f16[2048, 8, 2, 64][1024, 1, 512, 8]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1, [0, 3, 1, 2]), kwargs = {})
#   %cumsum : Tensor "f16[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.cumsum.default](args = (%permute, -1), kwargs = {})
#   return %cumsum
triton_per_fused_cumsum_permute_view_0 = async_compile.triton('triton_per_fused_cumsum_permute_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_cumsum_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 4194304, 'r0_': 8388608}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_cumsum_permute_view_0(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    x0 = (xindex % 8)
    x1 = xindex // 8
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*r0_2 + 512*x1), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp3, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_2 + 64*x3), tmp3, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/4z/c4zbtsjax5jgvrxxlkdelxivmouhxysue6wmoa4hugywj3tdnpxn.py
# Topologically Sorted Source Nodes: [getitem_4, pad, x_cumsum_1], Original ATen: [aten.select, aten.constant_pad_nd, aten.cumsum]
# Source node to ATen node mapping:
#   getitem_4 => select
#   pad => constant_pad_nd
#   x_cumsum_1 => cumsum_2
# Graph fragment:
#   %cumsum : Tensor "f16[2048, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %select : Tensor "f16[2048, 8, 2][1024, 128, 64]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.select.int](args = (%cumsum, 3, -1), kwargs = {})
#   %constant_pad_nd : Tensor "f16[2048, 8, 3][24, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%select, [1, 0], 0.0), kwargs = {})
#   %cumsum_2 : Tensor "f16[2048, 8, 3][24, 3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cumsum.default](args = (%constant_pad_nd, -1), kwargs = {})
#   return %cumsum_2
triton_per_fused_constant_pad_nd_cumsum_select_1 = async_compile.triton('triton_per_fused_constant_pad_nd_cumsum_select_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16384, 'r0_': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_constant_pad_nd_cumsum_select_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 73728}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_constant_pad_nd_cumsum_select_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 3
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 8)
    x1 = xindex // 8
    x3 = xindex
    tmp0 = ((-1) + r0_2).to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-449) + 64*x0 + 512*r0_2 + 1024*x1), r0_mask & tmp2, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp6, = tl.associative_scan((tmp5,), 1, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (r0_2 + 3*x3), tmp6, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/r3/cr3infbow2ep32zhvqor6qnk7d7mtlcy55rl2pwmloiiojwtfzpc.py
# Topologically Sorted Source Nodes: [tensor, states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
# Source node to ATen node mapping:
#   states => clone_3, permute_13, permute_14, unsqueeze_15
#   tensor => view
# Graph fragment:
#   %arg3_1 : Tensor "f16[2048, 128, 8, 64][65536, 512, 64, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %view : Tensor "f16[2048, 2, 64, 8, 64][65536, 32768, 512, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg3_1, [2048, 2, 64, 8, 64]), kwargs = {})
#   %unsqueeze_15 : Tensor "f16[2048, 2, 64, 8, 64, 1][65536, 32768, 512, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view, 5), kwargs = {})
#   %permute_13 : Tensor "f16[2048, 2, 8, 64, 1, 64][65536, 32768, 64, 1, 1, 512]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_15, [0, 1, 3, 4, 5, 2]), kwargs = {})
#   %permute_14 : Tensor "f16[2048, 2, 8, 64, 64, 1][65536, 32768, 64, 1, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %clone_3 : Tensor "f16[2048, 2, 8, 64, 64, 1][65536, 32768, 4096, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_14,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_permute_unsqueeze_view_2 = async_compile.triton('triton_poi_fused_clone_permute_unsqueeze_view_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2097152, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_clone_permute_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 268435456, 'x': 536870912}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_permute_unsqueeze_view_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2097152
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 32768*y1), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (x2 + 64*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/4l/c4lys2z4pjq6sjn43ddyg4lfs4ltfximz6ofmqbd5qytr36jhirz.py
# Topologically Sorted Source Nodes: [tensor_2, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.slice, aten.sub, aten.exp, aten.mul, aten.clone]
# Source node to ATen node mapping:
#   decay_states => exp_1
#   getitem_2 => slice_1
#   states => clone_4, mul_1, permute_11, permute_12, permute_15, unsqueeze_12, unsqueeze_13, unsqueeze_14
#   sub_1 => sub_2
#   tensor_2 => view_2
# Graph fragment:
#   %arg1_1 : Tensor "f16[2048, 128, 8, 16][16384, 128, 16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %cumsum : Tensor "f16[2048, 8, 2, 64][1024, 64, 512, 1]cuda:0" = PlaceHolder[target=cumsum]
#   %view_2 : Tensor "f16[2048, 2, 64, 8, 16][16384, 8192, 128, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg1_1, [2048, 2, 64, 8, 16]), kwargs = {})
#   %unsqueeze_12 : Tensor "f16[2048, 2, 64, 8, 16, 1][16384, 8192, 128, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%view_2, 5), kwargs = {})
#   %permute_11 : Tensor "f16[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_12, [0, 1, 3, 5, 4, 2]), kwargs = {})
#   %slice_1 : Tensor "f16[2048, 8, 2, 1][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%cumsum, 3, -1, 9223372036854775807), kwargs = {})
#   %sub_2 : Tensor "f16[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_1, %cumsum), kwargs = {})
#   %exp_1 : Tensor "f16[2048, 8, 2, 64][1024, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %unsqueeze_13 : Tensor "f16[2048, 8, 2, 64, 1][1024, 128, 64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%exp_1, 4), kwargs = {})
#   %unsqueeze_14 : Tensor "f16[2048, 8, 2, 64, 1, 1][1024, 128, 64, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_13, 5), kwargs = {})
#   %permute_12 : Tensor "f16[2048, 2, 8, 1, 1, 64][1024, 64, 128, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_14, [0, 2, 1, 4, 5, 3]), kwargs = {})
#   %mul_1 : Tensor "f16[2048, 2, 8, 1, 16, 64][16384, 8192, 16, 1, 1, 128]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_11, %permute_12), kwargs = {})
#   %permute_15 : Tensor "f16[2048, 2, 8, 64, 16, 1][16384, 8192, 16, 128, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [0, 1, 2, 5, 4, 3]), kwargs = {})
#   %clone_4 : Tensor "f16[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_3 = async_compile.triton('triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 205520896}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 64)
    x2 = ((xindex // 1024) % 8)
    x3 = xindex // 8192
    x5 = xindex // 1024
    x6 = xindex // 16
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2 + 128*x1 + 8192*x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (63 + 64*x5), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tl.store(out_ptr0 + (x7), tmp5, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/5o/c5oim2erswo53nxwjyi6kgty7uxcdmedqadnpa7rcpy2sjdqtiqd.py
# Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp]
# Source node to ATen node mapping:
#   decay_chunk => exp_2
#   getitem_5 => unsqueeze_16
#   getitem_6 => unsqueeze_17
#   invert_1 => bitwise_not_1
#   mask_1 => iota_2, iota_3, le_1, logical_and_1, sub_4, unsqueeze_18, unsqueeze_19
#   ones_1 => full_default_1
#   x_segsum_2 => sub_3
#   x_segsum_3 => full_default_2, where_1
# Graph fragment:
#   %cumsum_2 : Tensor "f16[2048, 8, 3][24, 3, 1]cuda:0" = PlaceHolder[target=cumsum_2]
#   %iota_2 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_18 : Tensor "i64[1, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_2, -2), kwargs = {})
#   %iota_3 : Tensor "i64[3][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (3,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_19 : Tensor "i64[3, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_3, -1), kwargs = {})
#   %sub_4 : Tensor "i64[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_18, %unsqueeze_19), kwargs = {})
#   %le_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_4, 0), kwargs = {})
#   %full_default_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([3, 3], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %logical_and_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_1, %full_default_1), kwargs = {})
#   %bitwise_not_1 : Tensor "b8[3, 3][3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%logical_and_1,), kwargs = {})
#   %full_default_2 : Tensor "f16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %unsqueeze_16 : Tensor "f16[2048, 8, 3, 1][24, 3, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 3), kwargs = {})
#   %unsqueeze_17 : Tensor "f16[2048, 8, 1, 3][24, 3, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cumsum_2, 2), kwargs = {})
#   %sub_3 : Tensor "f16[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_16, %unsqueeze_17), kwargs = {})
#   %where_1 : Tensor "f16[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%bitwise_not_1, %full_default_2, %sub_3), kwargs = {})
#   %exp_2 : Tensor "f16[2048, 8, 3, 3][72, 9, 3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%where_1,), kwargs = {})
#   return %exp_2
triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_4 = async_compile.triton('triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 786432}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 3)
    x1 = ((xindex // 3) % 3)
    x3 = xindex // 3
    x2 = xindex // 9
    x4 = xindex
    tmp7 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0 + 3*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = (x0 + ((-1)*x1)).to(tl.int64)
    tmp1 = (tmp0).to(tl.int64)
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 <= tmp2
    tmp4 = tl.full([1], True, tl.int1)
    tmp5 = tmp3 & tmp4
    tmp6 = tmp5 == 0
    tmp9 = tmp7 - tmp8
    tmp10 = tl.full([1], float("-inf"), tl.float32)
    tmp11 = tl.where(tmp6, tmp10, tmp9)
    tmp12 = libdevice.exp(tmp11)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_49_Mamba2ReturnFinalState_1792942157/zq/czqszclqw5g6dubaauqll7u2heayik6cb2upb3bpfmxzvk7u5upa.py
# Topologically Sorted Source Nodes: [initial_states, states, states_1, new_states], Original ATen: [aten.zeros_like, aten.view, aten.permute, aten.cat, aten.unsqueeze, aten.clone]
# Source node to ATen node mapping:
#   initial_states => full_default
#   new_states => clone_5, permute_18, permute_20, unsqueeze_22
#   states => permute_16, view_13, view_14
#   states_1 => cat
# Graph fragment:
#   %bmm_2 : Tensor "f16[32768, 64, 16][1024, 16, 1]cuda:0" = PlaceHolder[target=bmm_2]
#   %full_default : Tensor "f16[2048, 1, 8, 64, 16][8192, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([2048, 1, 8, 64, 16], 0), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_13 : Tensor "f16[2048, 2, 8, 64, 1, 16][16384, 8192, 1024, 16, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_2, [2048, 2, 8, 64, 1, 16]), kwargs = {})
#   %permute_16 : Tensor "f16[2048, 2, 8, 64, 16, 1][16384, 8192, 1024, 16, 1, 16]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 1, 2, 3, 5, 4]), kwargs = {})
#   %view_14 : Tensor "f16[2048, 2, 8, 64, 16][16384, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_16, [2048, 2, 8, 64, 16]), kwargs = {})
#   %cat : Tensor "f16[2048, 3, 8, 64, 16][24576, 8192, 1024, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%full_default, %view_14], 1), kwargs = {})
#   %unsqueeze_22 : Tensor "f16[2048, 3, 8, 64, 16, 1][24576, 8192, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat, 5), kwargs = {})
#   %permute_18 : Tensor "f16[2048, 1, 8, 64, 16, 3][24576, 1, 1024, 16, 1, 8192]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_22, [0, 5, 2, 3, 4, 1]), kwargs = {})
#   %permute_20 : Tensor "f16[2048, 8, 3, 64, 16, 1][24576, 1024, 8192, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_18, [0, 2, 5, 3, 4, 1]), kwargs = {})
#   %clone_5 : Tensor "f16[2048, 8, 3, 64, 16, 1][24576, 3072, 1024, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_20,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_5
triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_5 = async_compile.triton('triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 268435456}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = ((xindex // 1024) % 3)
    x0 = (xindex % 1024)
    x2 = ((xindex // 3072) % 8)
    x3 = xindex // 24576
    x4 = xindex
    tmp0 = (x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = (x1).to(tl.int64)
    tmp4 = (tmp3).to(tl.int64)
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.full([1], 0.0, tl.float32)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tmp0 >= tmp5
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (x0 + 1024*x2 + 8192*((-1) + x1) + 16384*x3), tmp10, other=0.0).to(tl.float32)
    tmp14 = tl.where(tmp6, tmp9, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        assert_size_stride(arg0_1, (2048, 128, 8), (1024, 8, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((2048, 8, 2, 64), (1024, 64, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [tensor_1, tensor_4, A_cumsum], Original ATen: [aten.view, aten.permute, aten.cumsum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_cumsum_permute_view_0.run(arg0_1, buf0, 32768, 64, stream=raw_stream0)
            del arg0_1
            buf1 = empty_strided_cuda((2048, 8, 3), (24, 3, 1), torch.float16)
            # Topologically Sorted Source Nodes: [getitem_4, pad, x_cumsum_1], Original ATen: [aten.select, aten.constant_pad_nd, aten.cumsum]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_constant_pad_nd_cumsum_select_1.run(buf0, buf1, 16384, 3, stream=raw_stream0)
            assert_size_stride(arg3_1, (2048, 128, 8, 64), (65536, 512, 64, 1), 'input')
            arg3_1 = copy_if_misaligned(arg3_1)
            buf2 = empty_strided_cuda((2048, 2, 8, 64, 64, 1), (65536, 32768, 4096, 64, 1, 1), torch.float16)
            # Topologically Sorted Source Nodes: [tensor, states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_permute_unsqueeze_view_2.run(arg3_1, buf2, 2097152, 64, stream=raw_stream0)
            del arg3_1
            assert_size_stride(arg1_1, (2048, 128, 8, 16), (16384, 128, 16, 1), 'input')
            buf3 = empty_strided_cuda((2048, 2, 8, 64, 16, 1), (16384, 8192, 1024, 16, 1, 1), torch.float16)
            # Topologically Sorted Source Nodes: [tensor_2, states, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.slice, aten.sub, aten.exp, aten.mul, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_exp_mul_permute_slice_sub_unsqueeze_view_3.run(arg1_1, buf0, buf3, 33554432, stream=raw_stream0)
            del arg1_1
            del buf0
            buf4 = empty_strided_cuda((32768, 64, 16), (1024, 16, 1), torch.float16)
            # Topologically Sorted Source Nodes: [tensor, states, tensor_2, getitem_2, sub_1, decay_states], Original ATen: [aten.view, aten.unsqueeze, aten.permute, aten.clone, aten._unsafe_view, aten.slice, aten.sub, aten.exp, aten.mul, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf2, (32768, 64, 64), (4096, 64, 1), 0), reinterpret_tensor(buf3, (32768, 64, 16), (1024, 16, 1), 0), out=buf4)
            del buf2
            del buf3
            buf5 = empty_strided_cuda((2048, 8, 3, 3), (72, 9, 3, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_bitwise_not_exp_masked_fill_ones_sub_tril_unsqueeze_4.run(buf1, buf5, 147456, stream=raw_stream0)
            del buf1
            buf6 = empty_strided_cuda((2048, 8, 3, 64, 16, 1), (24576, 3072, 1024, 16, 1, 1), torch.float16)
            # Topologically Sorted Source Nodes: [initial_states, states, states_1, new_states], Original ATen: [aten.zeros_like, aten.view, aten.permute, aten.cat, aten.unsqueeze, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_cat_clone_permute_unsqueeze_view_zeros_like_5.run(buf4, buf6, 50331648, stream=raw_stream0)
            del buf4
            buf7 = empty_strided_cuda((16384, 3, 1024), (3072, 1024, 1), torch.float16)
            # Topologically Sorted Source Nodes: [mask_1, ones_1, invert_1, x_segsum_3, getitem_5, getitem_6, x_segsum_2, decay_chunk, new_states, initial_states, states, states_1], Original ATen: [aten.tril, aten.ones, aten.bitwise_not, aten.masked_fill, aten.unsqueeze, aten.sub, aten.exp, aten.view, aten.zeros_like, aten.permute, aten.cat, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf5, (16384, 3, 3), (9, 3, 1), 0), reinterpret_tensor(buf6, (16384, 3, 1024), (3072, 1024, 1), 0), out=buf7)
            del buf5
            del buf6
        return (reinterpret_tensor(buf7, (2048, 8, 64, 16), (24576, 3072, 16, 1), 2048), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((2048, 128, 8), (1024, 8, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((2048, 128, 8, 16), (16384, 128, 16, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((2048, 128, 8, 64), (65536, 512, 64, 1), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
