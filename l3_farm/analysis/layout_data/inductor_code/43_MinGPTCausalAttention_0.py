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


# kernel path: /tmp/krisd/layout/cache/ind_43_MinGPTCausalAttention_1792942112/2b/c2b5n6dedc7eyhv7kvzxlz7guzijrmyx7biviitqqwd27hcgfopv.py
# Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear => view_1
#   matmul => clone
#   q_1 => permute_2
#   split => split
#   view_1 => view_3
# Graph fragment:
#   %addmm : Tensor "f16[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_3 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem, [128, 512, 8, 96]), kwargs = {})
#   %permute_2 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f16[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_split_transpose_view_0 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 512)
    x2 = ((xindex // 49152) % 8)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 2304*x1 + 1179648*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_43_MinGPTCausalAttention_1792942112/vf/cvf6i46za4yqjnuuvuuxfwtfqni5ipnffqsi4j3p4arelsyxl2ks.py
# Topologically Sorted Source Nodes: [linear, split, view, k_1, transpose_3, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   k_1 => permute_1
#   linear => view_1
#   matmul => clone_1
#   split => split
#   transpose_3 => permute_4
#   view => view_2
# Graph fragment:
#   %addmm : Tensor "f16[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_2 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_1, [128, 512, 8, 96]), kwargs = {})
#   %permute_1 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %permute_4 : Tensor "f16[128, 8, 96, 512][1179648, 96, 1, 2304]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f16[128, 8, 96, 512][393216, 49152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_split_transpose_view_1 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 100663296, 'x': 201326592}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 512
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0 + 2304*x2 + 1179648*y1), xmask & ymask).to(tl.float32)
    tl.store(out_ptr0 + (x2 + 512*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_43_MinGPTCausalAttention_1792942112/ze/czezpne4qxpqwoj4a6yte5p4nanwmowustgbnvoe735lgit5p2fm.py
# Topologically Sorted Source Nodes: [linear, split, view_2, v_1, y], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   linear => view_1
#   split => split
#   v_1 => permute_3
#   view_2 => view_4
#   y => clone_2
# Graph fragment:
#   %addmm : Tensor "f16[65536, 2304][2304, 1]cuda:0" = PlaceHolder[target=addmm]
#   %view_1 : Tensor "f16[128, 512, 2304][1179648, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm, [128, 512, 2304]), kwargs = {})
#   %split : [num_users=3] = call_function[target=torch.ops.aten.split.Tensor](args = (%view_1, 768, 2), kwargs = {})
#   %view_4 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [128, 512, 8, 96]), kwargs = {})
#   %permute_3 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f16[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_split_transpose_view_2 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_2', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 512)
    x2 = ((xindex // 49152) % 8)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 96*x2 + 2304*x1 + 1179648*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_43_MinGPTCausalAttention_1792942112/gp/cgpwdazett7d2vzjqsvh3tbx4vqqwbbnv6hfmiuernam37rjyj7g.py
# Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   att => mul
#   att_1 => full_default, where
#   att_2 => convert_element_type_5, convert_element_type_6, div, exp_default, sub_tensor
#   eq => eq
#   getitem_3 => slice_1, slice_2
#   matmul => view_7
# Graph fragment:
#   %arg3_1 : Tensor "f16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %bmm : Tensor "f16[1024, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %getitem_3 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_4 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_4]
#   %slice_1 : Tensor "f16[1, 1, 512, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg3_1, 2, 0, 512), kwargs = {})
#   %slice_2 : Tensor "f16[1, 1, 512, 512][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 3, 0, 512), kwargs = {})
#   %eq : Tensor "b8[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%slice_2, 0), kwargs = {})
#   %full_default : Tensor "f16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_7 : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 8, 512, 512]), kwargs = {})
#   %mul : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 0.10206207261596577), kwargs = {})
#   %where : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul), kwargs = {})
#   %convert_element_type_5 : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_5, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_5, %getitem_3), kwargs = {})
#   %exp_default : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_4), kwargs = {})
#   %convert_element_type_6 : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %getitem_3,%getitem_4,%expand_2
triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_3 = async_compile.triton('triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 524288, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 1611137024}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 524288
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
    x0 = (xindex % 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tl.full([1, 1], 0.0, tl.float32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
    tmp7 = tl.where(tmp2, tmp6, tmp5)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp13 = triton_helpers.fmax2(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = tmp9 - tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp19 = tmp8 - tmp13
    tmp20 = libdevice.exp(tmp19)
    tmp21 = (tmp20 / tmp18)
    tmp22 = tmp21.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp22, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_43_MinGPTCausalAttention_1792942112/es/cesvjqh2n5lztxcpytdpwcisnwvfi6icl6oz2t73njrcoiublmjl.py
# Topologically Sorted Source Nodes: [y, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_3
#   transpose_4 => permute_5
#   y => view_10
# Graph fragment:
#   %bmm_1 : Tensor "f16[1024, 512, 96][49152, 96, 1]cuda:0" = PlaceHolder[target=bmm_1]
#   %view_10 : Tensor "f16[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm_1, [128, 8, 512, 96]), kwargs = {})
#   %permute_5 : Tensor "f16[128, 512, 8, 96][393216, 96, 49152, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1, 3]), kwargs = {})
#   %clone_3 : Tensor "f16[128, 512, 8, 96][393216, 768, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused_clone_transpose_view_4 = async_compile.triton('triton_poi_fused_clone_transpose_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 8)
    x2 = ((xindex // 768) % 512)
    x3 = xindex // 393216
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2 + 49152*x1 + 393216*x3), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
        args.clear()
        assert_size_stride_grouped((arg2_1, arg0_1, arg1_1), ((2304, ), (128, 512, 768), (2304, 768)), ((1, ), (393216, 768, 1), (768, 1)), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_if_misaligned(arg0_1)
            buf0 = empty_strided_cuda((65536, 2304), (2304, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg2_1, reinterpret_tensor(arg0_1, (65536, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf0)
            del arg0_1
            del arg1_1
            del arg2_1
            buf1 = empty_strided_cuda((128, 8, 512, 96), (393216, 49152, 96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_0.run(buf0, buf1, 50331648, stream=raw_stream0)
            buf2 = empty_strided_cuda((128, 8, 96, 512), (393216, 49152, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view, k_1, transpose_3, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_1.run(buf0, buf2, 98304, 512, stream=raw_stream0)
            buf7 = empty_strided_cuda((128, 8, 512, 96), (393216, 49152, 96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view_2, v_1, y], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_2.run(buf0, buf7, 50331648, stream=raw_stream0)
            del buf0
            buf3 = empty_strided_cuda((1024, 512, 512), (262144, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul, view, k_1, transpose_3], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf1, (1024, 512, 96), (49152, 96, 1), 0), reinterpret_tensor(buf2, (1024, 96, 512), (49152, 512, 1), 0), out=buf3)
            del buf1
            del buf2
            assert_size_stride(arg3_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), 'input')
            buf6 = reinterpret_tensor(buf3, (128, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_3.run(buf6, arg3_1, 524288, 512, stream=raw_stream0)
            del arg3_1
            buf8 = empty_strided_cuda((1024, 512, 96), (49152, 96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, getitem_3, eq, att_1, matmul, att, att_2, y, view_2, v_1], Original ATen: [aten.view, aten.split, aten.slice, aten.eq, aten.masked_fill, aten.mul, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf6, (1024, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf7, (1024, 512, 96), (49152, 96, 1), 0), out=buf8)
            del buf6
            buf9 = reinterpret_tensor(buf7, (128, 512, 8, 96), (393216, 768, 96, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_4.run(buf8, buf9, 50331648, stream=raw_stream0)
            assert_size_stride_grouped((arg5_1, arg4_1), ((768, ), (768, 768)), ((1, ), (768, 1)), 'input')
            buf10 = reinterpret_tensor(buf8, (65536, 768), (768, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous, y_1, linear_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.addmm]
            extern_kernels.addmm(arg5_1, reinterpret_tensor(buf9, (65536, 768), (768, 1), 0), reinterpret_tensor(arg4_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
            del arg4_1
            del arg5_1
            del buf9
        return (reinterpret_tensor(buf10, (128, 512, 768), (393216, 768, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
