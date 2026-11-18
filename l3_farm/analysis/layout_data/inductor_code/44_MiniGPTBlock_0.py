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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/dl/cdlepb4rxozjuyvsrtjkyexdv3fvlsbonnuluwuy2fgmj7p2zp3g.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, convert_element_type, convert_element_type_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %arg2_1 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %getitem_1 : Tensor "f32[128, 512, 1][512, 1, 65536]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[128, 512, 1][512, 1, 65536]cuda:0" = PlaceHolder[target=buf1]
#   %arg0_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convert_element_type : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_1), kwargs = {})
#   %add : Tensor "f32[128, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : Tensor "f32[128, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %arg0_1), kwargs = {})
#   %add_1 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %arg1_1), kwargs = {})
#   %convert_element_type_1 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_1, torch.float16), kwargs = {})
#   return %getitem_1,%buf1,%convert_element_type_1
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 301992960}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = (tl.full([1, 1], 768, tl.int32)).to(tl.float32)
    tmp10 = (tmp8 / tmp9)
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp1 - tmp10
    tmp18 = tl.full([1, 1], 768.0, tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tl.full([1, 1], 1e-05, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 * tmp25
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp30, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/lc/clcuf3rblqzk4bzdvbvvjin72jo2dl4ddhhgutxbfgrzosyy375j.py
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
#   %view_3 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_2, [128, 512, 8, 96]), kwargs = {})
#   %permute_2 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_3, [0, 2, 1, 3]), kwargs = {})
#   %clone : Tensor "f16[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone
triton_poi_fused_clone_split_transpose_view_1 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/yp/cypdf4fneitfeuf64qagjkz5rmceavfctgxb3uwqm6cbphbjeiby.py
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
#   %view_2 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_3, [128, 512, 8, 96]), kwargs = {})
#   %permute_1 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_2, [0, 2, 1, 3]), kwargs = {})
#   %permute_4 : Tensor "f16[128, 8, 96, 512][1179648, 96, 1, 2304]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%permute_1, [0, 1, 3, 2]), kwargs = {})
#   %clone_1 : Tensor "f16[128, 8, 96, 512][393216, 49152, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_1
triton_poi_fused_clone_split_transpose_view_2 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_2', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 100663296, 'x': 201326592}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/o2/co2fhh2bn6klazuvtjnnfdabe7rzc22ikilq6jap3c22ee2sofar.py
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
#   %view_4 : Tensor "f16[128, 512, 8, 96][1179648, 2304, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_4, [128, 512, 8, 96]), kwargs = {})
#   %permute_3 : Tensor "f16[128, 8, 512, 96][1179648, 96, 2304, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_4, [0, 2, 1, 3]), kwargs = {})
#   %clone_2 : Tensor "f16[128, 8, 512, 96][393216, 49152, 96, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused_clone_split_transpose_view_3 = async_compile.triton('triton_poi_fused_clone_split_transpose_view_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_split_transpose_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_split_transpose_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/oj/cojkj5cm7crtgr6r43wdxxg2i37uudx7hnryqlcskfxbf5xadmho.py
# Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, aten._softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   att => mul_2
#   att_1 => full_default, where
#   att_2 => convert_element_type_7, convert_element_type_8, div, exp_default, sub_tensor
#   eq => eq
#   getitem_3 => slice_1, slice_2
#   matmul => view_7
# Graph fragment:
#   %arg5_1 : Tensor "f16[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = PlaceHolder[target=arg5_1]
#   %bmm : Tensor "f16[1024, 512, 512][262144, 512, 1]cuda:0" = PlaceHolder[target=bmm]
#   %getitem_7 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_7]
#   %getitem_8 : Tensor "f32[128, 8, 512, 1][4096, 512, 1, 524288]cuda:0" = PlaceHolder[target=getitem_8]
#   %slice_1 : Tensor "f16[1, 1, 512, 1024][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%arg5_1, 2, 0, 512), kwargs = {})
#   %slice_2 : Tensor "f16[1, 1, 512, 512][1048576, 1048576, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%slice_1, 3, 0, 512), kwargs = {})
#   %eq : Tensor "b8[1, 1, 512, 512][262144, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%slice_2, 0), kwargs = {})
#   %full_default : Tensor "f16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %view_7 : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bmm, [128, 8, 512, 512]), kwargs = {})
#   %mul_2 : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, 0.10206207261596577), kwargs = {})
#   %where : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %mul_2), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%where, torch.float32), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_7, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_7, %getitem_7), kwargs = {})
#   %exp_default : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div : Tensor "f32[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_8), kwargs = {})
#   %convert_element_type_8 : Tensor "f16[128, 8, 512, 512][2097152, 262144, 512, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float16), kwargs = {})
#   return %getitem_7,%getitem_8,%expand_2
triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_4 = async_compile.triton('triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_4', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 1611137024}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_4(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/hs/chsk4ynwsabe4ksry75njvfqcntdp3kxfmbz73yfgzbqwajps2zl.py
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
triton_poi_fused_clone_transpose_view_5 = async_compile.triton('triton_poi_fused_clone_transpose_view_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_transpose_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 301989888}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_transpose_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/df/cdftku73u6ykln3htxv6g6wg3bzcfkqe2exf6frashjjz4ybfawl.py
# Topologically Sorted Source Nodes: [linear_1, x, layer_norm_1], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_3, add_4, convert_element_type_14, convert_element_type_15, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
#   linear_1 => view_13
#   x => add_2
# Graph fragment:
#   %arg2_1 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %addmm_1 : Tensor "f16[65536, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %getitem_6 : Tensor "f32[128, 512, 1][512, 1, 65536]cuda:0" = PlaceHolder[target=getitem_6]
#   %buf16 : Tensor "f32[128, 512, 1][512, 1, 65536]cuda:0" = PlaceHolder[target=buf16]
#   %arg8_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f16[768][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %view_13 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 512, 768]), kwargs = {})
#   %add_2 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_13), kwargs = {})
#   %convert_element_type_14 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_14, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_14, %getitem_6), kwargs = {})
#   %add_3 : Tensor "f32[128, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_5, 1e-05), kwargs = {})
#   %rsqrt_1 : Tensor "f32[128, 512, 1][512, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_3 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %arg8_1), kwargs = {})
#   %add_4 : Tensor "f32[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %arg9_1), kwargs = {})
#   %convert_element_type_15 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_4, torch.float16), kwargs = {})
#   return %getitem_6,%buf16,%convert_element_type_15
triton_per_fused_add_native_layer_norm_view_6 = async_compile.triton('triton_per_fused_add_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 65536, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'in_ptr2': '*fp16', 'in_ptr3': '*fp16', 'out_ptr2': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_native_layer_norm_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': set(), 'tiling_scores': {'x': 0, 'r0_': 402656256}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r0_1 + 768*x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = (tl.full([1, 1], 768, tl.int32)).to(tl.float32)
    tmp12 = (tmp10 / tmp11)
    tmp13 = tmp4 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = tmp3 - tmp12
    tmp20 = tl.full([1, 1], 768.0, tl.float32)
    tmp21 = (tmp18 / tmp20)
    tmp22 = tl.full([1, 1], 1e-05, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp19 * tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp25 * tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp32, r0_mask)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/fe/cfexylxd3y5ebc6uay2cjqt7er7jcenghynko5onz4dsqm3r2u55.py
# Topologically Sorted Source Nodes: [linear_2, mul_1, pow_1, mul_2, add_1, mul_3, tanh, add_2, mul_4], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
# Source node to ATen node mapping:
#   add_1 => add_5
#   add_2 => add_6
#   linear_2 => view_15
#   mul_1 => mul_5
#   mul_2 => mul_6
#   mul_3 => mul_7
#   mul_4 => mul_8
#   pow_1 => pow_1
#   tanh => tanh
# Graph fragment:
#   %addmm_2 : Tensor "f16[65536, 3072][3072, 1]cuda:0" = PlaceHolder[target=addmm_2]
#   %view_15 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_2, [128, 512, 3072]), kwargs = {})
#   %mul_5 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, 0.5), kwargs = {})
#   %pow_1 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_15, 3.0), kwargs = {})
#   %mul_6 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%pow_1, 0.044715), kwargs = {})
#   %add_5 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, %mul_6), kwargs = {})
#   %mul_7 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.7978845608028654), kwargs = {})
#   %tanh : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_7,), kwargs = {})
#   %add_6 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1.0), kwargs = {})
#   %mul_8 : Tensor "f16[128, 512, 3072][1572864, 3072, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_6), kwargs = {})
#   return %mul_8
triton_poi_fused_add_mul_pow_tanh_view_7 = async_compile.triton('triton_poi_fused_add_mul_pow_tanh_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_mul_pow_tanh_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1207959552}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_tanh_view_7(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201326592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.full([1], 0.5, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp3 * tmp0
    tmp5 = tl.full([1], 0.044715, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp0 + tmp6
    tmp8 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tmp11 = tl.full([1], 1.0, tl.float32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp2 * tmp12
    tl.store(in_out_ptr0 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: /tmp/krisd/layout/cache/ind_44_MiniGPTBlock_1792942175/sd/csd7qn453lyoccf2iov32xzyxuikskhqx2vgtyjbkampcnvbum5w.py
# Topologically Sorted Source Nodes: [linear_1, x, linear_3, x_1], Original ATen: [aten.view, aten.add]
# Source node to ATen node mapping:
#   linear_1 => view_13
#   linear_3 => view_17
#   x => add_2
#   x_1 => add_7
# Graph fragment:
#   %arg2_1 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %addmm_1 : Tensor "f16[65536, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_1]
#   %addmm_3 : Tensor "f16[65536, 768][768, 1]cuda:0" = PlaceHolder[target=addmm_3]
#   %view_13 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_1, [128, 512, 768]), kwargs = {})
#   %add_2 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %view_13), kwargs = {})
#   %view_17 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%addmm_3, [128, 512, 768]), kwargs = {})
#   %add_7 : Tensor "f16[128, 512, 768][393216, 768, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_17), kwargs = {})
#   return %add_7
triton_poi_fused_add_view_8 = async_compile.triton('triton_poi_fused_add_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=142, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_view_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 503316480}, 'backend_hash': '1BF60CE8B02430F2377FD835145761CB03133B83A071E6D5757BC4D0808DDE95', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_view_8(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
        args.clear()
        assert_size_stride_grouped((arg2_1, arg0_1, arg1_1), ((128, 512, 768), (768, ), (768, )), ((393216, 768, 1), (1, ), (1, )), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg2_1 = copy_if_misaligned(arg2_1)
            buf3 = empty_strided_cuda((128, 512, 768), (393216, 768, 1), torch.float16)
            # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_native_layer_norm_0.run(arg2_1, arg0_1, arg1_1, buf3, 65536, 768, stream=raw_stream0)
            del arg0_1
            del arg1_1
            assert_size_stride_grouped((arg4_1, arg3_1), ((2304, ), (2304, 768)), ((1, ), (768, 1)), 'input')
            buf4 = empty_strided_cuda((65536, 2304), (2304, 1), torch.float16)
            # Topologically Sorted Source Nodes: [layer_norm, linear], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
            extern_kernels.addmm(arg4_1, reinterpret_tensor(buf3, (65536, 768), (768, 1), 0), reinterpret_tensor(arg3_1, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf4)
            del arg3_1
            del arg4_1
            buf5 = reinterpret_tensor(buf3, (128, 8, 512, 96), (393216, 49152, 96, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_1.run(buf4, buf5, 50331648, stream=raw_stream0)
            buf6 = empty_strided_cuda((128, 8, 96, 512), (393216, 49152, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view, k_1, transpose_3, matmul], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_2.run(buf4, buf6, 98304, 512, stream=raw_stream0)
            buf11 = empty_strided_cuda((128, 8, 512, 96), (393216, 49152, 96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view_2, v_1, y], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_split_transpose_view_3.run(buf4, buf11, 50331648, stream=raw_stream0)
            del buf4
            buf7 = empty_strided_cuda((1024, 512, 512), (262144, 512, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, view_1, q_1, matmul, view, k_1, transpose_3], Original ATen: [aten.view, aten.split, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf5, (1024, 512, 96), (49152, 96, 1), 0), reinterpret_tensor(buf6, (1024, 96, 512), (49152, 512, 1), 0), out=buf7)
            del buf5
            del buf6
            assert_size_stride(arg5_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), 'input')
            buf10 = reinterpret_tensor(buf7, (128, 8, 512, 512), (2097152, 262144, 512, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [getitem_3, eq, att_1, matmul, att, att_2], Original ATen: [aten.slice, aten.eq, aten.masked_fill, aten.view, aten.mul, aten._softmax, prims.prepare_softmax_online]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused__softmax_eq_masked_fill_mul_prepare_softmax_online_slice_view_4.run(buf10, arg5_1, 524288, 512, stream=raw_stream0)
            del arg5_1
            buf12 = empty_strided_cuda((1024, 512, 96), (49152, 96, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear, split, getitem_3, eq, att_1, matmul, att, att_2, y, view_2, v_1], Original ATen: [aten.view, aten.split, aten.slice, aten.eq, aten.masked_fill, aten.mul, aten._softmax, aten.transpose, aten.clone, aten._unsafe_view, aten.bmm]
            extern_kernels.bmm(reinterpret_tensor(buf10, (1024, 512, 512), (262144, 512, 1), 0), reinterpret_tensor(buf11, (1024, 512, 96), (49152, 96, 1), 0), out=buf12)
            del buf10
            buf13 = reinterpret_tensor(buf11, (128, 512, 8, 96), (393216, 768, 96, 1), 0); del buf11  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous], Original ATen: [aten.view, aten.transpose, aten.clone]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_clone_transpose_view_5.run(buf12, buf13, 50331648, stream=raw_stream0)
            assert_size_stride_grouped((arg7_1, arg6_1), ((768, ), (768, 768)), ((1, ), (768, 1)), 'input')
            buf14 = reinterpret_tensor(buf12, (65536, 768), (768, 1), 0); del buf12  # reuse
            # Topologically Sorted Source Nodes: [y, transpose_4, contiguous, y_1, linear_1], Original ATen: [aten.view, aten.transpose, aten.clone, aten.t, aten.addmm]
            extern_kernels.addmm(arg7_1, reinterpret_tensor(buf13, (65536, 768), (768, 1), 0), reinterpret_tensor(arg6_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf14)
            del arg6_1
            del arg7_1
            assert_size_stride_grouped((arg8_1, arg9_1), ((768, ), (768, )), ((1, ), (1, )), 'input')
            buf18 = reinterpret_tensor(buf13, (128, 512, 768), (393216, 768, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [linear_1, x, layer_norm_1], Original ATen: [aten.view, aten.add, aten.native_layer_norm]
            raw_stream0 = get_raw_stream(0)
            triton_per_fused_add_native_layer_norm_view_6.run(arg2_1, buf14, arg8_1, arg9_1, buf18, 65536, 768, stream=raw_stream0)
            del arg8_1
            del arg9_1
            assert_size_stride_grouped((arg11_1, arg10_1), ((3072, ), (3072, 768)), ((1, ), (768, 1)), 'input')
            buf19 = empty_strided_cuda((65536, 3072), (3072, 1), torch.float16)
            # Topologically Sorted Source Nodes: [linear_1, x, layer_norm_1, linear_2], Original ATen: [aten.view, aten.add, aten.native_layer_norm, aten.t, aten.addmm]
            extern_kernels.addmm(arg11_1, reinterpret_tensor(buf18, (65536, 768), (768, 1), 0), reinterpret_tensor(arg10_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf19)
            del arg10_1
            del arg11_1
            buf20 = reinterpret_tensor(buf19, (128, 512, 3072), (1572864, 3072, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [linear_2, mul_1, pow_1, mul_2, add_1, mul_3, tanh, add_2, mul_4], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_pow_tanh_view_7.run(buf20, 201326592, stream=raw_stream0)
            assert_size_stride_grouped((arg13_1, arg12_1), ((768, ), (768, 3072)), ((1, ), (3072, 1)), 'input')
            buf21 = reinterpret_tensor(buf18, (65536, 768), (768, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [linear_2, mul_1, pow_1, mul_2, add_1, mul_3, tanh, add_2, mul_4, linear_3], Original ATen: [aten.view, aten.mul, aten.pow, aten.add, aten.tanh, aten.t, aten.addmm]
            extern_kernels.addmm(arg13_1, reinterpret_tensor(buf20, (65536, 3072), (3072, 1), 0), reinterpret_tensor(arg12_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf21)
            del arg12_1
            del arg13_1
            del buf20
            buf22 = reinterpret_tensor(buf14, (128, 512, 768), (393216, 768, 1), 0); del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_1, x, linear_3, x_1], Original ATen: [aten.view, aten.add]
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_add_view_8.run(buf22, arg2_1, buf21, 50331648, stream=raw_stream0)
            del arg2_1
            del buf21
        return (buf22, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
