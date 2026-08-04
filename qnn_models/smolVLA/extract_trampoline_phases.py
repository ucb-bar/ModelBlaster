"""Extract trampoline-phase ONNX models from each v3 DSP segment.

A DSP segment like dsp_seg_01 contains 2 Conv ops (the heavy compute the HTA
handles) and ~14 non-Conv ops (LayerNorm, Add, Mul, Pow, Transpose, Reshape,
Split, MatMul — the "trampolines" that stay on CPU). When the scheduler
routes the segment to HTA, the runtime must execute:

   trampoline_phase_0 → Conv1 (HTA) → trampoline_phase_1 → Conv2 (HTA) → trampoline_phase_2

This script extracts each trampoline_phase as a standalone ONNX. For a segment
with N Conv ops it produces N+1 phase ONNX files. The phases together cover
every non-Conv op in the original segment; their I/O is wired so the runtime
can chain them with the HTA conv calls.

Phase i contains the ops strictly between Conv (i-1) and Conv i (Conv -1
is "before the first conv", Conv N is "after the last conv").

Inputs of phase i = (tensors consumed by phase ops) - (tensors produced by
phase ops or by prior phases internally) + (initializers stay as initializers).
External inputs include:
  - original segment inputs (if first consumed in this phase)
  - prior conv outputs (these enter the phase from outside)
  - prior phase outputs that haven't been consumed yet (handled by the runtime's
    tensor cache, same mechanism as the cross-segment handoff)

Outputs of phase i = tensors produced by phase ops that are either:
  - consumed by a later Conv (those become Conv inputs)
  - consumed by a later phase
  - final segment outputs

Usage:
   python extract_trampoline_phases.py \\
       --slices-dir vision_slices_v3/conv1x1 \\
       --out-dir vision_slices_v3/trampolines

For each dsp_seg_XX.onnx, emits:
   vision_slices_v3/trampolines/dsp_seg_XX_tramp_p0.onnx   (pre-conv1)
   vision_slices_v3/trampolines/dsp_seg_XX_tramp_p1.onnx   (between conv1 and conv2)
   vision_slices_v3/trampolines/dsp_seg_XX_tramp_p2.onnx   (post-conv2)
   vision_slices_v3/trampolines/dsp_seg_XX_phases.json     (phase metadata for the runtime)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper, numpy_helper, shape_inference, TensorProto


def _tensor_shape(model_or_graph, name: str) -> list[int] | None:
    """Find the shape of `name` from value_info, graph inputs/outputs, or
    initializers. Returns a list of dim sizes or None if not found."""
    graph = model_or_graph.graph if hasattr(model_or_graph, 'graph') else model_or_graph
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                dims.append(d.dim_value if d.dim_value > 0 else None)
            return dims
    for init in graph.initializer:
        if init.name == name:
            return list(init.dims)
    return None


def _tensor_dtype(model_or_graph, name: str) -> int:
    """Find tensor dtype (TensorProto.X). Default FLOAT."""
    graph = model_or_graph.graph if hasattr(model_or_graph, 'graph') else model_or_graph
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            return vi.type.tensor_type.elem_type
    for init in graph.initializer:
        if init.name == name:
            return init.data_type
    return TensorProto.FLOAT


def _make_tvi(name: str, dtype: int, shape: list[int] | None):
    """Make a TensorValueInfo with concrete shape (substitute 1 for unknowns)."""
    if shape is None:
        shape = []
    shape = [d if d is not None and d > 0 else 1 for d in shape]
    return helper.make_tensor_value_info(name, dtype, shape)


def extract_phases(seg_onnx_path: str, out_dir: str) -> dict:
    """Split a DSP segment into N+1 trampoline-phase ONNX models.

    Returns a dict describing the phases (for the runtime's metadata file).
    """
    model = onnx.load(seg_onnx_path)
    # Run shape inference so internal tensors get value_info entries — we
    # need shape info to declare phase I/O tensors properly.
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"  warn: shape_inference failed for {seg_onnx_path}: {e}")

    graph = model.graph
    seg_name = Path(seg_onnx_path).stem  # e.g. "dsp_seg_01"

    nodes = list(graph.node)
    conv_indices = [i for i, n in enumerate(nodes) if n.op_type == 'Conv']
    if not conv_indices:
        print(f"  skip {seg_name}: no Conv nodes (CPU-only segment)")
        return {}

    # Phase ranges: [0..conv0), (conv0..conv1), ..., (convN-1..end]
    phase_ranges = []
    prev = 0
    for ci in conv_indices:
        phase_ranges.append((prev, ci))  # ops up to but not including conv
        prev = ci + 1
    phase_ranges.append((prev, len(nodes)))  # ops after last conv

    init_map = {init.name: init for init in graph.initializer}
    seg_input_names = {inp.name for inp in graph.input}
    seg_output_names = {out.name for out in graph.output}

    # All tensors produced by each node (so we can identify external inputs).
    node_outputs = {}
    for n in nodes:
        for o in n.output:
            node_outputs[o] = n

    # For each phase, figure out which tensors are produced inside, which
    # are consumed but produced elsewhere, etc.
    phases_meta = []
    for phase_i, (start, end) in enumerate(phase_ranges):
        phase_nodes = nodes[start:end]
        produced_in_phase = set()
        for n in phase_nodes:
            for o in n.output:
                produced_in_phase.add(o)
        consumed_in_phase = set()
        for n in phase_nodes:
            for inp in n.input:
                if inp:  # skip empty (optional)
                    consumed_in_phase.add(inp)
        # External inputs = consumed but not produced internally, and not a
        # constant initializer. We'll include initializers that ARE used
        # as graph initializers in the phase ONNX.
        ext_inputs = consumed_in_phase - produced_in_phase - set(init_map.keys())
        used_inits = consumed_in_phase & set(init_map.keys())

        # Outputs of this phase = tensors produced in this phase that are:
        #   - the input to a future Conv (-> bridge to the conv), OR
        #   - consumed by a future phase (any later node that's not in this phase), OR
        #   - a final segment output
        downstream_consumers = defaultdict(list)
        for j, n in enumerate(nodes):
            if j < end:
                continue
            for inp in n.input:
                if inp:
                    downstream_consumers[inp].append(j)
        phase_outputs = set()
        for t in produced_in_phase:
            if t in seg_output_names:
                phase_outputs.add(t)
            elif downstream_consumers.get(t):
                phase_outputs.add(t)

        # Determine if this phase has any actual work
        empty = (len(phase_nodes) == 0)
        phases_meta.append({
            'phase_idx': phase_i,
            'start': start, 'end': end,
            'op_count': len(phase_nodes),
            'ops': [(n.op_type, n.name) for n in phase_nodes],
            'inputs': sorted(ext_inputs),
            'outputs': sorted(phase_outputs),
            'initializers': sorted(used_inits),
            'empty': empty,
        })

    # Build and save each phase ONNX
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for pm in phases_meta:
        if pm['empty']:
            saved.append({'phase_idx': pm['phase_idx'], 'file': None,
                            'inputs': [], 'outputs': [], 'op_count': 0})
            continue
        phase_nodes = nodes[pm['start']:pm['end']]
        # Build input ValueInfos
        new_inputs = []
        for nm in pm['inputs']:
            dt = _tensor_dtype(model, nm)
            sh = _tensor_shape(model, nm)
            new_inputs.append(_make_tvi(nm, dt, sh))
        # Build output ValueInfos
        new_outputs = []
        for nm in pm['outputs']:
            dt = _tensor_dtype(model, nm)
            sh = _tensor_shape(model, nm)
            new_outputs.append(_make_tvi(nm, dt, sh))
        # Carry the used initializers (weights, biases, constants)
        new_inits = [init_map[nm] for nm in pm['initializers']]

        new_graph = helper.make_graph(
            phase_nodes,
            f'{seg_name}_tramp_p{pm["phase_idx"]}',
            inputs=new_inputs,
            outputs=new_outputs,
            initializer=new_inits,
        )
        new_model = helper.make_model(new_graph,
            opset_imports=[helper.make_opsetid('', 17)],
            producer_name='extract_trampoline_phases')
        new_model.ir_version = 8

        fname = f'{seg_name}_tramp_p{pm["phase_idx"]}.onnx'
        out_path = os.path.join(out_dir, fname)
        try:
            onnx.checker.check_model(new_model, full_check=False)
        except Exception as e:
            print(f"  warn: checker failed for {fname}: {e}")
        onnx.save(new_model, out_path)
        saved.append({
            'phase_idx': pm['phase_idx'],
            'file': fname,
            'inputs': pm['inputs'],
            'outputs': pm['outputs'],
            'op_count': pm['op_count'],
            'op_types': sorted({op for op, _ in pm['ops']}),
        })

    # Write phase metadata next to the ONNX files
    meta = {
        'segment': seg_name,
        'n_convs': len(conv_indices),
        'conv_node_names': [nodes[i].name for i in conv_indices],
        'phases': saved,
    }
    meta_path = os.path.join(out_dir, f'{seg_name}_phases.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--slices-dir', required=True,
                    help='vision_slices_v3/conv1x1 (rewritten DSP segments)')
    ap.add_argument('--out-dir', required=True,
                    help='where to write the trampoline-phase ONNX files')
    args = ap.parse_args()

    slices_dir = Path(args.slices_dir)
    out_dir = Path(args.out_dir)

    all_meta = []
    for onnx_file in sorted(slices_dir.glob('dsp_seg_*.onnx')):
        seg_name = onnx_file.stem
        print(f'Extracting trampoline phases for {seg_name}...')
        meta = extract_phases(str(onnx_file), str(out_dir))
        if meta:
            all_meta.append(meta)
            # Brief summary per segment
            for ph in meta['phases']:
                if ph['file'] is None:
                    print(f'  phase {ph["phase_idx"]}: (empty)')
                else:
                    print(f'  phase {ph["phase_idx"]}: {ph["op_count"]} ops '
                          f'[{",".join(ph.get("op_types", []))}] '
                          f'in={len(ph["inputs"])} out={len(ph["outputs"])}')

    # Global summary
    summary_path = out_dir / 'all_phases.json'
    with open(summary_path, 'w') as f:
        json.dump(all_meta, f, indent=2)
    print(f'\nWrote {len(all_meta)} segment phase manifests + global summary at {summary_path}')

    # Op-type breakdown across all phases
    from collections import Counter
    op_counter = Counter()
    for m in all_meta:
        for ph in m['phases']:
            for op in ph.get('op_types', []):
                op_counter[op] += 1
    print('\nOp types appearing in trampoline phases:')
    for op, cnt in op_counter.most_common():
        print(f'  {op}: {cnt} phases')


if __name__ == '__main__':
    main()
