from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    _infer_slice_output_shape_and_resolved_params,
)


def optimize_transpose_slice_prepost_nhwc_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate strict NHWC<->NCHW transpose wrappers around rank-4 SLICE.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --SLICE(begin,size)--> y_nchw
      y_nchw --TRANSPOSE(0,2,3,1)--> z_nhwc

    Rewrite:
      x_nhwc --SLICE(begin',size')--> z_nhwc

    Safety:
    - Rank-4 only.
    - SLICE begin/size must be constant tensors.
    - Intermediate transpose/slice tensors must be strict single-consumer links.
    - begin/size tensors must be used only by the target SLICE.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            slice_out_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if slice_out_name in model_outputs:
                continue

            slice_idx = producers.get(slice_out_name, None)
            if slice_idx is None:
                continue
            slice_op = model_ir.operators[int(slice_idx)]
            if str(slice_op.op_type) != "SLICE" or len(slice_op.inputs) < 3 or len(slice_op.outputs) != 1:
                continue
            if str(slice_op.outputs[0]) != slice_out_name:
                continue
            if set(int(v) for v in consumers.get(slice_out_name, [])) != {int(post_idx)}:
                continue

            begin_name = str(slice_op.inputs[1])
            size_name = str(slice_op.inputs[2])
            begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(begin_name, None))
            size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(size_name, None))
            if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
                continue
            if set(int(v) for v in consumers.get(begin_name, [])) != {int(slice_idx)}:
                continue
            if set(int(v) for v in consumers.get(size_name, [])) != {int(slice_idx)}:
                continue

            pre_out_name = str(slice_op.inputs[0])
            if pre_out_name in model_outputs:
                continue
            pre_idx = producers.get(pre_out_name, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != pre_out_name
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            if set(int(v) for v in consumers.get(pre_out_name, [])) != {int(slice_idx)}:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
            post_out_tensor = model_ir.tensors.get(post_out_name, None)
            if (
                pre_input_tensor is None
                or post_out_tensor is None
                or len(list(pre_input_tensor.shape)) != 4
                or len(list(post_out_tensor.shape)) != 4
            ):
                continue

            pre_input_shape = [int(v) for v in list(pre_input_tensor.shape)]
            post_out_shape = [int(v) for v in list(post_out_tensor.shape)]
            begin_vals_i = [int(v) for v in list(begin_vals)]
            size_vals_i = [int(v) for v in list(size_vals)]

            # Select remapped params only when they reproduce the expected NHWC post shape.
            # This prevents double-remap corruption when constants were already converted.
            as_is_shape, _, _ = _infer_slice_output_shape_and_resolved_params(
                input_shape=pre_input_shape,
                begin_vals=begin_vals_i,
                size_vals=size_vals_i,
            )
            remapped_begin = [int(begin_vals_i[int(axis)]) for axis in perm_nchw_to_nhwc]
            remapped_size = [int(size_vals_i[int(axis)]) for axis in perm_nchw_to_nhwc]
            remapped_shape, _, _ = _infer_slice_output_shape_and_resolved_params(
                input_shape=pre_input_shape,
                begin_vals=remapped_begin,
                size_vals=remapped_size,
            )

            selected_begin: Optional[List[int]] = None
            selected_size: Optional[List[int]] = None
            if as_is_shape is not None and [int(v) for v in list(as_is_shape)] == post_out_shape:
                selected_begin = begin_vals_i
                selected_size = size_vals_i
            elif remapped_shape is not None and [int(v) for v in list(remapped_shape)] == post_out_shape:
                selected_begin = remapped_begin
                selected_size = remapped_size
            else:
                continue

            _write_const_ints_to_tensor(model_ir.tensors.get(begin_name, None), selected_begin)
            _write_const_ints_to_tensor(model_ir.tensors.get(size_name, None), selected_size)
            _replace_operator_input_at(
                model_ir=model_ir,
                op=slice_op,
                input_index=0,
                new_input_name=pre_input_name,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=slice_op,
                new_outputs=[post_out_name],
            )

            for remove_idx in sorted([int(post_idx), int(pre_idx)], reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_slice_prepost_nhwc_passthrough_chains": int(optimized)}
