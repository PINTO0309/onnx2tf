from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, normalize_onnx_shape


def _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove boundary NHWC->NCHW input transpose for a strict normalization prelude.

    Target:
      input_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --MUL(const[1,C,1,1]|scalar)--> y_nchw
      y_nchw --SUM(axis includes channel=1, keepDims=True)--> z_nchw
      z_nchw --RESHAPE([N,H,W,C'])--> z_nhwc

    Rewrite:
      input_nhwc --MUL(const_nhwc|scalar)--> y_nhwc
      y_nhwc --SUM(mapped_axes, keepDims=True)--> z_nhwc
      z_nhwc --RESHAPE(...)--> z_nhwc
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    def _normalize_axis(axis: int, rank: int) -> Optional[int]:
        try:
            axis_int = int(axis)
        except Exception:
            return None
        if axis_int < 0:
            axis_int += int(rank)
        if axis_int < 0 or axis_int >= int(rank):
            return None
        return int(axis_int)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
            ):
                continue

            input_name = str(pre_op.inputs[0])
            internal_name = str(pre_op.outputs[0])
            if input_name not in model_inputs:
                continue
            if input_name in model_outputs or internal_name in model_outputs:
                continue
            if not str(internal_name).endswith("_onnx_ncx_internal"):
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            internal_users = [int(v) for v in consumers.get(internal_name, [])]
            if len(internal_users) != 1:
                continue
            mul_idx = int(internal_users[0])
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            if str(mul_inputs[0]) == internal_name:
                mul_data_input_index = 0
                mul_const_name = str(mul_inputs[1])
            elif str(mul_inputs[1]) == internal_name:
                mul_data_input_index = 1
                mul_const_name = str(mul_inputs[0])
            else:
                continue

            mul_const_tensor = model_ir.tensors.get(str(mul_const_name), None)
            if mul_const_tensor is None or mul_const_tensor.data is None:
                continue
            mul_const_array = np.asarray(mul_const_tensor.data)
            if int(mul_const_array.size) != 1:
                if mul_const_array.ndim != 4:
                    continue
                const_shape = [int(v) for v in list(mul_const_array.shape)]
                if not (
                    int(const_shape[0]) == 1
                    and int(const_shape[2]) == 1
                    and int(const_shape[3]) == 1
                ):
                    continue

            mul_out_name = str(mul_op.outputs[0])
            if mul_out_name in model_outputs:
                continue
            mul_out_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_out_users) != 1:
                continue
            sum_idx = int(mul_out_users[0])
            sum_op = model_ir.operators[int(sum_idx)]
            if str(sum_op.op_type) != "SUM" or len(sum_op.inputs) < 2 or len(sum_op.outputs) != 1:
                continue
            if str(sum_op.inputs[0]) != mul_out_name:
                continue
            if not bool(sum_op.options.get("keepDims", False)):
                continue
            sum_axes_tensor = model_ir.tensors.get(str(sum_op.inputs[1]), None)
            sum_axes_vals = _read_const_ints_from_tensor(sum_axes_tensor)
            if sum_axes_vals is None or len(sum_axes_vals) == 0:
                continue

            sum_out_name = str(sum_op.outputs[0])
            if sum_out_name in model_outputs:
                continue
            sum_out_users = [int(v) for v in consumers.get(sum_out_name, [])]
            if len(sum_out_users) != 1:
                continue
            reshape_idx = int(sum_out_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != sum_out_name
            ):
                continue

            mapped_axes: List[int] = []
            axes_ok = True
            includes_channel_axis = False
            for axis in list(sum_axes_vals):
                norm_axis = _normalize_axis(int(axis), 4)
                if norm_axis is None:
                    axes_ok = False
                    break
                if int(norm_axis) == 1:
                    includes_channel_axis = True
                mapped_axes.append(int(perm_nhwc_to_nchw[int(norm_axis)]))
            if not axes_ok or not includes_channel_axis:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul_op,
                input_index=int(mul_data_input_index),
                new_input_name=str(input_name),
            )

            if int(mul_const_array.size) != 1:
                transposed_const = np.transpose(mul_const_array, axes=perm_nchw_to_nhwc)
                mul_const_tensor.data = np.asarray(transposed_const)
                const_shape, const_signature = normalize_onnx_shape(list(transposed_const.shape))
                mul_const_tensor.shape = [int(v) for v in list(const_shape)]
                mul_const_tensor.shape_signature = [int(v) for v in list(const_signature)]

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(sum_out_name), None),
                perm_nchw_to_nhwc,
            )

            _write_const_ints_to_tensor(sum_axes_tensor, [int(v) for v in list(mapped_axes)])
            if isinstance(sum_op.options, dict):
                sum_opts = dict(sum_op.options)
                for key in ["axis", "axes", "onnxRawAxes"]:
                    value = sum_opts.get(key, None)
                    if isinstance(value, list) and len(value) == len(mapped_axes):
                        sum_opts[key] = [int(v) for v in list(mapped_axes)]
                sum_op.options = sum_opts

            del model_ir.operators[int(pre_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_boundary_input_transpose_mul_sum_reshape_nhwc_chains": int(rewritten)}


def _optimize_boundary_input_transpose_batchmatmul_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Elide input-boundary transpose wrappers that only feed BATCH_MATMUL.

    Target:
      X_in(external layout) --TRANSPOSE(external->internal)--> X_internal
      BATCH_MATMUL(..., X_internal, ...)

    Rewrite:
      BATCH_MATMUL(..., X_in, ...)

    Safety:
    - Leading transpose input must be a model input tensor.
    - Model input tensor must be consumed only by that transpose.
    - Transpose output must be consumed only by BATCH_MATMUL ops.
    - Transpose must be one of NCX boundary permutations inserted by input
      layout adaptation (rank 3/4/5).
    """
    rewritten = 0
    ncx_boundary_perms = {
        (0, 2, 1),
        (0, 3, 1, 2),
        (0, 4, 1, 2, 3),
    }

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
            ):
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_input_name not in model_inputs:
                continue
            if pre_output_name in model_outputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            if tuple(int(v) for v in list(perm_pre)) not in ncx_boundary_perms:
                continue

            pre_input_users = [int(v) for v in consumers.get(pre_input_name, [])]
            if set(pre_input_users) != {int(pre_idx)}:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) == 0:
                continue
            if not all(
                str(model_ir.operators[int(user_idx)].op_type) == "BATCH_MATMUL"
                for user_idx in pre_users
            ):
                continue

            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=user_op,
                    new_inputs=[
                        pre_input_name if str(v) == pre_output_name else str(v)
                        for v in list(user_op.inputs)
                    ],
                )

            input_tensor = model_ir.tensors.get(pre_input_name, None)
            internal_tensor = model_ir.tensors.get(pre_output_name, None)
            if input_tensor is not None and internal_tensor is not None:
                if internal_tensor.shape is not None:
                    input_tensor.shape = [int(v) for v in list(internal_tensor.shape)]
                if internal_tensor.shape_signature is not None:
                    input_tensor.shape_signature = [
                        int(v) for v in list(internal_tensor.shape_signature)
                    ]

            del model_ir.operators[int(pre_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_boundary_input_transpose_batchmatmul_chains": int(rewritten)}
