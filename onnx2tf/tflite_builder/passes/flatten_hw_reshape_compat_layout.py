from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.flatten_hw_reshape_layout import (
    optimize_transpose_flatten_hw_reshape_nhwc_chains as _optimize_transpose_flatten_hw_reshape_nhwc_chains_pass,
)


def optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Collapse NHWC->NCHW->RESHAPE->TRANSPOSE flatten wrappers into one RESHAPE.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --RESHAPE([N,C,H*W])--> r
      r      --TRANSPOSE(0,2,1)--> y  (shape [N,H*W,C])

    Rewrite:
      x_nhwc --RESHAPE([N,H*W,C])--> y
    """
    indexed_stats = _optimize_transpose_flatten_hw_reshape_nhwc_chains_pass(
        model_ir,
        graph_index=ModelIRGraphIndex(model_ir),
        layout_state=layout_state,
    )
    rewritten = int(
        indexed_stats.get(
            "optimized_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",
            0,
        )
    )
    tensors_before_fallback_prune = (
        set(str(name) for name in model_ir.tensors)
        if layout_state is not None
        else set()
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_ncw_to_nwc = [0, 2, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            pre_users = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(pre_users) != 1:
                continue
            reshape_idx = int(pre_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != pre_out_name
            ):
                continue

            reshape_out_name = str(reshape_op.outputs[0])
            reshape_users = [int(v) for v in consumers.get(reshape_out_name, [])]
            if len(reshape_users) != 1:
                continue
            post_idx = int(reshape_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != reshape_out_name
            ):
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_ncw_to_nwc:
                continue

            post_out_name = str(post_op.outputs[0])
            if pre_out_name in model_outputs or reshape_out_name in model_outputs:
                continue

            in_tensor = model_ir.tensors.get(pre_in_name, None)
            pre_out_tensor = model_ir.tensors.get(pre_out_name, None)
            reshape_out_tensor = model_ir.tensors.get(reshape_out_name, None)
            post_out_tensor = model_ir.tensors.get(post_out_name, None)
            if (
                in_tensor is None
                or pre_out_tensor is None
                or reshape_out_tensor is None
                or post_out_tensor is None
            ):
                continue

            if (
                not _is_fully_known_positive_shape(in_tensor.shape)
                or not _is_fully_known_positive_shape(pre_out_tensor.shape)
                or not _is_fully_known_positive_shape(reshape_out_tensor.shape)
                or not _is_fully_known_positive_shape(post_out_tensor.shape)
            ):
                continue

            in_shape = [int(v) for v in list(in_tensor.shape)]
            pre_out_shape = [int(v) for v in list(pre_out_tensor.shape)]
            reshape_out_shape = [int(v) for v in list(reshape_out_tensor.shape)]
            post_out_shape = [int(v) for v in list(post_out_tensor.shape)]
            if len(in_shape) != 4 or len(pre_out_shape) != 4 or len(reshape_out_shape) != 3 or len(post_out_shape) != 3:
                continue

            n, h, w, c = in_shape
            if pre_out_shape != [n, c, h, w]:
                continue
            hw = int(h) * int(w)
            if reshape_out_shape != [n, c, hw]:
                continue
            if post_out_shape != [n, hw, c]:
                continue

            shape_tensor = model_ir.tensors.get(str(reshape_op.inputs[1]), None)
            shape_name = str(reshape_op.inputs[1])
            if (
                shape_name in model_outputs
                or shape_name in model_inputs
                or shape_tensor is None
                or bool(shape_tensor.is_variable)
                or any(
                    shape_name == str(output_name)
                    for candidate_op in model_ir.operators
                    for output_name in candidate_op.outputs
                )
                or set(int(value) for value in consumers.get(shape_name, []))
                != {int(reshape_idx)}
                or _read_const_ints_from_tensor(shape_tensor) is None
            ):
                continue
            target_shape = [int(n), int(hw), int(c)]
            if not _write_const_ints_to_tensor(shape_tensor, target_shape):
                continue

            if isinstance(reshape_op.options, dict):
                reshape_opts = dict(reshape_op.options)
                for key in ["newShape", "onnxRawNewShape"]:
                    if isinstance(reshape_opts.get(key, None), list):
                        reshape_opts[key] = [int(v) for v in target_shape]
                reshape_op.options = reshape_opts

            _set_operator_inputs(
                model_ir=model_ir,
                op=reshape_op,
                new_inputs=[pre_in_name] + [str(v) for v in list(reshape_op.inputs[1:])],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape_op,
                new_outputs=[post_out_name],
            )

            for remove_idx in sorted([int(post_idx), int(pre_idx)], reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    if layout_state is not None:
        layout_state.remove(
            tensors_before_fallback_prune - set(str(name) for name in model_ir.tensors)
        )
    return {
        "optimized_transpose_reshape_transpose_to_flatten_hw_nhwc_chains": int(rewritten)
    }
