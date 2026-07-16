from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.pre_unary_reshape_suffix_layout import (
    optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains as _optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains_pass,
)


def optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NHWC->NCHW unary wrappers that immediately feed
    [N,C,HW] reshape + [0,2,1] transpose suffixes.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --UNARY(1-input)-> u_nchw
      u_nchw --RESHAPE([N,C,HW])--> r
      r --TRANSPOSE([0,2,1])--> z

    Rewrite:
      x_nhwc --UNARY(1-input)-> u_nhwc
      u_nhwc --RESHAPE([N,HW,C])--> z

    Extended (pseudo-unary Swish):
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      s_nchw = LOGISTIC(x_nchw)
      y_nchw = MUL(x_nchw, s_nchw)
      y_nchw --RESHAPE([N,C,HW])--> r
      r --TRANSPOSE([0,2,1])--> z

    Rewrite:
      s_nhwc = LOGISTIC(x_nhwc)
      y_nhwc = MUL(x_nhwc, s_nhwc)
      y_nhwc --RESHAPE([N,HW,C])--> z
    """
    indexed_stats = (
        _optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains_pass(
            model_ir,
            graph_index=ModelIRGraphIndex(model_ir),
            layout_state=layout_state,
        )
    )
    rewritten = int(
        indexed_stats.get(
            "optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains",
            0,
        )
    )
    tensors_before_fallback_prune = (
        set(str(name) for name in model_ir.tensors)
        if layout_state is not None
        else set()
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_3d_nchw_to_nhwc = [0, 2, 1]
    unary_ops = {
        "RELU",
        "RELU6",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "CAST",
        "FLOOR",
        "CEIL",
        "ROUND",
    }

    def _swap_reshape_shape_tensor(reshape_op: OperatorIR) -> bool:
        if len(reshape_op.inputs) < 2:
            return False
        shape_name = str(reshape_op.inputs[1])
        shape_tensor = model_ir.tensors.get(shape_name, None)
        shape_vals = _read_const_ints_from_tensor(shape_tensor)
        if shape_vals is None or len(shape_vals) != 3:
            return False
        swapped = [int(shape_vals[0]), int(shape_vals[2]), int(shape_vals[1])]
        return _write_const_ints_to_tensor(shape_tensor, swapped)

    def _swap_reshape_option_shape(reshape_op: OperatorIR) -> bool:
        if not isinstance(reshape_op.options, dict):
            return False
        opts = dict(reshape_op.options)
        changed = False
        for key in ["newShape", "onnxRawNewShape"]:
            value = opts.get(key, None)
            if not isinstance(value, list) or len(value) != 3:
                continue
            swapped = [int(value[0]), int(value[2]), int(value[1])]
            if swapped != [int(v) for v in value]:
                opts[key] = swapped
                changed = True
        if changed:
            reshape_op.options = opts
        return changed

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for reshape_idx, reshape_op in enumerate(model_ir.operators):
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 1
                or len(reshape_op.outputs) != 1
            ):
                continue

            reshape_in_name = str(reshape_op.inputs[0])
            reshape_out_name = str(reshape_op.outputs[0])
            if reshape_in_name in model_outputs:
                continue

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
                or _read_transpose_perm(model_ir, post_op) != perm_3d_nchw_to_nhwc
            ):
                continue
            post_out_name = str(post_op.outputs[0])
            if reshape_out_name in model_outputs:
                continue

            unary_idx = producers.get(reshape_in_name, None)
            if unary_idx is None:
                continue
            unary_op = model_ir.operators[int(unary_idx)]
            rewrite_kind = ""
            swish_log_idx: Optional[int] = None
            swish_log_op: Optional[OperatorIR] = None
            swish_log_out_name = ""
            swish_mul_data_input_index: Optional[int] = None

            if (
                str(unary_op.op_type) in unary_ops
                and len(unary_op.inputs) == 1
                and len(unary_op.outputs) == 1
                and str(unary_op.outputs[0]) == reshape_in_name
            ):
                rewrite_kind = "unary"
                pre_out_name = str(unary_op.inputs[0])
            elif (
                str(unary_op.op_type) == "MUL"
                and len(unary_op.inputs) == 2
                and len(unary_op.outputs) == 1
                and str(unary_op.outputs[0]) == reshape_in_name
            ):
                mul_inputs = [str(v) for v in unary_op.inputs]
                for log_input_index in [0, 1]:
                    data_input_index = 1 - int(log_input_index)
                    candidate_log_out_name = str(mul_inputs[log_input_index])
                    candidate_pre_out_name = str(mul_inputs[data_input_index])
                    candidate_log_idx = producers.get(candidate_log_out_name, None)
                    if candidate_log_idx is None:
                        continue
                    candidate_log_op = model_ir.operators[int(candidate_log_idx)]
                    if (
                        str(candidate_log_op.op_type) != "LOGISTIC"
                        or len(candidate_log_op.inputs) != 1
                        or len(candidate_log_op.outputs) != 1
                        or str(candidate_log_op.outputs[0]) != candidate_log_out_name
                        or str(candidate_log_op.inputs[0]) != candidate_pre_out_name
                    ):
                        continue
                    candidate_log_users = [
                        int(v) for v in consumers.get(candidate_log_out_name, [])
                    ]
                    if set(candidate_log_users) != {int(unary_idx)}:
                        continue
                    rewrite_kind = "swish"
                    pre_out_name = candidate_pre_out_name
                    swish_log_idx = int(candidate_log_idx)
                    swish_log_op = candidate_log_op
                    swish_log_out_name = candidate_log_out_name
                    swish_mul_data_input_index = int(data_input_index)
                    break
            else:
                continue

            if rewrite_kind == "":
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
            pre_in_name = str(pre_op.inputs[0])

            if (
                pre_in_name in model_outputs
                or pre_out_name in model_outputs
                or reshape_in_name in model_outputs
            ):
                continue

            if rewrite_kind == "unary":
                if set(int(v) for v in consumers.get(pre_out_name, [])) != {
                    int(unary_idx)
                }:
                    continue
            elif rewrite_kind == "swish":
                expected_pre_users = (
                    {int(unary_idx), int(swish_log_idx)}
                    if swish_log_idx is not None
                    else {int(unary_idx)}
                )
                if (
                    set(int(v) for v in consumers.get(pre_out_name, []))
                    != expected_pre_users
                ):
                    continue
                if swish_log_out_name == "":
                    continue
                if set(int(v) for v in consumers.get(swish_log_out_name, [])) != {
                    int(unary_idx)
                }:
                    continue
            else:
                continue
            if set(int(v) for v in consumers.get(reshape_in_name, [])) != {
                int(reshape_idx)
            }:
                continue

            shape_changed = _swap_reshape_shape_tensor(reshape_op)
            opts_changed = _swap_reshape_option_shape(reshape_op)
            if not shape_changed and not opts_changed:
                continue

            if rewrite_kind == "unary":
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_inputs=[pre_in_name],
                )
            else:
                if swish_log_op is None or swish_mul_data_input_index is None:
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=swish_log_op,
                    new_inputs=[pre_in_name],
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=unary_op,
                    input_index=int(swish_mul_data_input_index),
                    new_input_name=pre_in_name,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(swish_log_out_name, None),
                    perm_nchw_to_nhwc,
                )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(reshape_in_name, None),
                perm_nchw_to_nhwc,
            )

            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape_op,
                new_outputs=[post_out_name],
            )

            old_reshape_out_tensor = model_ir.tensors.get(reshape_out_name, None)
            canonical_tensor = model_ir.tensors.get(post_out_name, None)
            if old_reshape_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_reshape_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(
                    old_reshape_out_tensor.quantization
                )
                canonical_tensor.shape = [
                    int(v) for v in list(old_reshape_out_tensor.shape)
                ]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_reshape_out_tensor.shape_signature)]
                    if old_reshape_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_reshape_out_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_3d_nchw_to_nhwc,
                )

            remove_indices: List[int] = []
            pre_remove_idx = next(
                (idx for idx, op in enumerate(model_ir.operators) if op is pre_op), None
            )
            if pre_remove_idx is not None:
                remove_indices.append(int(pre_remove_idx))
            post_remove_idx = next(
                (idx for idx, op in enumerate(model_ir.operators) if op is post_op),
                None,
            )
            if post_remove_idx is not None:
                remove_indices.append(int(post_remove_idx))
            for remove_idx in sorted(set(remove_indices), reverse=True):
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
        "optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains": int(
            rewritten
        )
    }
