from __future__ import annotations

from typing import Dict, List

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Collapse NCHW concat blocks that are bracketed by transpose adapters.

    Target:
      x_i_nhwc --TRANSPOSE(0,3,1,2)--> x_i_nchw
      CONCAT(axis=1, [x_i_nchw...]) -> y_nchw
      (optional unary chain) -> z_nchw
      z_nchw --TRANSPOSE(0,2,3,1)--> z_i_nhwc -> CONV_2D/DEPTHWISE_CONV_2D

    Rewrite:
      CONCAT(axis=3, [x_i_nhwc...]) -> y_nhwc
      (same unary chain in NHWC semantics) -> z_nhwc
      z_nhwc -> CONV_2D/DEPTHWISE_CONV_2D
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_ops = {"RELU", "RELU6", "LEAKY_RELU", "LOGISTIC", "TANH", "HARD_SWISH"}
    conv_like_ops = {"CONV_2D", "DEPTHWISE_CONV_2D"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_axis = int(concat_op.options.get("axis", 1))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue

            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue

            pre_transpose_indices: List[int] = []
            concat_inputs_nhwc: List[str] = []
            rewritable_inputs = True
            for concat_input_name in [str(v) for v in list(concat_op.inputs)]:
                pre_idx = producers.get(concat_input_name, None)
                if pre_idx is None:
                    rewritable_inputs = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != str(concat_input_name)
                    or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                    or str(pre_op.outputs[0]) in model_outputs
                ):
                    rewritable_inputs = False
                    break
                if set(int(v) for v in consumers.get(str(concat_input_name), [])) != {int(concat_idx)}:
                    rewritable_inputs = False
                    break
                pre_transpose_indices.append(int(pre_idx))
                concat_inputs_nhwc.append(str(pre_op.inputs[0]))
            if not rewritable_inputs or len(pre_transpose_indices) == 0:
                continue

            unary_output_names: List[str] = []
            nchw_tail_name = str(concat_out_name)
            while True:
                tail_users = [int(v) for v in consumers.get(str(nchw_tail_name), [])]
                if len(tail_users) != 1:
                    break
                unary_idx = int(tail_users[0])
                unary_op = model_ir.operators[int(unary_idx)]
                if (
                    str(unary_op.op_type) not in unary_ops
                    or len(unary_op.inputs) != 1
                    or len(unary_op.outputs) != 1
                    or str(unary_op.inputs[0]) != str(nchw_tail_name)
                    or str(unary_op.outputs[0]) in model_outputs
                ):
                    break
                nchw_tail_name = str(unary_op.outputs[0])
                unary_output_names.append(str(nchw_tail_name))

            tail_users = [int(v) for v in consumers.get(str(nchw_tail_name), [])]
            if len(tail_users) == 0:
                continue

            post_transpose_indices: List[int] = []
            valid_tail = True
            for post_idx in tail_users:
                post_op = model_ir.operators[int(post_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != str(nchw_tail_name)
                    or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                    or str(post_op.outputs[0]) in model_outputs
                ):
                    valid_tail = False
                    break
                post_out_name = str(post_op.outputs[0])
                post_out_users = [int(v) for v in consumers.get(post_out_name, [])]
                if len(post_out_users) == 0:
                    valid_tail = False
                    break
                if any(str(model_ir.operators[int(v)].op_type) not in conv_like_ops for v in post_out_users):
                    valid_tail = False
                    break
                post_transpose_indices.append(int(post_idx))
            if not valid_tail:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=[str(v) for v in list(concat_inputs_nhwc)],
            )
            concat_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(concat_out_name), None),
                perm_nchw_to_nhwc,
            )
            for unary_output_name in unary_output_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(unary_output_name), None),
                    perm_nchw_to_nhwc,
                )

            for post_idx in post_transpose_indices:
                post_op = model_ir.operators[int(post_idx)]
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, str(nchw_tail_name))

            remove_indices = sorted(
                list({int(v) for v in list(pre_transpose_indices) + list(post_transpose_indices)}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_concat_unary_fanout_conv_nhwc_chains": int(optimized)}
