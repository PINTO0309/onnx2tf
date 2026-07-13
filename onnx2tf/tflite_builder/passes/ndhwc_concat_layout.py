from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_transpose_pre_concat_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Convert NCDHW concat blocks back to NDHWC when they are wrapped by transpose adapters.

    Target:
      ... -> t_i_ncdhw (some inputs may be unary outputs from transpose-wrapped NDHWC)
      CONCAT(axis=1, [t_0_ncdhw, ...]) -> y_ncdhw
      y_ncdhw -> TRANSPOSE(0,2,3,4,1) -> y_ndhwc

    Rewrite:
      ... -> t_i_ndhwc
      CONCAT(axis=4, [t_0_ndhwc, ...]) -> y_ndhwc
    """
    optimized = 0
    perm_ndhwc_to_ncdhw = [0, 4, 1, 2, 3]
    perm_ncdhw_to_ndhwc = [0, 2, 3, 4, 1]
    unary_ops = {
        "LOGISTIC",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "ELU",
        "LEAKY_RELU",
        "TANH",
        "GELU",
        "HARD_SWISH",
        "ABS",
        "EXP",
        "NEG",
        "SQRT",
    }

    def _project_shape_after_ncdhw_to_ndhwc(tensor_name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return None
        shape = [int(v) for v in list(tensor.shape)]
        if len(shape) != 5:
            return None
        return _permute_shape(shape, perm_ncdhw_to_ndhwc)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue

            concat_axis_old = int(concat_op.options.get("axis", 1))
            if concat_axis_old < 0:
                concat_axis_old += 5
            if concat_axis_old != 1:
                continue

            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue

            concat_input_actions: List[Dict[str, Any]] = []
            rewritable = True
            for input_name in [str(v) for v in list(concat_op.inputs)]:
                input_producer_idx = producers.get(str(input_name), None)
                if input_producer_idx is None:
                    rewritable = False
                    break
                input_producer = model_ir.operators[int(input_producer_idx)]

                if (
                    str(input_producer.op_type) == "TRANSPOSE"
                    and len(input_producer.inputs) >= 2
                    and len(input_producer.outputs) == 1
                    and str(input_producer.outputs[0]) == str(input_name)
                    and _read_transpose_perm(model_ir, input_producer) == perm_ndhwc_to_ncdhw
                    and set(int(v) for v in consumers.get(str(input_name), [])) == {int(concat_idx)}
                    and str(input_name) not in model_outputs
                ):
                    concat_input_actions.append(
                        {
                            "kind": "direct",
                            "new_input_name": str(input_producer.inputs[0]),
                            "remove_index": int(input_producer_idx),
                        }
                    )
                    continue

                if (
                    str(input_producer.op_type) in unary_ops
                    and len(input_producer.inputs) == 1
                    and len(input_producer.outputs) == 1
                    and str(input_producer.outputs[0]) == str(input_name)
                    and set(int(v) for v in consumers.get(str(input_name), [])) == {int(concat_idx)}
                    and str(input_name) not in model_outputs
                ):
                    unary_input_name = str(input_producer.inputs[0])
                    pre_idx = producers.get(unary_input_name, None)
                    if pre_idx is None:
                        rewritable = False
                        break
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != unary_input_name
                        or _read_transpose_perm(model_ir, pre_op) != perm_ndhwc_to_ncdhw
                        or unary_input_name in model_outputs
                        or set(int(v) for v in consumers.get(unary_input_name, [])) != {int(input_producer_idx)}
                    ):
                        rewritable = False
                        break
                    concat_input_actions.append(
                        {
                            "kind": "unary",
                            "input_name": str(input_name),
                            "unary_index": int(input_producer_idx),
                            "new_unary_input_name": str(pre_op.inputs[0]),
                            "remove_index": int(pre_idx),
                        }
                    )
                    continue

                rewritable = False
                break

            if not rewritable or len(concat_input_actions) == 0:
                continue

            concat_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            for user_idx in concat_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_ncdhw_to_ndhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                    continue
                rewritable = False
                break
            if not rewritable or len(post_indices) == 0:
                continue

            nhwc_ref_shape: Optional[List[int]] = None
            nhwc_inputs_ok = True
            for action in concat_input_actions:
                action_kind = str(action.get("kind", ""))
                if action_kind == "direct":
                    projected_input_name = str(action["new_input_name"])
                    input_tensor = model_ir.tensors.get(projected_input_name, None)
                    if input_tensor is None or len(list(input_tensor.shape)) != 5:
                        nhwc_inputs_ok = False
                        break
                    shape = [int(v) for v in list(input_tensor.shape)]
                elif action_kind == "unary":
                    projected_shape = _project_shape_after_ncdhw_to_ndhwc(str(action["input_name"]))
                    if projected_shape is None:
                        nhwc_inputs_ok = False
                        break
                    shape = [int(v) for v in list(projected_shape)]
                else:
                    nhwc_inputs_ok = False
                    break
                if nhwc_ref_shape is None:
                    nhwc_ref_shape = list(shape)
                else:
                    for dim_idx in [0, 1, 2, 3]:
                        if int(shape[dim_idx]) != int(nhwc_ref_shape[dim_idx]):
                            nhwc_inputs_ok = False
                            break
                if not nhwc_inputs_ok:
                    break
            if not nhwc_inputs_ok:
                continue

            new_concat_inputs: List[str] = []
            remove_indices: List[int] = []
            for action in concat_input_actions:
                action_kind = str(action.get("kind", ""))
                remove_indices.append(int(action["remove_index"]))
                if action_kind == "direct":
                    new_concat_inputs.append(str(action["new_input_name"]))
                    continue
                if action_kind == "unary":
                    unary_idx = int(action["unary_index"])
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=model_ir.operators[int(unary_idx)],
                        new_inputs=[str(action["new_unary_input_name"])],
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(action["input_name"]), None),
                        perm_ncdhw_to_ndhwc,
                    )
                    new_concat_inputs.append(str(action["input_name"]))
                    continue
                rewritable = False
                break
            if not rewritable:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=[str(v) for v in new_concat_inputs],
            )
            concat_op.options["axis"] = 4

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=concat_op,
                new_outputs=[canonical_post_output_name],
            )
            for alias_post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_post_output_name, canonical_post_output_name)

            old_concat_tensor = model_ir.tensors.get(concat_out_name, None)
            canonical_post_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_concat_tensor is not None and canonical_post_tensor is not None:
                canonical_post_tensor.dtype = str(old_concat_tensor.dtype)
                canonical_post_tensor.quantization = _clone_quantization(old_concat_tensor.quantization)
                canonical_post_tensor.shape = [int(v) for v in list(old_concat_tensor.shape)]
                canonical_post_tensor.shape_signature = (
                    [int(v) for v in list(old_concat_tensor.shape_signature)]
                    if old_concat_tensor.shape_signature is not None
                    else [int(v) for v in list(old_concat_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_post_tensor,
                    perm_ncdhw_to_ndhwc,
                )

            remove_indices.extend(post_indices)
            for remove_idx in sorted(list({int(v) for v in remove_indices}), reverse=True):
                if int(remove_idx) == int(concat_idx):
                    continue
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pre_concat_ndhwc_chains": int(optimized)}
