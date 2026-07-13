from __future__ import annotations

from typing import Any, Dict, List

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Convert strict NCHW concat quantize bridges back to NHWC and remove adapter transposes.

    Target:
      x_i_nhwc --TRANSPOSE(0,3,1,2)--> x_i_nchw --DEQUANTIZE--> f_i_nchw
      CONCAT(axis=1, [f_0_nchw, ...]) -> f_cat_nchw
      f_cat_nchw --QUANTIZE--> q_cat_nchw --TRANSPOSE(0,2,3,1)--> q_cat_nhwc

    Rewrite:
      x_i_nhwc --DEQUANTIZE--> f_i_nhwc
      CONCAT(axis=3, [f_0_nhwc, ...]) -> f_cat_nhwc
      f_cat_nhwc --QUANTIZE--> q_cat_nhwc
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue

            concat_axis = int(concat_op.options.get("axis", 1))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue

            concat_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_users) != 1:
                continue
            q_idx = int(concat_users[0])
            q_op = model_ir.operators[int(q_idx)]
            if (
                str(q_op.op_type) != "QUANTIZE"
                or len(q_op.inputs) != 1
                or len(q_op.outputs) != 1
                or str(q_op.inputs[0]) != concat_out_name
            ):
                continue

            q_out_name = str(q_op.outputs[0])
            if q_out_name in model_outputs:
                continue
            q_users = [int(v) for v in consumers.get(q_out_name, [])]
            if len(q_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for q_user_idx in q_users:
                q_user_op = model_ir.operators[int(q_user_idx)]
                if (
                    str(q_user_op.op_type) == "TRANSPOSE"
                    and len(q_user_op.inputs) >= 2
                    and len(q_user_op.outputs) == 1
                    and str(q_user_op.inputs[0]) == q_out_name
                    and _read_transpose_perm(model_ir, q_user_op) == perm_nchw_to_nhwc
                    and str(q_user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(q_user_idx))
                    post_output_names.append(str(q_user_op.outputs[0]))
                else:
                    valid_posts = False
                    break
            if not valid_posts or len(post_indices) == 0:
                continue

            dq_rewrite_plans: List[Dict[str, Any]] = []
            pre_indices_to_remove: List[int] = []
            rewrite_ok = True
            for concat_input_name in [str(v) for v in list(concat_op.inputs)]:
                dq_idx = producers.get(str(concat_input_name), None)
                if dq_idx is None:
                    rewrite_ok = False
                    break
                dq_op = model_ir.operators[int(dq_idx)]
                if (
                    str(dq_op.op_type) != "DEQUANTIZE"
                    or len(dq_op.inputs) != 1
                    or len(dq_op.outputs) != 1
                    or str(dq_op.outputs[0]) != str(concat_input_name)
                    or str(concat_input_name) in model_outputs
                ):
                    rewrite_ok = False
                    break
                dq_users = [int(v) for v in consumers.get(str(concat_input_name), [])]
                if set(dq_users) != {int(concat_idx)}:
                    rewrite_ok = False
                    break

                pre_name = str(dq_op.inputs[0])
                pre_idx = producers.get(pre_name, None)
                if pre_idx is None:
                    rewrite_ok = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != pre_name
                    or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                ):
                    rewrite_ok = False
                    break
                if pre_name in model_outputs:
                    rewrite_ok = False
                    break

                dq_rewrite_plans.append(
                    {
                        "dq_idx": int(dq_idx),
                        "concat_input_name": str(concat_input_name),
                        "new_dq_input_name": str(pre_op.inputs[0]),
                    }
                )

                pre_users = [int(v) for v in consumers.get(pre_name, [])]
                if set(pre_users) == {int(dq_idx)}:
                    pre_indices_to_remove.append(int(pre_idx))

            if not rewrite_ok:
                continue

            for dq_plan in dq_rewrite_plans:
                dq_idx = int(dq_plan["dq_idx"])
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(dq_idx)],
                    new_inputs=[str(dq_plan["new_dq_input_name"])],
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(dq_plan["concat_input_name"]), None),
                    perm_nchw_to_nhwc,
                )

            concat_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(concat_out_name, None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=q_op,
                new_outputs=[canonical_post_output_name],
            )
            for alias_post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_post_output_name, canonical_post_output_name)

            old_q_out_tensor = model_ir.tensors.get(q_out_name, None)
            concat_out_tensor = model_ir.tensors.get(concat_out_name, None)
            canonical_post_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if canonical_post_tensor is not None:
                if old_q_out_tensor is not None:
                    canonical_post_tensor.dtype = str(old_q_out_tensor.dtype)
                    canonical_post_tensor.quantization = _clone_quantization(old_q_out_tensor.quantization)
                if concat_out_tensor is not None:
                    canonical_post_tensor.shape = [int(v) for v in list(concat_out_tensor.shape)]
                    canonical_post_tensor.shape_signature = (
                        [int(v) for v in list(concat_out_tensor.shape_signature)]
                        if concat_out_tensor.shape_signature is not None
                        else [int(v) for v in list(concat_out_tensor.shape)]
                    )

            remove_indices = sorted(
                list({*pre_indices_to_remove, *post_indices}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                if int(remove_idx) == int(concat_idx) or int(remove_idx) == int(q_idx):
                    continue
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": int(optimized)}
