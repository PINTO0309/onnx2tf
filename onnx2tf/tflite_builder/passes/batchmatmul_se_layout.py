from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR


def optimize_batchmatmul_reshape_se_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove SE-like transpose bridges after BATCH_MATMUL -> RESHAPE tails.

    Target:
      x_nchw = RESHAPE(batch_matmul_out, [N,C,H,W])
      x_nchw --MEAN([2,3],keepDims=1)--> m_nchw --T(0,2,3,1)--> m_nhwc --CONV--CONV--> g_nhwc
      g_nhwc --T(0,3,1,2)--> g_nchw --LOGISTIC--> s_nchw
      MUL(x_nchw, s_nchw) -> y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - rewrite RESHAPE output to NHWC: [N,H,W,C]
      - remap MEAN axes to NHWC and bypass all three transposes
      - keep LOGISTIC/MUL in NHWC and preserve post-transpose output tensor name
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_out_idx, post_out_op in enumerate(model_ir.operators):
            if (
                str(post_out_op.op_type) != "TRANSPOSE"
                or len(post_out_op.inputs) < 2
                or len(post_out_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_out_op) != perm_nchw_to_nhwc
            ):
                continue

            mul_out_name = str(post_out_op.inputs[0])
            if mul_out_name in model_outputs:
                continue
            mul_idx = producers.get(mul_out_name, None)
            if mul_idx is None:
                continue
            mul_op = model_ir.operators[int(mul_idx)]
            if (
                str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != mul_out_name
            ):
                continue

            mul_out_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_out_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for user_idx in mul_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != mul_out_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_posts = False
                    break
                post_indices.append(int(user_idx))
                post_output_names.append(str(user_op.outputs[0]))
            if not valid_posts or len(post_indices) == 0:
                continue

            source_name: Optional[str] = None
            gate_name: Optional[str] = None
            for lhs_name, rhs_name in [
                (str(mul_op.inputs[0]), str(mul_op.inputs[1])),
                (str(mul_op.inputs[1]), str(mul_op.inputs[0])),
            ]:
                lhs_idx = producers.get(lhs_name, None)
                if lhs_idx is None:
                    continue
                lhs_op = model_ir.operators[int(lhs_idx)]
                if (
                    str(lhs_op.op_type) == "RESHAPE"
                    and len(lhs_op.inputs) >= 2
                    and len(lhs_op.outputs) == 1
                    and str(lhs_op.outputs[0]) == lhs_name
                    and lhs_name not in model_outputs
                ):
                    source_name = str(lhs_name)
                    gate_name = str(rhs_name)
                    break
            if source_name is None or gate_name is None:
                continue

            source_idx = producers.get(source_name, None)
            if source_idx is None:
                continue
            source_op = model_ir.operators[int(source_idx)]
            if str(source_op.op_type) != "RESHAPE" or len(source_op.inputs) < 2:
                continue
            source_users = set(int(v) for v in consumers.get(source_name, []))
            if int(mul_idx) not in source_users:
                continue
            mean_candidates = [int(v) for v in source_users if int(v) != int(mul_idx)]
            if len(mean_candidates) != 1:
                continue

            mean_idx = int(mean_candidates[0])
            mean_op = model_ir.operators[int(mean_idx)]
            if (
                str(mean_op.op_type) != "MEAN"
                or len(mean_op.inputs) < 2
                or len(mean_op.outputs) != 1
                or str(mean_op.inputs[0]) != source_name
                or not bool(mean_op.options.get("keepDims", False))
            ):
                continue
            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_outputs:
                continue

            mean_post_users = [int(v) for v in consumers.get(mean_out_name, [])]
            if len(mean_post_users) != 1:
                continue
            mean_post_idx = int(mean_post_users[0])
            mean_post_op = model_ir.operators[int(mean_post_idx)]
            if (
                str(mean_post_op.op_type) != "TRANSPOSE"
                or len(mean_post_op.inputs) < 2
                or len(mean_post_op.outputs) != 1
                or str(mean_post_op.inputs[0]) != mean_out_name
                or _read_transpose_perm(model_ir, mean_post_op) != perm_nchw_to_nhwc
            ):
                continue

            conv1_input_name = str(mean_post_op.outputs[0])
            conv1_users = [int(v) for v in consumers.get(conv1_input_name, [])]
            if len(conv1_users) != 1:
                continue
            conv1_idx = int(conv1_users[0])
            conv1_op = model_ir.operators[int(conv1_idx)]
            if (
                str(conv1_op.op_type) != "CONV_2D"
                or len(conv1_op.inputs) < 1
                or len(conv1_op.outputs) != 1
                or str(conv1_op.inputs[0]) != conv1_input_name
            ):
                continue
            conv1_out_name = str(conv1_op.outputs[0])
            conv1_out_users = [int(v) for v in consumers.get(conv1_out_name, [])]
            if len(conv1_out_users) != 1:
                continue
            conv2_idx = int(conv1_out_users[0])
            conv2_op = model_ir.operators[int(conv2_idx)]
            if (
                str(conv2_op.op_type) != "CONV_2D"
                or len(conv2_op.inputs) < 1
                or len(conv2_op.outputs) != 1
                or str(conv2_op.inputs[0]) != conv1_out_name
            ):
                continue
            conv2_out_name = str(conv2_op.outputs[0])

            gate_idx = producers.get(gate_name, None)
            if gate_idx is None:
                continue
            gate_op = model_ir.operators[int(gate_idx)]
            if (
                str(gate_op.op_type) != "LOGISTIC"
                or len(gate_op.inputs) != 1
                or len(gate_op.outputs) != 1
                or str(gate_op.outputs[0]) != gate_name
            ):
                continue
            if set(int(v) for v in consumers.get(gate_name, [])) != {int(mul_idx)}:
                continue

            pre_gate_name = str(gate_op.inputs[0])
            pre_gate_idx = producers.get(pre_gate_name, None)
            if pre_gate_idx is None:
                continue
            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
            if (
                str(pre_gate_op.op_type) != "TRANSPOSE"
                or len(pre_gate_op.inputs) < 2
                or len(pre_gate_op.outputs) != 1
                or str(pre_gate_op.outputs[0]) != pre_gate_name
                or _read_transpose_perm(model_ir, pre_gate_op) != perm_nhwc_to_nchw
                or str(pre_gate_op.inputs[0]) != conv2_out_name
            ):
                continue
            if set(int(v) for v in consumers.get(pre_gate_name, [])) != {int(gate_idx)}:
                continue

            # Keep semantics when converting source from NCHW to NHWC by
            # transposing BATCH_MATMUL output layout (without inserting transpose ops).
            source_bmm_out_name = str(source_op.inputs[0])
            source_bmm_idx = producers.get(source_bmm_out_name, None)
            if source_bmm_idx is None:
                continue
            source_bmm_op = model_ir.operators[int(source_bmm_idx)]
            if (
                str(source_bmm_op.op_type) != "BATCH_MATMUL"
                or len(source_bmm_op.inputs) != 2
                or len(source_bmm_op.outputs) != 1
                or str(source_bmm_op.outputs[0]) != source_bmm_out_name
            ):
                continue
            if set(int(v) for v in consumers.get(source_bmm_out_name, [])) != {int(source_idx)}:
                continue

            source_bmm_options = dict(source_bmm_op.options) if isinstance(source_bmm_op.options, dict) else {}
            # This rewrite is tuned for the affine-input optimized attention path:
            #   adjX=False, adjY=True.
            if bool(source_bmm_options.get("adjX", False)):
                continue
            if not bool(source_bmm_options.get("adjY", False)):
                continue

            old_bmm_inputs = [str(v) for v in list(source_bmm_op.inputs)]
            _set_operator_inputs(
                model_ir=model_ir,
                op=source_bmm_op,
                new_inputs=[str(old_bmm_inputs[1]), str(old_bmm_inputs[0])],
            )
            old_adj_x = bool(source_bmm_options.get("adjX", False))
            old_adj_y = bool(source_bmm_options.get("adjY", False))
            source_bmm_options["adjX"] = bool(not old_adj_y)
            source_bmm_options["adjY"] = bool(not old_adj_x)
            source_bmm_op.options = source_bmm_options

            source_bmm_tensor = model_ir.tensors.get(source_bmm_out_name, None)
            if source_bmm_tensor is not None:
                bmm_shape = (
                    [int(v) for v in list(source_bmm_tensor.shape)]
                    if source_bmm_tensor.shape is not None
                    else None
                )
                if bmm_shape is not None and len(bmm_shape) >= 2:
                    bmm_shape[-1], bmm_shape[-2] = int(bmm_shape[-2]), int(bmm_shape[-1])
                    source_bmm_tensor.shape = [int(v) for v in bmm_shape]
                bmm_signature = (
                    [int(v) for v in list(source_bmm_tensor.shape_signature)]
                    if source_bmm_tensor.shape_signature is not None
                    else (
                        [int(v) for v in list(source_bmm_tensor.shape)]
                        if source_bmm_tensor.shape is not None
                        else None
                    )
                )
                if bmm_signature is not None and len(bmm_signature) >= 2:
                    bmm_signature[-1], bmm_signature[-2] = int(bmm_signature[-2]), int(bmm_signature[-1])
                    source_bmm_tensor.shape_signature = [int(v) for v in bmm_signature]

            # Rewrite RESHAPE output layout from NCHW to NHWC.
            source_shape_name = str(source_op.inputs[1])
            source_shape_tensor = model_ir.tensors.get(source_shape_name, None)
            source_shape_vals = _read_const_ints_from_tensor(source_shape_tensor)
            if source_shape_vals is None or len(source_shape_vals) != 4:
                continue
            source_new_shape = [
                int(source_shape_vals[0]),
                int(source_shape_vals[2]),
                int(source_shape_vals[3]),
                int(source_shape_vals[1]),
            ]
            _write_const_ints_to_tensor(source_shape_tensor, [int(v) for v in source_new_shape])

            if isinstance(source_op.options, dict):
                source_opts = dict(source_op.options)
                for key in ["newShape", "onnxRawNewShape"]:
                    value = source_opts.get(key, None)
                    if isinstance(value, list) and len(value) == 4:
                        source_opts[key] = [int(v) for v in source_new_shape]
                source_op.options = source_opts

            source_tensor = model_ir.tensors.get(source_name, None)
            if source_tensor is not None:
                source_tensor.shape = [int(v) for v in source_new_shape]
                source_tensor.shape_signature = [int(v) for v in source_new_shape]

            # Remap MEAN axes from NCHW to NHWC.
            mean_axes_tensor = model_ir.tensors.get(str(mean_op.inputs[1]), None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None or len(mean_axes_vals) == 0:
                continue
            mapped_axes = [int(perm_nhwc_to_nchw[int(v)]) for v in list(mean_axes_vals)]
            _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes])
            if isinstance(mean_op.options, dict):
                mean_opts = dict(mean_op.options)
                for key in ["axis", "axes", "onnxRawAxes"]:
                    value = mean_opts.get(key, None)
                    if isinstance(value, list) and len(value) == len(mapped_axes):
                        mean_opts[key] = [int(v) for v in mapped_axes]
                mean_op.options = mean_opts

            conv1_inputs = [str(v) for v in list(conv1_op.inputs)]
            conv1_inputs[0] = str(mean_out_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv1_op,
                new_inputs=conv1_inputs,
            )

            _replace_operator_input_at(
                model_ir=model_ir,
                op=gate_op,
                input_index=0,
                new_input_name=conv2_out_name,
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mean_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(gate_name, None),
                perm_nchw_to_nhwc,
            )

            representative_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[representative_output_name],
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_name, representative_output_name)

            old_mul_tensor = model_ir.tensors.get(mul_out_name, None)
            representative_tensor = model_ir.tensors.get(representative_output_name, None)
            if old_mul_tensor is not None and representative_tensor is not None:
                representative_tensor.dtype = str(old_mul_tensor.dtype)
                representative_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                representative_tensor.shape = [int(v) for v in list(old_mul_tensor.shape)]
                representative_tensor.shape_signature = (
                    [int(v) for v in list(old_mul_tensor.shape_signature)]
                    if old_mul_tensor.shape_signature is not None
                    else [int(v) for v in list(old_mul_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    representative_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {
                int(mean_post_idx),
                int(pre_gate_idx),
                *[int(v) for v in post_indices],
            }
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_batchmatmul_reshape_se_nhwc_chains": int(rewritten)}
