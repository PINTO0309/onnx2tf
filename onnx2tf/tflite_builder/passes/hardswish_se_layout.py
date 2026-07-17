from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_singleton_constant_tensor,
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


def optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove NCHW bridge transposes in activation + SE-conv + HardSigmoid gating blocks.

    Supported activation roots:
    - HARD_SWISH
    - RELU
    - RELU6
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            hsw_out_name: Optional[str] = None
            hsw_input_rewrites: List[Tuple[OperatorIR, int]] = []
            hsw_root_intermediate_names: List[str] = []

            # Pattern 1: direct activation root
            #   pre_output -> {HARD_SWISH|RELU|RELU6} -> hsw_out
            if len(pre_users) == 1:
                hsw_idx = int(pre_users[0])
                hsw_op = model_ir.operators[int(hsw_idx)]
                if (
                    str(hsw_op.op_type) in {"HARD_SWISH", "RELU", "RELU6"}
                    and len(hsw_op.inputs) == 1
                    and len(hsw_op.outputs) == 1
                    and str(hsw_op.inputs[0]) == pre_output_name
                ):
                    hsw_out_name = str(hsw_op.outputs[0])
                    hsw_input_rewrites.append((hsw_op, 0))
                    hsw_root_intermediate_names.append(str(hsw_out_name))

            # Pattern 2: decomposed hard-swish root
            #   pre_output --ADD(+const, fused=RELU6)--> a
            #              --MUL(a, const)-------------> b
            #              --MUL(pre_output, b)--------> hsw_out
            if hsw_out_name is None and len(pre_users) == 2:
                add_idx: Optional[int] = None
                mul_data_idx: Optional[int] = None
                add_data_input_index: Optional[int] = None
                mul_data_input_index: Optional[int] = None
                for user_idx in pre_users:
                    user_op = model_ir.operators[int(user_idx)]
                    user_type = str(user_op.op_type)
                    if user_type == "ADD" and len(user_op.inputs) == 2 and len(user_op.outputs) == 1:
                        add_inputs = [str(v) for v in list(user_op.inputs)]
                        if pre_output_name == add_inputs[0]:
                            side_name = add_inputs[1]
                            add_data_input_index = 0
                        elif pre_output_name == add_inputs[1]:
                            side_name = add_inputs[0]
                            add_data_input_index = 1
                        else:
                            continue
                        if not _is_singleton_constant_tensor(model_ir, side_name):
                            continue
                        fused_act = str(user_op.options.get("fusedActivationFunction", "NONE")).upper()
                        if fused_act != "RELU6":
                            continue
                        add_idx = int(user_idx)
                    elif user_type == "MUL" and len(user_op.inputs) == 2 and len(user_op.outputs) == 1:
                        mul_inputs = [str(v) for v in list(user_op.inputs)]
                        if pre_output_name == mul_inputs[0]:
                            mul_data_input_index = 0
                            mul_data_idx = int(user_idx)
                        elif pre_output_name == mul_inputs[1]:
                            mul_data_input_index = 1
                            mul_data_idx = int(user_idx)
                if (
                    add_idx is not None
                    and mul_data_idx is not None
                    and add_data_input_index is not None
                    and mul_data_input_index is not None
                ):
                    add_op = model_ir.operators[int(add_idx)]
                    mul_data_op = model_ir.operators[int(mul_data_idx)]
                    add_out_name = str(add_op.outputs[0])
                    add_users = [int(v) for v in consumers.get(add_out_name, [])]
                    if len(add_users) == 1:
                        mul_scale_idx = int(add_users[0])
                        mul_scale_op = model_ir.operators[int(mul_scale_idx)]
                        if str(mul_scale_op.op_type) == "MUL" and len(mul_scale_op.inputs) == 2 and len(mul_scale_op.outputs) == 1:
                            mul_scale_inputs = [str(v) for v in list(mul_scale_op.inputs)]
                            if add_out_name == mul_scale_inputs[0]:
                                mul_scale_side_name = mul_scale_inputs[1]
                            elif add_out_name == mul_scale_inputs[1]:
                                mul_scale_side_name = mul_scale_inputs[0]
                            else:
                                mul_scale_side_name = ""
                            if _is_singleton_constant_tensor(model_ir, mul_scale_side_name):
                                mul_scale_out_name = str(mul_scale_op.outputs[0])
                                mul_data_inputs = [str(v) for v in list(mul_data_op.inputs)]
                                if (
                                    (mul_data_inputs[0] == pre_output_name and mul_data_inputs[1] == mul_scale_out_name)
                                    or (mul_data_inputs[1] == pre_output_name and mul_data_inputs[0] == mul_scale_out_name)
                                ):
                                    hsw_out_name = str(mul_data_op.outputs[0])
                                    hsw_input_rewrites.append((add_op, int(add_data_input_index)))
                                    hsw_input_rewrites.append((mul_data_op, int(mul_data_input_index)))
                                    hsw_root_intermediate_names.extend(
                                        [str(add_out_name), str(mul_scale_out_name), str(hsw_out_name)]
                                    )

            if hsw_out_name is None:
                continue
            if hsw_out_name in model_outputs:
                continue

            hsw_users = [int(v) for v in consumers.get(hsw_out_name, [])]
            if len(hsw_users) != 2:
                continue

            mean_idx: Optional[int] = None
            residual_mul_idx: Optional[int] = None
            for user_idx in hsw_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == hsw_out_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean_idx = int(user_idx)
                elif (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and hsw_out_name in {str(v) for v in list(user_op.inputs)}
                ):
                    residual_mul_idx = int(user_idx)
            if mean_idx is None or residual_mul_idx is None:
                continue

            mean_op = model_ir.operators[int(mean_idx)]
            mean_axes_name = str(mean_op.inputs[1])
            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None or len(mean_axes_vals) == 0:
                continue
            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_outputs:
                continue

            mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
            if len(mean_users) != 1:
                continue
            post_mean_idx = int(mean_users[0])
            post_mean_op = model_ir.operators[int(post_mean_idx)]
            if (
                str(post_mean_op.op_type) != "TRANSPOSE"
                or len(post_mean_op.inputs) < 2
                or len(post_mean_op.outputs) != 1
                or str(post_mean_op.inputs[0]) != mean_out_name
                or _read_transpose_perm(model_ir, post_mean_op) != perm_nchw_to_nhwc
            ):
                continue
            post_mean_out_name = str(post_mean_op.outputs[0])
            if post_mean_out_name in model_outputs:
                continue

            post_mean_users = [int(v) for v in consumers.get(post_mean_out_name, [])]
            if len(post_mean_users) != 1:
                continue
            conv1_idx = int(post_mean_users[0])
            conv1_op = model_ir.operators[int(conv1_idx)]
            if (
                str(conv1_op.op_type) != "CONV_2D"
                or len(conv1_op.inputs) < 1
                or len(conv1_op.outputs) != 1
                or str(conv1_op.inputs[0]) != post_mean_out_name
            ):
                continue
            conv1_out_name = str(conv1_op.outputs[0])

            conv1_users = [int(v) for v in consumers.get(conv1_out_name, [])]
            if len(conv1_users) != 1:
                continue
            conv2_idx = int(conv1_users[0])
            conv2_op = model_ir.operators[int(conv2_idx)]
            if (
                str(conv2_op.op_type) != "CONV_2D"
                or len(conv2_op.inputs) < 1
                or len(conv2_op.outputs) != 1
                or str(conv2_op.inputs[0]) != conv1_out_name
            ):
                continue
            conv2_out_name = str(conv2_op.outputs[0])

            conv2_users = [int(v) for v in consumers.get(conv2_out_name, [])]
            if len(conv2_users) != 1:
                continue
            pre_gate_idx = int(conv2_users[0])
            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
            if (
                str(pre_gate_op.op_type) != "TRANSPOSE"
                or len(pre_gate_op.inputs) < 2
                or len(pre_gate_op.outputs) != 1
                or str(pre_gate_op.inputs[0]) != conv2_out_name
                or _read_transpose_perm(model_ir, pre_gate_op) != perm_nhwc_to_nchw
            ):
                continue
            pre_gate_out_name = str(pre_gate_op.outputs[0])

            pre_gate_users = [int(v) for v in consumers.get(pre_gate_out_name, [])]
            if len(pre_gate_users) != 1:
                continue
            hs_entry_idx = int(pre_gate_users[0])
            hs_entry_op = model_ir.operators[int(hs_entry_idx)]
            hs_data_op = hs_entry_op
            hs_data_input_index: Optional[int] = None
            hs_out_name: Optional[str] = None
            hs_gate_intermediate_names: List[str] = []

            # Pattern A:
            #   MUL(alpha) -> ADD(beta) -> RELU_0_TO_1
            if str(hs_entry_op.op_type) == "MUL":
                if len(hs_entry_op.inputs) != 2 or len(hs_entry_op.outputs) != 1:
                    continue
                hs_mul_inputs = [str(v) for v in list(hs_entry_op.inputs)]
                if pre_gate_out_name == hs_mul_inputs[0]:
                    hs_data_input_index = 0
                    hs_mul_side_name = hs_mul_inputs[1]
                elif pre_gate_out_name == hs_mul_inputs[1]:
                    hs_data_input_index = 1
                    hs_mul_side_name = hs_mul_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_mul_side_name):
                    continue
                hs_mul_out_name = str(hs_entry_op.outputs[0])
                hs_gate_intermediate_names.append(str(hs_mul_out_name))

                hs_mul_users = [int(v) for v in consumers.get(hs_mul_out_name, [])]
                if len(hs_mul_users) != 1:
                    continue
                hs_add_idx = int(hs_mul_users[0])
                hs_add_op = model_ir.operators[int(hs_add_idx)]
                if str(hs_add_op.op_type) != "ADD" or len(hs_add_op.inputs) != 2 or len(hs_add_op.outputs) != 1:
                    continue
                hs_add_inputs = [str(v) for v in list(hs_add_op.inputs)]
                if hs_mul_out_name == hs_add_inputs[0]:
                    hs_add_side_name = hs_add_inputs[1]
                elif hs_mul_out_name == hs_add_inputs[1]:
                    hs_add_side_name = hs_add_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_add_side_name):
                    continue
                hs_add_out_name = str(hs_add_op.outputs[0])
                hs_gate_intermediate_names.append(str(hs_add_out_name))

                hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                if len(hs_add_users) != 1:
                    continue
                hs_relu_idx = int(hs_add_users[0])
                hs_relu_op = model_ir.operators[int(hs_relu_idx)]
                if (
                    str(hs_relu_op.op_type) != "RELU_0_TO_1"
                    or len(hs_relu_op.inputs) != 1
                    or len(hs_relu_op.outputs) != 1
                    or str(hs_relu_op.inputs[0]) != hs_add_out_name
                ):
                    continue
                hs_out_name = str(hs_relu_op.outputs[0])
                hs_gate_intermediate_names.append(str(hs_out_name))
            # Pattern B:
            #   ADD(beta, fusedActivation=RELU6) -> MUL(alpha)
            elif str(hs_entry_op.op_type) == "ADD":
                if len(hs_entry_op.inputs) != 2 or len(hs_entry_op.outputs) != 1:
                    continue
                hs_add_inputs = [str(v) for v in list(hs_entry_op.inputs)]
                if pre_gate_out_name == hs_add_inputs[0]:
                    hs_data_input_index = 0
                    hs_add_side_name = hs_add_inputs[1]
                elif pre_gate_out_name == hs_add_inputs[1]:
                    hs_data_input_index = 1
                    hs_add_side_name = hs_add_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_add_side_name):
                    continue
                fused_act = str(hs_entry_op.options.get("fusedActivationFunction", "NONE")).upper()
                if fused_act != "RELU6":
                    continue
                hs_add_out_name = str(hs_entry_op.outputs[0])
                hs_gate_intermediate_names.append(str(hs_add_out_name))

                hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                if len(hs_add_users) != 1:
                    continue
                hs_mul_idx = int(hs_add_users[0])
                hs_mul_op = model_ir.operators[int(hs_mul_idx)]
                if str(hs_mul_op.op_type) != "MUL" or len(hs_mul_op.inputs) != 2 or len(hs_mul_op.outputs) != 1:
                    continue
                hs_mul_inputs = [str(v) for v in list(hs_mul_op.inputs)]
                if hs_add_out_name == hs_mul_inputs[0]:
                    hs_mul_side_name = hs_mul_inputs[1]
                elif hs_add_out_name == hs_mul_inputs[1]:
                    hs_mul_side_name = hs_mul_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_mul_side_name):
                    continue
                hs_out_name = str(hs_mul_op.outputs[0])
                hs_gate_intermediate_names.append(str(hs_out_name))
            else:
                continue

            if hs_data_input_index is None or hs_out_name is None:
                continue

            residual_mul_op = model_ir.operators[int(residual_mul_idx)]
            residual_inputs = [str(v) for v in list(residual_mul_op.inputs)]
            if not (
                (residual_inputs[0] == hsw_out_name and residual_inputs[1] == hs_out_name)
                or (residual_inputs[1] == hsw_out_name and residual_inputs[0] == hs_out_name)
            ):
                continue
            residual_out_name = str(residual_mul_op.outputs[0])
            if residual_out_name in model_outputs:
                continue

            residual_users = [int(v) for v in consumers.get(residual_out_name, [])]
            if len(residual_users) != 1:
                continue
            post_idx = int(residual_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != residual_out_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            post_output_name = str(post_op.outputs[0])

            mapped_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes_vals:
                a = int(axis)
                if a < 0:
                    a += 4
                if a < 0 or a >= 4:
                    valid_axes = False
                    break
                mapped_axes.append(int(perm_nhwc_to_nchw[int(a)]))
            if not valid_axes:
                continue
            if mean_axes_tensor is None or not _write_const_ints_to_tensor(mean_axes_tensor, mapped_axes):
                continue

            for rewrite_op, rewrite_input_index in hsw_input_rewrites:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=rewrite_op,
                    input_index=int(rewrite_input_index),
                    new_input_name=pre_input_name,
                )
            for root_name in hsw_root_intermediate_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(root_name), None),
                    perm_nchw_to_nhwc,
                )

            mean_inputs = [str(v) for v in list(mean_op.inputs)]
            mean_inputs[0] = hsw_out_name
            _set_operator_inputs(model_ir=model_ir, op=mean_op, new_inputs=mean_inputs)
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mean_out_name, None),
                perm_nchw_to_nhwc,
            )

            _set_operator_outputs(
                model_ir=model_ir,
                op=mean_op,
                new_outputs=[post_mean_out_name],
            )
            old_mean_tensor = model_ir.tensors.get(mean_out_name, None)
            post_mean_tensor = model_ir.tensors.get(post_mean_out_name, None)
            if post_mean_tensor is not None and old_mean_tensor is not None:
                post_mean_tensor.dtype = str(old_mean_tensor.dtype)
                post_mean_tensor.quantization = _clone_quantization(old_mean_tensor.quantization)
                post_mean_tensor.shape = [int(v) for v in list(old_mean_tensor.shape)]
                post_mean_tensor.shape_signature = (
                    [int(v) for v in list(old_mean_tensor.shape_signature)]
                    if old_mean_tensor.shape_signature is not None
                    else [int(v) for v in list(old_mean_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_mean_tensor,
                    perm_nchw_to_nhwc,
                )

            _replace_operator_input_at(
                model_ir=model_ir,
                op=hs_data_op,
                input_index=int(hs_data_input_index),
                new_input_name=conv2_out_name,
            )
            for name in hs_gate_intermediate_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(name, None),
                    perm_nchw_to_nhwc,
                )

            _set_operator_outputs(
                model_ir=model_ir,
                op=residual_mul_op,
                new_outputs=[post_output_name],
            )
            old_residual_tensor = model_ir.tensors.get(residual_out_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if post_output_tensor is not None and old_residual_tensor is not None:
                post_output_tensor.dtype = str(old_residual_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_residual_tensor.quantization)
                post_output_tensor.shape = [int(v) for v in list(old_residual_tensor.shape)]
                post_output_tensor.shape_signature = (
                    [int(v) for v in list(old_residual_tensor.shape_signature)]
                    if old_residual_tensor.shape_signature is not None
                    else [int(v) for v in list(old_residual_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = sorted(
                list({int(pre_idx), int(post_mean_idx), int(pre_gate_idx), int(post_idx)}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": int(rewritten)}
