from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _clone_quantization,
    _invert_perm,
    _is_per_tensor_quantization,
    _is_singleton_constant_tensor,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)


def _optimize_leading_input_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold leading input-boundary transpose chains through layout-agnostic ops.

    Target:
      X_in(NHWC) --T(P)--> X_ncx --(CAST|SUB|ADD|MUL|DIV|MAXIMUM|MINIMUM|ATAN2|QUANTIZE|DEQUANTIZE)*--> Y_ncx --T(inv(P))--> Y_nhwc
      X_in(NHWC) --T(P)--> X_ncx --(same chain)*--> Y_ncx (Y_ncx is model output)

    Rewrite:
      X_in(NHWC) --(same chain)*--> Y_nhwc

    Safety:
    - Leading transpose input must be a model input tensor.
    - Chain must be strictly linear (single-consumer on the main path).
    - Binary ops in the chain must use constant side inputs.
    - Quantization in the chain must remain per-tensor only.
    """
    rewritten = 0
    unary_passthrough_ops = {"CAST", "QUANTIZE", "DEQUANTIZE"}
    binary_passthrough_ops = {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "ATAN2"}

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(model_ir.inputs)
        model_outputs = set(model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE":
                continue
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            if pre_input_name not in model_inputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            # Build a strict linear passthrough chain.
            chain_indices: List[int] = []
            chain_ops: List[OperatorIR] = []
            chain_output_names: List[str] = []
            chain_binary_side_const_by_op: Dict[int, str] = {}
            current_tensor = pre_output_name
            while True:
                current_users = [int(v) for v in consumers.get(current_tensor, [])]
                if len(current_users) != 1:
                    break
                op_idx = int(current_users[0])
                op = model_ir.operators[op_idx]
                op_type = str(op.op_type)

                if op_type in unary_passthrough_ops:
                    if len(op.inputs) != 1 or len(op.outputs) != 1:
                        break
                    if str(op.inputs[0]) != current_tensor:
                        break
                    out_name = str(op.outputs[0])
                    out_tensor = model_ir.tensors.get(out_name, None)
                    if out_tensor is not None and not _is_per_tensor_quantization(out_tensor.quantization):
                        break
                    chain_indices.append(int(op_idx))
                    chain_ops.append(op)
                    chain_output_names.append(out_name)
                    current_tensor = out_name
                    if out_name in model_outputs:
                        break
                    continue

                if op_type in binary_passthrough_ops:
                    if len(op.inputs) != 2 or len(op.outputs) != 1:
                        break
                    input_0 = str(op.inputs[0])
                    input_1 = str(op.inputs[1])
                    if input_0 == current_tensor:
                        side_input_name = input_1
                    elif input_1 == current_tensor:
                        side_input_name = input_0
                    else:
                        break
                    side_tensor = model_ir.tensors.get(side_input_name, None)
                    if side_tensor is None or side_tensor.data is None:
                        break
                    side_array = np.asarray(side_tensor.data)
                    if int(side_array.size) != 1 and int(side_array.ndim) != int(len(perm_pre)):
                        break
                    if side_tensor is not None and not _is_per_tensor_quantization(side_tensor.quantization):
                        break
                    out_name = str(op.outputs[0])
                    out_tensor = model_ir.tensors.get(out_name, None)
                    if out_tensor is not None and not _is_per_tensor_quantization(out_tensor.quantization):
                        break
                    chain_indices.append(int(op_idx))
                    chain_ops.append(op)
                    chain_output_names.append(out_name)
                    chain_binary_side_const_by_op[int(op_idx)] = str(side_input_name)
                    current_tensor = out_name
                    if out_name in model_outputs:
                        break
                    continue

                break

            if len(chain_ops) == 0:
                continue

            # Chain must end with one inverse transpose or a model output.
            tail_users = [int(v) for v in consumers.get(current_tensor, [])]
            terminal_mode = ""
            post_idx: Optional[int] = None
            post_op: Optional[OperatorIR] = None
            post_output_name = ""
            if len(tail_users) == 1:
                post_idx = int(tail_users[0])
                if post_idx in set(chain_indices):
                    continue
                post_op = model_ir.operators[post_idx]
                if str(post_op.op_type) == "TRANSPOSE" and len(post_op.inputs) >= 2 and len(post_op.outputs) == 1:
                    if str(post_op.inputs[0]) == current_tensor:
                        perm_post = _read_transpose_perm(model_ir, post_op)
                        if perm_post is not None and perm_post == perm_post_expected:
                            terminal_mode = "inverse_post_transpose"
                            post_output_name = str(post_op.outputs[0])
            if terminal_mode == "" and current_tensor in model_outputs and len(tail_users) == 0:
                terminal_mode = "model_output"
                post_output_name = str(current_tensor)
            if terminal_mode == "":
                continue

            # Ensure chain topology remains linear on the main path.
            linear_ok = True
            previous_tensor_name = pre_output_name
            chain_index_to_pos = {int(idx): pos for pos, idx in enumerate(chain_indices)}
            for op_idx, op in zip(chain_indices, chain_ops):
                if previous_tensor_name not in set(str(v) for v in op.inputs):
                    linear_ok = False
                    break
                out_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
                if out_name == "":
                    linear_ok = False
                    break
                expected_users = [int(v) for v in consumers.get(out_name, [])]
                pos = int(chain_index_to_pos[int(op_idx)])
                if pos < len(chain_indices) - 1:
                    if expected_users != [int(chain_indices[pos + 1])]:
                        linear_ok = False
                        break
                else:
                    if terminal_mode == "inverse_post_transpose":
                        if post_idx is None or expected_users != [int(post_idx)]:
                            linear_ok = False
                            break
                    else:
                        if len(expected_users) != 0:
                            linear_ok = False
                            break
                previous_tensor_name = out_name
            if not linear_ok:
                continue

            old_last_name = str(chain_ops[-1].outputs[0])
            if terminal_mode == "inverse_post_transpose" and old_last_name in model_outputs:
                continue

            # Rewire chain head: consume model input directly.
            first_op = chain_ops[0]
            first_input_names = [str(v) for v in first_op.inputs]
            if first_input_names[0] == pre_output_name:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=first_op,
                    new_inputs=[pre_input_name] + first_input_names[1:],
                )
            elif len(first_input_names) > 1 and first_input_names[1] == pre_output_name:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=first_op,
                    new_inputs=[first_input_names[0], pre_input_name],
                )
            else:
                continue

            # Permute non-scalar side constants that carry NCX axis order.
            chain_index_set = set(int(v) for v in chain_indices)
            for op_idx, side_name in chain_binary_side_const_by_op.items():
                side_tensor = model_ir.tensors.get(str(side_name), None)
                if side_tensor is None or side_tensor.data is None:
                    continue
                side_array = np.asarray(side_tensor.data)
                if int(side_array.size) == 1 or int(side_array.ndim) != int(len(perm_post_expected)):
                    continue
                transposed_const = np.transpose(side_array, axes=perm_post_expected)
                side_users = [int(v) for v in consumers.get(str(side_name), [])]
                unique_to_chain = all(int(v) in chain_index_set for v in side_users)
                replacement_name = str(side_name)
                if unique_to_chain:
                    side_tensor.data = transposed_const
                    shape, signature = normalize_onnx_shape(list(transposed_const.shape))
                    side_tensor.shape = [int(v) for v in list(shape)]
                    side_tensor.shape_signature = [int(v) for v in list(signature)]
                else:
                    replacement_name = _unique_tensor_name(f"{side_name}_nhwc")
                    shape, signature = normalize_onnx_shape(list(transposed_const.shape))
                    model_ir.tensors[replacement_name] = TensorIR(
                        name=replacement_name,
                        dtype=str(side_tensor.dtype),
                        shape=[int(v) for v in list(shape)],
                        shape_signature=[int(v) for v in list(signature)],
                        data=transposed_const,
                        is_variable=False,
                        quantization=_clone_quantization(side_tensor.quantization),
                    )
                if replacement_name != str(side_name):
                    target_op = model_ir.operators[int(op_idx)]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=target_op,
                        new_inputs=[
                            replacement_name if str(inp) == str(side_name) else str(inp)
                            for inp in list(target_op.inputs)
                        ],
                    )

            # Chain tail now produces post-transpose output name directly.
            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=chain_ops[-1],
                    new_outputs=[post_output_name],
                )

            # Update metadata to NHWC-side layout.
            for out_name in chain_output_names[:-1]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_post_expected,
                )
            if terminal_mode == "inverse_post_transpose":
                old_last_tensor = model_ir.tensors.get(old_last_name, None)
                post_output_tensor = model_ir.tensors.get(post_output_name, None)
                if post_output_tensor is not None and old_last_tensor is not None:
                    post_output_tensor.dtype = str(old_last_tensor.dtype)
                    post_output_tensor.quantization = _clone_quantization(old_last_tensor.quantization)
                    _permute_tensor_metadata_if_rank_matches(
                        post_output_tensor,
                        perm_post_expected,
                    )
            else:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(post_output_name, None),
                    perm_post_expected,
                )

            # Remove boundary transposes.
            remove_set = {int(pre_idx)}
            if terminal_mode == "inverse_post_transpose" and post_idx is not None:
                remove_set.add(int(post_idx))
            remove_indices = sorted(list(remove_set), reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_leading_input_transpose_passthrough_chains": int(rewritten)}


def _optimize_asin_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold leading input TRANSPOSE through Asin/Acos decomposition chains.

    Target:
      X --TRANSPOSE(P)--> x_t
      x_t --MUL(x_t, x_t)--> x2
      c1 --SUB(x2)--> one_minus
      one_minus --SQRT--> denom
      ATAN2(x_t, denom) or ATAN2(denom, x_t) --> y_t
      (optional) y_t --TRANSPOSE(inv(P))--> Y

    Rewrite:
      X --MUL(X, X)--> x2
      c1 --SUB(x2)--> one_minus
      one_minus --SQRT--> denom
      ATAN2(X, denom) or ATAN2(denom, X) --> Y

    Safety:
    - Leading transpose input must be a model input tensor.
    - Strict local topology and single-consumer inner chain.
    - SUB constant side must be scalar-like.
    - Terminal must be graph output or one inverse post-transpose.
    """
    rewritten = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            if pre_input_name not in model_inputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_users = sorted({int(v) for v in consumers.get(pre_output_name, [])})
            if len(pre_users) != 2:
                continue

            mul_idx: Optional[int] = None
            atan2_idx: Optional[int] = None
            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "MUL":
                    mul_idx = int(user_idx)
                elif user_type == "ATAN2":
                    atan2_idx = int(user_idx)
            if mul_idx is None or atan2_idx is None:
                continue

            mul_op = model_ir.operators[int(mul_idx)]
            if len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            if not (mul_inputs[0] == pre_output_name and mul_inputs[1] == pre_output_name):
                continue
            mul_out_name = str(mul_op.outputs[0])
            mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_users) != 1:
                continue

            sub_idx = int(mul_users[0])
            sub_op = model_ir.operators[int(sub_idx)]
            if str(sub_op.op_type) != "SUB" or len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1:
                continue
            sub_inputs = [str(v) for v in list(sub_op.inputs)]
            if sub_inputs[1] != mul_out_name:
                continue
            sub_const_name = str(sub_inputs[0])
            if not _is_singleton_constant_tensor(model_ir, sub_const_name):
                continue
            sub_out_name = str(sub_op.outputs[0])
            sub_users = [int(v) for v in consumers.get(sub_out_name, [])]
            if len(sub_users) != 1:
                continue

            sqrt_idx = int(sub_users[0])
            sqrt_op = model_ir.operators[int(sqrt_idx)]
            if str(sqrt_op.op_type) != "SQRT" or len(sqrt_op.inputs) != 1 or len(sqrt_op.outputs) != 1:
                continue
            if str(sqrt_op.inputs[0]) != sub_out_name:
                continue
            sqrt_out_name = str(sqrt_op.outputs[0])
            sqrt_users = [int(v) for v in consumers.get(sqrt_out_name, [])]
            if sqrt_users != [int(atan2_idx)]:
                continue

            atan2_op = model_ir.operators[int(atan2_idx)]
            if len(atan2_op.inputs) != 2 or len(atan2_op.outputs) != 1:
                continue
            atan2_inputs = [str(v) for v in list(atan2_op.inputs)]
            if set(atan2_inputs) != {str(pre_output_name), str(sqrt_out_name)}:
                continue
            atan2_out_name = str(atan2_op.outputs[0])
            if atan2_out_name in model_inputs:
                continue

            terminal_mode = ""
            post_idx: Optional[int] = None
            post_output_name = str(atan2_out_name)
            atan2_out_users = [int(v) for v in consumers.get(atan2_out_name, [])]
            if atan2_out_name in model_outputs and len(atan2_out_users) == 0:
                terminal_mode = "model_output"
            elif len(atan2_out_users) == 1:
                candidate_post_idx = int(atan2_out_users[0])
                candidate_post_op = model_ir.operators[int(candidate_post_idx)]
                if (
                    str(candidate_post_op.op_type) == "TRANSPOSE"
                    and len(candidate_post_op.inputs) >= 2
                    and len(candidate_post_op.outputs) == 1
                    and str(candidate_post_op.inputs[0]) == str(atan2_out_name)
                ):
                    perm_post = _read_transpose_perm(model_ir, candidate_post_op)
                    if perm_post is not None and list(perm_post) == list(perm_post_expected):
                        terminal_mode = "inverse_post_transpose"
                        post_idx = int(candidate_post_idx)
                        post_output_name = str(candidate_post_op.outputs[0])
            if terminal_mode == "":
                continue

            # Rewire chain heads from transposed input to external input.
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul_op,
                new_inputs=[str(pre_input_name), str(pre_input_name)],
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=atan2_op,
                new_inputs=[
                    str(pre_input_name) if str(v) == str(pre_output_name) else str(v)
                    for v in list(atan2_op.inputs)
                ],
            )

            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=atan2_op,
                    new_outputs=[str(post_output_name)],
                )

            # Update metadata to external layout.
            for name in [str(mul_out_name), str(sub_out_name), str(sqrt_out_name)]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(name, None),
                    perm_post_expected,
                )
            if terminal_mode == "inverse_post_transpose":
                atan2_out_tensor = model_ir.tensors.get(str(atan2_out_name), None)
                post_output_tensor = model_ir.tensors.get(str(post_output_name), None)
                if post_output_tensor is not None and atan2_out_tensor is not None:
                    post_output_tensor.dtype = str(atan2_out_tensor.dtype)
                    post_output_tensor.quantization = _clone_quantization(atan2_out_tensor.quantization)
                    _permute_tensor_metadata_if_rank_matches(
                        post_output_tensor,
                        perm_post_expected,
                    )
            else:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(post_output_name), None),
                    perm_post_expected,
                )

            remove_indices = {int(pre_idx)}
            if post_idx is not None:
                remove_indices.add(int(post_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_asin_transpose_passthrough_chains": int(rewritten)}


def _optimize_hardsigmoid_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around standalone HardSigmoid-expanded chains.

    Target:
      X --TRANSPOSE(P)--> x_t
      x_t --MUL(alpha)--> m1 --ADD(beta)--> a1 --(MAXIMUM(0)->MINIMUM(1) | RELU_0_TO_1)--> h1
      h1 --TRANSPOSE(inv(P))--> Y

    Rewrite:
      X --MUL(alpha)--> m1 --ADD(beta)--> a1 --(MAXIMUM(0)->MINIMUM(1) | RELU_0_TO_1)--> Y

    Safety:
    - Strict HardSigmoid decomposition topology.
    - All side inputs must be singleton constants.
    - Main branch must be strictly linear.
    """
    rewritten = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE":
                continue
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) != 1:
                continue
            hs_mul_idx = int(pre_users[0])
            hs_mul_op = model_ir.operators[int(hs_mul_idx)]
            if str(hs_mul_op.op_type) != "MUL" or len(hs_mul_op.inputs) != 2 or len(hs_mul_op.outputs) != 1:
                continue

            hs_mul_inputs = [str(v) for v in list(hs_mul_op.inputs)]
            if pre_output_name == hs_mul_inputs[0]:
                hs_mul_data_input_index = 0
                hs_mul_side_name = hs_mul_inputs[1]
            elif pre_output_name == hs_mul_inputs[1]:
                hs_mul_data_input_index = 1
                hs_mul_side_name = hs_mul_inputs[0]
            else:
                continue
            if not _is_singleton_constant_tensor(model_ir, hs_mul_side_name):
                continue
            hs_mul_out_name = str(hs_mul_op.outputs[0])

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

            hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
            if len(hs_add_users) != 1:
                continue
            hs_clamp_idx = int(hs_add_users[0])
            hs_clamp_op = model_ir.operators[int(hs_clamp_idx)]
            hs_intermediate_names: List[str] = [str(hs_mul_out_name), str(hs_add_out_name)]
            if (
                str(hs_clamp_op.op_type) == "RELU_0_TO_1"
                and len(hs_clamp_op.inputs) == 1
                and len(hs_clamp_op.outputs) == 1
                and str(hs_clamp_op.inputs[0]) == hs_add_out_name
            ):
                hs_terminal_op = hs_clamp_op
                hs_out_name = str(hs_clamp_op.outputs[0])
            else:
                if str(hs_clamp_op.op_type) != "MAXIMUM" or len(hs_clamp_op.inputs) != 2 or len(hs_clamp_op.outputs) != 1:
                    continue
                hs_max_inputs = [str(v) for v in list(hs_clamp_op.inputs)]
                if hs_add_out_name == hs_max_inputs[0]:
                    hs_max_side_name = hs_max_inputs[1]
                elif hs_add_out_name == hs_max_inputs[1]:
                    hs_max_side_name = hs_max_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_max_side_name):
                    continue
                hs_max_out_name = str(hs_clamp_op.outputs[0])
                hs_intermediate_names.append(str(hs_max_out_name))

                hs_max_users = [int(v) for v in consumers.get(hs_max_out_name, [])]
                if len(hs_max_users) != 1:
                    continue
                hs_min_idx = int(hs_max_users[0])
                hs_min_op = model_ir.operators[int(hs_min_idx)]
                if str(hs_min_op.op_type) != "MINIMUM" or len(hs_min_op.inputs) != 2 or len(hs_min_op.outputs) != 1:
                    continue
                hs_min_inputs = [str(v) for v in list(hs_min_op.inputs)]
                if hs_max_out_name == hs_min_inputs[0]:
                    hs_min_side_name = hs_min_inputs[1]
                elif hs_max_out_name == hs_min_inputs[1]:
                    hs_min_side_name = hs_min_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_min_side_name):
                    continue
                hs_terminal_op = hs_min_op
                hs_out_name = str(hs_min_op.outputs[0])

            hs_out_users = [int(v) for v in consumers.get(hs_out_name, [])]
            if len(hs_out_users) != 1:
                continue
            post_idx = int(hs_out_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != hs_out_name
            ):
                continue
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or perm_post != perm_post_expected:
                continue
            post_output_name = str(post_op.outputs[0])

            _replace_operator_input_at(
                model_ir=model_ir,
                op=hs_mul_op,
                input_index=int(hs_mul_data_input_index),
                new_input_name=pre_input_name,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=hs_terminal_op,
                new_outputs=[post_output_name],
            )

            for intermediate_name in hs_intermediate_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(intermediate_name, None),
                    perm_post_expected,
                )

            old_hs_tensor = model_ir.tensors.get(hs_out_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if post_output_tensor is not None and old_hs_tensor is not None:
                post_output_tensor.dtype = str(old_hs_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_hs_tensor.quantization)
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_post_expected,
                )

            remove_indices = sorted(list({int(pre_idx), int(post_idx)}), reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_hardsigmoid_transpose_passthrough_chains": int(rewritten)}


def _optimize_erf_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold leading input TRANSPOSE through Erf polynomial decomposition chains.

    Target (strict local topology):
      X --TRANSPOSE(P)--> x_t
      x_t --ABS--> a
      x_t --SIGN--> s
      a --MUL(p)--> px --ADD(1)--> one_plus --DIV(1)--> t
      a --MUL(a)--> a2 --MUL(-1)--> na2 --EXP--> e
      Horner chain with t:
        m1=MUL(a5,t); a1=ADD(m1,a4); m2=MUL(a1,t); a2=ADD(m2,a3);
        m3=MUL(a2,t); a3=ADD(m3,a2); m4=MUL(a3,t); a4=ADD(m4,a1);
        poly=MUL(a4,t)
      poly --MUL(e)--> pe --SUB(1)--> om
      MUL(s, om) --> y_t
      (optional) y_t --TRANSPOSE(inv(P))--> Y

    Rewrite:
      Remove boundary transpose(s) and feed chain directly from X.
    """
    rewritten = 0

    def _single_user(
        *,
        tensor_name: str,
        consumers: Dict[str, List[int]],
    ) -> Optional[int]:
        users = sorted({int(v) for v in consumers.get(str(tensor_name), [])})
        if len(users) != 1:
            return None
        return int(users[0])

    def _is_scalar_const_tensor(name: str) -> bool:
        return _is_singleton_constant_tensor(model_ir, str(name))

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            pre_input_name = str(pre_op.inputs[0])
            if pre_input_name not in model_inputs:
                continue
            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_users = sorted({int(v) for v in consumers.get(pre_output_name, [])})
            if len(pre_users) != 2:
                continue
            abs_idx: Optional[int] = None
            sign_idx: Optional[int] = None
            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "ABS":
                    abs_idx = int(user_idx)
                elif user_type == "SIGN":
                    sign_idx = int(user_idx)
            if abs_idx is None or sign_idx is None:
                continue
            abs_op = model_ir.operators[int(abs_idx)]
            sign_op = model_ir.operators[int(sign_idx)]
            if len(abs_op.inputs) != 1 or len(abs_op.outputs) != 1:
                continue
            if len(sign_op.inputs) != 1 or len(sign_op.outputs) != 1:
                continue
            if str(abs_op.inputs[0]) != pre_output_name or str(sign_op.inputs[0]) != pre_output_name:
                continue
            abs_out = str(abs_op.outputs[0])
            sign_out = str(sign_op.outputs[0])

            abs_users = sorted({int(v) for v in consumers.get(abs_out, [])})
            if len(abs_users) != 2:
                continue
            mul_px_idx: Optional[int] = None
            mul_sq_idx: Optional[int] = None
            for user_idx in abs_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) != "MUL" or len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                    continue
                in0 = str(user_op.inputs[0])
                in1 = str(user_op.inputs[1])
                if in0 == abs_out and in1 == abs_out:
                    mul_sq_idx = int(user_idx)
                    continue
                if abs_out in {in0, in1}:
                    side_name = in1 if in0 == abs_out else in0
                    if _is_scalar_const_tensor(side_name):
                        mul_px_idx = int(user_idx)
            if mul_px_idx is None or mul_sq_idx is None:
                continue

            mul_px_op = model_ir.operators[int(mul_px_idx)]
            mul_sq_op = model_ir.operators[int(mul_sq_idx)]
            mul_px_out = str(mul_px_op.outputs[0])
            mul_sq_out = str(mul_sq_op.outputs[0])

            add_px_idx = _single_user(tensor_name=mul_px_out, consumers=consumers)
            if add_px_idx is None:
                continue
            add_px_op = model_ir.operators[int(add_px_idx)]
            if str(add_px_op.op_type) != "ADD" or len(add_px_op.inputs) != 2 or len(add_px_op.outputs) != 1:
                continue
            if mul_px_out not in {str(v) for v in list(add_px_op.inputs)}:
                continue
            add_px_side = str(add_px_op.inputs[1]) if str(add_px_op.inputs[0]) == mul_px_out else str(add_px_op.inputs[0])
            if not _is_scalar_const_tensor(add_px_side):
                continue
            add_px_out = str(add_px_op.outputs[0])

            div_t_idx = _single_user(tensor_name=add_px_out, consumers=consumers)
            if div_t_idx is None:
                continue
            div_t_op = model_ir.operators[int(div_t_idx)]
            if str(div_t_op.op_type) != "DIV" or len(div_t_op.inputs) != 2 or len(div_t_op.outputs) != 1:
                continue
            div_t_inputs = [str(v) for v in list(div_t_op.inputs)]
            if add_px_out == div_t_inputs[0]:
                div_t_side = div_t_inputs[1]
            elif add_px_out == div_t_inputs[1]:
                div_t_side = div_t_inputs[0]
            else:
                continue
            if not _is_scalar_const_tensor(div_t_side):
                continue
            t_out = str(div_t_op.outputs[0])

            mul_neg_idx = _single_user(tensor_name=mul_sq_out, consumers=consumers)
            if mul_neg_idx is None:
                continue
            mul_neg_op = model_ir.operators[int(mul_neg_idx)]
            if str(mul_neg_op.op_type) != "MUL" or len(mul_neg_op.inputs) != 2 or len(mul_neg_op.outputs) != 1:
                continue
            if mul_sq_out not in {str(v) for v in list(mul_neg_op.inputs)}:
                continue
            mul_neg_side = (
                str(mul_neg_op.inputs[1])
                if str(mul_neg_op.inputs[0]) == mul_sq_out
                else str(mul_neg_op.inputs[0])
            )
            if not _is_scalar_const_tensor(mul_neg_side):
                continue
            mul_neg_out = str(mul_neg_op.outputs[0])

            exp_idx = _single_user(tensor_name=mul_neg_out, consumers=consumers)
            if exp_idx is None:
                continue
            exp_op = model_ir.operators[int(exp_idx)]
            if str(exp_op.op_type) != "EXP" or len(exp_op.inputs) != 1 or len(exp_op.outputs) != 1:
                continue
            if str(exp_op.inputs[0]) != mul_neg_out:
                continue
            exp_out = str(exp_op.outputs[0])

            t_users = sorted({int(v) for v in consumers.get(t_out, [])})
            if len(t_users) != 5:
                continue

            s1_mul_idx: Optional[int] = None
            for user_idx in t_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) != "MUL" or len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                    continue
                in0 = str(user_op.inputs[0])
                in1 = str(user_op.inputs[1])
                side_name: Optional[str] = None
                if in0 == t_out:
                    side_name = in1
                elif in1 == t_out:
                    side_name = in0
                if side_name is None:
                    continue
                if _is_scalar_const_tensor(side_name):
                    s1_mul_idx = int(user_idx)
                    break
            if s1_mul_idx is None:
                continue

            chain_indices: List[int] = []
            chain_ops: List[OperatorIR] = []
            s1_mul_op = model_ir.operators[int(s1_mul_idx)]
            s1_mul_out = str(s1_mul_op.outputs[0])
            chain_indices.append(int(s1_mul_idx))
            chain_ops.append(s1_mul_op)
            current = str(s1_mul_out)

            chain_ok = True
            for _ in range(4):
                add_idx = _single_user(tensor_name=current, consumers=consumers)
                if add_idx is None:
                    chain_ok = False
                    break
                add_op = model_ir.operators[int(add_idx)]
                if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                    chain_ok = False
                    break
                add_inputs = [str(v) for v in list(add_op.inputs)]
                if current == add_inputs[0]:
                    add_side = add_inputs[1]
                elif current == add_inputs[1]:
                    add_side = add_inputs[0]
                else:
                    chain_ok = False
                    break
                if not _is_scalar_const_tensor(add_side):
                    chain_ok = False
                    break
                chain_indices.append(int(add_idx))
                chain_ops.append(add_op)
                add_out = str(add_op.outputs[0])

                mul_idx = _single_user(tensor_name=add_out, consumers=consumers)
                if mul_idx is None:
                    chain_ok = False
                    break
                mul_op = model_ir.operators[int(mul_idx)]
                if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                    chain_ok = False
                    break
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                if add_out == mul_inputs[0]:
                    mul_side = mul_inputs[1]
                elif add_out == mul_inputs[1]:
                    mul_side = mul_inputs[0]
                else:
                    chain_ok = False
                    break
                if str(mul_side) != str(t_out):
                    chain_ok = False
                    break
                chain_indices.append(int(mul_idx))
                chain_ops.append(mul_op)
                current = str(mul_op.outputs[0])
            if not chain_ok:
                continue
            poly_out = str(current)

            mul_poly_exp_idx = _single_user(tensor_name=poly_out, consumers=consumers)
            if mul_poly_exp_idx is None:
                continue
            mul_poly_exp_op = model_ir.operators[int(mul_poly_exp_idx)]
            if str(mul_poly_exp_op.op_type) != "MUL" or len(mul_poly_exp_op.inputs) != 2 or len(mul_poly_exp_op.outputs) != 1:
                continue
            mpex_inputs = [str(v) for v in list(mul_poly_exp_op.inputs)]
            if set(mpex_inputs) != {str(poly_out), str(exp_out)}:
                continue
            mul_poly_exp_out = str(mul_poly_exp_op.outputs[0])

            if sorted({int(v) for v in consumers.get(exp_out, [])}) != [int(mul_poly_exp_idx)]:
                continue

            sub_idx = _single_user(tensor_name=mul_poly_exp_out, consumers=consumers)
            if sub_idx is None:
                continue
            sub_op = model_ir.operators[int(sub_idx)]
            if str(sub_op.op_type) != "SUB" or len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1:
                continue
            sub_inputs = [str(v) for v in list(sub_op.inputs)]
            if mul_poly_exp_out == sub_inputs[0]:
                sub_side = sub_inputs[1]
            elif mul_poly_exp_out == sub_inputs[1]:
                sub_side = sub_inputs[0]
            else:
                continue
            if not _is_scalar_const_tensor(sub_side):
                continue
            sub_out = str(sub_op.outputs[0])

            final_mul_idx = _single_user(tensor_name=sub_out, consumers=consumers)
            if final_mul_idx is None:
                continue
            final_mul_op = model_ir.operators[int(final_mul_idx)]
            if str(final_mul_op.op_type) != "MUL" or len(final_mul_op.inputs) != 2 or len(final_mul_op.outputs) != 1:
                continue
            final_inputs = [str(v) for v in list(final_mul_op.inputs)]
            if set(final_inputs) != {str(sign_out), str(sub_out)}:
                continue
            if sorted({int(v) for v in consumers.get(sign_out, [])}) != [int(final_mul_idx)]:
                continue
            final_out = str(final_mul_op.outputs[0])

            terminal_mode = ""
            post_idx: Optional[int] = None
            post_output_name = str(final_out)
            final_users = sorted({int(v) for v in consumers.get(final_out, [])})
            if final_out in model_outputs and len(final_users) == 0:
                terminal_mode = "model_output"
            elif len(final_users) == 1:
                cand_idx = int(final_users[0])
                cand_op = model_ir.operators[int(cand_idx)]
                if (
                    str(cand_op.op_type) == "TRANSPOSE"
                    and len(cand_op.inputs) >= 2
                    and len(cand_op.outputs) == 1
                    and str(cand_op.inputs[0]) == str(final_out)
                ):
                    perm_post = _read_transpose_perm(model_ir, cand_op)
                    if perm_post is not None and list(perm_post) == list(perm_post_expected):
                        terminal_mode = "inverse_post_transpose"
                        post_idx = int(cand_idx)
                        post_output_name = str(cand_op.outputs[0])
            if terminal_mode == "":
                continue

            # Rewire chain head from transposed tensor to model input tensor.
            _set_operator_inputs(
                model_ir=model_ir,
                op=abs_op,
                new_inputs=[str(pre_input_name)],
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=sign_op,
                new_inputs=[str(pre_input_name)],
            )
            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=final_mul_op,
                    new_outputs=[str(post_output_name)],
                )

            # Update tensor metadata from transposed layout to source layout.
            chain_all_ops: List[OperatorIR] = [
                abs_op,
                sign_op,
                mul_px_op,
                add_px_op,
                div_t_op,
                mul_sq_op,
                mul_neg_op,
                exp_op,
            ] + chain_ops + [mul_poly_exp_op, sub_op, final_mul_op]
            seen_names: set = set()
            for op in chain_all_ops:
                if len(op.outputs) != 1:
                    continue
                out_name = str(op.outputs[0])
                if out_name in seen_names:
                    continue
                seen_names.add(out_name)
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_post_expected,
                )

            remove_indices = {int(pre_idx)}
            if post_idx is not None:
                remove_indices.add(int(post_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_erf_transpose_passthrough_chains": int(rewritten)}
