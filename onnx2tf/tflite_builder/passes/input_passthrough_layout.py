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
