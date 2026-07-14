from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _invert_perm,
    _is_per_tensor_quantization,
    _is_singleton_constant_tensor,
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
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)


def _optimize_leading_input_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers
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
                    graph_index=graph_index,
                )
            elif len(first_input_names) > 1 and first_input_names[1] == pre_output_name:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=first_op,
                    new_inputs=[first_input_names[0], pre_input_name],
                    graph_index=graph_index,
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
                        graph_index=graph_index,
                    )

            # Chain tail now produces post-transpose output name directly.
            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=chain_ops[-1],
                    new_outputs=[post_output_name],
                    graph_index=graph_index,
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
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_leading_input_transpose_passthrough_chains": int(rewritten)}


def _optimize_asin_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
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
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=atan2_op,
                new_inputs=[
                    str(pre_input_name) if str(v) == str(pre_output_name) else str(v)
                    for v in list(atan2_op.inputs)
                ],
                graph_index=graph_index,
            )

            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=atan2_op,
                    new_outputs=[str(post_output_name)],
                    graph_index=graph_index,
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
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_asin_transpose_passthrough_chains": int(rewritten)}


def _optimize_hardsigmoid_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers

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
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=hs_terminal_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
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
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_hardsigmoid_transpose_passthrough_chains": int(rewritten)}


def _optimize_erf_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

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
        consumers = graph_index.consumers
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
                graph_index=graph_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=sign_op,
                new_inputs=[str(pre_input_name)],
                graph_index=graph_index,
            )
            if terminal_mode == "inverse_post_transpose":
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=final_mul_op,
                    new_outputs=[str(post_output_name)],
                    graph_index=graph_index,
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
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_erf_transpose_passthrough_chains": int(rewritten)}


def run_input_unary_passthrough_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run the repeated leading-input, Asin, and Erf passthrough sequence."""

    leading_types = {
        "CAST",
        "QUANTIZE",
        "DEQUANTIZE",
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "ATAN2",
    }

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        has_transpose = False
        has_relevant = False
        relevant = leading_types | {"ABS", "SIGN"}
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            has_transpose = has_transpose or op_type == "TRANSPOSE"
            has_relevant = has_relevant or op_type in relevant
            if has_transpose and has_relevant:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _leading_transpose_users(pass_state: ModelIRPassState) -> List[set[str]]:
        model_inputs = {str(name) for name in pass_state.model_ir.inputs}
        results: List[set[str]] = []
        for pre_op in pass_state.model_ir.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.inputs[0]) not in model_inputs
            ):
                continue
            users = pass_state.graph_index.consumer_indices(str(pre_op.outputs[0]))
            results.append(
                {
                    str(pass_state.model_ir.operators[int(index)].op_type)
                    for index in users
                }
            )
        return results

    def _has_leading_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            len(user_types) == 1 and bool(user_types & leading_types)
            for user_types in _leading_transpose_users(pass_state)
        )

    def _has_asin_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            {"MUL", "ATAN2"}.issubset(user_types)
            for user_types in _leading_transpose_users(pass_state)
        )

    def _has_erf_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            {"ABS", "SIGN"}.issubset(user_types)
            for user_types in _leading_transpose_users(pass_state)
        )

    def _run_leading(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_leading_input_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_leading_input_transpose_passthrough_chains", 0)),
        }

    def _run_asin(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_asin_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_asin_transpose_passthrough_chains", 0)),
        }

    def _run_erf(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_erf_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_erf_transpose_passthrough_chains", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.leading_input_passthrough",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_leading,
                precondition=_has_leading_candidate,
                priority=10,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.asin_passthrough",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_asin,
                precondition=_has_asin_candidate,
                priority=20,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.erf_passthrough",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_erf,
                precondition=_has_erf_candidate,
                priority=30,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            "rewritten_leading_input_transpose_passthrough_chains": 0,
            "rewritten_asin_transpose_passthrough_chains": 0,
            "rewritten_erf_transpose_passthrough_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_hardswish_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold transpose wrappers around pseudo-op-expanded HardSwish-like chains.

    Target:
      X --TRANSPOSE(P)--> x_t
      x_t --ADD(c3)--> a --RELU6--> r --(DIV(c6)|MUL(c6_inv))--> d
        or
      x_t --ADD(c3)--> a --MUL(c6_inv)--> d
      MUL(x_t, d) --> y_t
      y_t --TRANSPOSE(inv(P))--> Y

    Rewrite:
      X --ADD(c3)--> a --RELU6--> r --(DIV(c6)|MUL(c6_inv))--> d
        or
      X --ADD(c3)--> a --MUL(c6_inv)--> d
      MUL(X, d) --> Y

    Safety:
    - Strict residual topology (ADD -> optional RELU6 -> (DIV|MUL const) -> residual MUL).
    - ADD/(DIV|MUL) side inputs must be singleton constants.
    - Single-consumer chain on the main path.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers

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
            if len(pre_users) != 2:
                continue

            add_idx = None
            mul_idx = None
            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "ADD":
                    add_idx = int(user_idx)
                elif user_type == "MUL":
                    mul_idx = int(user_idx)
            if add_idx is None or mul_idx is None:
                continue

            add_op = model_ir.operators[int(add_idx)]
            if len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            add_inputs = [str(v) for v in add_op.inputs]
            if pre_output_name == add_inputs[0]:
                add_data_input_index = 0
                add_side_input_name = add_inputs[1]
            elif pre_output_name == add_inputs[1]:
                add_data_input_index = 1
                add_side_input_name = add_inputs[0]
            else:
                continue
            if not _is_singleton_constant_tensor(model_ir, add_side_input_name):
                continue
            add_out_name = str(add_op.outputs[0])

            add_users = [int(v) for v in consumers.get(add_out_name, [])]
            if len(add_users) != 1:
                continue
            stage_out_name: Optional[str] = None
            intermediates_to_permute: List[str] = [add_out_name]

            stage1_idx = int(add_users[0])
            stage1_op = model_ir.operators[int(stage1_idx)]
            stage1_type = str(stage1_op.op_type)

            if stage1_type == "RELU6":
                if len(stage1_op.inputs) != 1 or len(stage1_op.outputs) != 1:
                    continue
                if str(stage1_op.inputs[0]) != add_out_name:
                    continue
                relu_out_name = str(stage1_op.outputs[0])
                intermediates_to_permute.append(relu_out_name)

                relu_users = [int(v) for v in consumers.get(relu_out_name, [])]
                if len(relu_users) != 1:
                    continue
                stage2_idx = int(relu_users[0])
                stage2_op = model_ir.operators[int(stage2_idx)]
                stage2_type = str(stage2_op.op_type)
                if stage2_type not in {"DIV", "MUL"}:
                    continue
                if len(stage2_op.inputs) != 2 or len(stage2_op.outputs) != 1:
                    continue
                stage2_inputs = [str(v) for v in stage2_op.inputs]
                if relu_out_name == stage2_inputs[0]:
                    stage2_side_input_name = stage2_inputs[1]
                elif relu_out_name == stage2_inputs[1]:
                    stage2_side_input_name = stage2_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, stage2_side_input_name):
                    continue
                stage_out_name = str(stage2_op.outputs[0])
                intermediates_to_permute.append(stage_out_name)
            elif stage1_type in {"DIV", "MUL"}:
                if len(stage1_op.inputs) != 2 or len(stage1_op.outputs) != 1:
                    continue
                stage1_inputs = [str(v) for v in stage1_op.inputs]
                if add_out_name == stage1_inputs[0]:
                    stage1_side_input_name = stage1_inputs[1]
                elif add_out_name == stage1_inputs[1]:
                    stage1_side_input_name = stage1_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, stage1_side_input_name):
                    continue
                stage_out_name = str(stage1_op.outputs[0])
                intermediates_to_permute.append(stage_out_name)
            else:
                continue

            if stage_out_name is None:
                continue
            stage_users = [int(v) for v in consumers.get(stage_out_name, [])]
            if len(stage_users) != 1 or int(stage_users[0]) != int(mul_idx):
                continue

            mul_op = model_ir.operators[int(mul_idx)]
            if len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_inputs = [str(v) for v in mul_op.inputs]
            if pre_output_name == mul_inputs[0] and stage_out_name == mul_inputs[1]:
                mul_data_input_index = 0
            elif pre_output_name == mul_inputs[1] and stage_out_name == mul_inputs[0]:
                mul_data_input_index = 1
            else:
                continue
            mul_out_name = str(mul_op.outputs[0])

            mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_users) != 1:
                continue
            post_idx = int(mul_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != mul_out_name:
                continue
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or perm_post != perm_post_expected:
                continue
            post_output_name = str(post_op.outputs[0])

            # Rewire the pseudo-HardSwish chain to consume NHWC source directly.
            _replace_operator_input_at(
                model_ir=model_ir,
                op=add_op,
                input_index=int(add_data_input_index),
                new_input_name=pre_input_name,
                graph_index=graph_index,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul_op,
                input_index=int(mul_data_input_index),
                new_input_name=pre_input_name,
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )

            # Update intermediate metadata from transposed layout to source layout.
            for intermediate_name in intermediates_to_permute:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(intermediate_name, None),
                    perm_post_expected,
                )

            # Keep final tensor metadata stable on the post-transpose name.
            pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
            old_mul_tensor = model_ir.tensors.get(mul_out_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if post_output_tensor is not None:
                if pre_input_tensor is not None:
                    post_output_tensor.shape = [int(v) for v in list(pre_input_tensor.shape)]
                    post_output_tensor.shape_signature = (
                        [int(v) for v in list(pre_input_tensor.shape_signature)]
                        if pre_input_tensor.shape_signature is not None
                        else [int(v) for v in list(pre_input_tensor.shape)]
                    )
                if old_mul_tensor is not None:
                    post_output_tensor.dtype = str(old_mul_tensor.dtype)
                    post_output_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)

            remove_indices = sorted(list({int(pre_idx), int(post_idx)}), reverse=True)
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_hardswish_transpose_passthrough_chains": int(rewritten)}


def _optimize_hardsigmoid_mul_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold transpose wrappers around HardSigmoid-expanded residual MUL chains.

    Target:
      X --TRANSPOSE(P)--> x_t
      x_t --MUL(alpha)--> m1 --ADD(beta)--> a1 --(MAXIMUM(0)->MINIMUM(1) | RELU_0_TO_1)--> h1
      MUL(x_t, h1) --> y_t
      y_t --TRANSPOSE(inv(P))--> Y
      (optional branch)
      y_t --MEAN(axes_nchw, keepDims=True)--> m_t --TRANSPOSE(inv(P))--> M

    Rewrite:
      X --MUL(alpha)--> m1 --ADD(beta)--> a1 --(MAXIMUM(0)->MINIMUM(1) | RELU_0_TO_1)--> h1
      MUL(X, h1) --> Y
      MEAN(Y, axes_nhwc, keepDims=True) --> M

    Safety:
    - Strict HardSigmoid decomposition topology on one branch.
    - HardSigmoid branch constants must be singleton constants.
    - Chain on the HardSigmoid branch must be linear.
    - Optional MEAN branch must be rank-4 constant-axes and only feed inverse post-transpose adapters.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

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
            if len(pre_users) != 2:
                continue

            def _try_match_pair(hs_entry_idx: int, residual_mul_idx: int) -> Optional[Dict[str, Any]]:
                hs_entry_op = model_ir.operators[int(hs_entry_idx)]
                hs_entry_type = str(hs_entry_op.op_type)
                hs_data_idx: Optional[int] = None
                hs_data_input_index: Optional[int] = None
                hs_out_name: Optional[str] = None
                hs_intermediate_names: List[str] = []

                if hs_entry_type == "MUL" and len(hs_entry_op.inputs) == 2 and len(hs_entry_op.outputs) == 1:
                    hs_mul_inputs = [str(v) for v in list(hs_entry_op.inputs)]
                    if pre_output_name == hs_mul_inputs[0]:
                        hs_data_input_index = 0
                        hs_mul_side_name = hs_mul_inputs[1]
                    elif pre_output_name == hs_mul_inputs[1]:
                        hs_data_input_index = 1
                        hs_mul_side_name = hs_mul_inputs[0]
                    else:
                        return None
                    if not _is_singleton_constant_tensor(model_ir, hs_mul_side_name):
                        return None
                    hs_data_idx = int(hs_entry_idx)
                    hs_mul_out_name = str(hs_entry_op.outputs[0])

                    hs_mul_users = [int(v) for v in consumers.get(hs_mul_out_name, [])]
                    if len(hs_mul_users) != 1:
                        return None
                    hs_add_idx = int(hs_mul_users[0])
                    hs_add_op = model_ir.operators[int(hs_add_idx)]
                    if str(hs_add_op.op_type) != "ADD" or len(hs_add_op.inputs) != 2 or len(hs_add_op.outputs) != 1:
                        return None
                    hs_add_inputs = [str(v) for v in list(hs_add_op.inputs)]
                    if hs_mul_out_name == hs_add_inputs[0]:
                        hs_add_side_name = hs_add_inputs[1]
                    elif hs_mul_out_name == hs_add_inputs[1]:
                        hs_add_side_name = hs_add_inputs[0]
                    else:
                        return None
                    if not _is_singleton_constant_tensor(model_ir, hs_add_side_name):
                        return None
                    hs_add_out_name = str(hs_add_op.outputs[0])

                    hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                    if len(hs_add_users) != 1:
                        return None
                    hs_clamp_idx = int(hs_add_users[0])
                    hs_clamp_op = model_ir.operators[int(hs_clamp_idx)]
                    hs_intermediate_names = [str(hs_mul_out_name), str(hs_add_out_name)]
                    if (
                        str(hs_clamp_op.op_type) == "RELU_0_TO_1"
                        and len(hs_clamp_op.inputs) == 1
                        and len(hs_clamp_op.outputs) == 1
                        and str(hs_clamp_op.inputs[0]) == hs_add_out_name
                    ):
                        hs_out_name = str(hs_clamp_op.outputs[0])
                    else:
                        if (
                            str(hs_clamp_op.op_type) != "MAXIMUM"
                            or len(hs_clamp_op.inputs) != 2
                            or len(hs_clamp_op.outputs) != 1
                        ):
                            return None
                        hs_max_inputs = [str(v) for v in list(hs_clamp_op.inputs)]
                        if hs_add_out_name == hs_max_inputs[0]:
                            hs_max_side_name = hs_max_inputs[1]
                        elif hs_add_out_name == hs_max_inputs[1]:
                            hs_max_side_name = hs_max_inputs[0]
                        else:
                            return None
                        if not _is_singleton_constant_tensor(model_ir, hs_max_side_name):
                            return None
                        hs_max_out_name = str(hs_clamp_op.outputs[0])
                        hs_intermediate_names.append(str(hs_max_out_name))

                        hs_max_users = [int(v) for v in consumers.get(hs_max_out_name, [])]
                        if len(hs_max_users) != 1:
                            return None
                        hs_min_idx = int(hs_max_users[0])
                        hs_min_op = model_ir.operators[int(hs_min_idx)]
                        if (
                            str(hs_min_op.op_type) != "MINIMUM"
                            or len(hs_min_op.inputs) != 2
                            or len(hs_min_op.outputs) != 1
                        ):
                            return None
                        hs_min_inputs = [str(v) for v in list(hs_min_op.inputs)]
                        if hs_max_out_name == hs_min_inputs[0]:
                            hs_min_side_name = hs_min_inputs[1]
                        elif hs_max_out_name == hs_min_inputs[1]:
                            hs_min_side_name = hs_min_inputs[0]
                        else:
                            return None
                        if not _is_singleton_constant_tensor(model_ir, hs_min_side_name):
                            return None
                        hs_out_name = str(hs_min_op.outputs[0])
                elif hs_entry_type == "ADD" and len(hs_entry_op.inputs) == 2 and len(hs_entry_op.outputs) == 1:
                    hs_add_inputs = [str(v) for v in list(hs_entry_op.inputs)]
                    if pre_output_name == hs_add_inputs[0]:
                        hs_data_input_index = 0
                        hs_add_side_name = hs_add_inputs[1]
                    elif pre_output_name == hs_add_inputs[1]:
                        hs_data_input_index = 1
                        hs_add_side_name = hs_add_inputs[0]
                    else:
                        return None
                    if not _is_singleton_constant_tensor(model_ir, hs_add_side_name):
                        return None
                    hs_data_idx = int(hs_entry_idx)
                    hs_add_out_name = str(hs_entry_op.outputs[0])
                    hs_intermediate_names.append(str(hs_add_out_name))

                    hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                    if len(hs_add_users) != 1:
                        return None
                    hs_mul_idx = int(hs_add_users[0])
                    hs_mul_op = model_ir.operators[int(hs_mul_idx)]
                    if str(hs_mul_op.op_type) != "MUL" or len(hs_mul_op.inputs) != 2 or len(hs_mul_op.outputs) != 1:
                        return None
                    hs_mul_inputs = [str(v) for v in list(hs_mul_op.inputs)]
                    if hs_add_out_name == hs_mul_inputs[0]:
                        hs_mul_side_name = hs_mul_inputs[1]
                    elif hs_add_out_name == hs_mul_inputs[1]:
                        hs_mul_side_name = hs_mul_inputs[0]
                    else:
                        return None
                    if not _is_singleton_constant_tensor(model_ir, hs_mul_side_name):
                        return None
                    hs_out_name = str(hs_mul_op.outputs[0])
                    hs_intermediate_names.append(str(hs_out_name))
                else:
                    return None

                if hs_data_idx is None or hs_data_input_index is None or hs_out_name is None:
                    return None

                residual_mul_op = model_ir.operators[int(residual_mul_idx)]
                if (
                    str(residual_mul_op.op_type) != "MUL"
                    or len(residual_mul_op.inputs) != 2
                    or len(residual_mul_op.outputs) != 1
                ):
                    return None
                residual_mul_inputs = [str(v) for v in list(residual_mul_op.inputs)]
                if pre_output_name == residual_mul_inputs[0] and hs_out_name == residual_mul_inputs[1]:
                    residual_mul_data_input_index = 0
                elif pre_output_name == residual_mul_inputs[1] and hs_out_name == residual_mul_inputs[0]:
                    residual_mul_data_input_index = 1
                else:
                    return None
                residual_mul_out_name = str(residual_mul_op.outputs[0])
                residual_mul_users = [int(v) for v in consumers.get(residual_mul_out_name, []) if int(v) != int(residual_mul_idx)]
                if len(residual_mul_users) == 0:
                    return None

                post_indices: List[int] = []
                post_output_names: List[str] = []
                legacy_users: List[int] = []
                mean_branches: List[Dict[str, Any]] = []
                valid_users = True
                for user_idx in residual_mul_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) == "TRANSPOSE"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == residual_mul_out_name
                    ):
                        perm_post = _read_transpose_perm(model_ir, user_op)
                        if perm_post is None or perm_post != perm_post_expected:
                            valid_users = False
                            break
                        post_indices.append(int(user_idx))
                        post_output_names.append(str(user_op.outputs[0]))
                    elif (
                        str(user_op.op_type) == "MEAN"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == residual_mul_out_name
                        and bool(user_op.options.get("keepDims", False))
                    ):
                        mean_idx = int(user_idx)
                        mean_axes_name = str(user_op.inputs[1])
                        mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                        mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                        if mean_axes_vals is None or len(mean_axes_vals) == 0:
                            valid_users = False
                            break
                        mean_out_name = str(user_op.outputs[0])
                        if mean_out_name in model_outputs:
                            valid_users = False
                            break
                        mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
                        if len(mean_users) == 0:
                            valid_users = False
                            break
                        # Keep legacy NCHW adapter path when MEAN branch terminates
                        # at graph output through a single inverse transpose.
                        if len(mean_users) == 1:
                            only_mean_user_op = model_ir.operators[int(mean_users[0])]
                            if (
                                str(only_mean_user_op.op_type) == "TRANSPOSE"
                                and len(only_mean_user_op.inputs) >= 2
                                and len(only_mean_user_op.outputs) == 1
                                and str(only_mean_user_op.inputs[0]) == mean_out_name
                                and _read_transpose_perm(model_ir, only_mean_user_op) == perm_post_expected
                                and str(only_mean_user_op.outputs[0]) in model_outputs
                            ):
                                legacy_users.append(int(mean_idx))
                                continue
                        mean_post_indices: List[int] = []
                        mean_post_output_names: List[str] = []
                        for mean_user_idx in mean_users:
                            mean_user_op = model_ir.operators[int(mean_user_idx)]
                            if (
                                str(mean_user_op.op_type) != "TRANSPOSE"
                                or len(mean_user_op.inputs) < 2
                                or len(mean_user_op.outputs) != 1
                                or str(mean_user_op.inputs[0]) != mean_out_name
                            ):
                                valid_users = False
                                break
                            mean_perm_post = _read_transpose_perm(model_ir, mean_user_op)
                            if mean_perm_post is None or mean_perm_post != perm_post_expected:
                                valid_users = False
                                break
                            mean_post_output_name = str(mean_user_op.outputs[0])
                            mean_post_indices.append(int(mean_user_idx))
                            mean_post_output_names.append(mean_post_output_name)
                        if not valid_users or len(mean_post_indices) == 0:
                            valid_users = False
                            break
                        mean_branches.append(
                            {
                                "mean_idx": int(mean_idx),
                                "mean_axes_name": str(mean_axes_name),
                                "mean_axes_vals": [int(v) for v in mean_axes_vals],
                                "mean_out_name": str(mean_out_name),
                                "mean_post_indices": [int(v) for v in mean_post_indices],
                                "mean_post_output_names": [str(v) for v in mean_post_output_names],
                            }
                        )
                    else:
                        legacy_users.append(int(user_idx))
                if not valid_users or len(post_indices) == 0:
                    return None

                return {
                    "hs_data_idx": int(hs_data_idx),
                    "hs_data_input_index": int(hs_data_input_index),
                    "hs_intermediate_names": [str(v) for v in hs_intermediate_names],
                    "residual_mul_idx": int(residual_mul_idx),
                    "residual_mul_data_input_index": int(residual_mul_data_input_index),
                    "residual_mul_out_name": str(residual_mul_out_name),
                    "post_indices": [int(v) for v in post_indices],
                    "post_output_names": [str(v) for v in post_output_names],
                    "legacy_users": [int(v) for v in legacy_users],
                    "mean_branches": list(mean_branches),
                }

            matched = _try_match_pair(int(pre_users[0]), int(pre_users[1]))
            if matched is None:
                matched = _try_match_pair(int(pre_users[1]), int(pre_users[0]))
            if matched is None:
                continue

            hs_data_op = model_ir.operators[int(matched["hs_data_idx"])]
            residual_mul_op = model_ir.operators[int(matched["residual_mul_idx"])]
            residual_mul_out_name = str(matched["residual_mul_out_name"])
            post_indices = [int(v) for v in list(matched.get("post_indices", []))]
            post_output_names = [str(v) for v in list(matched.get("post_output_names", []))]
            legacy_users = [int(v) for v in list(matched.get("legacy_users", []))]
            mean_branches = list(matched.get("mean_branches", []))
            if len(post_indices) == 0 or len(post_output_names) == 0:
                continue
            representative_output_name = str(post_output_names[0])

            _replace_operator_input_at(
                model_ir=model_ir,
                op=hs_data_op,
                input_index=int(matched["hs_data_input_index"]),
                new_input_name=pre_input_name,
                graph_index=graph_index,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=residual_mul_op,
                input_index=int(matched["residual_mul_data_input_index"]),
                new_input_name=pre_input_name,
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=residual_mul_op,
                new_outputs=[representative_output_name],
                graph_index=graph_index,
            )
            for alias_output_name in post_output_names[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    alias_output_name,
                    representative_output_name,
                    graph_index=graph_index,
                )

            for intermediate_name in [str(v) for v in list(matched.get("hs_intermediate_names", []))]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(intermediate_name, None),
                    perm_post_expected,
                )

            old_mul_tensor = model_ir.tensors.get(residual_mul_out_name, None)
            representative_tensor = model_ir.tensors.get(representative_output_name, None)
            if representative_tensor is not None and old_mul_tensor is not None:
                representative_tensor.dtype = str(old_mul_tensor.dtype)
                representative_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                _permute_tensor_metadata_if_rank_matches(
                    representative_tensor,
                    perm_post_expected,
                )

            mean_post_remove_indices: List[int] = []
            mean_rewrite_ok = True
            for mean_branch in mean_branches:
                mean_idx = int(mean_branch["mean_idx"])
                mean_axes_name = str(mean_branch["mean_axes_name"])
                mean_axes_vals = [int(v) for v in list(mean_branch["mean_axes_vals"])]
                mean_out_name = str(mean_branch["mean_out_name"])
                mean_post_indices = [int(v) for v in list(mean_branch["mean_post_indices"])]
                mean_post_output_names = [str(v) for v in list(mean_branch["mean_post_output_names"])]
                if len(mean_post_indices) == 0 or len(mean_post_output_names) == 0:
                    mean_rewrite_ok = False
                    break

                mapped_axes: List[int] = []
                valid_axes = True
                for axis in mean_axes_vals:
                    a = int(axis)
                    if a < 0:
                        a += 4
                    if a < 0 or a >= 4:
                        valid_axes = False
                        break
                    mapped_axes.append(int(perm_pre[int(a)]))
                if not valid_axes:
                    mean_rewrite_ok = False
                    break

                mean_op = model_ir.operators[int(mean_idx)]
                mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                if mean_axes_tensor is None or not _write_const_ints_to_tensor(mean_axes_tensor, mapped_axes):
                    mean_rewrite_ok = False
                    break

                mean_inputs = [str(v) for v in list(mean_op.inputs)]
                if len(mean_inputs) == 0:
                    mean_rewrite_ok = False
                    break
                mean_inputs[0] = representative_output_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mean_op,
                    new_inputs=mean_inputs,
                    graph_index=graph_index,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mean_out_name, None),
                    perm_post_expected,
                )

                representative_mean_output_name = str(mean_post_output_names[0])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=mean_op,
                    new_outputs=[representative_mean_output_name],
                    graph_index=graph_index,
                )
                for alias_mean_output_name in mean_post_output_names[1:]:
                    _replace_tensor_inputs(
                        model_ir,
                        alias_mean_output_name,
                        representative_mean_output_name,
                        graph_index=graph_index,
                    )

                old_mean_tensor = model_ir.tensors.get(mean_out_name, None)
                representative_mean_tensor = model_ir.tensors.get(representative_mean_output_name, None)
                if representative_mean_tensor is not None and old_mean_tensor is not None:
                    representative_mean_tensor.dtype = str(old_mean_tensor.dtype)
                    representative_mean_tensor.quantization = _clone_quantization(old_mean_tensor.quantization)
                    _permute_tensor_metadata_if_rank_matches(
                        representative_mean_tensor,
                        perm_post_expected,
                    )

                mean_post_remove_indices.extend(int(v) for v in mean_post_indices)

            if not mean_rewrite_ok:
                continue

            preserve_nchw_adapter = len(legacy_users) > 0 or residual_mul_out_name in model_ir.outputs
            if preserve_nchw_adapter:
                keep_post_idx = int(post_indices[0])
                keep_post_op = model_ir.operators[keep_post_idx]
                keep_perm_name = str(keep_post_op.inputs[1])
                keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                if keep_perm_tensor is not None:
                    keep_perm_tensor.data = np.asarray(perm_pre, dtype=np.int32)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_inputs=[representative_output_name, keep_perm_name],
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[residual_mul_out_name],
                    graph_index=graph_index,
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = sorted(
                list({int(pre_idx), *post_remove_indices, *[int(v) for v in mean_post_remove_indices]}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_hardsigmoid_mul_transpose_passthrough_chains": int(rewritten)}


def run_hard_activation_passthrough_cleanup(
    model_ir: ModelIR,
    *,
    include_hardswish: bool = True,
    include_hardsigmoid: bool = True,
    include_hardsigmoid_mul: bool = True,
    reverse_hardsigmoid_order: bool = False,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run hard-activation passthrough specs in the selected production order."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        has_transpose = False
        has_add = False
        has_mul = False
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            has_transpose = has_transpose or op_type == "TRANSPOSE"
            has_add = has_add or op_type == "ADD"
            has_mul = has_mul or op_type == "MUL"
            if has_transpose and has_mul and has_add:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _single_user(pass_state: ModelIRPassState, tensor_name: str) -> Optional[OperatorIR]:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        return pass_state.model_ir.operators[int(users[0])]

    def _binary_side_is_scalar(
        pass_state: ModelIRPassState,
        op: OperatorIR,
        data_name: str,
    ) -> bool:
        if len(op.inputs) != 2 or str(data_name) not in [str(v) for v in op.inputs]:
            return False
        inputs = [str(v) for v in op.inputs]
        side_name = inputs[1] if inputs[0] == str(data_name) else inputs[0]
        return _is_singleton_constant_tensor(pass_state.model_ir, side_name)

    def _has_inverse_post(
        pass_state: ModelIRPassState,
        tensor_name: str,
        perm_pre: List[int],
    ) -> bool:
        inverse = _invert_perm(perm_pre)
        if inverse is None:
            return False
        for index in pass_state.graph_index.consumer_indices(str(tensor_name)):
            op = pass_state.model_ir.operators[int(index)]
            if (
                str(op.op_type) == "TRANSPOSE"
                and _read_transpose_perm(pass_state.model_ir, op) == inverse
            ):
                return True
        return False

    def _match_hardsigmoid_terminal(
        pass_state: ModelIRPassState,
        *,
        source_name: str,
        mul_op: OperatorIR,
    ) -> Optional[str]:
        if str(mul_op.op_type) != "MUL" or len(mul_op.outputs) != 1:
            return None
        if not _binary_side_is_scalar(pass_state, mul_op, source_name):
            return None
        add_op = _single_user(pass_state, str(mul_op.outputs[0]))
        if (
            add_op is None
            or str(add_op.op_type) != "ADD"
            or len(add_op.outputs) != 1
            or not _binary_side_is_scalar(pass_state, add_op, str(mul_op.outputs[0]))
        ):
            return None
        clamp_op = _single_user(pass_state, str(add_op.outputs[0]))
        if clamp_op is None or len(clamp_op.outputs) != 1:
            return None
        if str(clamp_op.op_type) == "RELU_0_TO_1":
            return str(clamp_op.outputs[0])
        if (
            str(clamp_op.op_type) != "MAXIMUM"
            or not _binary_side_is_scalar(pass_state, clamp_op, str(add_op.outputs[0]))
        ):
            return None
        minimum_op = _single_user(pass_state, str(clamp_op.outputs[0]))
        if (
            minimum_op is None
            or str(minimum_op.op_type) != "MINIMUM"
            or len(minimum_op.outputs) != 1
            or not _binary_side_is_scalar(
                pass_state,
                minimum_op,
                str(clamp_op.outputs[0]),
            )
        ):
            return None
        return str(minimum_op.outputs[0])

    def _has_hardswish_candidate(pass_state: ModelIRPassState) -> bool:
        for pre_op in pass_state.model_ir.operators:
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1:
                continue
            source_name = str(pre_op.outputs[0])
            user_indices = pass_state.graph_index.consumer_indices(source_name)
            if len(user_indices) != 2:
                continue
            user_ops = [pass_state.model_ir.operators[int(index)] for index in user_indices]
            add_ops = [op for op in user_ops if str(op.op_type) == "ADD"]
            residual_ops = [op for op in user_ops if str(op.op_type) == "MUL"]
            if len(add_ops) != 1 or len(residual_ops) != 1:
                continue
            add_op = add_ops[0]
            residual_op = residual_ops[0]
            if not _binary_side_is_scalar(pass_state, add_op, source_name):
                continue
            branch_input_name = str(add_op.outputs[0])
            branch_op = _single_user(pass_state, branch_input_name)
            if branch_op is None:
                continue
            if str(branch_op.op_type) == "RELU6":
                if len(branch_op.outputs) != 1:
                    continue
                branch_input_name = str(branch_op.outputs[0])
                branch_op = _single_user(pass_state, branch_input_name)
                if branch_op is None:
                    continue
            if (
                str(branch_op.op_type) not in {"MUL", "DIV"}
                or len(branch_op.outputs) != 1
                or not _binary_side_is_scalar(pass_state, branch_op, branch_input_name)
            ):
                continue
            branch_out = str(branch_op.outputs[0])
            if set(str(v) for v in residual_op.inputs) != {source_name, branch_out}:
                continue
            perm_pre = _read_transpose_perm(pass_state.model_ir, pre_op)
            if perm_pre is not None and _has_inverse_post(
                pass_state,
                str(residual_op.outputs[0]),
                perm_pre,
            ):
                return True
        return False

    def _has_hardsigmoid_candidate(pass_state: ModelIRPassState) -> bool:
        for pre_op in pass_state.model_ir.operators:
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1:
                continue
            source_name = str(pre_op.outputs[0])
            users = pass_state.graph_index.consumer_indices(source_name)
            if len(users) != 1:
                continue
            terminal_name = _match_hardsigmoid_terminal(
                pass_state,
                source_name=source_name,
                mul_op=pass_state.model_ir.operators[int(users[0])],
            )
            perm_pre = _read_transpose_perm(pass_state.model_ir, pre_op)
            if terminal_name is not None and perm_pre is not None and _has_inverse_post(
                pass_state,
                terminal_name,
                perm_pre,
            ):
                return True
        return False

    def _has_hardsigmoid_mul_candidate(pass_state: ModelIRPassState) -> bool:
        for pre_op in pass_state.model_ir.operators:
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1:
                continue
            source_name = str(pre_op.outputs[0])
            users = pass_state.graph_index.consumer_indices(source_name)
            if len(users) != 2:
                continue
            for hs_index in users:
                hs_op = pass_state.model_ir.operators[int(hs_index)]
                terminal_name = _match_hardsigmoid_terminal(
                    pass_state,
                    source_name=source_name,
                    mul_op=hs_op,
                )
                if terminal_name is None:
                    continue
                residual_index = users[1] if int(users[0]) == int(hs_index) else users[0]
                residual_op = pass_state.model_ir.operators[int(residual_index)]
                if (
                    str(residual_op.op_type) != "MUL"
                    or len(residual_op.inputs) != 2
                    or len(residual_op.outputs) != 1
                    or set(str(v) for v in residual_op.inputs) != {source_name, terminal_name}
                ):
                    continue
                perm_pre = _read_transpose_perm(pass_state.model_ir, pre_op)
                if perm_pre is not None and _has_inverse_post(
                    pass_state,
                    str(residual_op.outputs[0]),
                    perm_pre,
                ):
                    return True
        return False

    def _run_hardswish(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_hardswish_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_hardswish_transpose_passthrough_chains", 0)),
        }

    def _run_hardsigmoid(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_hardsigmoid_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_hardsigmoid_transpose_passthrough_chains", 0)),
        }

    def _run_hardsigmoid_mul(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_hardsigmoid_mul_transpose_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("rewritten_hardsigmoid_mul_transpose_passthrough_chains", 0)
            ),
        }

    ordered: List[Tuple[str, Any, Any]] = []
    if include_hardswish:
        ordered.append(("layout.hardswish_passthrough", _run_hardswish, _has_hardswish_candidate))
    hardsigmoid_specs = [
        (
            "layout.hardsigmoid_passthrough",
            _run_hardsigmoid,
            _has_hardsigmoid_candidate,
        ),
        (
            "layout.hardsigmoid_mul_passthrough",
            _run_hardsigmoid_mul,
            _has_hardsigmoid_mul_candidate,
        ),
    ]
    if reverse_hardsigmoid_order:
        hardsigmoid_specs.reverse()
    for pass_id, callback, precondition in hardsigmoid_specs:
        if pass_id == "layout.hardsigmoid_passthrough" and not include_hardsigmoid:
            continue
        if pass_id == "layout.hardsigmoid_mul_passthrough" and not include_hardsigmoid_mul:
            continue
        ordered.append((pass_id, callback, precondition))
    if len(ordered) == 0:
        return {
            "rewritten_hardswish_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_mul_transpose_passthrough_chains": 0,
        }

    specs = [
        PassSpec(
            pass_id=pass_id,
            phase=PassPhase.LAYOUT_PLAN,
            callback=callback,
            precondition=precondition,
            priority=(index + 1) * 10,
            transactional=True,
        )
        for index, (pass_id, callback, precondition) in enumerate(ordered)
    ]
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details={
            "rewritten_hardswish_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_mul_transpose_passthrough_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
