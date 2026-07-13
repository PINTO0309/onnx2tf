from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
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
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR

def _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips in OSNet-like multi-branch gate blocks.

    Strict target motif (rank-4):
      - Several branch tensors:
          x_i_nhwc --T(0,3,1,2)--> x_i_nchw --RELU--> r_i_nchw
          r_i_nchw --MEAN(axes=[2,3], keepDims=1)--> gap_i_nchw
            -> ... -> RESHAPE([N,C,1,1]) -> LOGISTIC -> g_i_nchw
          MUL(r_i_nchw, g_i_nchw) -> m_i_nchw
      - Nested ADD tree over m_i:
          ADD(...ADD(m0, m1)..., mk) -> y_nchw
      - Output bridge:
          y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - Bypass each branch pre-transpose into RELU.
      - Remap MEAN axes from NCHW to NHWC.
      - Rewrite gate RESHAPE [N,C,1,1] -> [N,1,1,C] (or bypass gate pre-transpose).
      - Keep RELU/LOGISTIC/MUL/ADD in NHWC.
      - Remove all branch pre-transposes and terminal post-transpose bridges.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _is_singleton_spatial_nhwc_to_nchw_reshape(
        *,
        input_name: str,
        output_name: str,
    ) -> bool:
        in_tensor = model_ir.tensors.get(str(input_name), None)
        out_tensor = model_ir.tensors.get(str(output_name), None)
        if in_tensor is None or out_tensor is None:
            return False
        in_shape = [int(v) for v in list(in_tensor.shape)] if in_tensor.shape is not None else []
        out_shape = [int(v) for v in list(out_tensor.shape)] if out_tensor.shape is not None else []
        if len(in_shape) != 4 or len(out_shape) != 4:
            return False
        if any(int(v) < 0 for v in list(in_shape) + list(out_shape)):
            return False
        return (
            int(in_shape[0]) == int(out_shape[0])
            and int(in_shape[1]) == 1
            and int(in_shape[2]) == 1
            and int(in_shape[3]) == int(out_shape[1])
            and int(out_shape[2]) == 1
            and int(out_shape[3]) == 1
        )

    def _clone_const_tensor_for_operator_input(
        *,
        op: OperatorIR,
        input_index: int,
        tag: str,
    ) -> str:
        input_name = str(op.inputs[int(input_index)])
        tensor = model_ir.tensors.get(input_name, None)
        if tensor is None or tensor.data is None:
            return input_name
        cloned_name = f"{input_name}{tag}"
        suffix = 1
        while cloned_name in model_ir.tensors:
            cloned_name = f"{input_name}{tag}_{suffix}"
            suffix += 1
        model_ir.tensors[cloned_name] = TensorIR(
            name=cloned_name,
            dtype=str(tensor.dtype),
            shape=[int(v) for v in list(tensor.shape)] if tensor.shape is not None else [],
            shape_signature=(
                [int(v) for v in list(tensor.shape_signature)]
                if tensor.shape_signature is not None
                else (
                    [int(v) for v in list(tensor.shape)]
                    if tensor.shape is not None
                    else []
                )
            ),
            data=np.array(tensor.data, copy=True),
            is_variable=bool(tensor.is_variable),
            quantization=_clone_quantization(tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=cloned_name,
            graph_index=graph_index,
        )
        return str(cloned_name)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            add_root_out_name = str(post_op.inputs[0])
            if add_root_out_name in model_outputs:
                continue

            add_root_idx = producers.get(add_root_out_name, None)
            if add_root_idx is None:
                continue
            add_root_op = model_ir.operators[int(add_root_idx)]
            if (
                str(add_root_op.op_type) != "ADD"
                or len(add_root_op.inputs) != 2
                or len(add_root_op.outputs) != 1
                or str(add_root_op.outputs[0]) != add_root_out_name
            ):
                continue

            add_root_users = [int(v) for v in consumers.get(add_root_out_name, [])]
            if len(add_root_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for user_idx in add_root_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != add_root_out_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_posts = False
                    break
                post_indices.append(int(user_idx))
                post_output_names.append(str(user_op.outputs[0]))
            if not valid_posts or len(post_indices) == 0:
                continue

            # Collect nested ADD tree leaves (MUL outputs).
            visited_add_outputs: set[str] = set()
            add_indices: List[int] = []
            add_output_names: List[str] = []
            leaf_mul_indices: List[int] = []
            tree_valid = True
            stack: List[str] = [str(add_root_out_name)]
            while len(stack) > 0:
                tensor_name = str(stack.pop())
                prod_idx = producers.get(tensor_name, None)
                if prod_idx is None:
                    tree_valid = False
                    break
                prod_op = model_ir.operators[int(prod_idx)]
                prod_type = str(prod_op.op_type)
                if (
                    prod_type == "ADD"
                    and len(prod_op.inputs) == 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == tensor_name
                    and tensor_name not in model_outputs
                ):
                    if tensor_name in visited_add_outputs:
                        tree_valid = False
                        break
                    visited_add_outputs.add(tensor_name)
                    add_indices.append(int(prod_idx))
                    add_output_names.append(str(tensor_name))
                    stack.append(str(prod_op.inputs[0]))
                    stack.append(str(prod_op.inputs[1]))
                elif (
                    prod_type == "MUL"
                    and len(prod_op.inputs) == 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == tensor_name
                    and tensor_name not in model_outputs
                ):
                    leaf_mul_indices.append(int(prod_idx))
                else:
                    tree_valid = False
                    break
            if not tree_valid or len(leaf_mul_indices) < 2:
                continue

            add_index_set = set(int(v) for v in add_indices)
            # Root ADD output may fan out only to inverse post-transposes.
            if set(int(v) for v in consumers.get(add_root_out_name, [])) != set(int(v) for v in post_indices):
                continue
            # Internal ADD outputs must flow only into one ADD parent.
            internal_add_outputs_valid = True
            for add_out_name in add_output_names:
                if str(add_out_name) == str(add_root_out_name):
                    continue
                add_users = [int(v) for v in consumers.get(str(add_out_name), [])]
                if len(add_users) != 1 or int(add_users[0]) not in add_index_set:
                    internal_add_outputs_valid = False
                    break
            if not internal_add_outputs_valid:
                continue
            # Leaf MUL outputs must feed exactly one ADD parent.
            leaf_users_valid = True
            for leaf_mul_idx in leaf_mul_indices:
                leaf_mul_op = model_ir.operators[int(leaf_mul_idx)]
                leaf_out_name = str(leaf_mul_op.outputs[0])
                leaf_users = [int(v) for v in consumers.get(leaf_out_name, [])]
                if len(leaf_users) != 1 or int(leaf_users[0]) not in add_index_set:
                    leaf_users_valid = False
                    break
            if not leaf_users_valid:
                continue

            leaf_infos: List[Dict[str, Any]] = []
            leaf_match_valid = True
            for leaf_mul_idx in leaf_mul_indices:
                leaf_mul_op = model_ir.operators[int(leaf_mul_idx)]
                leaf_inputs = [str(v) for v in list(leaf_mul_op.inputs)]

                matched_info: Optional[Dict[str, Any]] = None
                for relu_input_index in [0, 1]:
                    relu_out_name = str(leaf_inputs[int(relu_input_index)])
                    gate_out_name = str(leaf_inputs[1 - int(relu_input_index)])

                    relu_idx = producers.get(relu_out_name, None)
                    if relu_idx is None:
                        continue
                    relu_op = model_ir.operators[int(relu_idx)]
                    if (
                        str(relu_op.op_type) != "RELU"
                        or len(relu_op.inputs) != 1
                        or len(relu_op.outputs) != 1
                        or str(relu_op.outputs[0]) != relu_out_name
                        or relu_out_name in model_outputs
                    ):
                        continue
                    relu_input_name = str(relu_op.inputs[0])
                    if relu_input_name in model_outputs:
                        continue

                    pre_idx = producers.get(relu_input_name, None)
                    if pre_idx is None:
                        continue
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != relu_input_name
                        or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                        or str(pre_op.inputs[0]) in model_outputs
                    ):
                        continue
                    if set(int(v) for v in consumers.get(relu_input_name, [])) != {int(relu_idx)}:
                        continue

                    relu_users = [int(v) for v in consumers.get(relu_out_name, [])]
                    if int(leaf_mul_idx) not in set(relu_users):
                        continue
                    mean_candidates: List[int] = []
                    for relu_user_idx in relu_users:
                        if int(relu_user_idx) == int(leaf_mul_idx):
                            continue
                        relu_user_op = model_ir.operators[int(relu_user_idx)]
                        if (
                            str(relu_user_op.op_type) == "MEAN"
                            and len(relu_user_op.inputs) >= 2
                            and len(relu_user_op.outputs) == 1
                            and str(relu_user_op.inputs[0]) == relu_out_name
                            and bool(relu_user_op.options.get("keepDims", False))
                            and str(relu_user_op.outputs[0]) not in model_outputs
                        ):
                            mean_candidates.append(int(relu_user_idx))
                    if len(mean_candidates) != 1:
                        continue

                    mean_idx = int(mean_candidates[0])
                    mean_op = model_ir.operators[int(mean_idx)]
                    mean_axes_name = str(mean_op.inputs[1])
                    mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                    mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                    if mean_axes_vals is None or len(mean_axes_vals) == 0:
                        continue
                    normalized_axes: List[int] = []
                    axes_valid = True
                    for axis in mean_axes_vals:
                        a = int(axis)
                        if a < 0:
                            a += 4
                        if a < 0 or a >= 4:
                            axes_valid = False
                            break
                        normalized_axes.append(int(a))
                    if not axes_valid or sorted(normalized_axes) != [2, 3]:
                        continue
                    mapped_axes = [int(perm_nhwc_to_nchw[int(v)]) for v in normalized_axes]
                    if sorted(mapped_axes) != [1, 2]:
                        continue

                    gate_idx = producers.get(gate_out_name, None)
                    if gate_idx is None:
                        continue
                    gate_op = model_ir.operators[int(gate_idx)]
                    if (
                        str(gate_op.op_type) != "LOGISTIC"
                        or len(gate_op.inputs) != 1
                        or len(gate_op.outputs) != 1
                        or str(gate_op.outputs[0]) != gate_out_name
                        or gate_out_name in model_outputs
                    ):
                        continue
                    if set(int(v) for v in consumers.get(gate_out_name, [])) != {int(leaf_mul_idx)}:
                        continue

                    gate_pre_out_name = str(gate_op.inputs[0])
                    if gate_pre_out_name in model_outputs:
                        continue
                    gate_pre_idx = producers.get(gate_pre_out_name, None)
                    if gate_pre_idx is None:
                        continue
                    gate_pre_op = model_ir.operators[int(gate_pre_idx)]
                    gate_pre_type = str(gate_pre_op.op_type)

                    gate_pre_input_name: Optional[str] = None
                    gate_pre_remove = False
                    gate_shape_tensor: Optional[TensorIR] = None
                    if (
                        gate_pre_type == "TRANSPOSE"
                        and len(gate_pre_op.inputs) >= 2
                        and len(gate_pre_op.outputs) == 1
                        and str(gate_pre_op.outputs[0]) == gate_pre_out_name
                        and _read_transpose_perm(model_ir, gate_pre_op) == perm_nhwc_to_nchw
                        and str(gate_pre_op.inputs[0]) not in model_outputs
                    ):
                        gate_pre_input_name = str(gate_pre_op.inputs[0])
                        gate_pre_remove = True
                    elif (
                        gate_pre_type == "RESHAPE"
                        and len(gate_pre_op.inputs) >= 2
                        and len(gate_pre_op.outputs) == 1
                        and str(gate_pre_op.outputs[0]) == gate_pre_out_name
                        and _is_singleton_spatial_nhwc_to_nchw_reshape(
                            input_name=str(gate_pre_op.inputs[0]),
                            output_name=str(gate_pre_op.outputs[0]),
                        )
                    ):
                        gate_shape_tensor = model_ir.tensors.get(str(gate_pre_op.inputs[1]), None)
                        gate_shape_vals = _read_const_ints_from_tensor(gate_shape_tensor)
                        if gate_shape_vals is None or len(gate_shape_vals) != 4:
                            continue
                        gate_pre_remove = False
                    else:
                        continue
                    if set(int(v) for v in consumers.get(gate_pre_out_name, [])) != {int(gate_idx)}:
                        continue

                    matched_info = {
                        "mul_idx": int(leaf_mul_idx),
                        "mul_op": leaf_mul_op,
                        "mul_out_name": str(leaf_mul_op.outputs[0]),
                        "relu_idx": int(relu_idx),
                        "relu_op": relu_op,
                        "relu_out_name": str(relu_out_name),
                        "pre_idx": int(pre_idx),
                        "pre_input_name": str(pre_op.inputs[0]),
                        "pre_out_name": str(relu_input_name),
                        "mean_idx": int(mean_idx),
                        "mean_op": mean_op,
                        "mean_out_name": str(mean_op.outputs[0]),
                        "mapped_axes": [int(v) for v in mapped_axes],
                        "gate_idx": int(gate_idx),
                        "gate_op": gate_op,
                        "gate_out_name": str(gate_out_name),
                        "gate_pre_idx": int(gate_pre_idx),
                        "gate_pre_type": str(gate_pre_type),
                        "gate_pre_input_name": str(gate_pre_input_name) if gate_pre_input_name is not None else None,
                        "gate_pre_out_name": str(gate_pre_out_name),
                        "gate_pre_remove": bool(gate_pre_remove),
                    }
                    break

                if matched_info is None:
                    leaf_match_valid = False
                    break
                leaf_infos.append(dict(matched_info))
            if not leaf_match_valid:
                continue

            pre_indices = [int(v["pre_idx"]) for v in leaf_infos]
            if len(set(pre_indices)) != len(pre_indices):
                continue

            # Rewrite branch-local ops to NHWC semantics.
            gate_pre_remove_indices: set[int] = set()
            tensor_metadata_targets: set[str] = set()
            for leaf in leaf_infos:
                relu_op = leaf["relu_op"]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=relu_op,
                    new_inputs=[str(leaf["pre_input_name"])],
                    graph_index=graph_index,
                )

                mean_op = leaf["mean_op"]
                mean_axes_name = _clone_const_tensor_for_operator_input(
                    op=mean_op,
                    input_index=1,
                    tag="__osnet_nhwc_axes",
                )
                mean_axes_tensor = model_ir.tensors.get(str(mean_axes_name), None)
                mapped_axes = [int(v) for v in list(leaf["mapped_axes"])]
                if not _write_const_ints_to_tensor(mean_axes_tensor, mapped_axes):
                    leaf_match_valid = False
                    break
                if isinstance(mean_op.options, dict):
                    mean_options = dict(mean_op.options)
                    for key in ["axis", "axes", "onnxRawAxes"]:
                        value = mean_options.get(key, None)
                        if isinstance(value, list) and len(value) == len(mapped_axes):
                            mean_options[key] = [int(v) for v in mapped_axes]
                    mean_op.options = mean_options

                gate_op = leaf["gate_op"]
                if bool(leaf["gate_pre_remove"]):
                    gate_pre_input_name = leaf["gate_pre_input_name"]
                    if gate_pre_input_name is None:
                        leaf_match_valid = False
                        break
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=gate_op,
                        new_inputs=[str(gate_pre_input_name)],
                        graph_index=graph_index,
                    )
                    gate_pre_remove_indices.add(int(leaf["gate_pre_idx"]))
                else:
                    gate_pre_idx = int(leaf["gate_pre_idx"])
                    gate_pre_op = model_ir.operators[int(gate_pre_idx)]
                    gate_shape_name = _clone_const_tensor_for_operator_input(
                        op=gate_pre_op,
                        input_index=1,
                        tag="__osnet_nhwc_shape",
                    )
                    gate_shape_tensor = model_ir.tensors.get(str(gate_shape_name), None)
                    gate_shape_vals = _read_const_ints_from_tensor(gate_shape_tensor)
                    if gate_shape_vals is None or len(gate_shape_vals) != 4:
                        leaf_match_valid = False
                        break
                    swapped_shape = [
                        int(gate_shape_vals[0]),
                        int(gate_shape_vals[2]),
                        int(gate_shape_vals[3]),
                        int(gate_shape_vals[1]),
                    ]
                    if not _write_const_ints_to_tensor(gate_shape_tensor, swapped_shape):
                        leaf_match_valid = False
                        break

                tensor_metadata_targets.update({
                    str(leaf["relu_out_name"]),
                    str(leaf["mean_out_name"]),
                    str(leaf["gate_pre_out_name"]),
                    str(leaf["gate_out_name"]),
                    str(leaf["mul_out_name"]),
                })
            if not leaf_match_valid:
                continue

            for add_output_name in add_output_names:
                tensor_metadata_targets.add(str(add_output_name))
            tensor_metadata_targets.add(str(add_root_out_name))

            for tensor_name in tensor_metadata_targets:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_root_op,
                new_outputs=[canonical_post_output_name],
                graph_index=graph_index,
            )
            _replace_tensor_inputs(
                model_ir,
                add_root_out_name,
                canonical_post_output_name,
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    str(alias_name),
                    canonical_post_output_name,
                    graph_index=graph_index,
                )

            old_root_tensor = model_ir.tensors.get(add_root_out_name, None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_root_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_root_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_root_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_root_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_root_tensor.shape_signature)]
                    if old_root_tensor.shape_signature is not None
                    else [int(v) for v in list(old_root_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = set(int(v["pre_idx"]) for v in leaf_infos)
            remove_indices.update(int(v) for v in gate_pre_remove_indices)
            remove_indices.update(int(v) for v in post_indices)
            for remove_idx in sorted(remove_indices, reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains": int(rewritten)}


def _has_multi_branch_gate_candidate(pass_state: ModelIRPassState) -> bool:
    model_ir = pass_state.model_ir
    graph_index = pass_state.graph_index
    model_outputs = {str(name) for name in model_ir.outputs}

    def _is_transpose(
        op: OperatorIR | None,
        perm: List[int],
    ) -> bool:
        return bool(
            op is not None
            and str(op.op_type) == "TRANSPOSE"
            and len(op.inputs) >= 2
            and len(op.outputs) == 1
            and _read_transpose_perm(model_ir, op) == perm
        )

    def _is_gate_leaf(mul_op: OperatorIR) -> bool:
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
        ):
            return False
        mul_index = graph_index.operator_index(mul_op)
        if mul_index is None:
            return False
        for relu_input_index in (0, 1):
            relu_name = str(mul_op.inputs[relu_input_index])
            gate_name = str(mul_op.inputs[1 - relu_input_index])
            relu_op = graph_index.producer(relu_name)
            gate_op = graph_index.producer(gate_name)
            relu_index = (
                graph_index.operator_index(relu_op)
                if relu_op is not None
                else None
            )
            gate_index = (
                graph_index.operator_index(gate_op)
                if gate_op is not None
                else None
            )
            if (
                relu_op is None
                or relu_index is None
                or str(relu_op.op_type) != "RELU"
                or len(relu_op.inputs) != 1
                or len(relu_op.outputs) != 1
                or relu_name in model_outputs
                or gate_op is None
                or gate_index is None
                or str(gate_op.op_type) != "LOGISTIC"
                or len(gate_op.inputs) != 1
                or len(gate_op.outputs) != 1
                or gate_name in model_outputs
                or set(graph_index.consumer_indices(gate_name)) != {mul_index}
            ):
                continue
            pre_name = str(relu_op.inputs[0])
            pre_op = graph_index.producer(pre_name)
            if (
                not _is_transpose(pre_op, [0, 3, 1, 2])
                or pre_name in model_outputs
                or set(graph_index.consumer_indices(pre_name)) != {relu_index}
            ):
                continue
            relu_users = graph_index.consumers_of(relu_name)
            mean_ops = [
                user
                for user in relu_users
                if str(user.op_type) == "MEAN"
                and len(user.inputs) >= 2
                and len(user.outputs) == 1
                and bool(user.options.get("keepDims", False))
            ]
            if (
                len(mean_ops) != 1
                or mul_index not in graph_index.consumer_indices(relu_name)
            ):
                continue
            gate_pre_name = str(gate_op.inputs[0])
            gate_pre_op = graph_index.producer(gate_pre_name)
            if gate_pre_op is None or str(gate_pre_op.op_type) not in {
                "TRANSPOSE",
                "RESHAPE",
            }:
                continue
            if (
                str(gate_pre_op.op_type) == "TRANSPOSE"
                and not _is_transpose(gate_pre_op, [0, 3, 1, 2])
            ):
                continue
            if set(graph_index.consumer_indices(gate_pre_name)) != {gate_index}:
                continue
            return True
        return False

    for post_op in model_ir.operators:
        if not _is_transpose(post_op, [0, 2, 3, 1]):
            continue
        root_name = str(post_op.inputs[0])
        if root_name in model_outputs or str(post_op.outputs[0]) in model_outputs:
            continue
        root_op = graph_index.producer(root_name)
        if root_op is None or str(root_op.op_type) != "ADD":
            continue
        root_users = graph_index.consumers_of(root_name)
        if not root_users or any(
            not _is_transpose(user, [0, 2, 3, 1])
            or str(user.outputs[0]) in model_outputs
            for user in root_users
        ):
            continue

        leaf_count = 0
        tree_valid = True
        visited: set[str] = set()
        stack = [root_name]
        while stack:
            tensor_name = str(stack.pop())
            if tensor_name in visited:
                tree_valid = False
                break
            visited.add(tensor_name)
            producer = graph_index.producer(tensor_name)
            if producer is None or len(producer.outputs) != 1:
                tree_valid = False
                break
            if str(producer.op_type) == "ADD" and len(producer.inputs) == 2:
                stack.extend(str(name) for name in producer.inputs)
                continue
            if str(producer.op_type) == "MUL" and _is_gate_leaf(producer):
                leaf_count += 1
                continue
            tree_valid = False
            break
        if tree_valid and leaf_count >= 2:
            return True
    return False


def run_multi_branch_gate_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate NHWC through guarded multi-branch gate/Add trees."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "RELU", "MEAN", "LOGISTIC", "MUL", "ADD"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = (
        "optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains"
    )

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.multi_branch_gate_add_tree_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_multi_branch_gate_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={stats_key: 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
