from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
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
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR

def _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate terminal NHWC->NCHW adapters before unary/MEAN suffixes.

    Target (strict chain):
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --(UNARY)*--> u_nchw
      u_nchw --MEAN(axes_nchw, keepDims=True)--> m_nchw
      m_nchw --RESHAPE--> y

    Rewrite:
      x_nhwc --(same UNARY)*--> u_nhwc
      u_nhwc --MEAN(axes_nhwc, keepDims=True)--> m_nhwc
      m_nhwc --(same RESHAPE)--> y

    Safety:
    - Pre-transpose direction is canonical NHWC->NCHW.
    - Unary chain is linear and layout-agnostic.
    - MEAN rank is 4 with constant axes.
    - Skip chains already handled by pre/post MEAN transpose elimination.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_passthrough_ops = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "FLOOR",
        "CEIL",
    }
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
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
            if len(pre_users) == 0:
                continue

            for first_user_idx in pre_users:
                first_user_op = model_ir.operators[int(first_user_idx)]
                first_user_type = str(first_user_op.op_type)
                if (
                    first_user_type != "MEAN"
                    and first_user_type not in unary_passthrough_ops
                ):
                    continue

                unary_chain_indices: List[int] = []
                unary_chain_ops: List[OperatorIR] = []
                current_tensor_name = pre_output_name
                current_op_idx = int(first_user_idx)
                mean_idx: Optional[int] = None
                valid_path = True

                while True:
                    current_op = model_ir.operators[int(current_op_idx)]
                    current_type = str(current_op.op_type)
                    if current_type in unary_passthrough_ops:
                        if len(current_op.inputs) != 1 or len(current_op.outputs) != 1:
                            valid_path = False
                            break
                        if str(current_op.inputs[0]) != current_tensor_name:
                            valid_path = False
                            break
                        unary_out_name = str(current_op.outputs[0])
                        if unary_out_name in model_outputs:
                            valid_path = False
                            break
                        unary_users = [int(v) for v in consumers.get(unary_out_name, [])]
                        if len(unary_users) != 1:
                            valid_path = False
                            break
                        unary_chain_indices.append(int(current_op_idx))
                        unary_chain_ops.append(current_op)
                        current_tensor_name = unary_out_name
                        current_op_idx = int(unary_users[0])
                        continue
                    if current_type == "MEAN":
                        mean_idx = int(current_op_idx)
                        break
                    valid_path = False
                    break
                if not valid_path or mean_idx is None:
                    continue

                mean_op = model_ir.operators[int(mean_idx)]
                if len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                    continue
                if str(mean_op.inputs[0]) != current_tensor_name:
                    continue
                keep_dims = bool(
                    mean_op.options.get(
                        "keepDims",
                        mean_op.options.get(
                            "keep_dims",
                            mean_op.options.get("keepdims", False),
                        ),
                    )
                )

                rank = 4
                pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                if (
                    pre_input_tensor is not None
                    and pre_input_tensor.shape is not None
                    and len(pre_input_tensor.shape) > 0
                ):
                    rank = int(len(list(pre_input_tensor.shape)))
                if rank != 4:
                    continue

                mean_axes_name = str(mean_op.inputs[1])
                mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                mean_axes = _read_const_ints_from_tensor(mean_axes_tensor)
                if mean_axes is None or len(mean_axes) == 0:
                    continue

                normalized_axes: List[int] = []
                valid_axes = True
                for axis in mean_axes:
                    a = int(axis)
                    if a < 0:
                        a += int(rank)
                    if a < 0 or a >= int(rank):
                        valid_axes = False
                        break
                    normalized_axes.append(int(a))
                if not valid_axes:
                    continue
                mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]

                mean_out_name = str(mean_op.outputs[0])
                if not keep_dims:
                    # Some lowered MEAN ops omit keepDims in options while still
                    # producing rank-preserving tensors. Accept those by shape.
                    mean_in_tensor = model_ir.tensors.get(current_tensor_name, None)
                    mean_out_tensor = model_ir.tensors.get(mean_out_name, None)
                    mean_in_shape = (
                        [int(v) for v in list(mean_in_tensor.shape)]
                        if mean_in_tensor is not None and mean_in_tensor.shape is not None
                        else None
                    )
                    mean_out_shape = (
                        [int(v) for v in list(mean_out_tensor.shape)]
                        if mean_out_tensor is not None and mean_out_tensor.shape is not None
                        else None
                    )
                    if (
                        mean_in_shape is None
                        or mean_out_shape is None
                        or len(mean_in_shape) != len(mean_out_shape)
                    ):
                        continue
                    keep_dims_by_shape = True
                    for axis in normalized_axes:
                        if int(axis) < 0 or int(axis) >= len(mean_out_shape):
                            keep_dims_by_shape = False
                            break
                        out_dim = int(mean_out_shape[int(axis)])
                        if out_dim > 0 and out_dim != 1:
                            keep_dims_by_shape = False
                            break
                    if not keep_dims_by_shape:
                        continue

                mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
                if len(mean_users) != 1:
                    continue

                tail_idx = int(mean_users[0])
                tail_op = model_ir.operators[int(tail_idx)]
                tail_type = str(tail_op.op_type)
                if (
                    tail_type == "TRANSPOSE"
                    and len(tail_op.inputs) >= 2
                    and len(tail_op.outputs) == 1
                    and str(tail_op.inputs[0]) == mean_out_name
                    and _read_transpose_perm(model_ir, tail_op) == perm_nchw_to_nhwc
                ):
                    # Leave paired pre/post transpose chains to dedicated rewrite.
                    continue
                if tail_type != "RESHAPE":
                    continue
                if len(tail_op.inputs) < 2 or len(tail_op.outputs) != 1:
                    continue
                if str(tail_op.inputs[0]) != mean_out_name:
                    continue
                if _read_const_ints_from_tensor(model_ir.tensors.get(str(tail_op.inputs[1]), None)) is None:
                    continue

                _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes])

                if len(unary_chain_ops) > 0:
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=unary_chain_ops[0],
                        new_inputs=[pre_input_name],
                        graph_index=graph_index,
                    )
                    for unary_op in unary_chain_ops:
                        _permute_tensor_metadata_if_rank_matches(
                            model_ir.tensors.get(str(unary_op.outputs[0]), None),
                            perm_nchw_to_nhwc,
                        )
                else:
                    mean_inputs = [str(v) for v in list(mean_op.inputs)]
                    mean_inputs[0] = pre_input_name
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=mean_op,
                        new_inputs=mean_inputs,
                        graph_index=graph_index,
                    )

                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mean_out_name, None),
                    perm_nchw_to_nhwc,
                )

                pre_remaining_users = [int(v) for v in pre_users if int(v) != int(first_user_idx)]
                remove_indices: List[int] = []
                if len(pre_remaining_users) == 0:
                    remove_indices.append(int(pre_idx))
                for remove_idx in sorted(remove_indices, reverse=True):
                    graph_index.remove_operator(int(remove_idx))

                rewritten += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_pre_unary_mean_terminal_nhwc_chains": int(rewritten)}


def run_terminal_mean_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate NHWC through a guarded unary/Mean/Reshape suffix."""
    unary_passthrough_ops = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "FLOOR",
        "CEIL",
    }

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MEAN", "RESHAPE"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = {str(name) for name in candidate_model.outputs}
        for pre_op in candidate_model.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(candidate_model, pre_op)
                != [0, 3, 1, 2]
            ):
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue
            pre_input_tensor = candidate_model.tensors.get(pre_input_name)
            rank = (
                len(pre_input_tensor.shape)
                if pre_input_tensor is not None and pre_input_tensor.shape
                else 4
            )
            if rank != 4:
                continue

            for first_user_idx in pass_state.graph_index.consumer_indices(
                pre_output_name
            ):
                current_tensor_name = pre_output_name
                current_op_idx = int(first_user_idx)
                while True:
                    current_op = candidate_model.operators[current_op_idx]
                    current_type = str(current_op.op_type)
                    if current_type not in unary_passthrough_ops:
                        break
                    if (
                        len(current_op.inputs) != 1
                        or len(current_op.outputs) != 1
                        or str(current_op.inputs[0]) != current_tensor_name
                    ):
                        current_op_idx = -1
                        break
                    current_tensor_name = str(current_op.outputs[0])
                    if current_tensor_name in model_outputs:
                        current_op_idx = -1
                        break
                    users = pass_state.graph_index.consumer_indices(
                        current_tensor_name
                    )
                    if len(users) != 1:
                        current_op_idx = -1
                        break
                    current_op_idx = int(users[0])
                if current_op_idx < 0:
                    continue
                mean_op = candidate_model.operators[current_op_idx]
                if (
                    str(mean_op.op_type) != "MEAN"
                    or len(mean_op.inputs) < 2
                    or len(mean_op.outputs) != 1
                    or str(mean_op.inputs[0]) != current_tensor_name
                ):
                    continue
                axes = _read_const_ints_from_tensor(
                    candidate_model.tensors.get(str(mean_op.inputs[1]))
                )
                if axes is None or not axes:
                    continue
                normalized_axes = [
                    int(axis) + rank if int(axis) < 0 else int(axis)
                    for axis in axes
                ]
                if any(axis < 0 or axis >= rank for axis in normalized_axes):
                    continue

                mean_output_name = str(mean_op.outputs[0])
                keep_dims = bool(
                    mean_op.options.get(
                        "keepDims",
                        mean_op.options.get(
                            "keep_dims",
                            mean_op.options.get("keepdims", False),
                        ),
                    )
                )
                if not keep_dims:
                    mean_input_tensor = candidate_model.tensors.get(
                        current_tensor_name
                    )
                    mean_output_tensor = candidate_model.tensors.get(
                        mean_output_name
                    )
                    if (
                        mean_input_tensor is None
                        or mean_output_tensor is None
                        or len(mean_input_tensor.shape)
                        != len(mean_output_tensor.shape)
                        or any(
                            int(mean_output_tensor.shape[axis]) > 0
                            and int(mean_output_tensor.shape[axis]) != 1
                            for axis in normalized_axes
                        )
                    ):
                        continue
                mean_users = pass_state.graph_index.consumer_indices(
                    mean_output_name
                )
                if len(mean_users) != 1:
                    continue
                tail_op = candidate_model.operators[int(mean_users[0])]
                if (
                    str(tail_op.op_type) == "RESHAPE"
                    and len(tail_op.inputs) >= 2
                    and len(tail_op.outputs) == 1
                    and str(tail_op.inputs[0]) == mean_output_name
                    and _read_const_ints_from_tensor(
                        candidate_model.tensors.get(str(tail_op.inputs[1]))
                    )
                    is not None
                ):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_pre_unary_mean_terminal_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.terminal_unary_mean_reshape",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_pre_unary_mean_terminal_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
