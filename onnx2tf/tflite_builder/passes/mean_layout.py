from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
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
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_transpose_mean_prepost_nhwc_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Eliminate NCHW round-trips around TRANSPOSE->MEAN->TRANSPOSE."""
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
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

            for mean_idx in pre_users:
                mean_op = model_ir.operators[int(mean_idx)]
                if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                    continue
                if str(mean_op.inputs[0]) != pre_output_name:
                    continue
                if not bool(mean_op.options.get("keepDims", False)):
                    continue

                mean_axes_name = str(mean_op.inputs[1])
                mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                if mean_axes_vals is None or len(mean_axes_vals) == 0:
                    continue

                rank = 4
                pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                if pre_input_tensor is not None and pre_input_tensor.shape is not None and len(pre_input_tensor.shape) > 0:
                    rank = int(len(list(pre_input_tensor.shape)))
                if rank != 4:
                    continue

                normalized_axes: List[int] = []
                valid_axes = True
                for axis in mean_axes_vals:
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
                if mean_out_name in model_outputs:
                    continue
                mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
                if len(mean_users) == 0:
                    continue

                post_indices: List[int] = []
                post_output_names: List[str] = []
                valid_posts = True
                for user_idx in mean_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) != "TRANSPOSE"
                        or len(user_op.inputs) < 2
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != mean_out_name
                        or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    ):
                        valid_posts = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                if not valid_posts or len(post_indices) == 0:
                    continue

                _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes])
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mean_op,
                    new_inputs=[pre_input_name, mean_axes_name],
                    graph_index=graph_index,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mean_out_name, None),
                    perm_nchw_to_nhwc,
                )

                representative_output_name = str(post_output_names[0])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=mean_op,
                    new_outputs=[representative_output_name],
                    graph_index=graph_index,
                )
                for alias_name in post_output_names[1:]:
                    _replace_tensor_inputs(
                        model_ir,
                        str(alias_name),
                        representative_output_name,
                        graph_index=graph_index,
                    )

                old_mean_tensor = model_ir.tensors.get(mean_out_name, None)
                representative_tensor = model_ir.tensors.get(representative_output_name, None)
                if old_mean_tensor is not None and representative_tensor is not None:
                    representative_tensor.dtype = str(old_mean_tensor.dtype)
                    representative_tensor.quantization = _clone_quantization(old_mean_tensor.quantization)
                    representative_tensor.shape = [int(v) for v in list(old_mean_tensor.shape)]
                    representative_tensor.shape_signature = (
                        [int(v) for v in list(old_mean_tensor.shape_signature)]
                        if old_mean_tensor.shape_signature is not None
                        else [int(v) for v in list(old_mean_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        representative_tensor,
                        perm_nchw_to_nhwc,
                    )

                remove_indices = set(int(v) for v in post_indices)
                pre_remaining_users = [int(v) for v in pre_users if int(v) != int(mean_idx)]
                if len(pre_remaining_users) == 0:
                    remove_indices.add(int(pre_idx))
                for remove_idx in sorted(list(remove_indices), reverse=True):
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
    return {"optimized_transpose_mean_prepost_nhwc_passthrough_chains": int(rewritten)}


def _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Keep a TRANSPOSE->MEAN->MUL->RESHAPE->ADD->CONV branch in NHWC."""
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
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
            if pre_input_name in model_outputs or pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) != 1:
                continue
            mean_idx = int(pre_users[0])
            mean_op = model_ir.operators[int(mean_idx)]
            if (
                str(mean_op.op_type) != "MEAN"
                or len(mean_op.inputs) < 2
                or len(mean_op.outputs) != 1
                or str(mean_op.inputs[0]) != pre_output_name
                or not bool(mean_op.options.get("keepDims", False))
            ):
                continue

            mean_axes_name = str(mean_op.inputs[1])
            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None:
                continue
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes_vals:
                a = int(axis)
                if a < 0:
                    a += 4
                if a < 0 or a >= 4:
                    valid_axes = False
                    break
                normalized_axes.append(int(a))
            if not valid_axes or sorted(normalized_axes) != [2, 3]:
                continue

            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_outputs:
                continue
            mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
            if len(mean_users) != 1:
                continue
            mul_idx = int(mean_users[0])
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_out_name = str(mul_op.outputs[0])
            if mul_out_name in model_outputs:
                continue

            mul_const_name: Optional[str] = None
            for input_name in [str(v) for v in list(mul_op.inputs)]:
                if str(input_name) == str(mean_out_name):
                    continue
                tensor = model_ir.tensors.get(str(input_name), None)
                if tensor is None or tensor.data is None:
                    continue
                const_arr = np.asarray(tensor.data)
                if const_arr.ndim == 4:
                    mul_const_name = str(input_name)
                    break
            if mul_const_name is None:
                continue

            mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_users) != 1:
                continue
            reshape_idx = int(mul_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if str(reshape_op.op_type) != "RESHAPE" or len(reshape_op.inputs) < 2 or len(reshape_op.outputs) != 1:
                continue
            if str(reshape_op.inputs[0]) != mul_out_name:
                continue
            reshape_out_name = str(reshape_op.outputs[0])
            if reshape_out_name in model_outputs:
                continue

            reshape_out_tensor = model_ir.tensors.get(reshape_out_name, None)
            if reshape_out_tensor is None or len(list(reshape_out_tensor.shape)) != 4:
                continue
            reshape_shape = [int(v) for v in list(reshape_out_tensor.shape)]
            if any(int(v) <= 0 for v in reshape_shape) or reshape_shape[1:3] != [1, 1]:
                continue

            reshape_users = [int(v) for v in consumers.get(reshape_out_name, [])]
            if len(reshape_users) != 1:
                continue
            add_idx = int(reshape_users[0])
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue

            add_const_ok = False
            for input_name in [str(v) for v in list(add_op.inputs)]:
                if str(input_name) == str(reshape_out_name):
                    continue
                tensor = model_ir.tensors.get(str(input_name), None)
                if tensor is None or tensor.data is None:
                    continue
                add_const_ok = True
                break
            if not add_const_ok:
                continue

            add_users = [int(v) for v in consumers.get(add_out_name, [])]
            if len(add_users) != 1:
                continue
            conv_op = model_ir.operators[int(add_users[0])]
            if str(conv_op.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                continue

            mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]
            _write_const_ints_to_tensor(mean_axes_tensor, mapped_axes)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mean_op,
                new_inputs=[str(pre_input_name), str(mean_axes_name)],
                graph_index=graph_index,
            )

            mul_const_tensor = model_ir.tensors.get(str(mul_const_name), None)
            if mul_const_tensor is not None and mul_const_tensor.data is not None:
                mul_const_arr = np.asarray(mul_const_tensor.data)
                if mul_const_arr.ndim == 4:
                    mul_const_arr = np.transpose(mul_const_arr, axes=perm_nchw_to_nhwc)
                    mul_const_tensor.data = np.asarray(mul_const_arr)
                    mul_const_tensor.shape = [int(v) for v in list(mul_const_arr.shape)]
                    mul_const_tensor.shape_signature = [int(v) for v in list(mul_const_arr.shape)]

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mean_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul_out_name), None),
                perm_nchw_to_nhwc,
            )

            add_inputs = [
                str(mul_out_name) if str(input_name) == str(reshape_out_name) else str(input_name)
                for input_name in list(add_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=add_inputs,
                graph_index=graph_index,
            )

            graph_index.remove_operator(int(reshape_idx))
            graph_index.remove_operator(int(pre_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains": int(rewritten)}


def run_transpose_mean_passthrough_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Collapse guarded NHWC/NCHW round-trips around keep-dims Mean."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_transpose = False
        found_mean = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            found_mean = found_mean or operator_type == "MEAN"
            if found_transpose and found_mean:
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
                or _read_transpose_perm(candidate_model, pre_op) != [0, 3, 1, 2]
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
            for mean_idx in pass_state.graph_index.consumer_indices(pre_output_name):
                mean_op = candidate_model.operators[int(mean_idx)]
                if (
                    str(mean_op.op_type) != "MEAN"
                    or len(mean_op.inputs) < 2
                    or len(mean_op.outputs) != 1
                    or str(mean_op.inputs[0]) != pre_output_name
                    or not bool(mean_op.options.get("keepDims", False))
                ):
                    continue
                axes = _read_const_ints_from_tensor(
                    candidate_model.tensors.get(str(mean_op.inputs[1]))
                )
                if axes is None or not axes:
                    continue
                normalized_axes = [int(axis) + rank if int(axis) < 0 else int(axis) for axis in axes]
                if any(axis < 0 or axis >= rank for axis in normalized_axes):
                    continue
                mean_output_name = str(mean_op.outputs[0])
                if mean_output_name in model_outputs:
                    continue
                mean_users = pass_state.graph_index.consumer_indices(mean_output_name)
                if mean_users and all(
                    str(candidate_model.operators[int(user_idx)].op_type) == "TRANSPOSE"
                    and len(candidate_model.operators[int(user_idx)].inputs) >= 2
                    and len(candidate_model.operators[int(user_idx)].outputs) == 1
                    and str(candidate_model.operators[int(user_idx)].inputs[0]) == mean_output_name
                    and _read_transpose_perm(
                        candidate_model,
                        candidate_model.operators[int(user_idx)],
                    ) == [0, 2, 3, 1]
                    for user_idx in mean_users
                ):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_mean_prepost_nhwc_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_mean_prepost_nhwc_passthrough_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.transpose_mean_prepost",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_mean_prepost_nhwc_passthrough_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def run_mean_mul_add_conv_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Propagate NHWC through a guarded Mean/Mul/Add/Conv branch."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MEAN", "MUL", "RESHAPE", "ADD"}
        found_conv = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            required.discard(operator_type)
            found_conv = found_conv or operator_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
            if not required and found_conv:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = {str(name) for name in candidate_model.outputs}

        def _single_consumer(tensor_name: str) -> Any | None:
            users = pass_state.graph_index.consumer_indices(tensor_name)
            return candidate_model.operators[int(users[0])] if len(users) == 1 else None

        for pre_op in candidate_model.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(candidate_model, pre_op) != [0, 3, 1, 2]
            ):
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_input_name in model_outputs or pre_output_name in model_outputs:
                continue
            mean_op = _single_consumer(pre_output_name)
            if (
                mean_op is None
                or str(mean_op.op_type) != "MEAN"
                or len(mean_op.inputs) < 2
                or len(mean_op.outputs) != 1
                or str(mean_op.inputs[0]) != pre_output_name
                or not bool(mean_op.options.get("keepDims", False))
            ):
                continue
            axes = _read_const_ints_from_tensor(
                candidate_model.tensors.get(str(mean_op.inputs[1]))
            )
            if axes is None:
                continue
            normalized_axes = [int(axis) + 4 if int(axis) < 0 else int(axis) for axis in axes]
            if sorted(normalized_axes) != [2, 3]:
                continue
            mean_output_name = str(mean_op.outputs[0])
            if mean_output_name in model_outputs:
                continue
            mul_op = _single_consumer(mean_output_name)
            if (
                mul_op is None
                or str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
            ):
                continue
            mul_constants = [
                candidate_model.tensors.get(str(name))
                for name in mul_op.inputs
                if str(name) != mean_output_name
            ]
            if not any(
                tensor is not None
                and tensor.data is not None
                and np.asarray(tensor.data).ndim == 4
                for tensor in mul_constants
            ):
                continue
            mul_output_name = str(mul_op.outputs[0])
            if mul_output_name in model_outputs:
                continue
            reshape_op = _single_consumer(mul_output_name)
            if (
                reshape_op is None
                or str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != mul_output_name
            ):
                continue
            reshape_output_name = str(reshape_op.outputs[0])
            reshape_tensor = candidate_model.tensors.get(reshape_output_name)
            if (
                reshape_output_name in model_outputs
                or reshape_tensor is None
                or len(reshape_tensor.shape) != 4
                or any(int(value) <= 0 for value in reshape_tensor.shape)
                or [int(value) for value in reshape_tensor.shape[1:3]] != [1, 1]
            ):
                continue
            add_op = _single_consumer(reshape_output_name)
            if (
                add_op is None
                or str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue
            add_output_name = str(add_op.outputs[0])
            if add_output_name in model_outputs:
                continue
            if not any(
                str(name) != reshape_output_name
                and (tensor := candidate_model.tensors.get(str(name))) is not None
                and tensor.data is not None
                for name in add_op.inputs
            ):
                continue
            conv_op = _single_consumer(add_output_name)
            if conv_op is not None and str(conv_op.op_type) in {
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
            }:
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.mean_mul_add_conv_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=20,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
