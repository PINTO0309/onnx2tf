from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    preflight_any_operator,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import (
    PassPhase,
    PassSpec,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
    _is_singleton_constant_tensor,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_singleton_constant_float,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


def prune_dead_operators(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    prune_tensors: bool = True,
) -> Dict[str, int]:
    """Remove operators that do not contribute to graph outputs."""

    if len(model_ir.operators) == 0:
        return {"removed_dead_operators": 0}

    live_tensors = set(model_ir.outputs)
    keep_flags = [False for _ in model_ir.operators]
    for op_index in range(len(model_ir.operators) - 1, -1, -1):
        op = model_ir.operators[op_index]
        outputs_live = any(
            output_name in live_tensors for output_name in op.outputs
        )

        # Some kernels mutate variable input tensors in place. Retain such an
        # operator when live graph state depends on that variable input.
        mutates_live_variable_input = False
        if not outputs_live:
            for input_name in op.inputs:
                if input_name not in live_tensors:
                    continue
                input_tensor = model_ir.tensors.get(str(input_name), None)
                if input_tensor is not None and bool(input_tensor.is_variable):
                    mutates_live_variable_input = True
                    break

        if outputs_live or mutates_live_variable_input:
            keep_flags[op_index] = True
            live_tensors.update(op.inputs)

    remove_indices = [
        index for index, keep in enumerate(keep_flags) if not keep
    ]
    if len(remove_indices) == 0:
        return {"removed_dead_operators": 0}

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    graph_index.remove_operators(remove_indices)
    if prune_tensors:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"removed_dead_operators": int(len(remove_indices))}


def _optimize_duplicate_transpose_fanout(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Deduplicate fan-out TRANSPOSE nodes with identical input and permutation.

    Target pattern:
      X --TRANSPOSE(P)--> Y0
      X --TRANSPOSE(P)--> Y1
      ...

    Rewritten:
      X --TRANSPOSE(P)--> Y0
      (all uses of Y1, ... are rewired to Y0; duplicate TRANSPOSE nodes removed)
    """
    removed_duplicates = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        canonical_by_key: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            perm = _read_transpose_perm(model_ir, op)
            if perm is None:
                continue

            key = (input_name, tuple(int(value) for value in perm))
            canonical_idx = canonical_by_key.get(key, None)
            if canonical_idx is None:
                canonical_by_key[key] = int(op_idx)
                continue

            if output_name in model_ir.outputs:
                continue

            canonical_op = model_ir.operators[int(canonical_idx)]
            if len(canonical_op.outputs) != 1:
                continue
            canonical_output = str(canonical_op.outputs[0])
            if canonical_output == output_name:
                continue

            canonical_tensor = model_ir.tensors.get(canonical_output, None)
            duplicate_tensor = model_ir.tensors.get(output_name, None)
            if canonical_tensor is not None and duplicate_tensor is not None:
                if canonical_tensor.shape == [1] and duplicate_tensor.shape != [1]:
                    canonical_tensor.shape = [
                        int(value) for value in duplicate_tensor.shape
                    ]
                    canonical_tensor.shape_signature = (
                        [
                            int(value)
                            for value in duplicate_tensor.shape_signature
                        ]
                        if duplicate_tensor.shape_signature is not None
                        else [int(value) for value in duplicate_tensor.shape]
                    )
                if (
                    canonical_tensor.quantization is None
                    and duplicate_tensor.quantization is not None
                ):
                    canonical_tensor.quantization = _clone_quantization(
                        duplicate_tensor.quantization
                    )
                if (
                    str(canonical_tensor.dtype) == "FLOAT32"
                    and str(duplicate_tensor.dtype) != "FLOAT32"
                ):
                    canonical_tensor.dtype = str(duplicate_tensor.dtype)

            _replace_tensor_inputs(
                model_ir,
                output_name,
                canonical_output,
                graph_index=graph_index,
            )
            graph_index.remove_operator(op_idx)
            removed_duplicates += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"removed_duplicate_transpose_fanout": int(removed_duplicates)}


def _optimize_duplicate_reshape_fanout(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Deduplicate fan-out RESHAPE nodes with identical input and target shape.

    Target pattern:
      X --RESHAPE(S)--> Y0
      X --RESHAPE(S)--> Y1
      ...

    Rewritten:
      X --RESHAPE(S)--> Y0
      (all uses of Y1, ... are rewired to Y0; duplicate RESHAPE nodes removed)
    """
    removed_duplicates = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _read_reshape_target_shape(op: OperatorIR) -> Optional[List[int]]:
        if str(op.op_type) != "RESHAPE":
            return None
        if isinstance(op.options, dict):
            new_shape = op.options.get("newShape", None)
            if isinstance(new_shape, list) and len(new_shape) > 0:
                try:
                    return [int(v) for v in list(new_shape)]
                except Exception:
                    pass
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            shape_values = _read_const_ints_from_tensor(shape_tensor)
            if shape_values is not None and len(shape_values) > 0:
                return [int(v) for v in list(shape_values)]
        return None

    while True:
        changed = False
        canonical_by_key: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "RESHAPE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            target_shape = _read_reshape_target_shape(op)
            if target_shape is None:
                continue

            key = (input_name, tuple(int(v) for v in list(target_shape)))
            canonical_idx = canonical_by_key.get(key, None)
            if canonical_idx is None:
                canonical_by_key[key] = int(op_idx)
                continue

            if output_name in model_ir.outputs:
                # Preserve user-visible graph output names.
                continue

            canonical_op = model_ir.operators[int(canonical_idx)]
            if len(canonical_op.outputs) != 1:
                continue
            canonical_output = str(canonical_op.outputs[0])
            if canonical_output == output_name:
                continue

            canonical_tensor = model_ir.tensors.get(canonical_output, None)
            duplicate_tensor = model_ir.tensors.get(output_name, None)
            if canonical_tensor is not None and duplicate_tensor is not None:
                if canonical_tensor.shape == [1] and duplicate_tensor.shape != [1]:
                    canonical_tensor.shape = [int(v) for v in list(duplicate_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(duplicate_tensor.shape_signature)]
                        if duplicate_tensor.shape_signature is not None
                        else [int(v) for v in list(duplicate_tensor.shape)]
                    )
                if canonical_tensor.quantization is None and duplicate_tensor.quantization is not None:
                    canonical_tensor.quantization = _clone_quantization(duplicate_tensor.quantization)
                if str(canonical_tensor.dtype) == "FLOAT32" and str(duplicate_tensor.dtype) != "FLOAT32":
                    canonical_tensor.dtype = str(duplicate_tensor.dtype)

            _replace_tensor_inputs(
                model_ir,
                output_name,
                canonical_output,
                graph_index=graph_index,
            )
            graph_index.remove_operator(op_idx)
            removed_duplicates += 1
            changed = True
            break

        if not changed:
            break

    if removed_duplicates > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "removed_duplicate_reshape_fanout": int(removed_duplicates),
    }


def run_duplicate_fanout_cleanup(
    model_ir: ModelIR,
    *,
    include_transpose: bool = True,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run duplicate layout-adapter cleanup as one ordered transaction group."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        reshape_count = 0
        transpose_count = 0
        for visited, op in enumerate(candidate_model.operators, start=1):
            if str(op.op_type) == "RESHAPE":
                reshape_count += 1
            elif include_transpose and str(op.op_type) == "TRANSPOSE":
                transpose_count += 1
            if reshape_count >= 2 or transpose_count >= 2:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _run_transpose(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_duplicate_transpose_fanout(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("removed_duplicate_transpose_fanout", 0)),
        }

    def _run_reshape(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_duplicate_reshape_fanout(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("removed_duplicate_reshape_fanout", 0)),
        }

    def _has_duplicate_transpose_candidate(pass_state: ModelIRPassState) -> bool:
        seen: set[Tuple[str, Tuple[int, ...]]] = set()
        for op in pass_state.model_ir.operators:
            if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
                continue
            perm = _read_transpose_perm(pass_state.model_ir, op)
            if perm is None:
                continue
            key = (str(op.inputs[0]), tuple(int(value) for value in perm))
            if key in seen:
                return True
            seen.add(key)
        return False

    def _has_duplicate_reshape_candidate(pass_state: ModelIRPassState) -> bool:
        input_counts: Dict[str, int] = {}
        for op in pass_state.model_ir.operators:
            if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2:
                continue
            input_name = str(op.inputs[0])
            input_counts[input_name] = int(input_counts.get(input_name, 0)) + 1
            if input_counts[input_name] > 1:
                return True
        return False

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_transpose:
        specs.append(
            PassSpec(
                pass_id="cleanup.duplicate_transpose_fanout",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                callback=_run_transpose,
                precondition=_has_duplicate_transpose_candidate,
                transactional=True,
            )
        )
    specs.append(
        PassSpec(
            pass_id="cleanup.duplicate_reshape_fanout",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=_run_reshape,
            precondition=_has_duplicate_reshape_candidate,
            transactional=True,
        )
    )
    default_details: Dict[str, int] = {
        "removed_duplicate_reshape_fanout": 0,
    }
    if include_transpose:
        default_details["removed_duplicate_transpose_fanout"] = 0
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_consecutive_reshape_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Remove redundant RESHAPE chains.

    Target:
      x --RESHAPE(no-op)--> y
      x --RESHAPE--> y --RESHAPE--> z

    Rewrite:
      x ----------------> y
      x -----------RESHAPE--> z

    Safety:
    - No-op reshape removal requires matching input/output shape and shape_signature.
    - For graph-output tensors, preserve visible output names.
    - Middle tensor `y` can have fan-out. In that case the second RESHAPE is
      rewired to the original source while keeping the first RESHAPE for other users.
    - First RESHAPE removal still requires `y` to be consumed only by the second RESHAPE.
    - `y` must not be a graph output.
    - Source and destination element counts must match with fully known static shapes.
    """
    rewritten = 0
    rewritten_fanout_bypass = 0
    removed_noop = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _reshape_depends_on_input_dims(reshape_op: OperatorIR) -> bool:
        """
        Return True when RESHAPE target depends on source tensor dimensions.
        Such reshapes must not be bypassed across an intermediate reshape,
        otherwise ONNX semantics of 0/-1 placeholders can change.
        """
        options = (
            dict(reshape_op.options)
            if isinstance(reshape_op.options, dict)
            else {}
        )
        new_shape = options.get("newShape", None)
        if new_shape is not None:
            try:
                concrete_values = [int(v) for v in np.asarray(new_shape).reshape(-1).tolist()]
            except Exception:
                concrete_values = []
            if concrete_values and all(int(v) > 0 for v in concrete_values):
                return False
            if (
                bool(options.get("layoutTransposeAsReshape", False))
                and concrete_values
                and int(concrete_values[0]) == -1
                and all(int(v) > 0 for v in concrete_values[1:])
            ):
                # A singleton-only layout transpose may preserve a dynamic
                # batch with one leading -1. Bypassing an earlier reshape is
                # safe because both reshapes preserve the total element count
                # and the inferred dimension remains the leading batch.
                return False
        raw_shape = options.get("onnxRawNewShape", None)
        if raw_shape is not None:
            try:
                values = [int(v) for v in np.asarray(raw_shape).reshape(-1).tolist()]
            except Exception:
                return True
            if any(int(v) <= 0 for v in values):
                return True
            return False
        if new_shape is None:
            return False
        try:
            values = [int(v) for v in np.asarray(new_shape).reshape(-1).tolist()]
        except Exception:
            return True
        return any(int(v) <= 0 for v in values)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        # 1) Remove single no-op reshape: input/output metadata is identical.
        for reshape_idx, reshape_op in enumerate(model_ir.operators):
            if str(reshape_op.op_type) != "RESHAPE" or len(reshape_op.inputs) < 1 or len(reshape_op.outputs) != 1:
                continue
            if bool(reshape_op.options.get("preserveDynamicShape", False)):
                continue
            if bool(reshape_op.options.get("preserveSemanticRank", False)):
                continue

            src_name = str(reshape_op.inputs[0])
            dst_name = str(reshape_op.outputs[0])
            src_tensor = model_ir.tensors.get(src_name, None)
            dst_tensor = model_ir.tensors.get(dst_name, None)
            if src_tensor is None or dst_tensor is None:
                continue
            if bool(src_tensor.is_variable) or bool(dst_tensor.is_variable):
                continue

            src_shape = [int(v) for v in list(src_tensor.shape)]
            dst_shape = [int(v) for v in list(dst_tensor.shape)]
            src_signature = (
                [int(v) for v in list(src_tensor.shape_signature)]
                if src_tensor.shape_signature is not None
                else [int(v) for v in list(src_shape)]
            )
            dst_signature = (
                [int(v) for v in list(dst_tensor.shape_signature)]
                if dst_tensor.shape_signature is not None
                else [int(v) for v in list(dst_shape)]
            )
            if src_shape != dst_shape or src_signature != dst_signature:
                continue

            # Keep graph-output tensor names stable when possible.
            if dst_name in model_outputs:
                if len(consumers.get(dst_name, [])) != 0:
                    continue
                producer_idx = producers.get(src_name, None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                producer_outputs = [str(v) for v in list(producer_op.outputs)]
                if src_name not in set(producer_outputs):
                    continue
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=producer_op,
                    new_outputs=[
                        str(dst_name) if str(v) == str(src_name) else str(v)
                        for v in producer_outputs
                    ],
                    graph_index=graph_index,
                )
            else:
                _replace_tensor_inputs(
                    model_ir=model_ir,
                    src_name=str(dst_name),
                    dst_name=str(src_name),
                    graph_index=graph_index,
                )

            graph_index.remove_operator(int(reshape_idx))
            removed_noop += 1
            changed = True
            break

        if changed:
            continue

        # 2) Bypass intermediate reshape for RESHAPE->RESHAPE links, even with fan-out.
        for first_idx, first_op in enumerate(model_ir.operators):
            if str(first_op.op_type) != "RESHAPE" or len(first_op.inputs) < 1 or len(first_op.outputs) != 1:
                continue
            if bool(first_op.options.get("preserveSemanticRank", False)):
                continue

            first_input_name = str(first_op.inputs[0])
            first_output_name = str(first_op.outputs[0])
            if first_output_name in model_outputs:
                continue

            first_input_tensor = model_ir.tensors.get(first_input_name, None)
            first_output_tensor = model_ir.tensors.get(first_output_name, None)
            if (
                first_input_tensor is None
                or first_output_tensor is None
                or bool(first_input_tensor.is_variable)
                or bool(first_output_tensor.is_variable)
                or not _is_fully_known_positive_shape(first_input_tensor.shape)
            ):
                continue
            first_input_shape = [int(v) for v in list(first_input_tensor.shape)]
            if len(first_input_shape) == 0:
                continue
            first_input_elements = int(np.prod(np.asarray(first_input_shape, dtype=np.int64)))

            first_users = [int(v) for v in consumers.get(first_output_name, []) if int(v) != int(first_idx)]
            if len(first_users) <= 1:
                continue

            for second_idx in first_users:
                second_op = model_ir.operators[int(second_idx)]
                if str(second_op.op_type) != "RESHAPE" or len(second_op.inputs) < 1 or len(second_op.outputs) != 1:
                    continue
                if bool(second_op.options.get("preserveSemanticRank", False)):
                    continue
                if str(second_op.inputs[0]) != first_output_name:
                    continue
                if _reshape_depends_on_input_dims(second_op):
                    continue

                second_output_name = str(second_op.outputs[0])
                second_output_tensor = model_ir.tensors.get(second_output_name, None)
                if second_output_tensor is None or not _is_fully_known_positive_shape(second_output_tensor.shape):
                    continue
                second_output_shape = [int(v) for v in list(second_output_tensor.shape)]
                if len(second_output_shape) == 0:
                    continue
                second_output_elements = int(np.prod(np.asarray(second_output_shape, dtype=np.int64)))
                if int(first_input_elements) != int(second_output_elements):
                    continue

                second_inputs = [str(v) for v in list(second_op.inputs)]
                if str(second_inputs[0]) == str(first_input_name):
                    continue
                second_inputs[0] = str(first_input_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=second_op,
                    new_inputs=second_inputs,
                    graph_index=graph_index,
                )

                rewritten += 1
                rewritten_fanout_bypass += 1
                changed = True
                break

            if changed:
                break

        if changed:
            continue

        # 3) Remove consecutive reshape chains when the middle tensor has a single user.
        for first_idx, first_op in enumerate(model_ir.operators):
            if str(first_op.op_type) != "RESHAPE" or len(first_op.inputs) < 1 or len(first_op.outputs) != 1:
                continue
            if bool(first_op.options.get("preserveSemanticRank", False)):
                continue

            first_input_name = str(first_op.inputs[0])
            first_output_name = str(first_op.outputs[0])
            if first_output_name in model_outputs:
                continue
            first_input_tensor = model_ir.tensors.get(first_input_name, None)
            first_output_tensor = model_ir.tensors.get(first_output_name, None)
            if (
                first_input_tensor is None
                or first_output_tensor is None
                or bool(first_input_tensor.is_variable)
                or bool(first_output_tensor.is_variable)
            ):
                continue

            first_users = [int(v) for v in consumers.get(first_output_name, [])]
            if len(first_users) != 1:
                continue

            second_idx = int(first_users[0])
            if int(second_idx) == int(first_idx):
                continue
            second_op = model_ir.operators[int(second_idx)]
            if str(second_op.op_type) != "RESHAPE" or len(second_op.inputs) < 1 or len(second_op.outputs) != 1:
                continue
            if bool(second_op.options.get("preserveSemanticRank", False)):
                continue
            if str(second_op.inputs[0]) != first_output_name:
                continue
            if _reshape_depends_on_input_dims(second_op):
                continue

            second_output_name = str(second_op.outputs[0])

            second_output_tensor = model_ir.tensors.get(second_output_name, None)
            if (
                first_input_tensor is None
                or second_output_tensor is None
                or not _is_fully_known_positive_shape(first_input_tensor.shape)
                or not _is_fully_known_positive_shape(second_output_tensor.shape)
            ):
                continue

            first_input_shape = [int(v) for v in list(first_input_tensor.shape)]
            second_output_shape = [int(v) for v in list(second_output_tensor.shape)]
            if len(first_input_shape) == 0 or len(second_output_shape) == 0:
                continue
            if int(np.prod(np.asarray(first_input_shape, dtype=np.int64))) != int(
                np.prod(np.asarray(second_output_shape, dtype=np.int64))
            ):
                continue

            second_inputs = [str(v) for v in list(second_op.inputs)]
            second_inputs[0] = str(first_input_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=second_op,
                new_inputs=second_inputs,
                graph_index=graph_index,
            )

            graph_index.remove_operator(int(first_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0 or removed_noop > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {
        "removed_noop_reshape_chains": int(removed_noop),
        "rewritten_consecutive_reshape_passthrough_chains": int(rewritten),
        "rewritten_fanout_bypass_reshape_passthrough_chains": int(rewritten_fanout_bypass),
    }


def run_consecutive_reshape_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run general no-op, fan-out, and consecutive Reshape cleanup."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_any_operator(
            candidate_model,
            lambda operator: str(operator.op_type) == "RESHAPE",
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(value) for value in candidate_model.outputs)
        for reshape_op in candidate_model.operators:
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 1
                or len(reshape_op.outputs) != 1
            ):
                continue
            src_name = str(reshape_op.inputs[0])
            dst_name = str(reshape_op.outputs[0])
            src_tensor = candidate_model.tensors.get(src_name)
            dst_tensor = candidate_model.tensors.get(dst_name)
            if (
                src_tensor is not None
                and dst_tensor is not None
                and not bool(src_tensor.is_variable)
                and not bool(dst_tensor.is_variable)
                and list(src_tensor.shape) == list(dst_tensor.shape)
                and list(src_tensor.shape_signature or src_tensor.shape)
                == list(dst_tensor.shape_signature or dst_tensor.shape)
                and not bool(reshape_op.options.get("preserveDynamicShape", False))
                and not bool(reshape_op.options.get("preserveSemanticRank", False))
            ):
                if dst_name not in model_outputs:
                    return True
                if (
                    len(pass_state.graph_index.consumer_indices(dst_name)) == 0
                    and pass_state.graph_index.producers.get(src_name) is not None
                ):
                    return True
            if dst_name in model_outputs or bool(
                reshape_op.options.get("preserveSemanticRank", False)
            ):
                continue
            for user_index in pass_state.graph_index.consumer_indices(dst_name):
                user_op = candidate_model.operators[int(user_index)]
                if (
                    str(user_op.op_type) == "RESHAPE"
                    and len(user_op.inputs) >= 1
                    and str(user_op.inputs[0]) == dst_name
                    and not bool(user_op.options.get("preserveSemanticRank", False))
                ):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_consecutive_reshape_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(sum(int(value) for value in stats.values())),
        }

    default_details = {
        "removed_noop_reshape_chains": 0,
        "rewritten_consecutive_reshape_passthrough_chains": 0,
        "rewritten_fanout_bypass_reshape_passthrough_chains": 0,
    }
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="cleanup.consecutive_reshape_passthrough",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                priority=10,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_maximum_minimum_relu0to1_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Replace clamp chains MAXIMUM(0.0) -> MINIMUM(1.0) with RELU_0_TO_1.

    Target:
      X --MAXIMUM(0.0)--> M --MINIMUM(1.0)--> Y

    Rewrite:
      X --RELU_0_TO_1--> Y

    Safety:
    - MAXIMUM and MINIMUM side inputs must be singleton constants.
    - MAXIMUM output must be consumed only by the matched MINIMUM.
    """
    rewritten = 0
    atol = 1e-6
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for min_idx, min_op in enumerate(model_ir.operators):
            if str(min_op.op_type) != "MINIMUM" or len(min_op.inputs) != 2 or len(min_op.outputs) != 1:
                continue

            min_input0 = str(min_op.inputs[0])
            min_input1 = str(min_op.inputs[1])
            if _is_singleton_constant_tensor(model_ir, min_input0):
                min_data_name = str(min_input1)
                min_const_name = str(min_input0)
            elif _is_singleton_constant_tensor(model_ir, min_input1):
                min_data_name = str(min_input0)
                min_const_name = str(min_input1)
            else:
                continue
            min_const_value = _read_singleton_constant_float(model_ir, min_const_name)
            if min_const_value is None or not np.isclose(float(min_const_value), 1.0, atol=atol):
                continue
            if min_data_name in model_outputs:
                continue

            max_idx = producers.get(min_data_name, None)
            if max_idx is None:
                continue
            max_op = model_ir.operators[int(max_idx)]
            if str(max_op.op_type) != "MAXIMUM" or len(max_op.inputs) != 2 or len(max_op.outputs) != 1:
                continue
            if str(max_op.outputs[0]) != str(min_data_name):
                continue

            max_input0 = str(max_op.inputs[0])
            max_input1 = str(max_op.inputs[1])
            if _is_singleton_constant_tensor(model_ir, max_input0):
                max_data_name = str(max_input1)
                max_const_name = str(max_input0)
            elif _is_singleton_constant_tensor(model_ir, max_input1):
                max_data_name = str(max_input0)
                max_const_name = str(max_input1)
            else:
                continue
            max_const_value = _read_singleton_constant_float(model_ir, max_const_name)
            if max_const_value is None or not np.isclose(float(max_const_value), 0.0, atol=atol):
                continue

            max_users = [int(v) for v in consumers.get(str(min_data_name), [])]
            if len(max_users) != 1 or int(max_users[0]) != int(min_idx):
                continue

            graph_index.replace_operator_type(min_idx, "RELU_0_TO_1")
            min_op.version = 1
            _set_operator_inputs(
                model_ir=model_ir,
                op=min_op,
                new_inputs=[str(max_data_name)],
                graph_index=graph_index,
            )
            min_op.options = {}

            graph_index.remove_operator(max_idx)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"rewritten_maximum_minimum_relu0to1_chains": int(rewritten)}


def run_clamp_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run scalar zero-to-one clamp canonicalization transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_required_op_types(candidate_model, {"MAXIMUM", "MINIMUM"})

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        for op in pass_state.model_ir.operators:
            if str(op.op_type) != "MINIMUM" or len(op.inputs) != 2:
                continue
            for input_name in op.inputs:
                producer_idx = pass_state.graph_index.producers.get(
                    str(input_name),
                    None,
                )
                if producer_idx is None:
                    continue
                producer = pass_state.model_ir.operators[int(producer_idx)]
                if str(producer.op_type) == "MAXIMUM":
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_maximum_minimum_relu0to1_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_maximum_minimum_relu0to1_chains", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.scalar_clamp_relu0to1",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"rewritten_maximum_minimum_relu0to1_chains": 0},
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_maximum_with_zero_input2_to_relu(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Rewrite float Maximum(data, scalar-zero) operators to Relu."""

    rewritten = 0
    atol = 1e-6
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    for op_index, op in enumerate(model_ir.operators):
        if str(op.op_type) != "MAXIMUM" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue

        data_name = str(op.inputs[0])
        const_name = str(op.inputs[1])
        if not _is_singleton_constant_tensor(model_ir, const_name):
            continue
        const_value = _read_singleton_constant_float(model_ir, const_name)
        if const_value is None or not np.isclose(float(const_value), 0.0, atol=atol):
            continue

        data_tensor = model_ir.tensors.get(data_name, None)
        if data_tensor is None:
            continue
        if str(data_tensor.dtype).upper() not in {"FLOAT16", "FLOAT32"}:
            continue

        graph_index.replace_operator_type(op_index, "RELU")
        op.version = 1
        _set_operator_inputs(
            model_ir=model_ir,
            op=op,
            new_inputs=[data_name],
            graph_index=graph_index,
        )
        op.options = {}
        rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"rewritten_maximum_with_zero_input2_to_relu": int(rewritten)}


def run_maximum_zero_relu_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run guarded Maximum(data, zero) canonicalization transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_any_operator(
            candidate_model,
            lambda op: str(op.op_type) == "MAXIMUM",
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) == "MAXIMUM"
            and len(op.inputs) == 2
            and len(op.outputs) == 1
            for op in pass_state.model_ir.operators
        )

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_maximum_with_zero_input2_to_relu(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("rewritten_maximum_with_zero_input2_to_relu", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.maximum_zero_relu",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"rewritten_maximum_with_zero_input2_to_relu": 0},
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_fold_consecutive_mul_constants_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold strict floating-point Mul(data, const) chains."""

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_constant_tensor(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        return tensor is not None and tensor.data is not None

    def _is_float_tensor_name(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        return tensor is not None and str(tensor.dtype).upper().startswith("FLOAT")

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(name) for name in model_ir.outputs)

        for second_mul_idx, second_mul_op in enumerate(model_ir.operators):
            if (
                str(second_mul_op.op_type) != "MUL"
                or len(second_mul_op.inputs) != 2
                or len(second_mul_op.outputs) != 1
            ):
                continue

            second_in0 = str(second_mul_op.inputs[0])
            second_in1 = str(second_mul_op.inputs[1])
            second_const_name: Optional[str] = None
            mid_name: Optional[str] = None
            second_const_input_index = -1
            mid_input_index = -1
            if _is_constant_tensor(second_in0):
                second_const_name = second_in0
                mid_name = second_in1
                second_const_input_index = 0
                mid_input_index = 1
            elif _is_constant_tensor(second_in1):
                second_const_name = second_in1
                mid_name = second_in0
                second_const_input_index = 1
                mid_input_index = 0
            if second_const_name is None or mid_name is None:
                continue
            if mid_name in model_outputs:
                continue

            mid_users = sorted({int(index) for index in consumers.get(mid_name, [])})
            if mid_users != [int(second_mul_idx)]:
                continue

            first_mul_idx = producers.get(mid_name, None)
            if first_mul_idx is None:
                continue
            first_mul_op = model_ir.operators[int(first_mul_idx)]
            if (
                str(first_mul_op.op_type) != "MUL"
                or len(first_mul_op.inputs) != 2
                or len(first_mul_op.outputs) != 1
                or str(first_mul_op.outputs[0]) != mid_name
            ):
                continue

            first_in0 = str(first_mul_op.inputs[0])
            first_in1 = str(first_mul_op.inputs[1])
            first_const_name: Optional[str] = None
            data_name: Optional[str] = None
            if _is_constant_tensor(first_in0):
                first_const_name = first_in0
                data_name = first_in1
            elif _is_constant_tensor(first_in1):
                first_const_name = first_in1
                data_name = first_in0
            if first_const_name is None or data_name is None:
                continue

            second_output_name = str(second_mul_op.outputs[0])
            if not all(
                _is_float_tensor_name(name)
                for name in (data_name, mid_name, second_output_name)
            ):
                continue

            first_const_tensor = model_ir.tensors.get(first_const_name, None)
            second_const_tensor = model_ir.tensors.get(second_const_name, None)
            if first_const_tensor is None or first_const_tensor.data is None:
                continue
            if second_const_tensor is None or second_const_tensor.data is None:
                continue
            first_const_data = np.asarray(first_const_tensor.data)
            second_const_data = np.asarray(second_const_tensor.data)
            if not np.issubdtype(first_const_data.dtype, np.floating):
                continue
            if not np.issubdtype(second_const_data.dtype, np.floating):
                continue

            fused_dtype = np.result_type(first_const_data.dtype, second_const_data.dtype)
            if not np.issubdtype(fused_dtype, np.floating):
                continue
            try:
                fused_const_data = (
                    first_const_data.astype(fused_dtype, copy=False)
                    * second_const_data.astype(fused_dtype, copy=False)
                )
            except Exception:
                continue
            if not np.all(np.isfinite(fused_const_data)):
                continue

            fused_const_name = _unique_tensor_name(f"{first_const_name}_mulfused")
            fused_shape, fused_signature = normalize_onnx_shape(
                list(fused_const_data.shape)
            )
            model_ir.tensors[fused_const_name] = TensorIR(
                name=fused_const_name,
                dtype=tflite_dtype_from_numpy(fused_const_data.dtype),
                shape=[int(dim) for dim in fused_shape],
                shape_signature=[int(dim) for dim in fused_signature],
                data=fused_const_data,
                is_variable=False,
                quantization=_clone_quantization(
                    first_const_tensor.quantization
                    if first_const_tensor.quantization is not None
                    else second_const_tensor.quantization
                ),
            )
            if layout_state is not None:
                layout_state.set(
                    fused_const_name,
                    logical=model_ir.tensors[fused_const_name].logical_layout,
                    physical=model_ir.tensors[fused_const_name].physical_layout,
                )

            new_inputs = [str(name) for name in second_mul_op.inputs]
            new_inputs[int(mid_input_index)] = str(data_name)
            new_inputs[int(second_const_input_index)] = str(fused_const_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=second_mul_op,
                new_inputs=new_inputs,
                graph_index=graph_index,
            )
            graph_index.remove_operator(int(first_mul_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_fold_consecutive_mul_constants_chains": int(rewritten)}


def run_consecutive_mul_constants_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run guarded consecutive floating Mul folding transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        mul_count = 0
        for visited, op in enumerate(candidate_model.operators, start=1):
            if str(op.op_type) != "MUL":
                continue
            mul_count += 1
            if mul_count >= 2:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        for op in pass_state.model_ir.operators:
            if str(op.op_type) != "MUL" or len(op.outputs) != 1:
                continue
            for consumer_idx in pass_state.graph_index.consumer_indices(
                str(op.outputs[0])
            ):
                consumer = pass_state.model_ir.operators[int(consumer_idx)]
                if str(consumer.op_type) == "MUL":
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_fold_consecutive_mul_constants_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_fold_consecutive_mul_constants_chains", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.fold_consecutive_mul_constants",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"optimized_fold_consecutive_mul_constants_chains": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_squeeze_unary_reshape_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold Squeeze(axis=0) -> unary -> shape-restoring Reshape."""

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    unary_ops = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "CAST",
        "FLOOR",
        "CEIL",
        "ROUND",
        "HARD_SWISH",
    }

    def _shape_list(name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.shape is None:
            return None
        return [int(dim) for dim in tensor.shape]

    def _shape_compatible(lhs: List[int], rhs: List[int]) -> bool:
        return len(lhs) == len(rhs) and all(
            int(left) < 0 or int(right) < 0 or int(left) == int(right)
            for left, right in zip(lhs, rhs)
        )

    while True:
        changed = False
        consumers = graph_index.consumers

        for squeeze_idx, squeeze_op in enumerate(model_ir.operators):
            if (
                str(squeeze_op.op_type) != "SQUEEZE"
                or len(squeeze_op.inputs) != 1
                or len(squeeze_op.outputs) != 1
            ):
                continue

            squeeze_in_name = str(squeeze_op.inputs[0])
            squeeze_out_name = str(squeeze_op.outputs[0])
            squeeze_users = [
                int(index) for index in consumers.get(squeeze_out_name, [])
            ]
            if len(squeeze_users) != 1:
                continue

            unary_idx = int(squeeze_users[0])
            unary_op = model_ir.operators[unary_idx]
            if (
                str(unary_op.op_type) not in unary_ops
                or len(unary_op.inputs) != 1
                or len(unary_op.outputs) != 1
                or str(unary_op.inputs[0]) != squeeze_out_name
            ):
                continue

            unary_out_name = str(unary_op.outputs[0])
            unary_users = [int(index) for index in consumers.get(unary_out_name, [])]
            reshape_user_indices = [
                user_idx
                for user_idx in unary_users
                if (
                    str(model_ir.operators[user_idx].op_type) == "RESHAPE"
                    and len(model_ir.operators[user_idx].inputs) >= 1
                    and len(model_ir.operators[user_idx].outputs) == 1
                    and str(model_ir.operators[user_idx].inputs[0]) == unary_out_name
                )
            ]
            if len(reshape_user_indices) != 1:
                continue

            reshape_idx = int(reshape_user_indices[0])
            reshape_op = model_ir.operators[reshape_idx]
            reshape_out_name = str(reshape_op.outputs[0])
            in_shape = _shape_list(squeeze_in_name)
            squeezed_shape = _shape_list(squeeze_out_name)
            reshape_out_shape = _shape_list(reshape_out_name)
            if in_shape is None or squeezed_shape is None or reshape_out_shape is None:
                continue
            if len(in_shape) != len(squeezed_shape) + 1 or len(in_shape) < 1:
                continue
            if int(in_shape[0]) != 1:
                continue
            if not _shape_compatible(squeezed_shape, in_shape[1:]):
                continue
            if not _shape_compatible(reshape_out_shape, in_shape):
                continue

            squeeze_options = (
                dict(squeeze_op.options)
                if isinstance(squeeze_op.options, dict)
                else {}
            )
            if "squeezeDims" in squeeze_options:
                raw_axes = np.asarray(
                    squeeze_options.get("squeezeDims", []),
                    dtype=np.int64,
                ).reshape(-1)
                normalized_axes = _normalize_squeeze_axes_for_rank(
                    [int(axis) for axis in raw_axes.tolist()],
                    len(in_shape),
                )
                if normalized_axes is None or normalized_axes != [0]:
                    continue

            if len(unary_users) == 1:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_inputs=[squeeze_in_name],
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_outputs=[reshape_out_name],
                    graph_index=graph_index,
                )
                for op_ref in (reshape_op, squeeze_op):
                    remove_idx = graph_index.operator_index(op_ref)
                    if remove_idx is not None:
                        graph_index.remove_operator(remove_idx)
            else:
                if unary_out_name in model_ir.outputs:
                    continue
                if reshape_out_name == unary_out_name:
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_inputs=[squeeze_in_name],
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_outputs=[reshape_out_name],
                    graph_index=graph_index,
                )
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=squeeze_op,
                    new_inputs=[reshape_out_name],
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=squeeze_op,
                    new_outputs=[unary_out_name],
                    graph_index=graph_index,
                )
                current_reshape_idx = graph_index.operator_index(reshape_op)
                if current_reshape_idx is not None:
                    graph_index.remove_operator(current_reshape_idx)
                current_squeeze_idx = graph_index.operator_index(squeeze_op)
                if current_squeeze_idx is None:
                    continue
                squeeze_ref = graph_index.remove_operator(current_squeeze_idx)
                current_unary_idx = graph_index.operator_index(unary_op)
                if current_unary_idx is None:
                    continue
                graph_index.insert_operator(current_unary_idx + 1, squeeze_ref)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "optimized_squeeze_unary_reshape_passthrough_chains": int(rewritten)
    }


def _optimize_squeeze_reshape_identity_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Remove redundant SQUEEZE -> RESHAPE chains that round-trip to input shape.

    Target:
      x --SQUEEZE--> s --RESHAPE--> y, where shape(y) == shape(x)

    Rewrite:
      replace all uses of y with x and remove SQUEEZE/RESHAPE.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _shape_list(name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.shape is None:
            return None
        return [int(v) for v in list(tensor.shape)]

    def _dims_compatible(a: int, b: int) -> bool:
        if int(a) < 0 or int(b) < 0:
            return True
        return int(a) == int(b)

    def _shape_compatible(a: List[int], b: List[int]) -> bool:
        if len(a) != len(b):
            return False
        return all(_dims_compatible(int(x), int(y)) for x, y in zip(a, b))

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for squeeze_idx, squeeze_op in enumerate(model_ir.operators):
            if str(squeeze_op.op_type) != "SQUEEZE" or len(squeeze_op.inputs) != 1 or len(squeeze_op.outputs) != 1:
                continue

            squeeze_in_name = str(squeeze_op.inputs[0])
            squeeze_out_name = str(squeeze_op.outputs[0])
            if squeeze_out_name in model_outputs:
                continue

            squeeze_users = [int(v) for v in consumers.get(squeeze_out_name, [])]
            if len(squeeze_users) != 1:
                continue

            reshape_idx = int(squeeze_users[0])
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 1
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != squeeze_out_name
            ):
                continue

            reshape_out_name = str(reshape_op.outputs[0])
            if reshape_out_name in model_outputs:
                continue
            if reshape_out_name == squeeze_in_name:
                continue

            in_shape = _shape_list(squeeze_in_name)
            squeezed_shape = _shape_list(squeeze_out_name)
            reshape_out_shape = _shape_list(reshape_out_name)
            if in_shape is None or squeezed_shape is None or reshape_out_shape is None:
                continue
            if not _shape_compatible(in_shape, reshape_out_shape):
                continue

            squeeze_options = dict(squeeze_op.options) if isinstance(squeeze_op.options, dict) else {}
            squeeze_axes: List[int]
            if "squeezeDims" in squeeze_options:
                raw_axes = np.asarray(squeeze_options.get("squeezeDims", []), dtype=np.int64).reshape(-1)
                normalized_axes = _normalize_squeeze_axes_for_rank(
                    [int(v) for v in raw_axes.tolist()],
                    len(in_shape),
                )
                if normalized_axes is None:
                    continue
                squeeze_axes = [int(v) for v in normalized_axes]
            else:
                # Keep this conservative for unknown shapes when axes are omitted.
                if any(int(v) < 0 for v in in_shape):
                    continue
                squeeze_axes = [int(idx) for idx, dim in enumerate(in_shape) if int(dim) == 1]

            if len(set(int(v) for v in squeeze_axes)) != len(squeeze_axes):
                continue
            valid_axes = True
            for axis in squeeze_axes:
                if axis < 0 or axis >= len(in_shape):
                    valid_axes = False
                    break
                dim = int(in_shape[int(axis)])
                if dim >= 0 and dim != 1:
                    valid_axes = False
                    break
            if not valid_axes:
                continue

            squeeze_axes_set = set(int(v) for v in squeeze_axes)
            expected_squeezed_shape = [
                int(dim) for idx, dim in enumerate(in_shape) if int(idx) not in squeeze_axes_set
            ]
            if not _shape_compatible(squeezed_shape, expected_squeezed_shape):
                continue

            _replace_tensor_inputs(
                model_ir,
                reshape_out_name,
                squeeze_in_name,
                graph_index=graph_index,
            )
            for remove_idx in sorted([int(reshape_idx), int(squeeze_idx)], reverse=True):
                graph_index.remove_operator(remove_idx)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_squeeze_reshape_identity_chains": int(rewritten)}


def run_squeeze_reshape_identity_cleanup(
    model_ir: ModelIR,
    *,
    include_unary_passthrough: bool = False,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run guarded Squeeze/Reshape round-trip removal transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_required_op_types(candidate_model, {"SQUEEZE", "RESHAPE"})

    def _has_unary_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) == "SQUEEZE" and len(op.outputs) == 1
            for op in pass_state.model_ir.operators
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        for op in pass_state.model_ir.operators:
            if str(op.op_type) != "SQUEEZE" or len(op.outputs) != 1:
                continue
            users = pass_state.graph_index.consumer_indices(str(op.outputs[0]))
            if len(users) != 1:
                continue
            consumer = pass_state.model_ir.operators[int(users[0])]
            if str(consumer.op_type) == "RESHAPE":
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_squeeze_reshape_identity_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("optimized_squeeze_reshape_identity_chains", 0)),
        }

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_squeeze_unary_reshape_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_squeeze_unary_reshape_passthrough_chains", 0)
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_unary_passthrough:
        specs.append(
            PassSpec(
                pass_id="cleanup.squeeze_unary_reshape_passthrough",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                priority=10,
                callback=_run_unary,
                precondition=_has_unary_candidate,
                transactional=True,
            )
        )
    specs.append(
        PassSpec(
            pass_id="cleanup.squeeze_reshape_identity",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            priority=20,
            callback=_run,
            precondition=_has_candidate,
            transactional=True,
        )
    )
    default_details = {"optimized_squeeze_reshape_identity_chains": 0}
    if include_unary_passthrough:
        default_details["optimized_squeeze_unary_reshape_passthrough_chains"] = 0
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
