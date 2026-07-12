from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import (
    PassPhase,
    PassSpec,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_singleton_constant_tensor,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_singleton_constant_float,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


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
) -> Dict[str, int]:
    """Run duplicate layout-adapter cleanup as one ordered transaction group."""

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
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_maximum_minimum_relu0to1_chains(model_ir: ModelIR) -> Dict[str, int]:
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
    graph_index = ModelIRGraphIndex(model_ir)

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

            min_op.op_type = "RELU_0_TO_1"
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

    _prune_unused_tensors(model_ir)
    return {"rewritten_maximum_minimum_relu0to1_chains": int(rewritten)}


def _optimize_squeeze_reshape_identity_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove redundant SQUEEZE -> RESHAPE chains that round-trip to input shape.

    Target:
      x --SQUEEZE--> s --RESHAPE--> y, where shape(y) == shape(x)

    Rewrite:
      replace all uses of y with x and remove SQUEEZE/RESHAPE.
    """
    rewritten = 0
    graph_index = ModelIRGraphIndex(model_ir)

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

    _prune_unused_tensors(model_ir)
    return {"optimized_squeeze_reshape_identity_chains": int(rewritten)}
