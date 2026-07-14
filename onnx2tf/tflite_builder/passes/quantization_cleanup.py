from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    preflight_any_operator,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _rename_tensor_globally,
    _set_operator_inputs,
    _set_operator_outputs,
    _shapes_equal,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _quantized_tensors_share_exact_grid(
    model_ir: ModelIR,
    lhs_name: str,
    rhs_name: str,
) -> bool:
    """Return whether two quantized tensors use exactly the same grid."""

    lhs = model_ir.tensors.get(str(lhs_name), None)
    rhs = model_ir.tensors.get(str(rhs_name), None)
    if lhs is None or rhs is None:
        return False
    if str(lhs.dtype).upper() != str(rhs.dtype).upper():
        return False
    lhs_q = lhs.quantization
    rhs_q = rhs.quantization
    if lhs_q is None or rhs_q is None:
        return False
    if int(lhs_q.quantized_dimension) != int(rhs_q.quantized_dimension):
        return False
    return bool(
        np.array_equal(
            np.asarray(lhs_q.scale, dtype=np.float64),
            np.asarray(rhs_q.scale, dtype=np.float64),
        )
        and np.array_equal(
            np.asarray(lhs_q.zero_point, dtype=np.int64),
            np.asarray(rhs_q.zero_point, dtype=np.int64),
        )
    )


def _optimize_concat_pre_quantize_dequantize(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Bypass exact-grid Q/DQ round trips immediately before Concat."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(
            str(operator.op_type) == "CONCATENATION" for operator in model_ir.operators
        ):
            _prune_unused_tensors(model_ir, layout_state=layout_state)
            return {"bypassed_concat_pre_quantize_dequantize": 0}
        active_index = ModelIRGraphIndex(model_ir)
    elif len(active_index.operator_indices("CONCATENATION")) == 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {"bypassed_concat_pre_quantize_dequantize": 0}

    model_outputs = {str(name) for name in model_ir.outputs}
    bypassed = 0

    while True:
        changed = False
        for concat_index in active_index.operator_indices("CONCATENATION"):
            concat = model_ir.operators[int(concat_index)]
            if len(concat.inputs) == 0:
                continue

            for input_position, concat_input_name in enumerate(
                [str(name) for name in concat.inputs]
            ):
                dequantize = active_index.producer(concat_input_name)
                dequantize_index = (
                    active_index.operator_index(dequantize)
                    if dequantize is not None
                    else None
                )
                if (
                    dequantize is None
                    or dequantize_index is None
                    or str(dequantize.op_type) != "DEQUANTIZE"
                    or len(dequantize.inputs) != 1
                    or len(dequantize.outputs) != 1
                    or str(dequantize.outputs[0]) != concat_input_name
                ):
                    continue

                quantized_name = str(dequantize.inputs[0])
                if (
                    quantized_name in model_outputs
                    or concat_input_name in model_outputs
                ):
                    continue

                quantize = active_index.producer(quantized_name)
                if (
                    quantize is None
                    or str(quantize.op_type) != "QUANTIZE"
                    or len(quantize.inputs) != 1
                    or len(quantize.outputs) != 1
                    or str(quantize.outputs[0]) != quantized_name
                ):
                    continue
                quantized_users = active_index.consumer_indices(quantized_name)
                if quantized_users != [int(dequantize_index)]:
                    continue

                float_name = str(quantize.inputs[0])
                float_tensor = model_ir.tensors.get(float_name)
                dequantized_tensor = model_ir.tensors.get(concat_input_name)
                if float_tensor is None or dequantized_tensor is None:
                    continue
                if str(float_tensor.dtype).upper().startswith("INT"):
                    continue
                if not _shapes_equal(
                    list(float_tensor.shape),
                    list(dequantized_tensor.shape),
                ):
                    continue

                source_dequantize = active_index.producer(float_name)
                if (
                    source_dequantize is None
                    or str(source_dequantize.op_type) != "DEQUANTIZE"
                    or len(source_dequantize.inputs) != 1
                    or len(source_dequantize.outputs) != 1
                    or str(source_dequantize.outputs[0]) != float_name
                ):
                    continue
                if not _quantized_tensors_share_exact_grid(
                    model_ir,
                    str(source_dequantize.inputs[0]),
                    quantized_name,
                ):
                    continue

                new_inputs = [str(name) for name in concat.inputs]
                new_inputs[int(input_position)] = float_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=concat,
                    new_inputs=new_inputs,
                    graph_index=active_index,
                )
                bypassed += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "bypassed_concat_pre_quantize_dequantize": int(bypassed),
    }


def _sanitize_terminal_transpose_before_dequantize(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Normalize and remove terminal Transpose/Dequantize boundaries."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"DEQUANTIZE", "TRANSPOSE"}
        for operator in model_ir.operators:
            required_types.discard(str(operator.op_type))
            if len(required_types) == 0:
                break
        if len(required_types) > 0:
            _prune_unused_tensors(model_ir)
            return {
                "sanitized_terminal_transpose_before_dequantize": 0,
                "removed_terminal_dequantize_transpose": 0,
            }
        active_index = ModelIRGraphIndex(model_ir)
    elif (
        len(active_index.operator_indices("DEQUANTIZE")) == 0
        or len(active_index.operator_indices("TRANSPOSE")) == 0
    ):
        _prune_unused_tensors(model_ir)
        return {
            "sanitized_terminal_transpose_before_dequantize": 0,
            "removed_terminal_dequantize_transpose": 0,
        }

    model_outputs = {str(name) for name in model_ir.outputs}
    sanitized = 0
    removed_terminal_dequantize_transpose = 0

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    while True:
        changed = False
        for dequantize_index in active_index.operator_indices("DEQUANTIZE"):
            dequantize = model_ir.operators[int(dequantize_index)]
            if len(dequantize.inputs) != 1 or len(dequantize.outputs) != 1:
                continue

            dequantize_input = str(dequantize.inputs[0])
            dequantize_output = str(dequantize.outputs[0])
            if dequantize_output not in model_outputs:
                continue
            if len(active_index.consumer_indices(dequantize_output)) > 0:
                continue
            if dequantize_input in model_outputs:
                continue

            transpose = active_index.producer(dequantize_input)
            transpose_index = (
                active_index.operator_index(transpose)
                if transpose is not None
                else None
            )
            if (
                transpose is None
                or transpose_index is None
                or str(transpose.op_type) != "TRANSPOSE"
                or len(transpose.inputs) < 2
                or len(transpose.outputs) != 1
                or str(transpose.outputs[0]) != dequantize_input
            ):
                continue

            quantized_input = str(transpose.inputs[0])
            if quantized_input in model_outputs:
                continue
            if active_index.consumer_indices(dequantize_input) != [
                int(dequantize_index)
            ]:
                continue

            quantized_tensor = model_ir.tensors.get(quantized_input)
            transposed_tensor = model_ir.tensors.get(dequantize_input)
            output_tensor = model_ir.tensors.get(dequantize_output)
            if (
                quantized_tensor is None
                or transposed_tensor is None
                or output_tensor is None
            ):
                continue
            if not _all_per_tensor_quantized([quantized_tensor, transposed_tensor]):
                continue

            quantized_shape = [int(value) for value in quantized_tensor.shape]
            quantized_signature = (
                [int(value) for value in quantized_tensor.shape_signature]
                if quantized_tensor.shape_signature is not None
                else list(quantized_shape)
            )
            if len(quantized_shape) == 0:
                continue
            permutation = _read_transpose_perm(model_ir, transpose)
            if permutation is None:
                continue
            expected_shape = _permute_shape(
                quantized_shape,
                permutation,
            )
            expected_signature = _permute_shape(
                quantized_signature,
                permutation,
            )
            if expected_shape is None or expected_signature is None:
                continue

            before_transpose = _unique_tensor_name(
                f"{dequantize_output}_before_transpose"
            )
            model_ir.tensors[before_transpose] = TensorIR(
                name=before_transpose,
                dtype=str(output_tensor.dtype),
                shape=list(quantized_shape),
                shape_signature=list(quantized_signature),
                data=None,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=dequantize,
                new_inputs=[quantized_input],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=dequantize,
                new_outputs=[before_transpose],
                graph_index=active_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=transpose,
                new_inputs=[before_transpose, str(transpose.inputs[1])],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=transpose,
                new_outputs=[dequantize_output],
                graph_index=active_index,
            )
            output_tensor.shape = [int(value) for value in expected_shape]
            output_tensor.shape_signature = [int(value) for value in expected_signature]

            if int(transpose_index) < int(dequantize_index):
                moved = active_index.remove_operator(int(transpose_index))
                active_index.insert_operator(int(dequantize_index), moved)

            sanitized += 1
            changed = True
            break

        if changed:
            continue

        for transpose_index in active_index.operator_indices("TRANSPOSE"):
            transpose = model_ir.operators[int(transpose_index)]
            if len(transpose.inputs) < 2 or len(transpose.outputs) != 1:
                continue

            pre_output = str(transpose.inputs[0])
            final_output = str(transpose.outputs[0])
            if final_output not in model_outputs:
                continue
            if len(active_index.consumer_indices(final_output)) > 0:
                continue
            if pre_output in model_outputs:
                continue
            if active_index.consumer_indices(pre_output) != [int(transpose_index)]:
                continue

            dequantize = active_index.producer(pre_output)
            if (
                dequantize is None
                or str(dequantize.op_type) != "DEQUANTIZE"
                or len(dequantize.inputs) != 1
                or len(dequantize.outputs) != 1
                or str(dequantize.outputs[0]) != pre_output
            ):
                continue

            _rename_tensor_globally(
                model_ir=model_ir,
                old_name=pre_output,
                new_name=final_output,
                graph_index=active_index,
            )
            current_transpose_index = active_index.operator_index(transpose)
            if current_transpose_index is None:
                continue
            active_index.remove_operator(int(current_transpose_index))
            removed_terminal_dequantize_transpose += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "sanitized_terminal_transpose_before_dequantize": int(sanitized),
        "removed_terminal_dequantize_transpose": int(
            removed_terminal_dequantize_transpose
        ),
    }


def _optimize_terminal_quantize_dequantize(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Remove a terminal Q/DQ round trip only across an exact quant grid."""

    removed_pairs = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers

        for q_idx, q_op in enumerate(model_ir.operators):
            if (
                str(q_op.op_type) != "QUANTIZE"
                or len(q_op.inputs) != 1
                or len(q_op.outputs) != 1
            ):
                continue

            float_input_name = str(q_op.inputs[0])
            quantized_name = str(q_op.outputs[0])
            if float_input_name in model_ir.inputs:
                continue

            quantized_users = consumers.get(quantized_name, [])
            if len(quantized_users) != 1:
                continue
            dq_idx = int(quantized_users[0])
            if dq_idx == q_idx:
                continue
            dq_op = model_ir.operators[dq_idx]
            if (
                str(dq_op.op_type) != "DEQUANTIZE"
                or len(dq_op.inputs) != 1
                or len(dq_op.outputs) != 1
                or str(dq_op.inputs[0]) != quantized_name
            ):
                continue

            float_output_name = str(dq_op.outputs[0])
            if float_output_name not in model_ir.outputs:
                continue
            if len(consumers.get(float_output_name, [])) > 0:
                continue

            float_input_users = consumers.get(float_input_name, [])
            if len(float_input_users) != 1 or int(float_input_users[0]) != q_idx:
                continue
            if float_input_name not in producers:
                continue
            if float_input_name in model_ir.outputs:
                continue
            float_producer = model_ir.operators[int(producers[float_input_name])]
            if (
                str(float_producer.op_type) != "DEQUANTIZE"
                or len(float_producer.inputs) != 1
                or len(float_producer.outputs) != 1
                or str(float_producer.outputs[0]) != float_input_name
            ):
                continue
            if not _quantized_tensors_share_exact_grid(
                model_ir,
                str(float_producer.inputs[0]),
                quantized_name,
            ):
                continue

            _rename_tensor_globally(
                model_ir=model_ir,
                old_name=float_input_name,
                new_name=float_output_name,
                layout_state=layout_state,
                graph_index=graph_index,
            )
            for remove_idx in sorted([q_idx, dq_idx], reverse=True):
                graph_index.remove_operator(remove_idx)
            removed_pairs += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "removed_terminal_quantize_dequantize_pairs": int(removed_pairs),
    }


def run_terminal_quantize_dequantize_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run exact-grid terminal Q/DQ cleanup transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_any_operator(
            candidate_model,
            lambda op: (
                str(op.op_type) == "QUANTIZE"
                and len(op.inputs) == 1
                and len(op.outputs) == 1
            ),
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return _preflight(pass_state.model_ir)

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_terminal_quantize_dequantize(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("removed_terminal_quantize_dequantize_pairs", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="cleanup.terminal_quantize_dequantize",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"removed_terminal_quantize_dequantize_pairs": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
