from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _permute_tensor_metadata_if_rank_matches,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


_NHWC_TO_NCHW = [0, 3, 1, 2]
_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class SwishQDQBranchRewriteResult:
    rewritten_branches: int
    removed_pre_transposes: int
    rewritten_tensors: frozenset[str]


def _is_swish_quantized_output(
    model_ir: ModelIR,
    tensor_name: str,
    *,
    producers: Mapping[str, int],
) -> bool:
    producer_index = producers.get(str(tensor_name))
    if producer_index is None:
        return False
    producer = model_ir.operators[int(producer_index)]
    if (
        str(producer.op_type) != "QUANTIZE"
        or len(producer.inputs) != 1
        or len(producer.outputs) != 1
        or str(producer.outputs[0]) != str(tensor_name)
    ):
        return False

    multiply_output = str(producer.inputs[0])
    multiply_index = producers.get(multiply_output)
    if multiply_index is None:
        return False
    multiply = model_ir.operators[int(multiply_index)]
    return bool(
        str(multiply.op_type) == "MUL"
        and len(multiply.inputs) == 2
        and len(multiply.outputs) == 1
        and str(multiply.outputs[0]) == multiply_output
    )


def _concat_has_quantize_transpose_tail(
    model_ir: ModelIR,
    concat_output: str,
    *,
    consumers: Mapping[str, Sequence[int]],
) -> bool:
    concat_users = [int(value) for value in consumers.get(str(concat_output), [])]
    if len(concat_users) != 1:
        return False
    quantize = model_ir.operators[int(concat_users[0])]
    if (
        str(quantize.op_type) != "QUANTIZE"
        or len(quantize.inputs) != 1
        or len(quantize.outputs) != 1
        or str(quantize.inputs[0]) != str(concat_output)
    ):
        return False

    quantized_output = str(quantize.outputs[0])
    quantized_users = [int(value) for value in consumers.get(quantized_output, [])]
    if len(quantized_users) == 0:
        return False
    return all(
        str(model_ir.operators[int(user_index)].op_type) == "TRANSPOSE"
        and _read_transpose_perm(
            model_ir,
            model_ir.operators[int(user_index)],
        )
        == _NCHW_TO_NHWC
        for user_index in quantized_users
    )


def _has_concat_closure_from_tensor(
    model_ir: ModelIR,
    tensor_name: str,
    *,
    consumers: Mapping[str, Sequence[int]],
) -> bool:
    """Find a conservative path to CONCAT-Q-inverse-Transpose closure."""

    allowed_passthrough_ops = {
        "DEQUANTIZE",
        "QUANTIZE",
        "LOGISTIC",
        "MUL",
        "ADD",
        "SUB",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "TRANSPOSE",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "MAX_POOL_2D",
        "AVERAGE_POOL_2D",
        "RESIZE_NEAREST_NEIGHBOR",
        "RESIZE_BILINEAR",
        "SHAPE",
        "SLICE",
        "STRIDED_SLICE",
        "RELU",
        "RELU6",
        "LEAKY_RELU",
        "HARD_SWISH",
        "CAST",
    }
    queue: List[Tuple[str, int]] = [(str(tensor_name), 0)]
    visited: set[str] = set()
    max_hops = 192

    while len(queue) > 0:
        current_name, depth = queue.pop(0)
        if current_name in visited:
            continue
        visited.add(current_name)
        if int(depth) > max_hops:
            continue

        for user_index in [int(value) for value in consumers.get(current_name, [])]:
            user = model_ir.operators[int(user_index)]
            user_type = str(user.op_type)

            if user_type == "CONCATENATION" and len(user.outputs) == 1:
                axis = int(user.options.get("axis", 1))
                if axis < 0:
                    axis += 4
                if (
                    axis == 1
                    and current_name in {str(value) for value in user.inputs}
                    and _concat_has_quantize_transpose_tail(
                        model_ir,
                        str(user.outputs[0]),
                        consumers=consumers,
                    )
                ):
                    return True

            if len(user.outputs) != 1 or user_type not in allowed_passthrough_ops:
                continue
            next_name = str(user.outputs[0])
            if next_name not in visited:
                queue.append((next_name, int(depth) + 1))

    return False


def rewrite_transpose_swish_qdq_nhwc_branches(
    model_ir: ModelIR,
    *,
    min_spatial_stage: int = 160,
    require_concat_closure: bool = False,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> SwishQDQBranchRewriteResult:
    """Rewrite guarded Transpose-wrapped quantized Swish branches to NHWC."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
            return SwishQDQBranchRewriteResult(0, 0, frozenset())
        active_index = ModelIRGraphIndex(model_ir)
    elif len(active_index.operator_indices("TRANSPOSE")) == 0:
        return SwishQDQBranchRewriteResult(0, 0, frozenset())

    model_outputs = {str(value) for value in model_ir.outputs}
    rewritten_tensors: set[str] = set()
    rewritten_branches = 0
    removed_pre_transposes = 0

    while True:
        changed = False
        # The rewrite changes only the two source edges and removes an unused
        # pre-Transpose. All downstream match edges remain valid in the
        # differentially maintained index.
        consumers = active_index.consumers
        producers = active_index.producers

        for pre_index in active_index.operator_indices("TRANSPOSE"):
            pre = model_ir.operators[int(pre_index)]
            if (
                len(pre.inputs) < 2
                or len(pre.outputs) != 1
                or _read_transpose_perm(model_ir, pre) != _NHWC_TO_NCHW
            ):
                continue

            pre_input = str(pre.inputs[0])
            pre_output = str(pre.outputs[0])
            if pre_output in model_outputs:
                continue
            pre_tensor = model_ir.tensors.get(pre_output)
            if pre_tensor is None:
                continue
            pre_shape = [int(value) for value in pre_tensor.shape]
            if len(pre_shape) != 4:
                continue
            if int(min_spatial_stage) > 0 and min(
                int(pre_shape[2]), int(pre_shape[3])
            ) < int(min_spatial_stage):
                continue

            pre_users = [int(value) for value in consumers.get(pre_output, [])]
            if len(pre_users) < 2:
                continue
            dequantize_candidates = [
                int(index)
                for index in pre_users
                if str(model_ir.operators[int(index)].op_type) == "DEQUANTIZE"
                and len(model_ir.operators[int(index)].inputs) == 1
                and len(model_ir.operators[int(index)].outputs) == 1
                and str(model_ir.operators[int(index)].inputs[0]) == pre_output
            ]
            if len(dequantize_candidates) < 2:
                continue

            used_dequantize: set[int] = set()
            local_rewritten = 0

            for logistic_dequantize_index in dequantize_candidates:
                if int(logistic_dequantize_index) in used_dequantize:
                    continue
                logistic_dequantize = model_ir.operators[int(logistic_dequantize_index)]
                logistic_dequantize_output = str(logistic_dequantize.outputs[0])
                if logistic_dequantize_output in model_outputs:
                    continue

                logistic_users = [
                    int(value)
                    for value in consumers.get(logistic_dequantize_output, [])
                ]
                if len(logistic_users) != 1:
                    continue
                logistic_index = int(logistic_users[0])
                logistic = model_ir.operators[int(logistic_index)]
                if (
                    str(logistic.op_type) != "LOGISTIC"
                    or len(logistic.inputs) != 1
                    or len(logistic.outputs) != 1
                    or str(logistic.inputs[0]) != logistic_dequantize_output
                ):
                    continue
                logistic_output = str(logistic.outputs[0])
                if logistic_output in model_outputs:
                    continue

                quantize_gate_users = [
                    int(value) for value in consumers.get(logistic_output, [])
                ]
                if len(quantize_gate_users) != 1:
                    continue
                quantize_gate_index = int(quantize_gate_users[0])
                quantize_gate = model_ir.operators[int(quantize_gate_index)]
                if (
                    str(quantize_gate.op_type) != "QUANTIZE"
                    or len(quantize_gate.inputs) != 1
                    or len(quantize_gate.outputs) != 1
                    or str(quantize_gate.inputs[0]) != logistic_output
                ):
                    continue
                quantized_gate = str(quantize_gate.outputs[0])
                if quantized_gate in model_outputs:
                    continue

                dequantize_gate_users = [
                    int(value) for value in consumers.get(quantized_gate, [])
                ]
                if len(dequantize_gate_users) != 1:
                    continue
                dequantize_gate_index = int(dequantize_gate_users[0])
                dequantize_gate = model_ir.operators[int(dequantize_gate_index)]
                if (
                    str(dequantize_gate.op_type) != "DEQUANTIZE"
                    or len(dequantize_gate.inputs) != 1
                    or len(dequantize_gate.outputs) != 1
                    or str(dequantize_gate.inputs[0]) != quantized_gate
                ):
                    continue
                dequantized_gate = str(dequantize_gate.outputs[0])
                if dequantized_gate in model_outputs:
                    continue

                data_dequantize_index: Optional[int] = None
                multiply_index: Optional[int] = None
                for candidate_index in dequantize_candidates:
                    if (
                        int(candidate_index) == int(logistic_dequantize_index)
                        or int(candidate_index) in used_dequantize
                    ):
                        continue
                    candidate = model_ir.operators[int(candidate_index)]
                    candidate_output = str(candidate.outputs[0])
                    candidate_users = [
                        int(value) for value in consumers.get(candidate_output, [])
                    ]
                    if len(candidate_users) != 1:
                        continue
                    candidate_multiply_index = int(candidate_users[0])
                    candidate_multiply = model_ir.operators[
                        int(candidate_multiply_index)
                    ]
                    if (
                        str(candidate_multiply.op_type) != "MUL"
                        or len(candidate_multiply.inputs) != 2
                        or len(candidate_multiply.outputs) != 1
                    ):
                        continue
                    candidate_inputs = {
                        str(value) for value in candidate_multiply.inputs
                    }
                    if (
                        candidate_output not in candidate_inputs
                        or dequantized_gate not in candidate_inputs
                    ):
                        continue
                    gate_users = [
                        int(value) for value in consumers.get(dequantized_gate, [])
                    ]
                    if gate_users != [int(candidate_multiply_index)]:
                        continue
                    data_dequantize_index = int(candidate_index)
                    multiply_index = int(candidate_multiply_index)
                    break

                if data_dequantize_index is None or multiply_index is None:
                    continue

                data_dequantize = model_ir.operators[int(data_dequantize_index)]
                dequantized_data = str(data_dequantize.outputs[0])
                multiply = model_ir.operators[int(multiply_index)]
                multiply_output = str(multiply.outputs[0])
                if multiply_output in model_outputs:
                    continue

                multiply_users = [
                    int(value) for value in consumers.get(multiply_output, [])
                ]
                if len(multiply_users) == 0:
                    continue

                has_quantized_multiply = False
                quantized_multiply = ""
                if len(multiply_users) == 1:
                    quantize_multiply_index = int(multiply_users[0])
                    quantize_multiply = model_ir.operators[int(quantize_multiply_index)]
                    if (
                        str(quantize_multiply.op_type) == "QUANTIZE"
                        and len(quantize_multiply.inputs) == 1
                        and len(quantize_multiply.outputs) == 1
                        and str(quantize_multiply.inputs[0]) == multiply_output
                    ):
                        has_quantized_multiply = True
                        quantized_multiply = str(quantize_multiply.outputs[0])
                        quantized_users = [
                            int(value)
                            for value in consumers.get(quantized_multiply, [])
                        ]
                        if len(quantized_users) == 0:
                            continue
                        safe_quantized_users = True
                        for user_index in quantized_users:
                            user = model_ir.operators[int(user_index)]
                            user_type = str(user.op_type)
                            if user_type == "TRANSPOSE":
                                if (
                                    _read_transpose_perm(model_ir, user)
                                    != _NCHW_TO_NHWC
                                    or str(user.outputs[0]) in model_outputs
                                ):
                                    safe_quantized_users = False
                                    break
                                continue
                            if (
                                bool(require_concat_closure)
                                and user_type == "DEQUANTIZE"
                                and len(user.inputs) == 1
                                and len(user.outputs) == 1
                                and str(user.inputs[0]) == quantized_multiply
                                and _has_concat_closure_from_tensor(
                                    model_ir,
                                    str(user.outputs[0]),
                                    consumers=consumers,
                                )
                            ):
                                continue
                            if user_type not in {
                                "ADD",
                                "MUL",
                                "SUB",
                                "DIV",
                                "MAXIMUM",
                                "MINIMUM",
                            }:
                                safe_quantized_users = False
                                break
                            other_inputs = [
                                str(value)
                                for value in user.inputs
                                if str(value) != quantized_multiply
                            ]
                            if len(other_inputs) != 1:
                                safe_quantized_users = False
                                break
                            other_name = str(other_inputs[0])
                            if other_name in rewritten_tensors:
                                continue
                            other_tensor = model_ir.tensors.get(other_name)
                            if (
                                other_tensor is not None
                                and other_tensor.data is not None
                            ):
                                continue
                            if _is_swish_quantized_output(
                                model_ir,
                                other_name,
                                producers=producers,
                            ):
                                continue
                            quantized_tensor = model_ir.tensors.get(quantized_multiply)
                            if (
                                quantized_tensor is not None
                                and other_tensor is not None
                            ):
                                quantized_shape = [
                                    int(value) for value in quantized_tensor.shape
                                ]
                                if (
                                    bool(require_concat_closure)
                                    and [int(value) for value in other_tensor.shape]
                                    == quantized_shape
                                ):
                                    continue
                                permuted_shape = (
                                    [
                                        int(quantized_shape[index])
                                        for index in _NCHW_TO_NHWC
                                    ]
                                    if len(quantized_shape) == 4
                                    else None
                                )
                                if (
                                    permuted_shape is not None
                                    and [int(value) for value in other_tensor.shape]
                                    == permuted_shape
                                ):
                                    continue
                            safe_quantized_users = False
                            break
                        if not safe_quantized_users:
                            continue

                if not has_quantized_multiply:
                    safe_float_users = True
                    for user_index in multiply_users:
                        user = model_ir.operators[int(user_index)]
                        user_type = str(user.op_type)
                        if user_type in {
                            "ADD",
                            "MUL",
                            "SUB",
                            "DIV",
                            "MAXIMUM",
                            "MINIMUM",
                        }:
                            continue
                        if user_type == "CONCATENATION":
                            axis = int(user.options.get("axis", 1))
                            if axis < 0:
                                axis += 4
                            if axis != 1:
                                safe_float_users = False
                                break
                            continue
                        if user_type == "TRANSPOSE":
                            if (
                                _read_transpose_perm(model_ir, user) != _NCHW_TO_NHWC
                                or str(user.outputs[0]) in model_outputs
                            ):
                                safe_float_users = False
                                break
                            continue
                        safe_float_users = False
                        break
                    if not safe_float_users:
                        continue

                if bool(require_concat_closure):
                    closure_input = (
                        quantized_multiply
                        if has_quantized_multiply
                        else multiply_output
                    )
                    if not _has_concat_closure_from_tensor(
                        model_ir,
                        closure_input,
                        consumers=consumers,
                    ):
                        continue

                _set_operator_inputs(
                    model_ir=model_ir,
                    op=logistic_dequantize,
                    new_inputs=[pre_input],
                    graph_index=active_index,
                )
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=data_dequantize,
                    new_inputs=[pre_input],
                    graph_index=active_index,
                )

                for tensor_name in [
                    logistic_dequantize_output,
                    logistic_output,
                    quantized_gate,
                    dequantized_gate,
                    dequantized_data,
                    multiply_output,
                ]:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(tensor_name),
                        _NCHW_TO_NHWC,
                    )
                    rewritten_tensors.add(tensor_name)

                if has_quantized_multiply:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(quantized_multiply),
                        _NCHW_TO_NHWC,
                    )
                    rewritten_tensors.add(quantized_multiply)

                used_dequantize.add(int(logistic_dequantize_index))
                used_dequantize.add(int(data_dequantize_index))
                rewritten_branches += 1
                local_rewritten += 1
                changed = True

            if local_rewritten <= 0:
                continue

            if len(active_index.consumer_indices(pre_output)) == 0:
                active_index.remove_operator(int(pre_index))
                removed_pre_transposes += 1
                changed = True
                break

        if not changed:
            break

    return SwishQDQBranchRewriteResult(
        rewritten_branches=int(rewritten_branches),
        removed_pre_transposes=int(removed_pre_transposes),
        rewritten_tensors=frozenset(rewritten_tensors),
    )
