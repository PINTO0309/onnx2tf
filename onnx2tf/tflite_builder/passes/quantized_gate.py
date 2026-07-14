from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


@dataclass(frozen=True)
class _LogisticGateBranch:
    dequantize_gate_index: int
    dequantize_gate_op: OperatorIR
    quantize_gate_index: int
    quantize_gate_op: OperatorIR
    logistic_index: int
    logistic_op: OperatorIR
    dequantize_logistic_index: int
    dequantize_logistic_op: OperatorIR


def _match_logistic_gate_branch(
    model_ir: ModelIR,
    *,
    producers: Dict[str, int],
    dequantize_gate_index: int,
    dequantize_gate_op: OperatorIR,
) -> Optional[_LogisticGateBranch]:
    """Match the Q-DQ input branch produced by DQ-LOGISTIC-Q."""

    quantized_gate = str(dequantize_gate_op.inputs[0])
    quantize_gate_index = producers.get(quantized_gate)
    if quantize_gate_index is None:
        return None
    quantize_gate_op = model_ir.operators[int(quantize_gate_index)]
    if (
        str(quantize_gate_op.op_type) != "QUANTIZE"
        or len(quantize_gate_op.inputs) != 1
        or len(quantize_gate_op.outputs) != 1
        or str(quantize_gate_op.outputs[0]) != quantized_gate
    ):
        return None

    logistic_output = str(quantize_gate_op.inputs[0])
    logistic_index = producers.get(logistic_output)
    if logistic_index is None:
        return None
    logistic_op = model_ir.operators[int(logistic_index)]
    if (
        str(logistic_op.op_type) != "LOGISTIC"
        or len(logistic_op.inputs) != 1
        or len(logistic_op.outputs) != 1
        or str(logistic_op.outputs[0]) != logistic_output
    ):
        return None

    dequantized_logistic = str(logistic_op.inputs[0])
    dequantize_logistic_index = producers.get(dequantized_logistic)
    if dequantize_logistic_index is None:
        return None
    dequantize_logistic_op = model_ir.operators[int(dequantize_logistic_index)]
    if (
        str(dequantize_logistic_op.op_type) != "DEQUANTIZE"
        or len(dequantize_logistic_op.inputs) != 1
        or len(dequantize_logistic_op.outputs) != 1
        or str(dequantize_logistic_op.outputs[0]) != dequantized_logistic
    ):
        return None

    return _LogisticGateBranch(
        dequantize_gate_index=int(dequantize_gate_index),
        dequantize_gate_op=dequantize_gate_op,
        quantize_gate_index=int(quantize_gate_index),
        quantize_gate_op=quantize_gate_op,
        logistic_index=int(logistic_index),
        logistic_op=logistic_op,
        dequantize_logistic_index=int(dequantize_logistic_index),
        dequantize_logistic_op=dequantize_logistic_op,
    )


def optimize_transpose_dequant_logistic_mul_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Remove inverse Transposes around a quantized logistic-gated MUL."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
            return {
                "removed_transpose_dequant_logistic_mul_quantize_bridges": 0,
            }
        active_index = ModelIRGraphIndex(model_ir)

    pre_permutation = [0, 3, 1, 2]
    post_permutation = [0, 2, 3, 1]
    model_outputs = {str(name) for name in model_ir.outputs}
    removed_bridges = 0

    while True:
        changed = False
        producers = active_index.producers
        for post_index in active_index.operator_indices("TRANSPOSE"):
            post_op = model_ir.operators[int(post_index)]
            if (
                len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != post_permutation
            ):
                continue

            quantized_multiply_output = str(post_op.inputs[0])
            post_output = str(post_op.outputs[0])
            if (
                quantized_multiply_output in model_outputs
                or post_output in model_outputs
            ):
                continue

            quantize_multiply_index = producers.get(quantized_multiply_output)
            if quantize_multiply_index is None:
                continue
            quantize_multiply_op = model_ir.operators[int(quantize_multiply_index)]
            if (
                str(quantize_multiply_op.op_type) != "QUANTIZE"
                or len(quantize_multiply_op.inputs) != 1
                or len(quantize_multiply_op.outputs) != 1
                or str(quantize_multiply_op.outputs[0]) != quantized_multiply_output
            ):
                continue

            multiply_output = str(quantize_multiply_op.inputs[0])
            if multiply_output in model_outputs:
                continue
            multiply_index = producers.get(multiply_output)
            if multiply_index is None:
                continue
            multiply_op = model_ir.operators[int(multiply_index)]
            if (
                str(multiply_op.op_type) != "MUL"
                or len(multiply_op.inputs) != 2
                or len(multiply_op.outputs) != 1
                or str(multiply_op.outputs[0]) != multiply_output
            ):
                continue

            gate_branch: Optional[_LogisticGateBranch] = None
            data_branch: Optional[tuple[int, OperatorIR]] = None
            branches_are_ambiguous = False

            for multiply_input in [str(name) for name in multiply_op.inputs]:
                candidate_dequantize_index = producers.get(multiply_input)
                if candidate_dequantize_index is None:
                    continue
                candidate_dequantize_op = model_ir.operators[
                    int(candidate_dequantize_index)
                ]
                if (
                    str(candidate_dequantize_op.op_type) != "DEQUANTIZE"
                    or len(candidate_dequantize_op.inputs) != 1
                    or len(candidate_dequantize_op.outputs) != 1
                    or str(candidate_dequantize_op.outputs[0]) != multiply_input
                ):
                    continue

                candidate_gate_branch = _match_logistic_gate_branch(
                    model_ir,
                    producers=producers,
                    dequantize_gate_index=int(candidate_dequantize_index),
                    dequantize_gate_op=candidate_dequantize_op,
                )
                if candidate_gate_branch is not None:
                    if gate_branch is not None:
                        branches_are_ambiguous = True
                        break
                    gate_branch = candidate_gate_branch
                    continue

                if data_branch is not None:
                    branches_are_ambiguous = True
                    break
                data_branch = (
                    int(candidate_dequantize_index),
                    candidate_dequantize_op,
                )

            if branches_are_ambiguous or gate_branch is None or data_branch is None:
                continue
            dequantize_data_index, dequantize_data_op = data_branch
            dequantize_gate_index = gate_branch.dequantize_gate_index
            dequantize_gate_op = gate_branch.dequantize_gate_op
            quantize_gate_index = gate_branch.quantize_gate_index
            quantize_gate_op = gate_branch.quantize_gate_op
            logistic_index = gate_branch.logistic_index
            logistic_op = gate_branch.logistic_op
            dequantize_logistic_index = gate_branch.dequantize_logistic_index
            dequantize_logistic_op = gate_branch.dequantize_logistic_op

            transposed_input = str(dequantize_data_op.inputs[0])
            if str(dequantize_logistic_op.inputs[0]) != transposed_input:
                continue
            pre_index = producers.get(transposed_input)
            if pre_index is None:
                continue
            pre_op = model_ir.operators[int(pre_index)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != transposed_input
                or _read_transpose_perm(model_ir, pre_op) != pre_permutation
            ):
                continue
            source_input = str(pre_op.inputs[0])

            dequantized_data = str(dequantize_data_op.outputs[0])
            dequantized_logistic = str(dequantize_logistic_op.outputs[0])
            logistic_output = str(logistic_op.outputs[0])
            quantized_gate_output = str(quantize_gate_op.outputs[0])
            dequantized_gate = str(dequantize_gate_op.outputs[0])
            observable_intermediates = {
                transposed_input,
                dequantized_data,
                dequantized_logistic,
                logistic_output,
                quantized_gate_output,
                dequantized_gate,
                multiply_output,
                quantized_multiply_output,
            }
            if source_input in model_outputs or any(
                name in model_outputs for name in observable_intermediates
            ):
                continue

            if set(active_index.consumer_indices(transposed_input)) != {
                int(dequantize_data_index),
                int(dequantize_logistic_index),
            }:
                continue
            required_single_consumers = [
                (dequantized_data, multiply_index),
                (dequantized_logistic, logistic_index),
                (logistic_output, quantize_gate_index),
                (quantized_gate_output, dequantize_gate_index),
                (dequantized_gate, multiply_index),
                (multiply_output, quantize_multiply_index),
            ]
            if any(
                set(active_index.consumer_indices(name)) != {int(index)}
                for name, index in required_single_consumers
            ):
                continue

            post_indices: List[int] = []
            post_outputs: List[str] = []
            quantized_multiply_users = active_index.consumer_indices(
                quantized_multiply_output
            )
            if len(quantized_multiply_users) == 0:
                continue
            valid_posts = True
            for user_index in quantized_multiply_users:
                user_op = model_ir.operators[int(user_index)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != quantized_multiply_output
                    or _read_transpose_perm(model_ir, user_op) != post_permutation
                ):
                    valid_posts = False
                    break
                candidate_output = str(user_op.outputs[0])
                if candidate_output in model_outputs:
                    valid_posts = False
                    break
                post_indices.append(int(user_index))
                post_outputs.append(candidate_output)
            if not valid_posts or len(post_indices) == 0:
                continue

            source_tensor = model_ir.tensors.get(source_input)
            transposed_tensor = model_ir.tensors.get(transposed_input)
            quantized_gate_tensor = model_ir.tensors.get(quantized_gate_output)
            quantized_multiply_tensor = model_ir.tensors.get(quantized_multiply_output)
            if not _all_per_tensor_quantized(
                [
                    source_tensor,
                    transposed_tensor,
                    quantized_gate_tensor,
                    quantized_multiply_tensor,
                ]
            ):
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=dequantize_data_op,
                new_inputs=[source_input],
                graph_index=active_index,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=dequantize_logistic_op,
                new_inputs=[source_input],
                graph_index=active_index,
            )

            for tensor_name in [
                dequantized_data,
                dequantized_logistic,
                logistic_output,
                quantized_gate_output,
                dequantized_gate,
                multiply_output,
                quantized_multiply_output,
            ]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(tensor_name),
                    post_permutation,
                )

            canonical_output = str(post_outputs[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=quantize_multiply_op,
                new_outputs=[canonical_output],
                graph_index=active_index,
            )
            for alias_output in post_outputs[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    str(alias_output),
                    canonical_output,
                    graph_index=active_index,
                )

            canonical_tensor = model_ir.tensors.get(canonical_output)
            if canonical_tensor is not None and quantized_multiply_tensor is not None:
                canonical_tensor.dtype = str(quantized_multiply_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(
                    quantized_multiply_tensor.quantization
                )
                canonical_tensor.shape = [
                    int(value) for value in quantized_multiply_tensor.shape
                ]
                canonical_tensor.shape_signature = (
                    [int(value) for value in quantized_multiply_tensor.shape_signature]
                    if quantized_multiply_tensor.shape_signature is not None
                    else [int(value) for value in quantized_multiply_tensor.shape]
                )

            active_index.remove_operators(
                [int(pre_index), *[int(index) for index in post_indices]]
            )
            removed_bridges += 1
            changed = True
            break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_logistic_mul_quantize_bridges": int(removed_bridges),
    }
