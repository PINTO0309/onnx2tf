from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list


def _match_affine_layer_norm_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    is_constant_tensor_name_fn: Callable[[str], bool],
    canonical_codegen_name_fn: Callable[[str], str],
    next_unique_attr_name_fn: Callable[[str], str],
    op_index: int,
    op: OperatorIR,
) -> Optional[Dict[str, Any]]:
    if str(op.op_type) != "ADD" or len(op.inputs) < 2 or len(op.outputs) != 1:
        return None
    output_name = str(op.outputs[0])
    canonical_output_name = canonical_codegen_name_fn(output_name)
    if "fakelayernorm" not in canonical_output_name or not canonical_output_name.endswith(
        "add"
    ):
        return None
    beta_input_name = ""
    mul_output_name = ""
    for input_name in op.inputs[:2]:
        input_tensor_name = str(input_name)
        canonical_input_name = canonical_codegen_name_fn(input_tensor_name)
        if (
            is_constant_tensor_name_fn(input_tensor_name)
            and "fakelayernorm_beta" in canonical_input_name
        ):
            beta_input_name = input_tensor_name
        else:
            mul_output_name = input_tensor_name
    if beta_input_name == "" or mul_output_name == "":
        return None
    mul_op_index = producer_index.get(str(mul_output_name), None)
    if mul_op_index is None:
        return None
    mul_op = model_ir.operators[int(mul_op_index)]
    if (
        str(mul_op.op_type) != "MUL"
        or len(mul_op.inputs) < 2
        or len(mul_op.outputs) != 1
    ):
        return None
    if str(mul_op.outputs[0]) != mul_output_name:
        return None
    gamma_input_name = ""
    input_name = ""
    for mul_input_name in mul_op.inputs[:2]:
        candidate_name = str(mul_input_name)
        canonical_candidate_name = canonical_codegen_name_fn(candidate_name)
        if (
            is_constant_tensor_name_fn(candidate_name)
            and "fakelayernorm_gamma" in canonical_candidate_name
        ):
            gamma_input_name = candidate_name
        else:
            input_name = candidate_name
    if gamma_input_name == "" or input_name == "":
        return None
    gamma_tensor = model_ir.tensors.get(str(gamma_input_name), None)
    beta_tensor = model_ir.tensors.get(str(beta_input_name), None)
    if gamma_tensor is None or beta_tensor is None:
        return None
    attr_stem = re.sub(
        r"(?i)(?:[/_])?FakeLayerNorm(?:[/_])add$", "", output_name
    )
    attr_stem = re.sub(r"^bert[/_]", "", attr_stem, flags=re.IGNORECASE)
    attr_name = next_unique_attr_name_fn(f"{attr_stem}_layer_norm")
    return {
        "attr_name": attr_name,
        "input_name": str(input_name),
        "output_name": output_name,
        "gamma_name": str(gamma_input_name),
        "beta_name": str(beta_input_name),
        "gamma_shape": [int(v) for v in list(gamma_tensor.shape)],
        "gamma_dtype": str(gamma_tensor.dtype).upper(),
        "mul_op_index": int(mul_op_index),
    }


def _match_swish_activation_pattern_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
    consumer_indices: Sequence[int],
) -> Optional[Tuple[str, Set[int]]]:
    consumer_index_list = [int(idx) for idx in consumer_indices]
    if len(consumer_index_list) != 2:
        return None
    logistic_idx: Optional[int] = None
    mul_idx: Optional[int] = None
    logistic_output_name: Optional[str] = None
    for consumer_idx in consumer_index_list:
        consumer_op = model_ir.operators[int(consumer_idx)]
        consumer_type = str(consumer_op.op_type)
        if (
            consumer_type == "LOGISTIC"
            and len(consumer_op.inputs) == 1
            and len(consumer_op.outputs) == 1
            and str(consumer_op.inputs[0]) == str(tensor_name)
        ):
            logistic_idx = int(consumer_idx)
            logistic_output_name = str(consumer_op.outputs[0])
            continue
        if (
            consumer_type == "MUL"
            and len(consumer_op.inputs) == 2
            and len(consumer_op.outputs) == 1
            and str(tensor_name)
            in {str(name) for name in list(consumer_op.inputs)}
        ):
            mul_idx = int(consumer_idx)
    if logistic_idx is None or mul_idx is None or logistic_output_name is None:
        return None
    mul_op = model_ir.operators[int(mul_idx)]
    mul_input_names = [str(name) for name in list(mul_op.inputs)]
    if logistic_output_name not in mul_input_names:
        return None
    if set(mul_input_names) != {str(tensor_name), str(logistic_output_name)}:
        return None
    if consumer_index.get(str(logistic_output_name), []) != [int(mul_idx)]:
        return None
    return (str(mul_op.outputs[0]), {int(logistic_idx), int(mul_idx)})


def _match_if_axis0_tensor_mux_slice_for_codegen(
    *,
    model_ir: ModelIR,
    producer_by_output_name: Dict[str, OperatorIR],
    op: OperatorIR,
) -> Optional[Dict[str, str]]:
    if str(op.op_type) != "SLICE" or len(op.inputs) < 3:
        return None

    def _unwrap_axis0_concat_prefix(tensor_name: str) -> Optional[str]:
        producer = producer_by_output_name.get(str(tensor_name), None)
        if producer is None:
            return str(tensor_name)
        if (
            str(producer.op_type) != "CONCATENATION"
            or int(producer.options.get("axis", 0)) != 0
        ):
            return str(tensor_name)
        concat_inputs = [str(v) for v in list(producer.inputs)]
        if len(concat_inputs) != 2:
            return None
        tail_values = _constant_int_list(
            model_ir.tensors.get(concat_inputs[1], None)
        )
        if tail_values is None:
            return None
        return concat_inputs[0]

    merged_name = str(op.inputs[0])
    begin_name = str(op.inputs[1])
    size_name = str(op.inputs[2])

    merged_producer = producer_by_output_name.get(merged_name, None)
    if merged_producer is None:
        return None
    if (
        str(merged_producer.op_type) != "CONCATENATION"
        or int(merged_producer.options.get("axis", 0)) != 0
    ):
        return None
    merged_inputs = [str(v) for v in list(merged_producer.inputs)]
    if len(merged_inputs) != 2:
        return None

    begin_axis0_name = _unwrap_axis0_concat_prefix(begin_name)
    size_axis0_name = _unwrap_axis0_concat_prefix(size_name)
    if begin_axis0_name is None or size_axis0_name is None:
        return None

    begin_axis0_producer = producer_by_output_name.get(begin_axis0_name, None)
    size_axis0_producer = producer_by_output_name.get(size_axis0_name, None)
    if begin_axis0_producer is None or size_axis0_producer is None:
        return None
    if (
        str(begin_axis0_producer.op_type) != "MUL"
        or str(size_axis0_producer.op_type) != "ADD"
    ):
        return None

    begin_inputs = [str(v) for v in list(begin_axis0_producer.inputs)]
    size_axis0_inputs = [str(v) for v in list(size_axis0_producer.inputs)]
    if len(begin_inputs) != 2 or len(size_axis0_inputs) != 2:
        return None

    not_cond_i32_name = begin_inputs[0]
    then_first_dim_name = begin_inputs[1]
    size_then_name = size_axis0_inputs[0]
    size_else_name = size_axis0_inputs[1]

    size_then_producer = producer_by_output_name.get(size_then_name, None)
    size_else_producer = producer_by_output_name.get(size_else_name, None)
    if size_then_producer is None or size_else_producer is None:
        return None
    if (
        str(size_then_producer.op_type) != "MUL"
        or str(size_else_producer.op_type) != "MUL"
    ):
        return None

    size_then_inputs = [str(v) for v in list(size_then_producer.inputs)]
    size_else_inputs = [str(v) for v in list(size_else_producer.inputs)]
    if len(size_then_inputs) != 2 or len(size_else_inputs) != 2:
        return None

    cond_i32_name = size_then_inputs[0]
    size_then_first_dim_name = size_then_inputs[1]
    size_else_not_cond_name = size_else_inputs[0]
    else_first_dim_name = size_else_inputs[1]
    if (
        then_first_dim_name != size_then_first_dim_name
        or not_cond_i32_name != size_else_not_cond_name
    ):
        return None

    not_cond_i32_producer = producer_by_output_name.get(
        not_cond_i32_name, None
    )
    cond_i32_producer = producer_by_output_name.get(cond_i32_name, None)
    if not_cond_i32_producer is None or cond_i32_producer is None:
        return None
    if (
        str(not_cond_i32_producer.op_type) != "SUB"
        or str(cond_i32_producer.op_type) != "CAST"
    ):
        return None

    not_cond_inputs = [str(v) for v in list(not_cond_i32_producer.inputs)]
    cond_inputs = [str(v) for v in list(cond_i32_producer.inputs)]
    if len(not_cond_inputs) != 2 or len(cond_inputs) != 1:
        return None
    if not_cond_inputs[1] != cond_i32_name:
        return None
    cond_name = cond_inputs[0]

    then_first_dim_values = _constant_int_list(
        model_ir.tensors.get(then_first_dim_name, None)
    )
    else_first_dim_values = _constant_int_list(
        model_ir.tensors.get(else_first_dim_name, None)
    )
    if (
        then_first_dim_values is None
        or else_first_dim_values is None
        or len(then_first_dim_values) != 1
        or len(else_first_dim_values) != 1
    ):
        return None

    return {
        "cond_name": cond_name,
        "then_name": merged_inputs[0],
        "else_name": merged_inputs[1],
    }
