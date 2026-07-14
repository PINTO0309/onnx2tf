from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


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
