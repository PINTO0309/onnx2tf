from __future__ import annotations

from typing import Callable, Dict, List, Optional

from onnx2tf.tflite_builder.ir import ModelIR


def _is_identity_nms_postprocess_gather_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_expr_aliases: Dict[str, str],
    producer_index: Dict[str, int],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    params_name: str,
    indices_name: str,
) -> bool:
    params_alias = tensor_expr_aliases.get(str(params_name), "")
    if not str(params_alias).startswith("_nms_selected_indices_valid_"):
        return False
    indices_producer_index = producer_index.get(str(indices_name), None)
    if indices_producer_index is None:
        return False
    indices_producer = model_ir.operators[int(indices_producer_index)]
    if (
        str(indices_producer.op_type) != "RANGE"
        or len(indices_producer.inputs) < 3
    ):
        return False
    start_literal = scalar_literal_expr_fn(str(indices_producer.inputs[0]))
    delta_literal = scalar_literal_expr_fn(str(indices_producer.inputs[2]))
    return start_literal == "0" and delta_literal == "1"


def _range_only_feeds_identity_nms_postprocess_gathers_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    is_identity_nms_postprocess_gather_fn: Callable[[str, str], bool],
    output_name: str,
) -> bool:
    consumers = consumer_index.get(str(output_name), [])
    if len(consumers) == 0:
        return False
    for consumer_idx in consumers:
        consumer_op = model_ir.operators[int(consumer_idx)]
        if str(consumer_op.op_type) != "GATHER" or len(consumer_op.inputs) < 2:
            return False
        if str(consumer_op.inputs[1]) != str(output_name):
            return False
        if not is_identity_nms_postprocess_gather_fn(
            str(consumer_op.inputs[0]),
            str(output_name),
        ):
            return False
    return True
