from __future__ import annotations

from typing import Dict, List


def build_signature_defs(
    schema_tflite: Dict,
    tensor_index_map: Dict[str, int],
    input_names: List[str],
    output_names: List[str],
) -> List[object]:
    signature = schema_tflite["SignatureDefT"]()
    signature.signatureKey = "serving_default"
    signature.subgraphIndex = 0

    signature_inputs = []
    for name in input_names:
        tm = schema_tflite["TensorMapT"]()
        tm.name = name
        tm.tensorIndex = tensor_index_map[name]
        signature_inputs.append(tm)
    signature.inputs = signature_inputs

    signature_outputs = []
    for name in output_names:
        tm = schema_tflite["TensorMapT"]()
        tm.name = name
        tm.tensorIndex = tensor_index_map[name]
        signature_outputs.append(tm)
    signature.outputs = signature_outputs

    return [signature]
