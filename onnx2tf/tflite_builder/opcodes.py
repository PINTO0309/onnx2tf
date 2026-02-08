from __future__ import annotations

from typing import Dict, List, Tuple

from onnx2tf.tflite_builder.ir import OperatorIR


def build_operator_codes(
    schema_tflite: Dict,
    operators: List[OperatorIR],
) -> Tuple[List[object], Dict[Tuple[str, int], int]]:
    operator_codes = []
    op_index_map: Dict[Tuple[str, int], int] = {}
    for op in operators:
        key = (op.op_type, op.version)
        if key in op_index_map:
            continue
        oc = schema_tflite["OperatorCodeT"]()
        builtin_code = getattr(schema_tflite["BuiltinOperator"], op.op_type)
        oc.builtinCode = builtin_code
        oc.deprecatedBuiltinCode = builtin_code
        oc.version = op.version
        op_index_map[key] = len(operator_codes)
        operator_codes.append(oc)
    return operator_codes, op_index_map
