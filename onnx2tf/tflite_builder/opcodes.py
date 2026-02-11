from __future__ import annotations

from typing import Dict, List, Tuple

from onnx2tf.tflite_builder.ir import OperatorIR


def operator_code_key(op: OperatorIR) -> Tuple[str, int, str]:
    custom_code = ""
    if op.op_type == "CUSTOM":
        custom_code = str(op.options.get("customCode", "CUSTOM"))
    return (str(op.op_type), int(op.version), custom_code)


def build_operator_codes(
    schema_tflite: Dict,
    operators: List[OperatorIR],
) -> Tuple[List[object], Dict[Tuple[str, int, str], int]]:
    operator_codes = []
    op_index_map: Dict[Tuple[str, int, str], int] = {}
    for op in operators:
        key = operator_code_key(op)
        if key in op_index_map:
            continue
        oc = schema_tflite["OperatorCodeT"]()
        builtin_code = getattr(schema_tflite["BuiltinOperator"], op.op_type)
        oc.builtinCode = builtin_code
        oc.deprecatedBuiltinCode = builtin_code
        oc.version = int(op.version)
        if op.op_type == "CUSTOM":
            oc.customCode = str(op.options.get("customCode", "CUSTOM"))
        op_index_map[key] = len(operator_codes)
        operator_codes.append(oc)
    return operator_codes, op_index_map
