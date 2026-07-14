from __future__ import annotations

from typing import Dict, List, Set, Tuple

from onnx2tf.tflite_builder.ir import ModelIR


def collect_custom_op_artifact_metadata(
    model_ir: ModelIR,
) -> Tuple[List[str], List[Dict[str, str]]]:
    custom_ops_used: Set[str] = set()
    custom_op_nodes: List[Dict[str, str]] = []
    custom_op_nodes_seen: Set[Tuple[str, str, str]] = set()
    for op in model_ir.operators:
        if str(op.op_type) != "CUSTOM":
            continue
        custom_ops_used.add(str(op.options.get("customCode", "CUSTOM")))
        options = op.options if isinstance(op.options, dict) else {}
        custom_code = str(options.get("customCode", "CUSTOM")).strip()
        if custom_code == "":
            custom_code = "CUSTOM"
        onnx_op = str(options.get("onnxOp", "")).strip()
        onnx_node_name = str(options.get("onnxNodeName", "")).strip()
        key = (custom_code, onnx_op, onnx_node_name)
        if key in custom_op_nodes_seen:
            continue
        custom_op_nodes_seen.add(key)
        custom_op_nodes.append(
            {
                "custom_code": custom_code,
                "onnx_op": onnx_op,
                "onnx_node_name": onnx_node_name,
            }
        )
    custom_op_nodes.sort(
        key=lambda value: (
            str(value.get("custom_code", "")),
            str(value.get("onnx_op", "")),
            str(value.get("onnx_node_name", "")),
        )
    )
    return sorted(custom_ops_used), custom_op_nodes
