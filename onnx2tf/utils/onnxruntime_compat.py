from __future__ import annotations

from typing import Dict

import onnx


_MICROSOFT_CONTRIB_OPS = {
    "FusedConv",
    "FusedMatMul",
    "Gelu",
    "GroupNorm",
    "Inverse",
    "MultiHeadAttention",
    "QGemm",
    "QLinearAdd",
    "QLinearAveragePool",
    "QLinearConcat",
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearMul",
    "QLinearSigmoid",
    "QLinearSoftmax",
}


def _default_domain_opset(model: onnx.ModelProto) -> int | None:
    versions = [
        int(opset.version)
        for opset in model.opset_import
        if str(opset.domain) in {"", "ai.onnx"}
    ]
    return min(versions) if versions else None


def prepare_onnx_graph_for_onnxruntime(
    onnx_graph: onnx.ModelProto,
) -> tuple[onnx.ModelProto, Dict[str, int]]:
    """Build an evaluation-only graph compatible with current ONNX Runtime."""

    prepared = onnx.ModelProto()
    prepared.CopyFrom(onnx_graph)

    default_opset = _default_domain_opset(prepared)
    if default_opset is not None and int(default_opset) < 7:
        try:
            prepared = onnx.version_converter.convert_version(prepared, 7)
            default_opset = _default_domain_opset(prepared)
        except Exception:
            pass

    rewritten: Dict[str, int] = {}
    for node in prepared.graph.node:
        if str(node.domain) not in {"", "ai.onnx"}:
            continue
        is_contrib = str(node.op_type) in _MICROSOFT_CONTRIB_OPS
        is_legacy_grid_sample = (
            str(node.op_type) == "GridSample"
            and default_opset is not None
            and int(default_opset) < 16
        )
        if not is_contrib and not is_legacy_grid_sample:
            continue
        node.domain = "com.microsoft"
        op_type = str(node.op_type)
        rewritten[op_type] = int(rewritten.get(op_type, 0)) + 1

    if rewritten and not any(
        str(opset.domain) == "com.microsoft" for opset in prepared.opset_import
    ):
        prepared.opset_import.append(
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        )
    return prepared, rewritten
