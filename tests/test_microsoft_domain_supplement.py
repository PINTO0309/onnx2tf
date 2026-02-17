import numpy as np
import onnx
from onnx import helper, numpy_helper

from onnx2tf.onnx2tf import (
    _prepare_onnx_graph_for_runtime_checks,
    _supplement_microsoft_domain_for_selected_ops,
)


def _make_fused_matmul_model(*, domain: str = "") -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
        name="w",
    )
    fused = helper.make_node(
        "FusedMatMul",
        ["x", "w"],
        ["y"],
        name="FusedMatMulNode",
        domain=domain,
    )
    graph = helper.make_graph([fused], "fused_matmul_graph", [x], [y], initializer=[w])
    opsets = [helper.make_operatorsetid("", 12)]
    if domain == "com.microsoft":
        opsets.append(helper.make_operatorsetid("com.microsoft", 1))
    return helper.make_model(graph, opset_imports=opsets)


def test_supplement_microsoft_domain_for_fused_matmul_default_domain() -> None:
    model = _make_fused_matmul_model(domain="")

    rewritten = _supplement_microsoft_domain_for_selected_ops(onnx_model=model)

    assert rewritten == {"FusedMatMul": 1}
    assert model.graph.node[0].domain == "com.microsoft"
    assert any(
        opset.domain == "com.microsoft" and int(opset.version) == 1
        for opset in model.opset_import
    )


def test_supplement_microsoft_domain_skips_already_tagged_node() -> None:
    model = _make_fused_matmul_model(domain="com.microsoft")

    rewritten = _supplement_microsoft_domain_for_selected_ops(onnx_model=model)

    assert rewritten == {}
    assert model.graph.node[0].domain == "com.microsoft"
    assert sum(1 for opset in model.opset_import if opset.domain == "com.microsoft") == 1


def test_prepare_runtime_checks_applies_domain_supplement_on_copy() -> None:
    source = _make_fused_matmul_model(domain="")

    prepared = _prepare_onnx_graph_for_runtime_checks(source_onnx_graph=source)

    assert prepared is not None
    assert source.graph.node[0].domain == ""
    assert prepared.graph.node[0].domain == "com.microsoft"
