import onnx
import pytest
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.preprocess import (
    clear_preprocess_rules,
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def test_preprocess_pipeline_no_rules_is_noop() -> None:
    clear_preprocess_rules()
    model = _make_add_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert len(preprocessed.graph.node) == len(model.graph.node)
    assert report["schema_version"] == 1
    assert report["summary"]["registered_rule_count"] == 0
    assert report["summary"]["executed_rule_count"] == 0
    assert report["applied_rules"] == []


def test_preprocess_pipeline_unknown_rule_id_fails() -> None:
    clear_preprocess_rules()
    model = _make_add_model()
    with pytest.raises(ValueError):
        run_preprocess_pipeline(
            onnx_graph=model,
            enabled_rule_ids=["missing_rule"],
        )


def _make_hardswish_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("HardSwish", ["x"], ["y"], name="HardSwishNode")
    graph = helper.make_graph([node], "hardswish_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])


def _make_gelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Gelu", ["x"], ["y"], name="GeluNode")
    graph = helper.make_graph([node], "gelu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)])


def _make_pow_square_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    exponent = helper.make_tensor(
        "pow_exp",
        TensorProto.FLOAT,
        [1],
        [2.0],
    )
    node = helper.make_node("Pow", ["x", "pow_exp"], ["y"], name="PowNode")
    graph = helper.make_graph([node], "pow_graph", [x], [y], [exponent])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def test_preprocess_wave1_rewrites_hardswish() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_hardswish_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["executed_rule_count"] >= 1
    assert report["summary"]["changed_rule_count"] >= 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert "HardSwish" not in ops
    assert ops == ["Add", "Clip", "Div", "Mul"]


def test_preprocess_wave1_rewrites_gelu_to_tanh_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_gelu_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["changed_rule_count"] >= 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert "Gelu" not in ops
    assert ops == ["Mul", "Mul", "Mul", "Add", "Mul", "Tanh", "Add", "Mul", "Mul"]


def test_preprocess_wave1_rewrites_pow_square() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_pow_square_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["changed_rule_count"] >= 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert "Pow" not in ops
    assert ops == ["Mul"]
