import onnx
import pytest
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.preprocess import (
    clear_preprocess_rules,
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

