from onnx import helper

from onnx2tf.tflite_builder.core.node import NodeView


def test_node_view_exposes_typed_input_and_output_metadata() -> None:
    node = helper.make_node(
        "Add",
        inputs=["left", "", "right"],
        outputs=["sum", ""],
        name="add_node",
    )

    view = NodeView(
        node,
        input_name_remap={"left": "left_nhwc"},
        shape_map={"left": [1, 3], "sum": [1, 3]},
        dtype_map={"left": "FLOAT32", "sum": "FLOAT32"},
    )

    assert view.name == "add_node"
    assert view.op == "Add"
    assert [value.name for value in view.inputs] == ["left_nhwc", "", "right"]
    assert [value.onnx_name for value in view.inputs] == ["left", "", "right"]
    assert view.inputs[0].shape == [1, 3]
    assert view.inputs[0].dtype == "FLOAT32"
    assert [value.name for value in view.outputs] == ["sum"]
    assert view.outputs[0].onnx_name == "sum"
    assert view.outputs[0].shape == [1, 3]
    assert view.outputs[0].dtype == "FLOAT32"


def test_node_view_value_metadata_is_instance_local() -> None:
    node = helper.make_node("Add", inputs=["left", "right"], outputs=["sum"])

    view = NodeView(node)
    view.inputs[0].name = "remapped_left"

    assert view.name == "Add"
    assert view.inputs[0].name == "remapped_left"
    assert view.inputs[1].name == "right"
