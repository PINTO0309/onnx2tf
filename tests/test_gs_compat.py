import numpy as np
import onnx
from onnx import helper, numpy_helper

import onnx2tf.gs as gs


def _make_linear_model() -> onnx.ModelProto:
    x_vi = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])
    y_vi = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), name="w")
    add = helper.make_node("Add", ["x", "w"], ["z"], name="add")
    relu = helper.make_node("Relu", ["z"], ["y"], name="relu")
    graph = helper.make_graph([add, relu], "g", [x_vi], [y_vi], initializer=[w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _make_constant_node_model() -> onnx.ModelProto:
    x_vi = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    y_vi = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    const_attr = numpy_helper.from_array(np.asarray([2.0], dtype=np.float32), name="const_value")
    const = helper.make_node("Constant", [], ["c"], name="const", value=const_attr)
    add = helper.make_node("Add", ["x", "c"], ["y"], name="add")
    graph = helper.make_graph([const, add], "g_const", [x_vi], [y_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _make_if_subgraph_model() -> onnx.ModelProto:
    cond_vi = helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y_vi = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])

    then_out_vi = helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [1])
    then_const_value = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="then_val")
    then_const = helper.make_node("Constant", [], ["then_out"], name="then_const", value=then_const_value)
    then_graph = helper.make_graph([then_const], "then_branch", [], [then_out_vi])

    else_out_vi = helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [1])
    else_const_value = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="else_val")
    else_const = helper.make_node("Constant", [], ["else_out"], name="else_const", value=else_const_value)
    else_graph = helper.make_graph([else_const], "else_branch", [], [else_out_vi])

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="if_node",
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph([if_node], "if_graph", [cond_vi], [y_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _make_loop_subgraph_model() -> onnx.ModelProto:
    trip_vi = helper.make_tensor_value_info("trip_count", onnx.TensorProto.INT64, [])
    cond_vi = helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    v_init_vi = helper.make_tensor_value_info("v_init", onnx.TensorProto.FLOAT, [1])
    v_out_vi = helper.make_tensor_value_info("v_out", onnx.TensorProto.FLOAT, [1])

    iter_vi = helper.make_tensor_value_info("iter_num", onnx.TensorProto.INT64, [])
    cond_in_vi = helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, [])
    v_in_vi = helper.make_tensor_value_info("v_in", onnx.TensorProto.FLOAT, [1])
    cond_out_vi = helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, [])
    body_v_out_vi = helper.make_tensor_value_info("body_v_out", onnx.TensorProto.FLOAT, [1])

    cond_identity = helper.make_node("Identity", ["cond_in"], ["cond_out"], name="loop_cond_identity")
    v_identity = helper.make_node("Identity", ["v_in"], ["body_v_out"], name="loop_v_identity")
    body_graph = helper.make_graph(
        [cond_identity, v_identity],
        "loop_body",
        [iter_vi, cond_in_vi, v_in_vi],
        [cond_out_vi, body_v_out_vi],
    )

    loop_node = helper.make_node(
        "Loop",
        ["trip_count", "cond", "v_init"],
        ["v_out"],
        name="loop_node",
        body=body_graph,
    )
    graph = helper.make_graph([loop_node], "loop_graph", [trip_vi, cond_vi, v_init_vi], [v_out_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _make_scan_subgraph_model() -> onnx.ModelProto:
    state_init_vi = helper.make_tensor_value_info("state_init", onnx.TensorProto.FLOAT, [1])
    scan_in_vi = helper.make_tensor_value_info("scan_input", onnx.TensorProto.FLOAT, [2, 1])
    state_out_vi = helper.make_tensor_value_info("state_out", onnx.TensorProto.FLOAT, [1])
    scan_out_vi = helper.make_tensor_value_info("scan_out", onnx.TensorProto.FLOAT, [2, 1])

    body_state_in_vi = helper.make_tensor_value_info("body_state_in", onnx.TensorProto.FLOAT, [1])
    body_scan_in_vi = helper.make_tensor_value_info("body_scan_in", onnx.TensorProto.FLOAT, [1])
    body_state_out_vi = helper.make_tensor_value_info("body_state_out", onnx.TensorProto.FLOAT, [1])
    body_scan_out_vi = helper.make_tensor_value_info("body_scan_out", onnx.TensorProto.FLOAT, [1])

    add = helper.make_node(
        "Add",
        ["body_state_in", "body_scan_in"],
        ["body_state_out"],
        name="scan_add",
    )
    identity = helper.make_node("Identity", ["body_state_out"], ["body_scan_out"], name="scan_identity")
    body_graph = helper.make_graph(
        [add, identity],
        "scan_body",
        [body_state_in_vi, body_scan_in_vi],
        [body_state_out_vi, body_scan_out_vi],
    )

    scan_node = helper.make_node(
        "Scan",
        ["state_init", "scan_input"],
        ["state_out", "scan_out"],
        name="scan_node",
        body=body_graph,
        num_scan_inputs=1,
    )
    graph = helper.make_graph([scan_node], "scan_graph", [state_init_vi, scan_in_vi], [state_out_vi, scan_out_vi])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model


def _assert_subgraph_roundtrip(model: onnx.ModelProto, op_type: str, attr_names: list[str]) -> None:
    graph = gs.import_onnx(model)
    node = next(node for node in graph.nodes if node.op == op_type)
    for attr_name in attr_names:
        assert isinstance(node.attrs[attr_name], gs.Graph)

    exported = gs.export_onnx(graph, domain=model.domain, ir_version=model.ir_version)
    onnx.checker.check_model(exported)

    reimported = gs.import_onnx(exported)
    re_node = next(node for node in reimported.nodes if node.op == op_type)
    for attr_name in attr_names:
        assert isinstance(re_node.attrs[attr_name], gs.Graph)
        assert len(re_node.attrs[attr_name].nodes) > 0


def test_import_onnx_and_navigation_helpers():
    model = _make_linear_model()
    graph = gs.import_onnx(model)

    assert len(graph.nodes) == 2
    add_node = graph.nodes[0]
    relu_node = graph.nodes[1]

    assert add_node.op == "Add"
    assert relu_node.op == "Relu"
    assert isinstance(add_node.inputs[1], gs.Constant)
    assert add_node.o() is relu_node
    assert relu_node.i() is add_node

    z_tensor = add_node.outputs[0]
    assert z_tensor.i() is add_node
    assert z_tensor.o() is relu_node


def test_constant_output_exposes_constant_as_producer():
    model = _make_constant_node_model()
    graph = gs.import_onnx(model)

    const_node = next(node for node in graph.nodes if node.op == "Constant")
    c_tensor = const_node.outputs[0]
    producer = c_tensor.i()

    assert isinstance(producer, gs.Constant)
    assert np.allclose(producer.values, np.asarray([2.0], dtype=np.float32))


def test_export_roundtrip_model_is_valid():
    model = _make_linear_model()
    graph = gs.import_onnx(model)
    exported = gs.export_onnx(graph, domain=model.domain, ir_version=model.ir_version)

    onnx.checker.check_model(exported)
    reimported = gs.import_onnx(exported)
    assert len(reimported.nodes) == 2
    assert [node.op for node in reimported.nodes] == ["Add", "Relu"]


def test_toposort_cleanup_and_node_ids():
    x = gs.Variable(name="x", dtype=np.dtype("float32"), shape=[1])
    z = gs.Variable(name="z", dtype=np.dtype("float32"), shape=[1])
    y = gs.Variable(name="y", dtype=np.dtype("float32"), shape=[1])

    node1 = gs.Node(op="Add", name="n1", inputs=[x, gs.Constant(name="k", values=np.asarray([1.0], dtype=np.float32))], outputs=[z])
    node2 = gs.Node(op="Relu", name="n2", inputs=[z], outputs=[y])

    dead_in = gs.Variable(name="dead_in", dtype=np.dtype("float32"), shape=[1])
    dead_out = gs.Variable(name="dead_out", dtype=np.dtype("float32"), shape=[1])
    dead = gs.Node(op="Relu", name="dead", inputs=[dead_in], outputs=[dead_out])

    graph = gs.Graph(nodes=[node2, dead, node1], inputs=[x], outputs=[y], opset=13)

    graph.toposort()
    sorted_names = [node.name for node in graph.nodes]
    assert sorted_names.index("n1") < sorted_names.index("n2")

    graph.cleanup()
    assert [node.name for node in graph.nodes] == ["n1", "n2"]

    with graph.node_ids():
        assert [node.id for node in graph.nodes] == [0, 1]


def test_if_subgraph_roundtrip():
    model = _make_if_subgraph_model()
    _assert_subgraph_roundtrip(model, "If", ["then_branch", "else_branch"])


def test_loop_subgraph_roundtrip():
    model = _make_loop_subgraph_model()
    _assert_subgraph_roundtrip(model, "Loop", ["body"])


def test_scan_subgraph_roundtrip():
    model = _make_scan_subgraph_model()
    _assert_subgraph_roundtrip(model, "Scan", ["body"])
