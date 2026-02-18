import onnx
import pytest
import numpy as np
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.preprocess import (
    CONSTANT_FOLD_RULE_ID,
    NORMALIZE_ATTRS_RULE_ID,
    PATTERN_FUSION_WAVE2_RULE_ID,
    QUANT_CHAIN_FUSION_WAVE3_RULE_ID,
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


def _make_relu_clip_relu6_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Relu", ["x"], ["r"], name="ReluNode")
    n1 = helper.make_node("Clip", ["r"], ["y"], name="ClipNode", min=0.0, max=6.0)
    graph = helper.make_graph([n0, n1], "relu_clip_chain_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gelu_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    c_sqrt2 = helper.make_tensor("c_sqrt2", TensorProto.FLOAT, [1], [float(np.sqrt(2.0))])
    c_one = helper.make_tensor("c_one", TensorProto.FLOAT, [1], [1.0])
    c_half = helper.make_tensor("c_half", TensorProto.FLOAT, [1], [0.5])
    n0 = helper.make_node("Div", ["x", "c_sqrt2"], ["d0"], name="DivNode")
    n1 = helper.make_node("Erf", ["d0"], ["e0"], name="ErfNode")
    n2 = helper.make_node("Add", ["e0", "c_one"], ["a0"], name="AddNode")
    n3 = helper.make_node("Mul", ["x", "a0"], ["m0"], name="MulNode0")
    n4 = helper.make_node("Mul", ["m0", "c_half"], ["y"], name="MulNode1")
    graph = helper.make_graph([n0, n1, n2, n3, n4], "gelu_chain_graph", [x], [y], [c_sqrt2, c_one, c_half])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_interleaved_gelu_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    side = helper.make_tensor_value_info("side", TensorProto.FLOAT, [1, 3])
    c_sqrt2 = helper.make_tensor("c_sqrt2", TensorProto.FLOAT, [1], [float(np.sqrt(2.0))])
    c_one = helper.make_tensor("c_one", TensorProto.FLOAT, [1], [1.0])
    c_half = helper.make_tensor("c_half", TensorProto.FLOAT, [1], [0.5])
    nodes = [
        helper.make_node("Div", ["x", "c_sqrt2"], ["d0"], name="DivNode"),
        helper.make_node("Identity", ["s"], ["s0"], name="SideIdentity0"),
        helper.make_node("Erf", ["d0"], ["e0"], name="ErfNode"),
        helper.make_node("Identity", ["s0"], ["s1"], name="SideIdentity1"),
        helper.make_node("Add", ["e0", "c_one"], ["a0"], name="AddNode"),
        helper.make_node("Identity", ["s1"], ["s2"], name="SideIdentity2"),
        helper.make_node("Mul", ["x", "a0"], ["m0"], name="MulNode0"),
        helper.make_node("Identity", ["s2"], ["side"], name="SideIdentity3"),
        helper.make_node("Mul", ["m0", "c_half"], ["y"], name="MulNode1"),
    ]
    graph = helper.make_graph(
        nodes,
        "interleaved_gelu_chain_graph",
        [x, s],
        [y, side],
        [c_sqrt2, c_one, c_half],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_space_to_depth_chain_model(*, with_fanout_conflict: bool = False) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])
    extras = []
    outputs = [y]
    shape1 = helper.make_tensor("shape1", TensorProto.INT64, [6], [1, 2, 2, 2, 2, 2])
    shape2 = helper.make_tensor("shape2", TensorProto.INT64, [4], [1, 8, 2, 2])
    n0 = helper.make_node("Reshape", ["x", "shape1"], ["r1"], name="ReshapeNode0")
    n1 = helper.make_node("Transpose", ["r1"], ["t1"], name="TransposeNode", perm=[0, 1, 3, 5, 2, 4])
    n2 = helper.make_node("Reshape", ["t1", "shape2"], ["y"], name="ReshapeNode1")
    nodes = [n0, n1, n2]
    if with_fanout_conflict:
        z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 2, 2, 2, 2])
        extras.append(z)
        outputs.append(z)
        nodes.append(helper.make_node("Identity", ["t1"], ["z"], name="IdentityConflict"))
    graph = helper.make_graph(nodes, "s2d_chain_graph", [x], outputs, [shape1, shape2], value_info=extras)
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_attr_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])
    n0 = helper.make_node("Transpose", ["x"], ["y"], name="TransposeAttrNode", perm=[0, 2, 1])
    graph = helper.make_graph([n0], "transpose_attr_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_axes_attr_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    n0 = helper.make_node("ReduceSum", ["x"], ["y"], name="ReduceAxesAttrNode", axes=[1], keepdims=1)
    graph = helper.make_graph([n0], "reduce_axes_attr_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_axis_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    n0 = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxAxisNode", axis=1)
    graph = helper.make_graph([n0], "softmax_axis_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_axes_concat_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    c0 = helper.make_tensor("c0", TensorProto.INT64, [1], [1])
    c1 = helper.make_tensor("c1", TensorProto.INT64, [0], [])
    n0 = helper.make_node("Concat", ["c0", "c1"], ["axes"], name="AxesConcatNode", axis=0)
    n1 = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceConstFoldNode", keepdims=1)
    graph = helper.make_graph([n0, n1], "reduce_axes_concat_const_graph", [x], [y], [c0, c1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_dq_bn_prelu_q_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2])

    x_scale = helper.make_tensor("x_scale", TensorProto.FLOAT, [1], [0.1])
    x_zp = helper.make_tensor("x_zp", TensorProto.INT8, [1], [0])
    bn_scale = helper.make_tensor("bn_scale", TensorProto.FLOAT, [2], [1.2, 0.7])
    bn_bias = helper.make_tensor("bn_bias", TensorProto.FLOAT, [2], [0.1, -0.2])
    bn_mean = helper.make_tensor("bn_mean", TensorProto.FLOAT, [2], [0.3, -0.4])
    bn_var = helper.make_tensor("bn_var", TensorProto.FLOAT, [2], [1.0, 1.5])
    prelu_slope = helper.make_tensor("prelu_slope", TensorProto.FLOAT, [2], [0.25, 0.1])
    y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [1], [0.2])
    y_zp = helper.make_tensor("y_zp", TensorProto.INT8, [1], [0])

    n0 = helper.make_node("DequantizeLinear", ["x", "x_scale", "x_zp"], ["x_dq"], name="DQNode")
    n1 = helper.make_node(
        "BatchNormalization",
        ["x_dq", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        ["x_bn"],
        name="BNNode",
        epsilon=1e-5,
    )
    n2 = helper.make_node("PRelu", ["x_bn", "prelu_slope"], ["x_prelu"], name="PReluNode")
    n3 = helper.make_node("QuantizeLinear", ["x_prelu", "y_scale", "y_zp"], ["y"], name="QNode")
    graph = helper.make_graph(
        [n0, n1, n2, n3],
        "dq_bn_prelu_q_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zp,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            prelu_slope,
            y_scale,
            y_zp,
        ],
    )
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


def test_preprocess_wave1_keeps_gelu_op() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_gelu_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["changed_rule_count"] == 0
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["Gelu"]


def test_preprocess_wave1_rewrites_pow_square() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_pow_square_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["changed_rule_count"] >= 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert "Pow" not in ops
    assert ops == ["Mul"]


def test_pattern_fusion_wave2_rewrites_relu_clip_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_relu_clip_relu6_chain_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[PATTERN_FUSION_WAVE2_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["Clip"]


def test_pattern_fusion_wave2_rewrites_gelu_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_gelu_chain_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[PATTERN_FUSION_WAVE2_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["Gelu"]


def test_pattern_fusion_wave2_rewrites_interleaved_gelu_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_interleaved_gelu_chain_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[PATTERN_FUSION_WAVE2_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops.count("Gelu") == 1
    assert ops.count("Identity") == 4


def test_preprocess_default_rules_keep_fused_gelu_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_gelu_chain_model()
    preprocessed, report = run_preprocess_pipeline(onnx_graph=model)
    assert report["summary"]["changed_rule_count"] >= 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["Gelu"]


def test_pattern_fusion_wave2_rewrites_space_to_depth_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_space_to_depth_chain_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[PATTERN_FUSION_WAVE2_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["SpaceToDepth"]
    attrs = {str(a.name): a for a in preprocessed.graph.node[0].attribute}
    assert "blocksize" in attrs
    assert int(attrs["blocksize"].i) == 2


def test_pattern_fusion_wave2_invalid_rewrite_raises_reason_code() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_space_to_depth_chain_model(with_fanout_conflict=True)
    with pytest.raises(ValueError, match="reason_code=space_to_depth_fanout_conflict"):
        run_preprocess_pipeline(
            onnx_graph=model,
            enabled_rule_ids=[PATTERN_FUSION_WAVE2_RULE_ID],
        )


def test_normalize_attrs_adds_transpose_perm_input() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_transpose_attr_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[NORMALIZE_ATTRS_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    node = preprocessed.graph.node[0]
    assert str(node.op_type) == "Transpose"
    assert len(node.input) == 2
    perm_name = str(node.input[1])
    init_names = {str(i.name) for i in preprocessed.graph.initializer}
    assert perm_name in init_names


def test_normalize_attrs_adds_reduce_axes_input() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_reduce_axes_attr_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[NORMALIZE_ATTRS_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    node = preprocessed.graph.node[0]
    assert str(node.op_type) == "ReduceSum"
    assert len(node.input) == 2


def test_normalize_attrs_inserts_softmax_transpose_bridge() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_softmax_axis_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[NORMALIZE_ATTRS_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["Transpose", "Softmax", "Transpose"]


def test_constant_fold_rewrites_concat_axes_to_constant() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_reduce_axes_concat_const_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[CONSTANT_FOLD_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops[0] == "Constant"
    assert "ReduceSum" in ops


def test_quant_chain_fusion_wave3_rewrites_dq_bn_prelu_q_chain() -> None:
    clear_preprocess_rules()
    register_default_preprocess_rules()
    model = _make_dq_bn_prelu_q_chain_model()
    preprocessed, report = run_preprocess_pipeline(
        onnx_graph=model,
        enabled_rule_ids=[QUANT_CHAIN_FUSION_WAVE3_RULE_ID],
    )
    assert report["summary"]["changed_rule_count"] == 1
    ops = [str(node.op_type) for node in preprocessed.graph.node]
    assert ops == ["DequantizeLinear", "Mul", "Add", "PRelu", "QuantizeLinear"]
