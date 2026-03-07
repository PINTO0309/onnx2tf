import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnx2tf.tflite_builder.lower_from_onnx2tf as lower_from_onnx2tf
from onnx2tf.tflite_builder.lower_from_onnx2tf import build_op_coverage_report


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _mark_tensor_external_data(
    tensor: onnx.TensorProto,
    *,
    location: str,
) -> onnx.TensorProto:
    tensor.data_location = onnx.TensorProto.EXTERNAL
    del tensor.external_data[:]
    location_entry = tensor.external_data.add()
    location_entry.key = "location"
    location_entry.value = str(location)
    return tensor


def _make_external_data_initializer_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = _mark_tensor_external_data(
        numpy_helper.from_array(np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32), name="w"),
        location="weights.bin",
    )
    node = helper.make_node("Add", ["x", "w"], ["y"], name="ExternalAddNode")
    graph = helper.make_graph([node], "external_add_graph", [x], [y], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_max_model() -> onnx.ModelProto:
    x0 = helper.make_tensor_value_info("x0", TensorProto.FLOAT, [1, 3, 4])
    x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Max", ["x0", "x1"], ["y"], name="MaxNode")
    graph = helper.make_graph([node], "max_graph", [x0, x1], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_log_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Log", ["x"], ["y"], name="LogNode")
    graph = helper.make_graph([node], "log_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scatter_elements_model() -> onnx.ModelProto:
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3, 4])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3, 4])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    node = helper.make_node(
        "ScatterElements",
        ["data", "indices", "updates"],
        ["y"],
        name="ScatterElementsNode",
        axis=1,
    )
    graph = helper.make_graph([node], "scatter_elements_graph", [data, indices, updates], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_roi_align_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 8, 8])
    rois = helper.make_tensor_value_info("rois", TensorProto.FLOAT, [3, 4])
    batch_indices = helper.make_tensor_value_info("batch_indices", TensorProto.INT64, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2, 2, 2])
    node = helper.make_node(
        "RoiAlign",
        ["x", "rois", "batch_indices"],
        ["y"],
        name="RoiAlignNode",
        mode="avg",
        output_height=2,
        output_width=2,
        sampling_ratio=2,
        spatial_scale=1.0,
    )
    graph = helper.make_graph([node], "roi_align_graph", [x, rois, batch_indices], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_axis_non_last_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxNode", axis=1)
    graph = helper.make_graph([node], "softmax_axis_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_dynamic_quantize_linear_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])
    y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [])
    y_zero = helper.make_tensor_value_info("y_zero", TensorProto.UINT8, [])
    node = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "y_scale", "y_zero"],
        name="DynamicQuantizeLinearNode",
    )
    graph = helper.make_graph([node], "dynamic_quantize_linear_graph", [x], [y, y_scale, y_zero])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shape_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2])
    node = helper.make_node(
        "Shape",
        ["x"],
        ["y"],
        name="ShapeNode",
        start=1,
        end=3,
    )
    graph = helper.make_graph([node], "shape_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def _make_constant_of_shape_model() -> onnx.ModelProto:
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    shape = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="cos_shape")
    value = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="cos_value")
    node = helper.make_node(
        "ConstantOfShape",
        ["cos_shape"],
        ["y"],
        name="ConstantOfShapeNode",
        value=value,
    )
    graph = helper.make_graph([node], "constant_of_shape_graph", [], [y], initializer=[shape])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_fused_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
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
        name="fmm_w",
    )
    node = helper.make_node(
        "FusedMatMul",
        ["x", "fmm_w"],
        ["y"],
        name="FusedMatMulNode",
        alpha=0.125,
        transA=0,
        transB=1,
    )
    graph = helper.make_graph([node], "fused_matmul_graph", [x], [y], initializer=[w])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_fused_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="B")
    node = helper.make_node(
        "FusedConv",
        ["x", "W", "B"],
        ["y"],
        name="FusedConvNode",
        domain="com.microsoft",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        activation="Relu",
    )
    graph = helper.make_graph([node], "fused_conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_qlinear_concat_slice_softmax_unknown_rank_model() -> onnx.ModelProto:
    x0_q = helper.make_tensor_value_info("x0_q", TensorProto.UINT8, [])
    x1_q = helper.make_tensor_value_info("x1_q", TensorProto.UINT8, [])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4, 2])
    part = helper.make_tensor_value_info("part", TensorProto.FLOAT, [1, 4, 1])

    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="x_scale")
    x_zero = numpy_helper.from_array(np.asarray([128], dtype=np.uint8), name="x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="y_scale")
    y_zero = numpy_helper.from_array(np.asarray([128], dtype=np.uint8), name="y_zero")
    starts = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="steps")

    nodes = [
        helper.make_node(
            "QLinearConcat",
            [
                "y_scale",
                "y_zero",
                "x0_q",
                "x_scale",
                "x_zero",
                "x1_q",
                "x_scale",
                "x_zero",
            ],
            ["y_q"],
            name="QCatUnknownRank",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "y_scale", "y_zero"],
            ["y"],
            name="DQUnknownRank",
        ),
        helper.make_node(
            "Softmax",
            ["y"],
            ["scores"],
            name="SoftmaxUnknownRank",
            axis=2,
        ),
        helper.make_node(
            "Slice",
            ["y", "starts", "ends", "axes", "steps"],
            ["part"],
            name="SliceUnknownRank",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_slice_softmax_unknown_rank_graph",
        [x0_q, x1_q],
        [scores, part],
        initializer=[x_scale, x_zero, y_scale, y_zero, starts, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_axes_nonconst_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    axes = helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    node = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceNode", keepdims=1)
    graph = helper.make_graph([node], "reduce_nonconst_axes_graph", [x, axes], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_min_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8])
    axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="reduce_min_axes")
    node = helper.make_node(
        "ReduceMin",
        ["x", "reduce_min_axes"],
        ["y"],
        name="ReduceMinNode",
        keepdims=1,
    )
    graph = helper.make_graph(
        [node],
        "reduce_min_graph",
        [x],
        [y],
        initializer=[axes],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_cumprod_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    axis = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="cumprod_axis")
    node = helper.make_node(
        "CumProd",
        ["x", "cumprod_axis"],
        ["y"],
        name="CumProdNode",
        exclusive=0,
        reverse=0,
    )
    graph = helper.make_graph(
        [node],
        "cumprod_graph",
        [x],
        [y],
        initializer=[axis],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_layer_normalization_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 16])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 16])
    scale = numpy_helper.from_array(np.ones((16,), dtype=np.float32), name="ln_scale")
    bias = numpy_helper.from_array(np.zeros((16,), dtype=np.float32), name="ln_bias")
    node = helper.make_node(
        "LayerNormalization",
        ["x", "ln_scale", "ln_bias"],
        ["y"],
        name="LayerNormalizationNode",
        axis=-1,
        epsilon=1e-5,
        stash_type=1,
    )
    graph = helper.make_graph(
        [node],
        "layer_normalization_graph",
        [x],
        [y],
        initializer=[scale, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])


def _make_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 5, 7])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 1, 1])
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"], name="GlobalAveragePoolNode")
    graph = helper.make_graph([node], "global_average_pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_grouped_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 5, 5])
    w = numpy_helper.from_array(
        np.ones((6, 2, 3, 3), dtype=np.float32),
        name="W",
    )
    b = numpy_helper.from_array(np.zeros((6,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="GroupedConvNode",
        group=2,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "grouped_conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_nonconst_rhs_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_nonconst_rhs_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_custom_candidate_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumCustomNode",
        equation="ii,jk->kj",
    )
    graph = helper.make_graph([node], "einsum_custom_candidate_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_transposed_output_builtin_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumTransposedOutputNode",
        equation="ij,jk->kj",
    )
    graph = helper.make_graph([node], "einsum_transposed_output_builtin_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_const_rhs_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    node = helper.make_node(
        "Einsum",
        ["x", "W"],
        ["z"],
        name="EinsumConstNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_const_rhs_graph", [x], [z], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_erf_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Erf", ["x"], ["y"], name="ErfNode")
    graph = helper.make_graph([node], "erf_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_if_nms_guard_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, ["N", 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["N"])
    idxs = helper.make_tensor_value_info("idxs", TensorProto.INT64, ["N"])
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    keep = helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])

    top_then_empty = numpy_helper.from_array(np.asarray([], dtype=np.int64), name="top_then_empty")
    top_then_output = helper.make_tensor_value_info("top_then_empty", TensorProto.INT64, [0])
    top_then_graph = helper.make_graph(
        [],
        "if_top_then",
        [],
        [top_then_output],
        initializer=[top_then_empty],
    )

    nested_then_axis = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="nested_then_axis")
    nested_then_node = helper.make_node(
        "Squeeze",
        ["nms_gathered", "nested_then_axis"],
        ["nested_then_out"],
        name="NestedThenSqueeze",
    )
    nested_then_output = helper.make_tensor_value_info("nested_then_out", TensorProto.INT64, ["K"])
    nested_then_graph = helper.make_graph(
        [nested_then_node],
        "if_nested_then",
        [],
        [nested_then_output],
        initializer=[nested_then_axis],
    )
    nested_else_node = helper.make_node(
        "Identity",
        ["nms_gathered"],
        ["nested_else_out"],
        name="NestedElseIdentity",
    )
    nested_else_output = helper.make_tensor_value_info("nested_else_out", TensorProto.INT64, ["K", 1])
    nested_else_graph = helper.make_graph(
        [nested_else_node],
        "if_nested_else",
        [],
        [nested_else_output],
    )

    eq_lhs = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="eq_lhs")
    eq_rhs = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="eq_rhs")
    unsqueeze_scores_axes = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="unsqueeze_scores_axes")
    add_one = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="add_one")
    unsqueeze_offsets_axes = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="unsqueeze_offsets_axes")
    unsqueeze_boxes_axes = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="unsqueeze_boxes_axes")
    nms_max_output = numpy_helper.from_array(np.asarray([9223372036854775807], dtype=np.int64), name="nms_max_output")
    nms_iou = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="nms_iou")
    nms_gather_index = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="nms_gather_index")

    else_nodes = [
        helper.make_node("ReduceMax", ["boxes"], ["max_coordinate"], name="IfElseReduceMax", keepdims=0),
        helper.make_node("Cast", ["idxs"], ["idxs_f"], name="IfElseCast", to=TensorProto.FLOAT),
        helper.make_node("Equal", ["eq_lhs", "eq_rhs"], ["nested_cond"], name="IfElseEqual"),
        helper.make_node("Unsqueeze", ["scores", "unsqueeze_scores_axes"], ["scores_nms"], name="IfElseUnsqueezeScores"),
        helper.make_node("Add", ["max_coordinate", "add_one"], ["max_plus_one"], name="IfElseAdd"),
        helper.make_node("Mul", ["idxs_f", "max_plus_one"], ["offsets"], name="IfElseMul"),
        helper.make_node("Unsqueeze", ["offsets", "unsqueeze_offsets_axes"], ["offsets_2d"], name="IfElseUnsqueezeOffsets"),
        helper.make_node("Add", ["boxes", "offsets_2d"], ["boxes_for_nms"], name="IfElseAddBoxes"),
        helper.make_node("Unsqueeze", ["boxes_for_nms", "unsqueeze_boxes_axes"], ["boxes_nms"], name="IfElseUnsqueezeBoxes"),
        helper.make_node(
            "NonMaxSuppression",
            ["boxes_nms", "scores_nms", "nms_max_output", "nms_iou"],
            ["nms_selected"],
            name="IfElseNMS",
        ),
        helper.make_node("Gather", ["nms_selected", "nms_gather_index"], ["nms_gathered"], name="IfElseGather", axis=1),
        helper.make_node(
            "If",
            ["nested_cond"],
            ["keep"],
            name="IfElseNestedIf",
            then_branch=nested_then_graph,
            else_branch=nested_else_graph,
        ),
    ]
    else_output = helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])
    top_else_graph = helper.make_graph(
        else_nodes,
        "if_top_else",
        [],
        [else_output],
        initializer=[
            eq_lhs,
            eq_rhs,
            unsqueeze_scores_axes,
            add_one,
            unsqueeze_offsets_axes,
            unsqueeze_boxes_axes,
            nms_max_output,
            nms_iou,
            nms_gather_index,
        ],
    )

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["keep"],
        name="IfNmsGuardNode",
        then_branch=top_then_graph,
        else_branch=top_else_graph,
    )
    graph = helper.make_graph(
        [if_node],
        "if_nms_guard_graph",
        [boxes, scores, idxs, cond],
        [keep],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_tile_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6])
    repeats = numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="repeats")
    node = helper.make_node("Tile", ["x", "repeats"], ["y"], name="TileNode")
    graph = helper.make_graph([node], "tile_graph", [x], [y], initializer=[repeats])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scatter_nd_model() -> onnx.ModelProto:
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    indices = numpy_helper.from_array(
        np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        name="indices",
    )
    node = helper.make_node(
        "ScatterND",
        ["data", "indices", "updates"],
        ["output"],
        name="ScatterNDNode",
    )
    graph = helper.make_graph(
        [node],
        "scatter_nd_graph",
        [data, updates],
        [output],
        initializer=[indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def test_op_coverage_report_keys_compatibility_snapshot() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_add_model(),
        output_file_name="add_cov_snapshot",
    )
    assert sorted(report.keys()) == [
        "conversion_error",
        "custom_lowered_nodes",
        "custom_op_candidate_ops",
        "custom_op_policy",
        "graph_custom_ops",
        "graph_node_reports",
        "graph_ops",
        "graph_summary",
        "graph_supported_ops",
        "graph_unsupported_ops",
        "preprocess_report",
        "registry_extra_outside_schema_range",
        "registry_missing_from_schema_range",
        "schema_onnx_ops_target_range",
        "schema_policy_counts",
        "schema_policy_matrix",
        "schema_unresolved_ops",
        "schema_version",
        "supported_onnx_ops_registry",
        "target_opset_max",
        "target_opset_min",
        "unsupported_nodes",
        "unsupported_reason_counts",
    ]
    assert sorted(report["custom_op_policy"].keys()) == [
        "allow_custom_ops",
        "allowlist_builtin_supported_ops",
        "allowlist_custom_candidate_ops",
        "allowlist_unknown_ops",
        "candidate_count",
        "candidate_count_excluding_builtin_supported",
        "candidate_ops_now_builtin_supported",
        "custom_op_allowlist",
    ]
    assert sorted(report["preprocess_report"].keys()) == [
        "applied_rules",
        "enabled_rule_ids",
        "pipeline_version",
        "registered_rule_ids",
        "schema_version",
        "summary",
    ]
    assert sorted(report["preprocess_report"]["summary"].keys()) == [
        "changed_rule_count",
        "enabled_rule_count",
        "executed_rule_count",
        "registered_rule_count",
        "total_matched_nodes",
        "total_rewritten_nodes",
    ]


def test_op_coverage_report_skips_infer_shapes_for_external_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    infer_shapes_called = False
    original_to_array = lower_from_onnx2tf.numpy_helper.to_array

    def _unexpected_infer_shapes(model: onnx.ModelProto) -> onnx.ModelProto:
        nonlocal infer_shapes_called
        infer_shapes_called = True
        return model

    def _to_array_with_external_data(tensor: onnx.TensorProto) -> np.ndarray:
        if (
            isinstance(tensor, onnx.TensorProto)
            and tensor.name == "w"
            and tensor.data_location == onnx.TensorProto.EXTERNAL
        ):
            return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        return original_to_array(tensor)

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", _unexpected_infer_shapes)
    monkeypatch.setattr(lower_from_onnx2tf.numpy_helper, "to_array", _to_array_with_external_data)

    report = build_op_coverage_report(
        onnx_graph=_make_external_data_initializer_add_model(),
        output_file_name="external_data_cov_snapshot",
    )

    assert infer_shapes_called is False
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_summary"]["unsupported_nodes"] == 0


def test_op_coverage_reason_code_snapshot_validation_failures() -> None:
    softmax_report = build_op_coverage_report(
        onnx_graph=_make_softmax_axis_non_last_model(),
        output_file_name="softmax_reason_snapshot",
    )
    assert softmax_report["unsupported_reason_counts"] == {}
    assert softmax_report["graph_summary"]["unsupported_nodes"] == 0
    assert softmax_report["graph_summary"]["supported_nodes"] == 1
    assert softmax_report["graph_node_reports"][0]["onnx_op"] == "Softmax"
    assert softmax_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"

    reduce_report = build_op_coverage_report(
        onnx_graph=_make_reduce_axes_nonconst_model(),
        output_file_name="reduce_reason_snapshot",
    )
    assert reduce_report["unsupported_reason_counts"] == {"requires_constant_input": 1}
    assert reduce_report["unsupported_nodes"][0]["reason_code"] == "requires_constant_input"
    assert reduce_report["unsupported_nodes"][0]["onnx_op"] == "ReduceSum"


def test_op_coverage_reduce_min_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_reduce_min_model(),
        output_file_name="reduce_min_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert report["graph_node_reports"][0]["onnx_op"] == "ReduceMin"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_cumprod_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_cumprod_model(),
        output_file_name="cumprod_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert report["graph_node_reports"][0]["onnx_op"] == "CumProd"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_if_nms_guard_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_if_nms_guard_model(),
        output_file_name="if_nms_guard_builtin_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["If"],
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert report["graph_node_reports"][0]["onnx_op"] == "If"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_dynamic_quantize_linear_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_dynamic_quantize_linear_model(),
        output_file_name="dynamic_quantize_linear_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "DynamicQuantizeLinear"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_shape_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_shape_model(),
        output_file_name="shape_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "Shape"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_constant_of_shape_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_constant_of_shape_model(),
        output_file_name="constant_of_shape_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "ConstantOfShape"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_fused_matmul_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_fused_matmul_model(),
        output_file_name="fused_matmul_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "FusedMatMul"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_fused_conv_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_fused_conv_model(),
        output_file_name="fused_conv_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "FusedConv"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_axis_rank_inference_avoids_custom_fallback_on_unknown_shapes() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_qlinear_concat_slice_softmax_unknown_rank_model(),
        output_file_name="axis_rank_inference_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert all(
        node.get("dispatch_mode") == "builtin"
        for node in report["graph_node_reports"]
    )


def test_op_coverage_reason_code_snapshot_custom_policy_paths() -> None:
    default_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_disabled_snapshot",
    )
    assert default_report["unsupported_reason_counts"] == {"custom_op_candidate_disabled": 1}
    assert default_report["unsupported_nodes"][0]["reason_code"] == "custom_op_candidate_disabled"

    allowlist_block_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_allowlist_block_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["TopK"],
    )
    assert allowlist_block_report["unsupported_reason_counts"] == {"custom_op_not_in_allowlist": 1}
    assert allowlist_block_report["unsupported_nodes"][0]["reason_code"] == "custom_op_not_in_allowlist"

    allowlist_custom_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_allowed_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert allowlist_custom_report["unsupported_reason_counts"] == {}
    assert allowlist_custom_report["graph_summary"]["custom_lowered_nodes"] == 1
    assert allowlist_custom_report["graph_custom_ops"] == ["Einsum"]

    builtin_report = build_op_coverage_report(
        onnx_graph=_make_einsum_nonconst_rhs_model(),
        output_file_name="einsum_builtin_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert builtin_report["unsupported_reason_counts"] == {}
    assert builtin_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert builtin_report["graph_custom_ops"] == []
    assert any(
        node["onnx_op"] == "Einsum" and node.get("dispatch_mode") == "builtin"
        for node in builtin_report["graph_node_reports"]
    )

    transposed_builtin_report = build_op_coverage_report(
        onnx_graph=_make_einsum_transposed_output_builtin_model(),
        output_file_name="einsum_transposed_output_builtin_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert transposed_builtin_report["unsupported_reason_counts"] == {}
    assert transposed_builtin_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert transposed_builtin_report["graph_custom_ops"] == []
    assert any(
        node["onnx_op"] == "Einsum" and node.get("dispatch_mode") == "builtin"
        for node in transposed_builtin_report["graph_node_reports"]
    )


def test_op_coverage_global_average_pool_and_grouped_conv_builtin_dispatch() -> None:
    global_avg_report = build_op_coverage_report(
        onnx_graph=_make_global_average_pool_model(),
        output_file_name="global_average_pool_builtin_snapshot",
    )
    assert global_avg_report["unsupported_reason_counts"] == {}
    assert global_avg_report["graph_summary"]["unsupported_nodes"] == 0
    assert global_avg_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert global_avg_report["graph_custom_ops"] == []
    assert global_avg_report["graph_node_reports"][0]["onnx_op"] == "GlobalAveragePool"
    assert global_avg_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"

    grouped_conv_default_report = build_op_coverage_report(
        onnx_graph=_make_grouped_conv_model(),
        output_file_name="grouped_conv_default_snapshot",
    )
    assert grouped_conv_default_report["unsupported_reason_counts"] == {}
    assert grouped_conv_default_report["graph_summary"]["unsupported_nodes"] == 0

    grouped_conv_report = build_op_coverage_report(
        onnx_graph=_make_grouped_conv_model(),
        output_file_name="grouped_conv_builtin_snapshot",
        disable_group_convolution=True,
    )
    assert grouped_conv_report["unsupported_reason_counts"] == {}
    assert grouped_conv_report["graph_summary"]["unsupported_nodes"] == 0
    assert grouped_conv_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert grouped_conv_report["graph_custom_ops"] == []
    assert grouped_conv_report["graph_node_reports"][0]["onnx_op"] == "Conv"
    assert grouped_conv_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_layer_normalization_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_layer_normalization_model(),
        output_file_name="layer_normalization_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert any(
        node["onnx_op"] == "LayerNormalization" and node.get("dispatch_mode") == "builtin"
        for node in report["graph_node_reports"]
    )


def test_op_coverage_erf_tile_scatternd_builtin_dispatch() -> None:
    cases = [
        (_make_erf_model(), "Erf"),
        (_make_tile_model(), "Tile"),
        (_make_scatter_nd_model(), "ScatterND"),
    ]
    for model, op_name in cases:
        report = build_op_coverage_report(
            onnx_graph=model,
            output_file_name=f"{op_name.lower()}_builtin_snapshot",
        )
        assert report["unsupported_reason_counts"] == {}
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert report["graph_custom_ops"] == []
        assert any(
            node["onnx_op"] == op_name and node.get("dispatch_mode") == "builtin"
            for node in report["graph_node_reports"]
        )


def test_op_coverage_max_log_builtin_dispatch() -> None:
    cases = [
        (_make_max_model(), "Max"),
        (_make_log_model(), "Log"),
        (_make_scatter_elements_model(), "ScatterElements"),
        (_make_roi_align_model(), "RoiAlign"),
    ]
    for model, op_name in cases:
        report = build_op_coverage_report(
            onnx_graph=model,
            output_file_name=f"{op_name.lower()}_builtin_snapshot",
        )
        assert report["unsupported_reason_counts"] == {}
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert report["graph_custom_ops"] == []
        assert any(
            node["onnx_op"] == op_name and node.get("dispatch_mode") == "builtin"
            for node in report["graph_node_reports"]
        )
