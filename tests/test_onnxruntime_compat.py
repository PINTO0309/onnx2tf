from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.onnx2tf import (
    _run_onnxsim_inplace_safely,
    _sanitize_onnx_graph_names_inplace,
)
from onnx2tf.utils.onnxruntime_compat import prepare_onnx_graph_for_onnxruntime


def _grid_sample_model(*, opset: int, include_inverse: bool):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])
    nodes = []
    grid_input = "x"
    if include_inverse:
        nodes.append(helper.make_node("Inverse", ["x"], ["x_inv"], name="inverse"))
        grid_input = "x_inv"
    nodes.append(helper.make_node("GridSample", [grid_input, "grid"], ["y"], name="grid"))
    return helper.make_model(
        helper.make_graph(nodes, "grid_sample", [x, grid], [y]),
        opset_imports=[helper.make_operatorsetid("", int(opset))],
    )


def _missing_torchvision_nms_capture_model() -> onnx.ModelProto:
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [4, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [4])
    segment0 = helper.make_tensor_value_info("segment0", TensorProto.FLOAT, [2])
    segment1 = helper.make_tensor_value_info("segment1", TensorProto.FLOAT, [2])
    keep = helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])
    k = numpy_helper.from_array(np.asarray(2, dtype=np.int64), name="k")
    offset = numpy_helper.from_array(np.asarray(2, dtype=np.int64), name="offset")
    prefilter = numpy_helper.from_array(np.arange(4, dtype=np.int64), name="prefilter")
    final_filter = numpy_helper.from_array(np.arange(4, dtype=np.int64), name="final_filter")
    max_output = numpy_helper.from_array(np.asarray([4], dtype=np.int64), name="max_output")
    iou = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="iou")
    gather_column = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="gather_column")
    empty = numpy_helper.from_array(np.asarray([], dtype=np.int64), name="empty")

    top_nodes = [
        helper.make_node("TopK", ["segment0", "k"], ["values0", "indices0"], name="topk0"),
        helper.make_node("TopK", ["segment1", "k"], ["values1", "indices1"], name="topk1"),
        helper.make_node("Add", ["indices1", "offset"], ["indices1_offset"], name="offset1"),
        helper.make_node("Concat", ["indices0", "indices1_offset"], ["global_indices"], name="topk_indices", axis=0),
        helper.make_node("Gather", ["boxes", "global_indices"], ["box_candidates"], name="box_candidates", axis=0),
        helper.make_node("Gather", ["scores", "global_indices"], ["score_candidates"], name="score_candidates", axis=0),
        helper.make_node("Gather", ["box_candidates", "prefilter"], ["boxes_prefiltered"], name="boxes_prefiltered", axis=0),
        helper.make_node("Gather", ["score_candidates", "prefilter"], ["scores_prefiltered"], name="scores_prefiltered", axis=0),
        helper.make_node("Gather", ["boxes_prefiltered", "final_filter"], ["boxes_final"], name="boxes_final", axis=0),
    ]
    else_nodes = [
        helper.make_node("ReduceMax", ["boxes_final"], ["max_coordinate"], keepdims=0),
        helper.make_node("Cast", ["missing_levels"], ["levels_float"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["boxes_final"], ["boxes_nms"], axes=[0]),
        helper.make_node("Unsqueeze", ["missing_scores"], ["scores_nms"], axes=[0, 1]),
        helper.make_node("NonMaxSuppression", ["boxes_nms", "scores_nms", "max_output", "iou"], ["selected"]),
        helper.make_node("Gather", ["selected", "gather_column"], ["selected_column"], axis=1),
        helper.make_node("Squeeze", ["selected_column"], ["keep"], axes=[1]),
    ]
    else_graph = helper.make_graph(
        else_nodes,
        "else_graph",
        [],
        [helper.make_tensor_value_info("keep", TensorProto.INT64, ["K"])],
    )
    then_graph = helper.make_graph(
        [],
        "then_graph",
        [],
        [helper.make_tensor_value_info("empty", TensorProto.INT64, [0])],
        initializer=[empty],
    )
    top_nodes.append(
        helper.make_node(
            "If",
            ["cond"],
            ["keep"],
            name="nms_guard",
            then_branch=then_graph,
            else_branch=else_graph,
        )
    )
    model = helper.make_model(
        helper.make_graph(
            top_nodes,
            "missing_nms_captures",
            [cond, boxes, scores, segment0, segment1],
            [keep],
            initializer=[k, offset, prefilter, final_filter, max_output, iou, gather_column],
        ),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    model.ir_version = 10
    return model


def _missing_arange_delta_capture_model() -> onnx.ModelProto:
    trip_count = numpy_helper.from_array(np.asarray(3, dtype=np.int64), name="trip_count")
    condition = numpy_helper.from_array(np.asarray(True), name="condition")
    start = numpy_helper.from_array(np.asarray(0, dtype=np.int32), name="start")
    body = helper.make_graph(
        [
            helper.make_node("Identity", ["cond"], ["cond_out"], name="keep_running"),
            helper.make_node("Add", ["prev", "tf/arange/delta:0"], ["current"], name="step"),
            helper.make_node("Identity", ["prev"], ["range"], name="scan"),
        ],
        "arange_body",
        [
            helper.make_tensor_value_info("iteration", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("prev", TensorProto.INT32, []),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info("current", TensorProto.INT32, []),
            helper.make_tensor_value_info("range", TensorProto.INT32, []),
        ],
    )
    loop = helper.make_node(
        "Loop",
        ["trip_count", "condition", "start"],
        ["final", "range_values"],
        name="tf/arange_loop",
        body=body,
    )
    model = helper.make_model(
        helper.make_graph(
            [loop],
            "missing_arange_delta",
            [],
            [helper.make_tensor_value_info("range_values", TensorProto.INT32, [3])],
            initializer=[trip_count, condition, start],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


def _missing_torchvision_paste_masks_capture_model() -> onnx.ModelProto:
    raw_masks = helper.make_tensor_value_info(
        "raw_masks",
        TensorProto.FLOAT,
        [2, 1, 2, 2],
    )
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [2, 4])
    boxes_out = helper.make_tensor_value_info(
        "boxes_out",
        TensorProto.FLOAT,
        [2, 4],
    )
    masks_out = helper.make_tensor_value_info(
        "masks_out",
        TensorProto.FLOAT,
        [2, 4, 4],
    )
    pads = numpy_helper.from_array(
        np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64),
        name="pads",
    )
    condition = numpy_helper.from_array(np.asarray(True), name="condition")
    carried = numpy_helper.from_array(
        np.zeros((0, 4, 4), dtype=np.float32),
        name="carried",
    )
    coordinate_indices = [
        numpy_helper.from_array(
            np.asarray(index, dtype=np.int64),
            name=f"coordinate_{index}",
        )
        for index in range(4)
    ]
    body = helper.make_graph(
        [
            helper.make_node(
                "Gather",
                ["padded_masks", "iteration"],
                ["mask_item"],
                axis=0,
            ),
            helper.make_node(
                "Gather",
                ["missing_expanded_boxes", "iteration"],
                ["box_item"],
                axis=0,
            ),
            *[
                helper.make_node(
                    "Gather",
                    ["box_item", f"coordinate_{index}"],
                    [f"box_coordinate_{index}"],
                    axis=0,
                )
                for index in range(4)
            ],
            helper.make_node("Identity", ["cond"], ["cond_out"]),
            helper.make_node(
                "Concat",
                ["previous_masks", "mask_item"],
                ["next_masks"],
                axis=0,
            ),
        ],
        "paste_masks_body",
        [
            helper.make_tensor_value_info("iteration", TensorProto.INT64, []),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "previous_masks",
                TensorProto.FLOAT,
                [None, 4, 4],
            ),
        ],
        [
            helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
            helper.make_tensor_value_info(
                "next_masks",
                TensorProto.FLOAT,
                [None, 4, 4],
            ),
        ],
        initializer=coordinate_indices,
    )
    nodes = [
        helper.make_node("Identity", ["boxes"], ["boxes_out"]),
        helper.make_node("Pad", ["raw_masks", "pads"], ["padded_masks"]),
        helper.make_node("Shape", ["padded_masks"], ["mask_shape"]),
        helper.make_node(
            "Gather",
            ["mask_shape", "coordinate_0"],
            ["trip_count"],
            axis=0,
        ),
        helper.make_node(
            "Loop",
            ["trip_count", "condition", "carried"],
            ["masks_out"],
            name="paste_masks_loop",
            body=body,
        ),
    ]
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "missing_paste_masks_capture",
            [raw_masks, boxes],
            [boxes_out, masks_out],
            initializer=[pads, condition, carried, *coordinate_indices],
        ),
        opset_imports=[helper.make_operatorsetid("", 14)],
    )
    model.ir_version = 10
    return model


def test_prepare_onnxruntime_graph_redomains_legacy_grid_sample_and_inverse() -> None:
    model = _grid_sample_model(opset=10, include_inverse=True)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"GridSample": 1, "Inverse": 1}
    assert [node.domain for node in prepared.graph.node] == [
        "com.microsoft",
        "com.microsoft",
    ]
    assert [node.domain for node in model.graph.node] == ["", ""]
    assert any(opset.domain == "com.microsoft" for opset in prepared.opset_import)


def test_prepare_onnxruntime_graph_keeps_standard_grid_sample_at_opset_16() -> None:
    model = _grid_sample_model(opset=16, include_inverse=False)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {}
    assert prepared.graph.node[0].domain == ""


def test_prepare_onnxruntime_graph_upgrades_legacy_downsample_upsample() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32),
        name="scales",
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Upsample",
                    ["x", "scales"],
                    ["y"],
                    mode="linear",
                )
            ],
            "legacy_downsample",
            [x],
            [y],
            [scales],
        ),
        opset_imports=[helper.make_operatorsetid("", 9)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"LegacyUpsample": 1}
    assert model.graph.node[0].op_type == "Upsample"
    assert any(node.op_type == "Resize" for node in prepared.graph.node)
    assert all(node.op_type != "Upsample" for node in prepared.graph.node)
    assert next(
        opset.version
        for opset in prepared.opset_import
        if opset.domain in {"", "ai.onnx"}
    ) == 11
    ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )


def test_prepare_onnxruntime_graph_decomposes_group_norm_with_swish() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 4])
    scale = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    bias = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
    node = helper.make_node(
        "GroupNorm",
        ["x", "scale", "bias"],
        ["y"],
        name="group_norm",
        activation=1,
        epsilon=1e-5,
        groups=2,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "group_norm",
            [x_info],
            [y_info],
            [
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        ),
        opset_imports=[
            helper.make_operatorsetid("", 14),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"GroupNorm": 1}
    assert all(node.op_type != "GroupNorm" for node in prepared.graph.node)
    assert model.graph.node[0].op_type == "GroupNorm"
    x = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 8.0
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": x})[0]
    grouped = x.reshape(1, 2, 2, 2, 2)
    mean = np.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = np.mean(np.square(grouped - mean), axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + 1e-5)).reshape(x.shape)
    affine = normalized * scale.reshape(1, 1, 1, 4) + bias.reshape(1, 1, 1, 4)
    expected = affine / (1.0 + np.exp(-affine))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_prepare_onnxruntime_graph_repairs_if_sequenceconstruct_tensor_alias() -> None:
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x1_info = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 2])
    x2_info = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [2, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])

    def branch(*, name: str, include_x2: bool):
        one = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32))
        nodes = [
            helper.make_node(
                "Constant",
                [],
                [f"{name}_one"],
                name=f"{name}_constant",
                value=one,
            ),
            helper.make_node(
                "Add",
                ["x1", f"{name}_one"],
                [f"{name}_x1"],
                name=f"{name}_add_x1",
            ),
        ]
        sequence_inputs = [f"{name}_x1"]
        value_infos = [
            helper.make_tensor_value_info(
                f"{name}_x1", TensorProto.FLOAT, [1, 2]
            )
        ]
        if include_x2:
            nodes.append(
                helper.make_node(
                    "Add",
                    ["x2", f"{name}_one"],
                    [f"{name}_x2"],
                    name=f"{name}_add_x2",
                )
            )
            sequence_inputs.append(f"{name}_x2")
            value_infos.append(
                helper.make_tensor_value_info(
                    f"{name}_x2", TensorProto.FLOAT, [2, 2]
                )
            )
        sequence_output = f"{name}_sequence"
        nodes.append(
            helper.make_node(
                "SequenceConstruct",
                sequence_inputs,
                [sequence_output],
                name=f"{name}_sequence_node",
            )
        )
        graph = helper.make_graph(
            nodes,
            name,
            [],
            [helper.make_tensor_value_info(sequence_output, TensorProto.FLOAT, [])],
            value_info=value_infos,
        )
        return graph

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="if_sequence",
        then_branch=branch(name="then", include_x2=False),
        else_branch=branch(name="else", include_x2=True),
    )
    model = helper.make_model(
        helper.make_graph(
            [if_node],
            "if_sequence",
            [cond_info, x1_info, x2_info],
            [y_info],
        ),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"IfSequenceConstruct": 1}
    prepared_if = prepared.graph.node[0]
    branch_terminals = {
        attribute.name: attribute.g.node[-1].op_type
        for attribute in prepared_if.attribute
    }
    assert branch_terminals == {
        "else_branch": "Concat",
        "then_branch": "Identity",
    }
    session = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    x1 = np.asarray([[1.0, 2.0]], dtype=np.float32)
    x2 = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    actual_then = session.run(["y"], {"cond": np.asarray(True), "x1": x1, "x2": x2})[0]
    actual_else = session.run(["y"], {"cond": np.asarray(False), "x1": x1, "x2": x2})[0]
    np.testing.assert_array_equal(actual_then, x1 + 1.0)
    np.testing.assert_array_equal(actual_else, np.concatenate([x1 + 1.0, x2 + 1.0], axis=0))


def test_prepare_onnxruntime_graph_rewrites_integer_matmul_like_direct_lowerer() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 2])
    weights = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    node = helper.make_node("MatMul", ["x", "weights"], ["y"], name="matmul")
    model = helper.make_model(
        helper.make_graph(
            [node],
            "integer_matmul",
            [x_info],
            [y_info],
            [numpy_helper.from_array(weights, name="weights")],
        ),
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"IntegerMatMul": 1}
    assert [node.op_type for node in prepared.graph.node] == [
        "Cast",
        "Cast",
        "MatMul",
        "Cast",
    ]
    x = np.asarray([[2, 3], [4, 5]], dtype=np.uint8)
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": x})[0]
    expected = np.matmul(x.astype(np.float32), weights.astype(np.float32)).astype(
        np.uint8
    )
    np.testing.assert_array_equal(actual, expected)


def test_prepare_onnxruntime_graph_folds_optional_has_element_tensor_alias() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y_info = helper.make_tensor_value_info("y", TensorProto.BOOL, [])
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "OptionalHasElement",
                    ["x"],
                    ["has_element"],
                    name="optional_has_element",
                ),
                helper.make_node(
                    "Identity", ["has_element"], ["y"], name="identity"
                ),
            ],
            "optional_tensor_alias",
            [x_info],
            [y_info],
        ),
        opset_imports=[helper.make_operatorsetid("", 15)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"TensorOptionalHasElement": 1}
    assert prepared.graph.node[0].op_type == "Constant"
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": np.asarray([1.0, 2.0], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.asarray(True, dtype=np.bool_))


def test_prepare_onnxruntime_graph_materializes_unknown_rank_fused_conv_io() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
    weights = np.ones((4, 3, 3, 3), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)
    node = helper.make_node(
        "FusedConv",
        ["x", "weights", "bias"],
        ["y"],
        name="fused_conv",
        activation="Relu",
        pads=[1, 1, 1, 1],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "unknown_rank_fused_conv",
            [x_info],
            [y_info],
            [
                numpy_helper.from_array(weights, name="weights"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        ),
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"FusedConv": 1, "UnknownRankConv": 1}
    input_dims = prepared.graph.input[0].type.tensor_type.shape.dim
    output_dims = prepared.graph.output[0].type.tensor_type.shape.dim
    assert len(input_dims) == 4
    assert input_dims[1].dim_value == 3
    assert len(output_dims) == 4
    assert output_dims[1].dim_value == 4
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": np.ones((1, 3, 1, 1), dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.full((1, 4, 1, 1), 3.0, dtype=np.float32))

def test_prepare_onnx_graph_recursively_toposorts_control_flow_subgraphs() -> None:
    condition = helper.make_tensor_value_info(
        "condition",
        TensorProto.BOOL,
        [],
    )
    source = helper.make_tensor_value_info("source", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    branch_output = helper.make_tensor_value_info(
        "branch_output",
        TensorProto.FLOAT,
        [1],
    )
    one = numpy_helper.from_array(
        np.asarray([1.0], dtype=np.float32),
        name="one",
    )
    then_branch = helper.make_graph(
        [
            helper.make_node("Relu", ["sum"], ["branch_output"]),
            helper.make_node("Add", ["x", "one"], ["sum"]),
        ],
        "then_branch",
        [],
        [branch_output],
        initializer=[one],
    )
    else_branch = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["branch_output"])],
        "else_branch",
        [],
        [branch_output],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["y"],
                then_branch=then_branch,
                else_branch=else_branch,
            ),
            helper.make_node("Add", ["source", "one"], ["x"]),
        ],
        "recursive_toposort",
        [condition, source],
        [y],
        initializer=[one],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert [node.op_type for node in prepared.graph.node] == ["Add", "If"]
    if_node = prepared.graph.node[1]
    prepared_then = next(
        attribute.g
        for attribute in if_node.attribute
        if attribute.name == "then_branch"
    )
    assert [node.op_type for node in prepared_then.node] == ["Add", "Relu"]
    assert rewritten["TopologicalSort"] == 2
    session = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        None,
        {
            "condition": np.asarray(True),
            "source": np.asarray([-3.0], dtype=np.float32),
        },
    )[0]
    np.testing.assert_array_equal(actual, np.asarray([0.0], dtype=np.float32))


def test_recursive_name_sanitization_updates_control_flow_outer_captures() -> None:
    condition = helper.make_tensor_value_info("condition", TensorProto.BOOL, [])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    branch_output = helper.make_tensor_value_info(
        "branch:output",
        TensorProto.FLOAT,
        [1],
    )
    then_branch = helper.make_graph(
        [helper.make_node("Identity", ["/captured:value"], ["branch:output"])],
        "then/branch",
        [],
        [branch_output],
    )
    else_branch = helper.make_graph(
        [helper.make_node("Neg", ["/captured:value"], ["branch:output"])],
        "else/branch",
        [],
        [branch_output],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["/captured:value"]),
                helper.make_node(
                    "If",
                    ["condition"],
                    ["y"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                ),
            ],
            "outer_capture_sanitize",
            [condition, x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10

    _sanitize_onnx_graph_names_inplace(
        onnx_model=model,
        rewrite_leading_slash=True,
    )

    assert model.graph.node[0].output[0] == "wa/captured__value"
    if_node = model.graph.node[1]
    for attribute in if_node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            assert attribute.g.node[0].input[0] == "wa/captured__value"
            assert attribute.g.node[0].output[0] == "branch__output"
    actual = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(
        None,
        {
            "condition": np.asarray(True),
            "x": np.asarray([2.0], dtype=np.float32),
        },
    )[0]
    np.testing.assert_array_equal(actual, np.asarray([2.0], dtype=np.float32))


def test_failed_onnxsim_does_not_partially_overwrite_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "model.onnx"
    source_path.write_bytes(b"original-model")

    def fail_after_partial_output(command, *, stderr):
        assert command[1] == str(source_path)
        assert command[2] != str(source_path)
        Path(command[2]).write_bytes(b"partial-model")
        raise RuntimeError("onnxsim failed")

    monkeypatch.setattr(
        "onnx2tf.onnx2tf.subprocess.check_output",
        fail_after_partial_output,
    )
    with pytest.raises(RuntimeError, match="onnxsim failed"):
        _run_onnxsim_inplace_safely(
            input_onnx_file_path=str(source_path),
            append_param=[],
        )

    assert source_path.read_bytes() == b"original-model"
    assert list(tmp_path.glob("*.onnx2tf_onnxsim_*.onnx")) == []


def test_prepare_onnxruntime_repairs_missing_torchvision_nms_guard_captures() -> None:
    model = _missing_torchvision_nms_capture_model()
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(model)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten["TorchVisionNmsGuardCaptures"] == 1
    onnx.checker.check_model(prepared)
    repair_nodes = {
        str(node.name): node
        for node in prepared.graph.node
        if "repair_nms" in str(node.name)
    }
    assert set(repair_nodes) == {
        "nms_guard_repair_nms_scores",
        "nms_guard_repair_nms_levels_prefilter",
        "nms_guard_repair_nms_levels_final",
    }
    assert list(repair_nodes["nms_guard_repair_nms_scores"].input) == [
        "scores_prefiltered",
        "final_filter",
    ]
    levels = next(
        numpy_helper.to_array(initializer)
        for initializer in prepared.graph.initializer
        if str(initializer.name) == "nms_guard_repair_nms_levels"
    )
    np.testing.assert_array_equal(
        levels,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )


def test_prepare_onnxruntime_repairs_missing_arange_default_delta_capture() -> None:
    model = _missing_arange_delta_capture_model()
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(model)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten["ArangeLoopDeltaCaptures"] == 1
    onnx.checker.check_model(prepared)
    delta = next(
        numpy_helper.to_array(initializer)
        for initializer in prepared.graph.initializer
        if str(initializer.name) == "tf/arange/delta:0"
    )
    assert delta.dtype == np.int32
    assert delta.shape == ()
    assert int(delta) == 1
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {})[0]
    np.testing.assert_array_equal(actual, np.asarray([0, 1, 2], dtype=np.int32))


def test_prepare_onnxruntime_repairs_missing_paste_masks_box_capture() -> None:
    model = _missing_torchvision_paste_masks_capture_model()
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(model)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten["TorchVisionPasteMasksLoopCaptures"] == 1
    onnx.checker.check_model(prepared)
    repair_cast = next(
        node
        for node in prepared.graph.node
        if str(node.name) == "paste_masks_loop_repair_paste_masks_cast"
    )
    assert list(repair_cast.output) == ["missing_expanded_boxes"]
    prepared.graph.output.append(
        helper.make_tensor_value_info(
            "missing_expanded_boxes",
            TensorProto.INT64,
            [2, 4],
        )
    )
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(
        ["missing_expanded_boxes"],
        {
            "raw_masks": np.ones((2, 1, 2, 2), dtype=np.float32),
            "boxes": np.asarray(
                [[2.0, 2.0, 6.0, 6.0], [1.0, 2.0, 5.0, 6.0]],
                dtype=np.float32,
            ),
        },
    )[0]
    np.testing.assert_array_equal(
        actual,
        np.asarray([[0, 0, 8, 8], [-1, 0, 7, 8]], dtype=np.int64),
    )


def test_prepare_onnxruntime_does_not_guess_ambiguous_paste_masks_capture() -> None:
    model = _missing_torchvision_paste_masks_capture_model()
    pads = next(
        initializer
        for initializer in model.graph.initializer
        if str(initializer.name) == "pads"
    )
    pads.CopyFrom(
        numpy_helper.from_array(
            np.asarray([0, 0, 1, 2, 0, 0, 1, 1], dtype=np.int64),
            name="pads",
        )
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert "TorchVisionPasteMasksLoopCaptures" not in rewritten
    assert all(
        str(node.name) != "paste_masks_loop_repair_paste_masks_cast"
        for node in prepared.graph.node
    )
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(prepared)
