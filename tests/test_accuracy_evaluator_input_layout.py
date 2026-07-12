import numpy as np
from onnx import TensorProto, helper

from onnx2tf.onnx2tf import _merge_runtime_shape_hints
from onnx2tf.tflite_builder.accuracy_evaluator import (
    _adapt_input_layout_for_tflite_input,
    _apply_shape_hints_to_runtime_graph_inputs,
    _collect_onnx_input_specs,
)


def test_runtime_shape_hints_include_overwrite_input_shape() -> None:
    merged = _merge_runtime_shape_hints(
        shape_hints=[
            "x:1,3,224,224",
            "import/tokens:0:1,16",
        ],
        overwrite_input_shape=[
            "x:1,3,480,640",
            "mask:1,16",
        ],
    )

    assert merged == [
        "x:1,3,480,640",
        "import/tokens:0:1,16",
        "mask:1,16",
    ]


def _detail(shape_signature):
    return {
        "shape_signature": np.asarray(shape_signature, dtype=np.int32),
    }


def test_adapt_input_layout_rank3_ncw_to_nwc() -> None:
    x = np.arange(1 * 17 * 2, dtype=np.float32).reshape(1, 17, 2)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 2, 17]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 1)))


def test_adapt_input_layout_rank4_nchw_to_nhwc() -> None:
    x = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 4, 5, 3]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 3, 1)))


def test_adapt_input_layout_rank5_ncdhw_to_ndhwc() -> None:
    x = np.arange(1 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(1, 2, 3, 4, 5)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 3, 4, 5, 2]))
    np.testing.assert_array_equal(adapted, np.transpose(x, (0, 2, 3, 4, 1)))


def test_adapt_input_layout_noop_when_shape_matches() -> None:
    x = np.arange(1 * 2 * 17, dtype=np.float32).reshape(1, 2, 17)
    adapted = _adapt_input_layout_for_tflite_input(x, _detail([1, 2, 17]))
    np.testing.assert_array_equal(adapted, x)


def test_collect_input_specs_applies_hint_to_colon_suffixed_name() -> None:
    x = helper.make_tensor_value_info(
        "import/input_images:0",
        TensorProto.FLOAT,
        ["N", "H", "W", 3],
    )
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", "H", "W", 3])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", [x.name], [y.name])],
            "colon_shape_hint",
            [x],
            [y],
        )
    )

    specs = _collect_onnx_input_specs(
        model,
        shape_hints=["import/input_images:0:1,320,320,3"],
    )

    assert specs[0][0] == "import/input_images:0"
    assert specs[0][2] == (1, 320, 320, 3)


def test_shape_hint_restores_materialized_singleton_on_runtime_copy() -> None:
    x = helper.make_tensor_value_info(
        "import/input_images__0",
        TensorProto.FLOAT,
        [1, 1, 1, 3],
    )
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 3])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", [x.name], [y.name])],
            "materialized_shape_hint",
            [x],
            [y],
        )
    )

    _apply_shape_hints_to_runtime_graph_inputs(
        model,
        ["import/input_images:0:1,320,320,3"],
    )

    assert [
        int(dim.dim_value)
        for dim in model.graph.input[0].type.tensor_type.shape.dim
    ] == [1, 320, 320, 3]
