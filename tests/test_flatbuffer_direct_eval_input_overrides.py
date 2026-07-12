from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.eval_input_overrides import (
    build_attention_control_input_overrides,
)


def _attention_control_model() -> onnx.ModelProto:
    spatial_shapes = helper.make_tensor_value_info(
        "spatial_shapes",
        TensorProto.FLOAT,
        [4, 2],
    )
    valid_ratios = helper.make_tensor_value_info(
        "valid_ratios",
        TensorProto.FLOAT,
        [2, 4, 2],
    )
    mask = helper.make_tensor_value_info(
        "mask_control",
        TensorProto.FLOAT,
        [2, 11097],
    )
    values = helper.make_tensor_value_info(
        "values",
        TensorProto.FLOAT,
        [2, 11097, 256],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [2, 8352, 256],
    )
    split_sizes = numpy_helper.from_array(
        np.asarray([8352, 2088, 522, 135], dtype=np.int64),
        name="split_sizes",
    )
    nodes = [
        helper.make_node(
            "Split",
            ["values", "split_sizes"],
            ["output", "level1", "level2", "level3"],
            axis=1,
        ),
        helper.make_node(
            "Cast",
            ["mask_control"],
            ["mask_bool"],
            to=TensorProto.BOOL,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "attention_controls",
        [spatial_shapes, valid_ratios, mask, values],
        [output],
        [split_sizes],
    )
    return helper.make_model(graph)


def test_builds_static_multiscale_attention_control_values() -> None:
    model = _attention_control_model()
    input_specs = [
        ("spatial_shapes", np.dtype(np.float32), (4, 2)),
        ("valid_ratios", np.dtype(np.float32), (2, 4, 2)),
        ("mask_control", np.dtype(np.float32), (2, 11097)),
        ("values", np.dtype(np.float32), (2, 11097, 256)),
    ]

    overrides = build_attention_control_input_overrides(
        onnx_graph=model,
        input_specs=input_specs,
    )

    np.testing.assert_array_equal(
        overrides["spatial_shapes"],
        np.asarray(
            [[72, 116], [36, 58], [18, 29], [9, 15]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        overrides["valid_ratios"],
        np.ones((2, 4, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        overrides["mask_control"],
        np.zeros((2, 11097), dtype=np.float32),
    )
    assert "values" not in overrides


def test_deduplicates_equivalent_split_pyramid_candidates() -> None:
    model = _attention_control_model()
    duplicate = onnx.NodeProto()
    duplicate.CopyFrom(model.graph.node[0])
    duplicate.name = "duplicate_split"
    duplicate.output[:] = ["duplicate0", "duplicate1", "duplicate2", "duplicate3"]
    model.graph.node.append(duplicate)

    overrides = build_attention_control_input_overrides(
        onnx_graph=model,
        input_specs=[
            ("spatial_shapes", np.dtype(np.float32), (4, 2)),
            ("values", np.dtype(np.float32), (2, 11097, 256)),
        ],
    )

    np.testing.assert_array_equal(
        overrides["spatial_shapes"],
        np.asarray(
            [[72, 116], [36, 58], [18, 29], [9, 15]],
            dtype=np.float32,
        ),
    )


def test_derives_spatial_shapes_from_unique_sequence_pyramid_total() -> None:
    model = _attention_control_model()
    del model.graph.node[0]

    overrides = build_attention_control_input_overrides(
        onnx_graph=model,
        input_specs=[
            ("spatial_shapes", np.dtype(np.float32), (4, 2)),
            ("values", np.dtype(np.float32), (2, 11097, 256)),
        ],
    )

    np.testing.assert_array_equal(
        overrides["spatial_shapes"],
        np.asarray(
            [[72, 116], [36, 58], [18, 29], [9, 15]],
            dtype=np.float32,
        ),
    )


def test_does_not_guess_non_pyramidal_spatial_shapes() -> None:
    model = _attention_control_model()
    initializer = next(
        item for item in model.graph.initializer if item.name == "split_sizes"
    )
    initializer.CopyFrom(
        numpy_helper.from_array(
            np.asarray([8352, 2000, 500, 125], dtype=np.int64),
            name="split_sizes",
        )
    )

    overrides = build_attention_control_input_overrides(
        onnx_graph=model,
        input_specs=[
            ("spatial_shapes", np.dtype(np.float32), (4, 2)),
            ("values", np.dtype(np.float32), (2, 10977, 256)),
        ],
    )

    assert "spatial_shapes" not in overrides
