from __future__ import annotations

import copy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _sanitize_squeeze_axes_with_static_input_shapes,
)
from onnx2tf.tflite_builder.passes.squeeze_shape_sanitization import (
    sanitize_squeeze_axes_with_static_input_shapes,
)


def _model(
    *,
    shape: list[int],
    signature: list[int] | None,
    axes: list[int],
    data: np.ndarray | None = None,
) -> ModelIR:
    model_ir = ModelIR("squeeze_shape_sanitization")
    model_ir.inputs = [] if data is not None else ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(
            name="x",
            dtype="FLOAT32",
            shape=list(shape),
            shape_signature=(
                None if signature is None else list(signature)
            ),
            data=data,
            is_variable=False,
        ),
        "y": TensorIR(
            name="y",
            dtype="FLOAT32",
            shape=[99],
            shape_signature=None,
        ),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["x"],
            outputs=["y"],
            options={"squeezeDims": list(axes)},
        )
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple:
    operator = model_ir.operators[0]
    return (
        tuple(model_ir.tensors["x"].shape),
        (
            None
            if model_ir.tensors["x"].shape_signature is None
            else tuple(model_ir.tensors["x"].shape_signature)
        ),
        tuple(model_ir.tensors["y"].shape),
        (
            None
            if model_ir.tensors["y"].shape_signature is None
            else tuple(model_ir.tensors["y"].shape_signature)
        ),
        tuple(operator.options.get("squeezeDims", [])),
    )


def test_nonconstant_input_repairs_axis_and_output_metadata() -> None:
    model_ir = _model(
        shape=[2, 1, 8, 3],
        signature=[-1, 1, 8, 3],
        axes=[0, 0, -4, 9],
    )

    stats = sanitize_squeeze_axes_with_static_input_shapes(model_ir)

    assert stats == {
        "sanitized_squeeze_axes_with_static_input_shapes": 1,
        "repaired_squeeze_input_singleton_dims": 1,
        "updated_squeeze_output_shapes": 1,
    }
    assert _snapshot(model_ir) == (
        (1, 1, 8, 3),
        (1, 1, 8, 3),
        (1, 8, 3),
        (1, 8, 3),
        (0,),
    )


def test_constant_payload_rejects_non_singleton_axis() -> None:
    model_ir = _model(
        shape=[2, 3],
        signature=[2, 3],
        axes=[0],
        data=np.arange(6, dtype=np.float32).reshape(2, 3),
    )

    stats = sanitize_squeeze_axes_with_static_input_shapes(model_ir)

    assert stats == {
        "sanitized_squeeze_axes_with_static_input_shapes": 1,
        "repaired_squeeze_input_singleton_dims": 0,
        "updated_squeeze_output_shapes": 1,
    }
    assert _snapshot(model_ir) == (
        (2, 3),
        (2, 3),
        (2, 3),
        (2, 3),
        (),
    )


def test_compatibility_wrapper_matches_owner_and_result_is_idempotent() -> None:
    direct_model = _model(
        shape=[2, 1, 8, 3],
        signature=[-1, 1, 8, 3],
        axes=[0],
    )
    wrapper_model = copy.deepcopy(direct_model)

    direct_stats = sanitize_squeeze_axes_with_static_input_shapes(direct_model)
    wrapper_stats = _sanitize_squeeze_axes_with_static_input_shapes(wrapper_model)

    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model) == _snapshot(direct_model)
    assert sanitize_squeeze_axes_with_static_input_shapes(direct_model) == {
        "sanitized_squeeze_axes_with_static_input_shapes": 0,
        "repaired_squeeze_input_singleton_dims": 0,
        "updated_squeeze_output_shapes": 0,
    }
