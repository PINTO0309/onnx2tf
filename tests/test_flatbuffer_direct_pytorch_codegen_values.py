from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_codegen_values import (
    _conv_block_activation_config,
    _conv_block_activation_config_from_fused_name,
    _is_small_inline_constant_tensor,
    _python_literal_for_constant_tensor,
    _scalar_literal_for_constant_tensor,
    _torch_dtype_literal,
    _torch_pad_literal_for_constant_tensor,
)
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError


def _tensor(data: object, *, dtype: str = "FLOAT32") -> TensorIR:
    array = np.asarray(data)
    return TensorIR(
        name="constant",
        dtype=dtype,
        shape=list(array.shape),
        shape_signature=list(array.shape),
        data=array,
    )


def test_inline_constant_policy_bounds_shape_size_and_dtype() -> None:
    assert _is_small_inline_constant_tensor(_tensor(np.arange(32), dtype="INT64"))
    assert not _is_small_inline_constant_tensor(_tensor(np.arange(33), dtype="INT64"))
    assert not _is_small_inline_constant_tensor(_tensor(np.ones((1, 1, 1))))
    assert not _is_small_inline_constant_tensor(_tensor([1], dtype="STRING"))
    assert not _is_small_inline_constant_tensor(
        TensorIR(name="dynamic", dtype="FLOAT32", shape=[1])
    )


def test_python_constant_literal_preserves_nested_and_non_finite_values() -> None:
    tensor = _tensor([[1.5, np.nan], [np.inf, -np.inf]])

    assert _python_literal_for_constant_tensor(tensor) == (
        "[[1.5, float('nan')], [float('inf'), float('-inf')]]"
    )
    assert _python_literal_for_constant_tensor(_tensor(np.asarray(7))) == "7"
    assert _python_literal_for_constant_tensor(_tensor(np.arange(33))) is None


def test_scalar_literal_requires_one_value_and_handles_non_finite_values() -> None:
    assert _scalar_literal_for_constant_tensor(None) is None
    assert _scalar_literal_for_constant_tensor(_tensor([1, 2], dtype="INT64")) is None
    assert _scalar_literal_for_constant_tensor(_tensor([3], dtype="INT64")) == "3"
    assert _scalar_literal_for_constant_tensor(_tensor([np.inf])) == "float('inf')"


def test_torch_pad_literal_reverses_axes_and_applies_axis_permutation() -> None:
    pads = _tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype="INT64")

    assert _torch_pad_literal_for_constant_tensor(pads) == "[3, 7, 2, 6, 1, 5, 0, 4]"
    assert (
        _torch_pad_literal_for_constant_tensor(
            pads,
            axis_permutation=[0, 3, 1, 2],
        )
        == "[2, 6, 1, 5, 3, 7, 0, 4]"
    )
    assert (
        _torch_pad_literal_for_constant_tensor(_tensor([[0, 0], [1, 2]], dtype="INT64"))
        == "[1, 2]"
    )


def test_torch_dtype_literal_maps_supported_values_and_rejects_unknown() -> None:
    assert _torch_dtype_literal("float32") == "torch.float32"
    assert _torch_dtype_literal("INT64") == "torch.int64"
    with pytest.raises(ModelIRPyTorchExportError, match="Unsupported dtype"):
        _torch_dtype_literal("COMPLEX64")


@pytest.mark.parametrize(
    ("op_type", "expected"),
    [
        ("RELU", ("relu", None)),
        ("RELU6", ("relu6", None)),
        ("RELU_N1_TO_1", ("relu_n1_to_1", None)),
        ("RELU_0_TO_1", ("relu_0_to_1", None)),
        ("TANH", ("tanh", None)),
        ("LOGISTIC", ("sigmoid", None)),
        ("ABS", ("none", None)),
    ],
)
def test_conv_block_activation_policy(
    op_type: str, expected: tuple[str, float | None]
) -> None:
    op = OperatorIR(op_type=op_type, inputs=["x"], outputs=["y"])

    assert _conv_block_activation_config(op) == expected
    assert _conv_block_activation_config_from_fused_name(op_type) == expected


def test_leaky_relu_activation_policy_preserves_explicit_and_default_alpha() -> None:
    explicit = OperatorIR(
        op_type="LEAKY_RELU",
        inputs=["x"],
        outputs=["y"],
        options={"alpha": 0.125},
    )

    assert _conv_block_activation_config(explicit) == ("leaky_relu", 0.125)
    assert _conv_block_activation_config_from_fused_name("LEAKY_RELU") == (
        "leaky_relu",
        0.2,
    )
    assert _conv_block_activation_config_from_fused_name("LEAKY_RELU", alpha=0.125) == (
        "leaky_relu",
        0.125,
    )
