from __future__ import annotations

import builtins

import numpy as np
import pytest

from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder.pytorch_export_support import (
    _build_torchscript_image_input_from_nhwc,
    _parse_torchscript_shape_hints,
    _resolve_torchscript_trace_shape,
)


def test_parse_torchscript_shape_hints_ignores_malformed_values() -> None:
    assert _parse_torchscript_shape_hints(
        ["image:1,224,224,3", "missing_shape", "bad:1,x", ":1,2"]
    ) == {"image": [1, 224, 224, 3]}


def test_resolve_trace_shape_rejects_unresolved_dynamic_dimension() -> None:
    with pytest.raises(ModelIRPyTorchExportError, match="positive values"):
        _resolve_torchscript_trace_shape(
            input_name="image",
            shape_values=[-1, 3, 8, 8],
            shape_hint=[-1, 3, 8, 8],
        )


def test_nhwc_image_input_resize_does_not_import_tensorflow(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocking_import(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise AssertionError("TensorFlow import is forbidden")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)
    data = np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)

    result = _build_torchscript_image_input_from_nhwc(
        data=data,
        expected_shape=(1, 3, 4, 4),
        np_dtype=np.dtype(np.float32),
    )

    assert result.shape == (1, 3, 4, 4)
    assert result.dtype == np.float32
