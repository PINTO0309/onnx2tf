from types import SimpleNamespace

import pytest

from onnx2tf.tflite_builder.op_registry import NodeValidationError, _validate_grid_sample


def _io(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _node(attrs: dict) -> SimpleNamespace:
    return SimpleNamespace(
        name="GridSample_test",
        op="GridSample",
        attrs=attrs,
        inputs=[_io("x"), _io("grid")],
        outputs=[_io("y")],
    )


class _Ctx:
    def __init__(self, *, shapes: dict[str, list[int]], dtypes: dict[str, str]) -> None:
        self._shapes = shapes
        self._dtypes = dtypes

    def get_tensor_shape(self, name: str) -> list[int]:
        return self._shapes[name]

    def get_tensor_dtype(self, name: str) -> str:
        return self._dtypes[name]


def test_validate_grid_sample_rank5_accepts_xyz_grid() -> None:
    node = _node(attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": 0})
    ctx = _Ctx(
        shapes={
            "x": [1, 2, 3, 4, 5],
            "grid": [1, 6, 7, 8, 3],
            "y": [1, 2, 6, 7, 8],
        },
        dtypes={"x": "FLOAT32", "grid": "FLOAT32", "y": "FLOAT32"},
    )
    _validate_grid_sample(node, ctx)


def test_validate_grid_sample_rank5_accepts_border_padding_mode() -> None:
    node = _node(attrs={"mode": "bilinear", "padding_mode": "border", "align_corners": 1})
    ctx = _Ctx(
        shapes={
            "x": [1, 2, 3, 4, 5],
            "grid": [1, 6, 7, 8, 3],
            "y": [1, 2, 6, 7, 8],
        },
        dtypes={"x": "FLOAT32", "grid": "FLOAT32", "y": "FLOAT32"},
    )
    _validate_grid_sample(node, ctx)


def test_validate_grid_sample_rank5_rejects_bad_grid_last_dim() -> None:
    node = _node(attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": 1})
    ctx = _Ctx(
        shapes={
            "x": [1, 2, 3, 4, 5],
            "grid": [1, 6, 7, 8, 2],
            "y": [1, 2, 6, 7, 8],
        },
        dtypes={"x": "FLOAT16", "grid": "FLOAT16", "y": "FLOAT16"},
    )
    with pytest.raises(NodeValidationError):
        _validate_grid_sample(node, ctx)
