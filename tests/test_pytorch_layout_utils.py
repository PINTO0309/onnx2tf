from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import TensorIR
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _clone_tensor,
    _compose_axis_permutations,
    _inverse_axis_permutation,
    _normalize_constant_pad_pairs,
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
    _permute_tensor_to_channel_first_inplace,
)


def test_pytorch_layout_permutations_are_inverse_for_supported_ranks() -> None:
    for rank in (3, 4, 5):
        cl_to_cf = _perm_cl_to_cf(rank)
        cf_to_cl = _perm_cf_to_cl(rank)
        assert cl_to_cf is not None
        assert cf_to_cl is not None
        # Identity compositions are represented as no-op (`None`).
        assert _compose_axis_permutations(cl_to_cf, cf_to_cl) is None
        assert _inverse_axis_permutation(cl_to_cf) == cf_to_cl


def test_pytorch_layout_shape_and_pad_helpers_are_deterministic() -> None:
    assert _permute_shape([1, 2, 3, 4], [0, 3, 1, 2]) == [1, 4, 2, 3]
    assert _normalize_constant_pad_pairs(
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7]),
    ) == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_pytorch_layout_tensor_clone_and_permute_preserve_provenance() -> None:
    tensor = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[-1, 2, 3, 4],
        data=np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
        logical_layout="NHWC",
        physical_layout="NHWC",
        onnx_tensor_name="onnx_x",
    )
    cloned = _clone_tensor(tensor)
    assert cloned is not tensor
    assert cloned.onnx_tensor_name == "onnx_x"
    assert cloned.physical_layout == "NHWC"

    assert _permute_tensor_to_channel_first_inplace(cloned) is True
    assert cloned.shape == [1, 4, 2, 3]
    assert cloned.shape_signature == [-1, 4, 2, 3]
    assert list(np.asarray(cloned.data).shape) == [1, 4, 2, 3]
    assert cloned.physical_layout == "NCHW"
