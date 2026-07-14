from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _clone_tensor,
    _collect_kernel_weight_tensor_names,
    _compose_axis_permutations,
    _inverse_axis_permutation,
    _normalize_constant_pad_pairs,
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
    _permute_tensor_to_channel_first_inplace,
    _preferred_reshape_target_values,
    _preferred_reshape_target_values_for_op,
    _tensor_name_suggests_channel_last_layout_for_codegen,
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


def test_pytorch_preferred_reshape_target_policy_is_shared() -> None:
    tensor = TensorIR(
        name="feature_nhwc",
        dtype="FLOAT32",
        shape=[1, 1, 64, 1],
        shape_signature=[-1, 1, 64, 1],
        logical_layout="NCHW",
    )
    model_ir = ModelIR(name="reshape_target")
    model_ir.tensors["feature_nhwc"] = tensor
    reshape = OperatorIR("RESHAPE", ["x"], ["feature_nhwc"])

    assert _tensor_name_suggests_channel_last_layout_for_codegen(
        "feature_NHWC"
    ) is True
    assert _preferred_reshape_target_values(tensor) == [-1, 1, 1, 64]
    assert _preferred_reshape_target_values_for_op(
        model_ir=model_ir,
        op=reshape,
    ) == [-1, 1, 1, 64]


def test_kernel_weight_collection_uses_shared_op_family_index() -> None:
    class CountingOperators(list[OperatorIR]):
        def __init__(self, values: list[OperatorIR]) -> None:
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iteration_count += 1
            return super().__iter__()

    model_ir = ModelIR(
        name="indexed_kernel_weights",
        operators=[
            OperatorIR("ADD", ["x", "y"], ["sum"]),
            OperatorIR("CONV_2D", ["sum", "weight"], ["output"]),
        ],
    )
    graph_index = ModelIRGraphIndex(model_ir)
    counting_operators = CountingOperators(model_ir.operators)
    model_ir.operators = counting_operators

    assert _collect_kernel_weight_tensor_names(
        model_ir,
        graph_index=graph_index,
    ) == {"weight"}
    assert counting_operators.iteration_count == 0
