from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_constant_alias_policy import (
    _binary_trailing_axis_constant_buffer_alias_shape_for_codegen,
    _channel_first_rank4_constant_buffer_alias_shape_for_codegen,
    _constant_permute_for_broadcast_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    layout: str = "UNKNOWN",
    constant: bool = False,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=layout,
        data=(np.ones(shape, dtype=np.float32) if constant else None),
    )


def test_constant_buffer_alias_shapes_require_the_proven_axis() -> None:
    model_ir = ModelIR(
        name="constant_alias_shapes",
        tensors={
            "vector": _tensor("vector", [4], constant=True),
            "channel_last": _tensor(
                "channel_last", [1, 2, 3, 4], layout="NHWC"
            ),
            "rank4": _tensor("rank4", [1, 1, 1, 4], constant=True),
        },
    )

    assert _binary_trailing_axis_constant_buffer_alias_shape_for_codegen(
        model_ir=model_ir,
        producer_index={},
        inlined_constant_tensor_names=set(),
        tensor_name="vector",
        other_tensor_name="channel_last",
    ) == [1, 1, 1, 4]
    assert _channel_first_rank4_constant_buffer_alias_shape_for_codegen(
        model_ir=model_ir,
        producer_index={},
        inlined_constant_tensor_names=set(),
        tensor_name="rank4",
    ) == [1, 4, 1, 1]


def test_constant_broadcast_permutation_prefers_declared_layout() -> None:
    model_ir = ModelIR(
        name="constant_broadcast",
        tensors={
            "constant": _tensor(
                "constant", [1, 3, 4, 2], layout="NCHW", constant=True
            ),
            "peer": _tensor("peer", [1, 4, 2, 3]),
            "singleton": _tensor(
                "singleton", [1, 384, 1], constant=True
            ),
            "singleton_peer": _tensor("singleton_peer", [1, 1, 384]),
        },
    )

    assert _constant_permute_for_broadcast_for_codegen(
        model_ir=model_ir,
        tensor_name="constant",
        other_tensor_name="peer",
    ) == [0, 2, 3, 1]
    assert (
        _constant_permute_for_broadcast_for_codegen(
            model_ir=model_ir,
            tensor_name="singleton",
            other_tensor_name="singleton_peer",
        )
        is None
    )
