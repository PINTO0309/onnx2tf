from __future__ import annotations

import numpy as np
import pytest
from onnx import helper

import onnx2tf.tflite_builder.core.shape_readiness as shape_readiness_module
from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.node import NodeView
from onnx2tf.tflite_builder.core.shape_readiness import (
    reconcile_shape_sensitive_inputs_on_demand,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _context(
    *,
    shape: list[int],
    raw_shape=None,
    data=None,
) -> LoweringContext:
    tensor = TensorIR(
        name="input",
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
    )
    return LoweringContext(
        model_ir=ModelIR(name="shape_readiness", tensors={"input": tensor}),
        shape_map={} if raw_shape is None else {"input": list(raw_shape)},
        dtype_map={"input": "FLOAT32"},
        constants={},
    )


def _node(op_type: str) -> NodeView:
    return NodeView(
        helper.make_node(
            op_type,
            ["input"],
            ["output"],
            name="shape_sensitive_node",
        )
    )


@pytest.mark.parametrize(
    "op_type",
    [
        "Attention",
        "Gather",
        "GatherElements",
        "LayerNormalization",
        "MatMul",
        "MultiHeadAttention",
    ],
)
def test_shape_readiness_reconciles_each_supported_op(
    monkeypatch,
    op_type: str,
) -> None:
    calls = []

    def reconcile(model_ir):
        calls.append(model_ir)
        return {"reconciled_static_tensor_shapes": 3}

    monkeypatch.setattr(
        shape_readiness_module,
        "reconcile_static_tensor_shapes",
        reconcile,
    )
    ctx = _context(shape=[1])

    result = reconcile_shape_sensitive_inputs_on_demand(
        node=_node(op_type),
        ctx=ctx,
    )

    assert calls == [ctx.model_ir]
    assert result == {"reconciled_static_tensor_shapes": 3}


@pytest.mark.parametrize(
    ("op_type", "shape", "raw_shape", "data"),
    [
        ("Add", [1], None, None),
        ("MatMul", [1, 2], None, None),
        ("MatMul", [1], [1], None),
        ("MatMul", [1], None, np.asarray([1.0], dtype=np.float32)),
    ],
)
def test_shape_readiness_noop_paths_return_exact_zero(
    monkeypatch,
    op_type: str,
    shape: list[int],
    raw_shape,
    data,
) -> None:
    def unexpected_reconcile(_model_ir):
        raise AssertionError("stable shape readiness path reconciled")

    monkeypatch.setattr(
        shape_readiness_module,
        "reconcile_static_tensor_shapes",
        unexpected_reconcile,
    )

    result = reconcile_shape_sensitive_inputs_on_demand(
        node=_node(op_type),
        ctx=_context(shape=shape, raw_shape=raw_shape, data=data),
    )

    assert result == {"reconciled_static_tensor_shapes": 0}
