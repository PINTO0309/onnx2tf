from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.constant_fold import (
    run_constant_input_fold_cleanup,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _constant_pad_pool_cast_model() -> ModelIR:
    input_data = np.asarray(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        dtype=np.float32,
    )
    pads = np.asarray(
        [[0, 0], [1, 1], [1, 1], [0, 0]],
        dtype=np.int32,
    )
    model_ir = ModelIR("constant_pad_pool_cast")
    model_ir.outputs = ["cast_out"]
    model_ir.tensors = {
        "input": _tensor("input", "FLOAT32", [1, 2, 2, 1], data=input_data),
        "pads": _tensor("pads", "INT32", [4, 2], data=pads),
        "padded": _tensor("padded", "FLOAT32", [1, 4, 4, 1]),
        "pooled": _tensor("pooled", "FLOAT32", [1, 2, 2, 1]),
        "cast_out": _tensor("cast_out", "FLOAT16", [1, 2, 2, 1]),
    }
    model_ir.operators = [
        OperatorIR("PAD", ["input", "pads"], ["padded"]),
        OperatorIR(
            "AVERAGE_POOL_2D",
            ["padded"],
            ["pooled"],
            options={
                "padding": "VALID",
                "strideH": 2,
                "strideW": 2,
                "filterHeight": 2,
                "filterWidth": 2,
            },
        ),
        OperatorIR(
            "CAST",
            ["pooled"],
            ["cast_out"],
            options={"inDataType": "FLOAT32", "outDataType": "FLOAT16"},
        ),
    ]
    return model_ir


def test_constant_input_fold_group_materializes_pad_pool_cast_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _constant_pad_pool_cast_model()
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_constant_input_fold_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert stats == {
        "optimized_constant_input_pad_chains": 1,
        "optimized_constant_input_pool_chains": 1,
        "optimized_constant_input_cast_chains": 1,
    }
    assert refresh_count == 1
    assert model_ir.operators == []
    assert set(model_ir.tensors) == {"cast_out"}
    assert model_ir.tensors["cast_out"].dtype == "FLOAT16"
    np.testing.assert_array_equal(
        model_ir.tensors["cast_out"].data,
        np.asarray(
            [[[[0.25], [0.5]], [[0.75], [1.0]]]],
            dtype=np.float16,
        ),
    )
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert [event["code"] for event in diagnostics] == [
        "canonicalize.constant_input_pad",
        "canonicalize.constant_input_pool",
        "canonicalize.constant_input_cast",
    ]
    assert all(event["status"] == "changed" for event in diagnostics)


def test_constant_input_fold_group_preserves_runtime_accumulator_cast() -> None:
    model_ir = ModelIR("preserve_runtime_cast")
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "constant": _tensor(
            "constant",
            "INT32",
            [2],
            data=np.asarray([1, 2], dtype=np.int32),
        ),
        "out": _tensor("out", "FLOAT32", [2]),
    }
    model_ir.operators = [
        OperatorIR(
            "CAST",
            ["constant"],
            ["out"],
            options={
                "inDataType": "INT32",
                "outDataType": "FLOAT32",
                "preserveRuntimeCastForQuantizedAccumulator": True,
            },
        )
    ]

    stats = run_constant_input_fold_cleanup(model_ir)

    assert stats == {
        "optimized_constant_input_pad_chains": 0,
        "optimized_constant_input_pool_chains": 0,
        "optimized_constant_input_cast_chains": 0,
    }
    assert [op.op_type for op in model_ir.operators] == ["CAST"]
    assert model_ir.tensors["out"].data is None
