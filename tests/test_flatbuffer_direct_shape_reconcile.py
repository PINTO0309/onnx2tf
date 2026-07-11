from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_consecutive_reshape_passthrough_chains,
    _reconcile_static_tensor_shapes,
    _replace_expand_dims_and_squeeze_with_reshape,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
    )


def test_reconcile_strided_slice_repairs_lowered_squeeze_target() -> None:
    model_ir = ModelIR("strided_slice_squeeze_reconcile")
    model_ir.tensors = {
        "x": _tensor("x", [1, 32, 64, 1, 16]),
        "begin": _tensor(
            "begin",
            [5],
            dtype="INT32",
            data=np.asarray([0, 0, 0, 0, 0], dtype=np.int32),
        ),
        "end": _tensor(
            "end",
            [5],
            dtype="INT32",
            data=np.asarray(
                [2147483647, 2147483647, 2147483647, 1, 16],
                dtype=np.int32,
            ),
        ),
        "strides": _tensor(
            "strides",
            [5],
            dtype="INT32",
            data=np.ones((5,), dtype=np.int32),
        ),
        "slice": _tensor("slice", [1, 1, 1, 1, 16]),
        "squeezed": _tensor("squeezed", [1, 1, 1, 16]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="STRIDED_SLICE",
            inputs=["x", "begin", "end", "strides"],
            outputs=["slice"],
            options={
                "beginMask": 0,
                "endMask": 0,
                "ellipsisMask": 0,
                "newAxisMask": 0,
                "shrinkAxisMask": 0,
            },
        ),
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["slice"],
            outputs=["squeezed"],
            options={"squeezeDims": [3]},
        ),
    ]

    _replace_expand_dims_and_squeeze_with_reshape(model_ir)
    stats = _reconcile_static_tensor_shapes(model_ir)

    reshape_op = model_ir.operators[1]
    assert str(reshape_op.op_type) == "RESHAPE"
    assert reshape_op.options["onnxSqueezeDims"] == [3]
    assert reshape_op.options["newShape"] == [1, 32, 64, 16]
    assert model_ir.tensors["slice"].shape == [1, 32, 64, 1, 16]
    assert model_ir.tensors["squeezed"].shape == [1, 32, 64, 16]
    np.testing.assert_array_equal(
        model_ir.tensors[str(reshape_op.inputs[1])].data,
        np.asarray([1, 32, 64, 16], dtype=np.int32),
    )
    assert stats["reconciled_static_tensor_shapes"] >= 2


def test_lowered_squeeze_keeps_semantic_rank_after_expand_dims() -> None:
    model_ir = ModelIR("expand_squeeze_semantic_rank")
    model_ir.tensors = {
        "core": _tensor("core", [8, 16, 1, 2, 64]),
        "axis": _tensor(
            "axis",
            [1],
            dtype="INT32",
            data=np.asarray([0], dtype=np.int32),
        ),
        "expanded": _tensor("expanded", [1, 8, 16, 1, 2, 64]),
        "squeezed": _tensor("squeezed", [1, 8, 16, 2, 64]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=["core", "axis"],
            outputs=["expanded"],
        ),
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["expanded"],
            outputs=["squeezed"],
            options={"squeezeDims": [3]},
        ),
    ]

    _replace_expand_dims_and_squeeze_with_reshape(model_ir)
    stats = _optimize_consecutive_reshape_passthrough_chains(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    assert stats["rewritten_consecutive_reshape_passthrough_chains"] == 0
    assert [str(op.op_type) for op in model_ir.operators] == ["RESHAPE", "RESHAPE"]
    assert str(model_ir.operators[1].inputs[0]) == "expanded"
    assert model_ir.operators[1].options["newShape"] == [1, 8, 16, 2, 64]
    assert model_ir.tensors["squeezed"].shape == [1, 8, 16, 2, 64]
