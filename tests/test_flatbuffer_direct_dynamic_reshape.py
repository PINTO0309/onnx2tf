from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _infer_batch_matmul_output_shape_and_signature,
    _reconcile_static_tensor_shapes,
    _restore_placeholder_matmul_flattened_inputs,
    _resolve_dynamic_reshape_shapes,
)


def test_final_reshape_preserves_raw_minus_one_when_static_metadata_is_stale() -> None:
    model_ir = ModelIR("stale_static_high_rank_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 10, 4096, 1],
        shape_signature=[1, 10, 4096, 1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 5, 64, 64, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 64, 64, 2],
        shape_signature=[1, 5, 64, 64, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={
                "newShape": [1, 5, 64, 64, 2],
                "onnxRawNewShape": [1, -1, 64, 64, 2],
                "allowZero": False,
            },
        )
    ]

    stats = _resolve_dynamic_reshape_shapes(
        model_ir,
        prefer_runtime_inferable_from_onnx_raw=True,
    )

    assert stats == {"resolved_dynamic_reshape_shapes": 1}
    assert model_ir.operators[0].options["newShape"] == [1, -1, 64, 64, 2]
    assert np.asarray(model_ir.tensors["shape"].data).tolist() == [1, -1, 64, 64, 2]
    assert model_ir.tensors["y"].shape == [1, 1, 64, 64, 2]
    assert model_ir.tensors["y"].shape_signature == [1, -1, 64, 64, 2]


def test_shape_reconciliation_repairs_stale_flatten_shape_constant() -> None:
    model_ir = ModelIR("stale_flatten_shape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1024, 7, 7],
        shape_signature=[-1, 1024, 7, 7],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, 1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={"newShape": [1, 1], "onnxFlattenAxis": 1},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)

    assert stats == {"reconciled_static_tensor_shapes": 1}
    assert model_ir.operators[0].options["newShape"] == [-1, 50176]
    assert np.asarray(model_ir.tensors["shape"].data).tolist() == [-1, 50176]
    assert model_ir.tensors["y"].shape == [1, 50176]
    assert model_ir.tensors["y"].shape_signature == [-1, 50176]


def test_batch_matmul_shape_inference_accepts_unbatched_rhs() -> None:
    output_shape, output_signature = _infer_batch_matmul_output_shape_and_signature(
        shape_a=[1, 56, 56, 96],
        shape_b=[96, 384],
        signature_a=[-1, 56, 56, 96],
        signature_b=[96, 384],
        adj_x=False,
        adj_y=False,
    )

    assert output_shape == [1, 56, 56, 384]
    assert output_signature == [-1, 56, 56, 384]


def test_restores_placeholder_matmul_flatten_after_rank_recovery() -> None:
    model_ir = ModelIR("placeholder_matmul_flatten")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 56, 56, 384],
        shape_signature=[-1, 56, 56, 384],
    )
    model_ir.tensors["flatten_shape"] = TensorIR(
        name="flatten_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([-1, 384], dtype=np.int32),
    )
    model_ir.tensors["x_flat"] = TensorIR(
        name="x_flat",
        dtype="FLOAT32",
        shape=[3136, 384],
        shape_signature=[-1, 384],
    )
    model_ir.tensors["weights"] = TensorIR(
        name="weights",
        dtype="FLOAT32",
        shape=[384, 96],
        shape_signature=[384, 96],
        data=np.ones((384, 96), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[3136, 96],
        shape_signature=[-1, 96],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "flatten_shape"],
            outputs=["x_flat"],
            options={
                "newShape": [-1, 384],
                "preserveDynamicShape": True,
                "onnxMatMulFlattenedPlaceholder": True,
            },
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["x_flat", "weights"],
            outputs=["y"],
            options={"adjX": False, "adjY": False},
        ),
    ]

    stats = _restore_placeholder_matmul_flattened_inputs(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    assert stats == {"restored_placeholder_matmul_flattened_inputs": 1}
    assert [op.op_type for op in model_ir.operators] == ["BATCH_MATMUL"]
    assert model_ir.operators[0].inputs == ["x", "weights"]
    assert model_ir.tensors["y"].shape == [1, 56, 56, 96]
    assert model_ir.tensors["y"].shape_signature == [-1, 56, 56, 96]
