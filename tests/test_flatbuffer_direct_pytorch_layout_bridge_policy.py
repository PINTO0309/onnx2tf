from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_layout_bridge_policy import (
    _fold_single_consumer_public_input_bridge_for_codegen,
    _has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen,
    _is_batchless_rank3_public_output_transpose_for_codegen,
    _match_single_consumer_layout_bridge_transpose_for_codegen,
)


def _tensor(name: str, shape: list[int], logical_layout: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=logical_layout,
    )


def test_fold_single_consumer_public_input_bridge() -> None:
    bridge_name = "input_public_layout_bridge"
    model_ir = ModelIR(
        name="public_input_bridge",
        operators=[
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["input"],
                outputs=[bridge_name],
                options={"perm": [0, 3, 1, 2]},
            )
        ],
        inputs=["input"],
    )

    assert _fold_single_consumer_public_input_bridge_for_codegen(
        model_ir=model_ir,
        producer_index={bridge_name: 0},
        consumer_index={bridge_name: [1]},
        public_layout_bridge_tensor_names={bridge_name},
        public_input_names={"input"},
        tensor_name=bridge_name,
        downstream_permute=None,
    ) == ("input", [0, 3, 1, 2], 0)


def test_match_single_consumer_layout_bridge_requires_exact_layout_perm() -> None:
    model_ir = ModelIR(
        name="layout_bridge",
        tensors={
            "source": _tensor("source", [1, 3, 4, 5], "NCHW"),
            "output": _tensor("output", [1, 4, 5, 3], "NHWC"),
        },
        operators=[
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["source"],
                outputs=["output"],
                options={"perm": [0, 2, 3, 1]},
            )
        ],
    )

    assert _match_single_consumer_layout_bridge_transpose_for_codegen(
        model_ir=model_ir,
        consumer_index={"source": [0]},
        tensor_name="source",
        required_output_layout="NHWC",
    ) == ("output", 0)
    assert (
        _match_single_consumer_layout_bridge_transpose_for_codegen(
            model_ir=model_ir,
            consumer_index={"source": [0, 1]},
            tensor_name="source",
        )
        is None
    )


def test_same_shape_transpose_follows_spatial_reduction_hint() -> None:
    transpose = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["input"],
        outputs=["bridge"],
        options={"perm": [0, 2, 3, 1]},
    )
    reduction = OperatorIR(
        op_type="MEAN",
        inputs=["bridge", "axes"],
        outputs=["output"],
    )
    model_ir = ModelIR(
        name="same_shape_hint",
        tensors={
            "input": _tensor("input", [1, 3, 3, 3], "UNKNOWN"),
            "bridge": _tensor("bridge", [1, 3, 3, 3], "UNKNOWN"),
            "axes": TensorIR(
                name="axes",
                dtype="INT32",
                shape=[2],
                data=np.asarray([1, 2], dtype=np.int32),
            ),
            "output": _tensor("output", [1, 3], "UNKNOWN"),
        },
        operators=[transpose, reduction],
    )

    assert _has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen(
        model_ir=model_ir,
        consumer_index={"bridge": [1]},
        op=transpose,
    )


def test_batchless_rank3_public_output_honors_boundary_marker() -> None:
    transpose = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["input"],
        outputs=["output"],
        options={"perm": [0, 2, 1]},
    )
    model_ir = ModelIR(
        name="batchless_public_output",
        tensors={
            "input": _tensor("input", [1, 8, 8], "NCW"),
            "output": _tensor("output", [1, 8, 8], "NWC"),
        },
        operators=[transpose],
        outputs=["output"],
    )

    assert _is_batchless_rank3_public_output_transpose_for_codegen(
        model_ir=model_ir,
        producer_index={},
        batchless_rank3_public_boundary_names={"output"},
        op=transpose,
    )
