from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_reduction_policy import (
    _channel_first_reduction_plan_for_codegen,
    _direct_mean_reduction_expr_for_codegen,
    _normalized_constant_reduction_axes_for_codegen,
)


def test_channel_first_spatial_reduction_plan() -> None:
    model_ir = ModelIR(
        name="spatial_reduction",
        tensors={
            "input": TensorIR(
                name="input",
                dtype="FLOAT32",
                shape=[1, 2, 3, 4],
                logical_layout="NHWC",
            ),
            "axes": TensorIR(
                name="axes",
                dtype="INT32",
                shape=[2],
                data=np.asarray([1, 2], dtype=np.int32),
            ),
            "output": TensorIR(
                name="output",
                dtype="FLOAT32",
                shape=[1, 4],
            ),
        },
    )
    op = OperatorIR(
        op_type="MEAN",
        inputs=["input", "axes"],
        outputs=["output"],
    )

    assert _channel_first_reduction_plan_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={"input": "input_cf"},
        op=op,
        input_name="input",
    ) == ("input_cf", [2, 3])


def test_reduction_axis_normalization_and_direct_mean_expression() -> None:
    assert _normalized_constant_reduction_axes_for_codegen(
        axis_values=[-1, 1, 1],
        rank=4,
    ) == [1, 3]
    assert _direct_mean_reduction_expr_for_codegen(
        normalized_constant_reduction_axes_fn=(
            lambda axes, rank: _normalized_constant_reduction_axes_for_codegen(
                axis_values=axes,
                rank=rank,
            )
        ),
        input_expr="input_cf",
        axes=[2, 3],
        input_rank=4,
        keepdims=True,
    ) == "torch.mean(input_cf, dim=[2, 3], keepdim=True)"
