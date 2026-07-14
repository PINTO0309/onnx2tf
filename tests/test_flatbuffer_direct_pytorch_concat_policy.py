import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_concat_policy import (
    _can_fold_channel_last_alias_slice_consumer_for_codegen,
    _can_keep_channel_first_slice_output_for_codegen,
    _channel_first_concat_input_expr_for_codegen,
    _is_valid_concat_axis_for_channel_first_shapes_for_codegen,
    _resolve_concat_axis_for_channel_first_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    layout: str = "UNKNOWN",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=layout,
        data=data,
    )


def test_concat_input_expression_converts_channel_last_rank4() -> None:
    model_ir = ModelIR(
        name="concat_expr",
        tensors={"input": _tensor("input", [1, 8, 8, 3], layout="NHWC")},
    )

    assert _channel_first_concat_input_expr_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={},
        tensor_name="input",
        tensor_expr_fn=lambda name: f"expr_{name}",
    ) == "expr_input.permute(0, 3, 1, 2).contiguous()"


def test_concat_axis_policy_resolves_channel_first_shapes() -> None:
    op = OperatorIR(
        op_type="CONCATENATION",
        inputs=["lhs", "rhs"],
        outputs=["output"],
        options={"axis": 1},
    )
    model_ir = ModelIR(name="concat_axis", operators=[op])
    shapes = {
        "lhs": [1, 2, 4, 4],
        "rhs": [1, 3, 4, 4],
        "output": [1, 5, 4, 4],
    }

    assert _is_valid_concat_axis_for_channel_first_shapes_for_codegen(
        input_shapes=[shapes["lhs"], shapes["rhs"]],
        output_shape=shapes["output"],
        axis=1,
    )
    assert _resolve_concat_axis_for_channel_first_for_codegen(
        model_ir=model_ir,
        op=op,
        channel_first_shape_for_tensor_fn=lambda name: shapes.get(name),
        tensor_shape_list_fn=lambda name: shapes.get(name),
    ) == (1, [1, 5, 4, 4], [0, 1, 2, 3])


def test_slice_concat_policy_requires_static_slice_and_cf_consumers() -> None:
    slice_op = OperatorIR(
        op_type="SLICE",
        inputs=["input", "begin", "end"],
        outputs=["slice_output"],
    )
    concat_op = OperatorIR(
        op_type="CONCATENATION",
        inputs=["slice_output", "other"],
        outputs=["concat_output"],
        options={"axis": 1},
    )
    model_ir = ModelIR(
        name="slice_concat",
        tensors={
            "begin": _tensor(
                "begin",
                [1],
                data=np.asarray([0], dtype=np.int32),
            ),
            "end": _tensor(
                "end",
                [1],
                data=np.asarray([2], dtype=np.int32),
            ),
            "concat_output": _tensor(
                "concat_output",
                [1, 4, 4, 4],
                layout="NCHW",
            ),
        },
        operators=[slice_op, concat_op],
    )

    assert _can_fold_channel_last_alias_slice_consumer_for_codegen(
        model_ir=model_ir,
        op=slice_op,
        expected_input_name="input",
    )
    assert _can_keep_channel_first_slice_output_for_codegen(
        model_ir=model_ir,
        consumer_index={"slice_output": [1]},
        output_name="slice_output",
        resolve_concat_axis_for_channel_first_fn=lambda _op: (
            1,
            [1, 4, 4, 4],
            [0, 1, 2, 3],
        ),
    )
