from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains,
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
        is_variable=False if data is not None else True,
    )


def _model(*, gate_fanout: bool) -> ModelIR:
    model_ir = ModelIR("multi_branch_gate")
    model_ir.inputs = [
        "x0_nhwc",
        "x1_nhwc",
        "gate0_nhwc",
        "gate1_nhwc",
    ]
    model_ir.outputs = ["z"] + (["gate_side"] if gate_fanout else [])
    model_ir.tensors = {
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "axes": _tensor(
            "axes",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "root_nchw": _tensor("root_nchw", [1, 4, 3, 5]),
        "y_nhwc": _tensor("y_nhwc", [1, 3, 5, 4]),
        "z": _tensor("z", [1, 3, 5, 4]),
    }
    operators: list[OperatorIR] = []
    for branch in range(2):
        names = {
            "x_nhwc": f"x{branch}_nhwc",
            "x_nchw": f"x{branch}_nchw",
            "relu": f"relu{branch}_nchw",
            "mean": f"mean{branch}_nchw",
            "gate_nhwc": f"gate{branch}_nhwc",
            "gate_nchw": f"gate{branch}_nchw",
            "gate": f"sigmoid{branch}_nchw",
            "mul": f"mul{branch}_nchw",
        }
        for name in [names["x_nhwc"], names["gate_nhwc"]]:
            model_ir.tensors[name] = _tensor(name, [1, 3, 5, 4])
        for name in [names["x_nchw"], names["relu"], names["mul"]]:
            model_ir.tensors[name] = _tensor(name, [1, 4, 3, 5])
        for name in [names["mean"], names["gate_nchw"], names["gate"]]:
            model_ir.tensors[name] = _tensor(name, [1, 4, 1, 1])
        operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [names["x_nhwc"], "to_nchw"],
                    [names["x_nchw"]],
                ),
                OperatorIR("RELU", [names["x_nchw"]], [names["relu"]]),
                OperatorIR(
                    "MEAN",
                    [names["relu"], "axes"],
                    [names["mean"]],
                    options={"keepDims": True, "axes": [2, 3]},
                ),
                OperatorIR(
                    "TRANSPOSE",
                    [names["gate_nhwc"], "to_nchw"],
                    [names["gate_nchw"]],
                ),
                OperatorIR(
                    "LOGISTIC",
                    [names["gate_nchw"]],
                    [names["gate"]],
                ),
                OperatorIR(
                    "MUL",
                    [names["relu"], names["gate"]],
                    [names["mul"]],
                ),
            ]
        )
    operators.extend(
        [
            OperatorIR("ADD", ["mul0_nchw", "mul1_nchw"], ["root_nchw"]),
            OperatorIR("TRANSPOSE", ["root_nchw", "to_nhwc"], ["y_nhwc"]),
            OperatorIR("RELU", ["y_nhwc"], ["z"]),
        ]
    )
    if gate_fanout:
        model_ir.tensors["gate_side"] = _tensor(
            "gate_side",
            [1, 4, 1, 1],
        )
        operators.append(
            OperatorIR("IDENTITY", ["sigmoid0_nchw"], ["gate_side"])
        )
    model_ir.operators = operators
    return model_ir


def test_multi_branch_gate_layout_characterization() -> None:
    model_ir = _model(gate_fanout=False)

    stats = _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains"
    ] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model_ir.operators)
    cloned_axes = [
        tensor.data
        for name, tensor in model_ir.tensors.items()
        if name.startswith("axes__osnet_nhwc_axes")
    ]
    assert len(cloned_axes) == 2
    for axes in cloned_axes:
        np.testing.assert_array_equal(
            axes,
            np.asarray([1, 2], dtype=np.int32),
        )
    add_op = next(
        operator for operator in model_ir.operators if operator.op_type == "ADD"
    )
    assert add_op.outputs == ["y_nhwc"]


def test_multi_branch_gate_layout_rejects_gate_fanout() -> None:
    model_ir = _model(gate_fanout=True)

    stats = _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains"
    ] == 0
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 5
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([2, 3], dtype=np.int32),
    )
