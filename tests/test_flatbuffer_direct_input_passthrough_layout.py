from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    _optimize_asin_transpose_passthrough_chains,
    _optimize_erf_transpose_passthrough_chains,
    _optimize_hardsigmoid_transpose_passthrough_chains,
    _optimize_leading_input_transpose_passthrough_chains,
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
        is_variable=data is None,
    )


def _make_mul_roundtrip(*, add_input_fanout: bool = False) -> ModelIR:
    model_ir = ModelIR("input_passthrough_mul")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"] + (["side"] if add_input_fanout else [])
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 3, 2, 2]),
        "scale": _tensor(
            "scale",
            [1, 3, 1, 1],
            data=np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1),
        ),
        "mul_nchw": _tensor("mul_nchw", [1, 3, 2, 2]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "to_nchw"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "scale"],
            outputs=["mul_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["mul_nchw", "to_nhwc"],
            outputs=["y"],
        ),
    ]
    if add_input_fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 2, 2, 3])
        model_ir.operators.append(
            OperatorIR(op_type="IDENTITY", inputs=["x_nchw"], outputs=["side"])
        )
    return model_ir


def test_input_passthrough_moves_linear_binary_chain_to_nhwc() -> None:
    model_ir = _make_mul_roundtrip()

    stats = _optimize_leading_input_transpose_passthrough_chains(model_ir)

    assert stats == {
        "rewritten_leading_input_transpose_passthrough_chains": 1
    }
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "MUL"
    assert model_ir.operators[0].inputs == ["x", "scale"]
    assert model_ir.operators[0].outputs == ["y"]
    assert model_ir.tensors["scale"].shape == [1, 1, 1, 3]
    assert model_ir.tensors["scale"].data is not None
    assert list(model_ir.tensors["scale"].data.shape) == [1, 1, 1, 3]
    assert "x_nchw" not in model_ir.tensors
    assert "mul_nchw" not in model_ir.tensors


def test_input_passthrough_preserves_main_path_fanout() -> None:
    model_ir = _make_mul_roundtrip(add_input_fanout=True)

    stats = _optimize_leading_input_transpose_passthrough_chains(model_ir)

    assert stats == {
        "rewritten_leading_input_transpose_passthrough_chains": 0
    }
    assert [operator.op_type for operator in model_ir.operators[:3]] == [
        "TRANSPOSE",
        "MUL",
        "TRANSPOSE",
    ]
    assert model_ir.operators[1].inputs == ["x_nchw", "scale"]


def _make_asin_roundtrip(*, singleton_sub_constant: bool = True) -> ModelIR:
    model_ir = ModelIR("input_passthrough_asin")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    constant = (
        np.asarray(1.0, dtype=np.float32)
        if singleton_sub_constant
        else np.asarray([1.0, 1.0], dtype=np.float32)
    )
    constant_shape = [] if singleton_sub_constant else [2]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 3, 2, 2]),
        "x2": _tensor("x2", [1, 3, 2, 2]),
        "one": _tensor("one", constant_shape, data=constant),
        "one_minus": _tensor("one_minus", [1, 3, 2, 2]),
        "denom": _tensor("denom", [1, 3, 2, 2]),
        "asin_nchw": _tensor("asin_nchw", [1, 3, 2, 2]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "to_nchw"],
            outputs=["x_nchw"],
        ),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "x_nchw"], outputs=["x2"]),
        OperatorIR(op_type="SUB", inputs=["one", "x2"], outputs=["one_minus"]),
        OperatorIR(op_type="SQRT", inputs=["one_minus"], outputs=["denom"]),
        OperatorIR(
            op_type="ATAN2",
            inputs=["x_nchw", "denom"],
            outputs=["asin_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["asin_nchw", "to_nhwc"],
            outputs=["y"],
        ),
    ]
    return model_ir


def test_asin_passthrough_moves_decomposition_to_nhwc() -> None:
    model_ir = _make_asin_roundtrip()

    stats = _optimize_asin_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_asin_transpose_passthrough_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "SUB",
        "SQRT",
        "ATAN2",
    ]
    assert model_ir.operators[0].inputs == ["x", "x"]
    assert model_ir.operators[-1].inputs == ["x", "denom"]
    assert model_ir.operators[-1].outputs == ["y"]
    for name in ["x2", "one_minus", "denom"]:
        assert model_ir.tensors[name].shape == [1, 2, 2, 3]


def test_asin_passthrough_rejects_nonsingleton_sub_constant() -> None:
    model_ir = _make_asin_roundtrip(singleton_sub_constant=False)

    stats = _optimize_asin_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_asin_transpose_passthrough_chains": 0}
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert model_ir.operators[1].inputs == ["x_nchw", "x_nchw"]


def _make_hardsigmoid_roundtrip(*, inverse_output_perm: bool = True) -> ModelIR:
    model_ir = ModelIR("input_passthrough_hardsigmoid")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    output_perm = [0, 2, 3, 1] if inverse_output_perm else [0, 3, 1, 2]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 3, 2, 2]),
        "alpha": _tensor(
            "alpha",
            [],
            data=np.asarray(0.2, dtype=np.float32),
        ),
        "scaled": _tensor("scaled", [1, 3, 2, 2]),
        "beta": _tensor(
            "beta",
            [],
            data=np.asarray(0.5, dtype=np.float32),
        ),
        "biased": _tensor("biased", [1, 3, 2, 2]),
        "clamped": _tensor("clamped", [1, 3, 2, 2]),
        "to_output": _tensor(
            "to_output",
            [4],
            dtype="INT32",
            data=np.asarray(output_perm, dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "to_nchw"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "alpha"],
            outputs=["scaled"],
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["scaled", "beta"],
            outputs=["biased"],
        ),
        OperatorIR(
            op_type="RELU_0_TO_1",
            inputs=["biased"],
            outputs=["clamped"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["clamped", "to_output"],
            outputs=["y"],
        ),
    ]
    return model_ir


def test_hardsigmoid_passthrough_moves_relu_decomposition_to_nhwc() -> None:
    model_ir = _make_hardsigmoid_roundtrip()

    stats = _optimize_hardsigmoid_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_hardsigmoid_transpose_passthrough_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "ADD",
        "RELU_0_TO_1",
    ]
    assert model_ir.operators[0].inputs == ["x", "alpha"]
    assert model_ir.operators[-1].outputs == ["y"]
    assert model_ir.tensors["scaled"].shape == [1, 2, 2, 3]
    assert model_ir.tensors["biased"].shape == [1, 2, 2, 3]


def test_hardsigmoid_passthrough_rejects_noninverse_output_perm() -> None:
    model_ir = _make_hardsigmoid_roundtrip(inverse_output_perm=False)

    stats = _optimize_hardsigmoid_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_hardsigmoid_transpose_passthrough_chains": 0}
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert model_ir.operators[-1].op_type == "TRANSPOSE"


def _make_erf_roundtrip(*, scalar_abs_multiplier: bool = True) -> ModelIR:
    model_ir = ModelIR("input_passthrough_erf")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 3, 2, 2]),
        "abs": _tensor("abs", [1, 3, 2, 2]),
        "sign": _tensor("sign", [1, 3, 2, 2]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 3]),
    }

    def add_scalar(name: str, value: float) -> None:
        model_ir.tensors[name] = _tensor(
            name,
            [],
            data=np.asarray(value, dtype=np.float32),
        )

    def add_value(name: str) -> None:
        model_ir.tensors[name] = _tensor(name, [1, 3, 2, 2])

    for name, value in {
        "p": 0.3275911,
        "one_a": 1.0,
        "one_div": 1.0,
        "negative_one": -1.0,
        "a5": 0.254829592,
        "a4": -0.284496736,
        "a3": 1.421413741,
        "a2": -1.453152027,
        "a1": 1.061405429,
        "one_sub": 1.0,
    }.items():
        add_scalar(name, value)
    if not scalar_abs_multiplier:
        model_ir.tensors["p"] = _tensor(
            "p",
            [2],
            data=np.asarray([0.3275911, 0.3275911], dtype=np.float32),
        )

    for name in [
        "px",
        "one_plus",
        "t",
        "square",
        "negative_square",
        "exp",
        "h0",
        "add0",
        "h1",
        "add1",
        "h2",
        "add2",
        "h3",
        "add3",
        "poly",
        "poly_exp",
        "one_minus",
        "erf_nchw",
    ]:
        add_value(name)

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "to_nchw"],
            outputs=["x_nchw"],
        ),
        OperatorIR(op_type="ABS", inputs=["x_nchw"], outputs=["abs"]),
        OperatorIR(op_type="SIGN", inputs=["x_nchw"], outputs=["sign"]),
        OperatorIR(op_type="MUL", inputs=["abs", "p"], outputs=["px"]),
        OperatorIR(op_type="ADD", inputs=["px", "one_a"], outputs=["one_plus"]),
        OperatorIR(op_type="DIV", inputs=["one_div", "one_plus"], outputs=["t"]),
        OperatorIR(op_type="MUL", inputs=["abs", "abs"], outputs=["square"]),
        OperatorIR(
            op_type="MUL",
            inputs=["square", "negative_one"],
            outputs=["negative_square"],
        ),
        OperatorIR(op_type="EXP", inputs=["negative_square"], outputs=["exp"]),
        OperatorIR(op_type="MUL", inputs=["a5", "t"], outputs=["h0"]),
    ]
    current = "h0"
    coefficient_names = ["a4", "a3", "a2", "a1"]
    output_names = ["h1", "h2", "h3", "poly"]
    for index, (coefficient, output_name) in enumerate(
        zip(coefficient_names, output_names)
    ):
        add_name = f"add{index}"
        model_ir.operators.append(
            OperatorIR(
                op_type="ADD",
                inputs=[current, coefficient],
                outputs=[add_name],
            )
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="MUL",
                inputs=[add_name, "t"],
                outputs=[output_name],
            )
        )
        current = output_name
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="MUL",
                inputs=["poly", "exp"],
                outputs=["poly_exp"],
            ),
            OperatorIR(
                op_type="SUB",
                inputs=["one_sub", "poly_exp"],
                outputs=["one_minus"],
            ),
            OperatorIR(
                op_type="MUL",
                inputs=["sign", "one_minus"],
                outputs=["erf_nchw"],
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["erf_nchw", "to_nhwc"],
                outputs=["y"],
            ),
        ]
    )
    return model_ir


def test_erf_passthrough_moves_polynomial_decomposition_to_nhwc() -> None:
    model_ir = _make_erf_roundtrip()

    stats = _optimize_erf_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_erf_transpose_passthrough_chains": 1}
    assert model_ir.operators[0].op_type == "ABS"
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].op_type == "SIGN"
    assert model_ir.operators[1].inputs == ["x"]
    assert model_ir.operators[-1].op_type == "MUL"
    assert model_ir.operators[-1].outputs == ["y"]
    for name in ["abs", "sign", "poly", "poly_exp", "one_minus"]:
        assert model_ir.tensors[name].shape == [1, 2, 2, 3]


def test_erf_passthrough_rejects_nonsingleton_abs_multiplier() -> None:
    model_ir = _make_erf_roundtrip(scalar_abs_multiplier=False)

    stats = _optimize_erf_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_erf_transpose_passthrough_chains": 0}
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert model_ir.operators[1].inputs == ["x_nchw"]
