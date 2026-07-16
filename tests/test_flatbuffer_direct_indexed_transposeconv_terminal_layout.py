from __future__ import annotations

import copy
import pickle
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.conv_output_passthrough_layout import (
    optimize_transposeconv_output_channel1_terminal_transpose_chains,
)


_N = 2
_H = 3
_W = 4
_NHWC = [_N, _H, _W, 1]
_NCHW = [_N, 1, _H, _W]
_OUTPUT = [_N, _H, _W]
_SIDE_NCHW = [1, 1, 1, _W]
_SIDE_NHWC = [1, 1, _W, 1]
_TERMINAL_UNARIES = (
    "QUANTIZE",
    "DEQUANTIZE",
    "CAST",
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "LOGISTIC",
    "TANH",
    "GELU",
    "HARD_SWISH",
    "ABS",
    "NEG",
    "SQRT",
    "EXP",
)
_BINARIES = ("ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM")


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    dtype: str = "FLOAT32",
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
    layout: str = LOGICAL_LAYOUT_UNKNOWN,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
        logical_layout=layout,
        physical_layout=layout,
    )


def _make_model(
    *,
    axes: tuple[int, ...] = (1,),
    dynamic: bool = False,
    unary_after: str | None = None,
    binary_before: str | None = None,
    main_input_index: int = 0,
    scalar_after: bool = False,
    shared_constant: bool = False,
) -> ModelIR:
    model_ir = ModelIR("indexed_transposeconv_terminal")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    nhwc_signature = [_N, -1, -1, 1] if dynamic else list(_NHWC)
    nchw_signature = [_N, 1, -1, -1] if dynamic else list(_NCHW)
    output_signature = [_N, -1, -1] if dynamic else list(_OUTPUT)
    model_ir.tensors = {
        "x": _tensor("x", _NHWC, signature=nhwc_signature, layout=LOGICAL_LAYOUT_NHWC),
        "filter": _tensor(
            "filter",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        ),
        "bias": _tensor(
            "bias",
            [1],
            data=np.asarray([0.0], dtype=np.float32),
        ),
        "deconv_nhwc": _tensor(
            "deconv_nhwc",
            _NHWC,
            signature=nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int64),
            dtype="INT64",
        ),
        "deconv_nchw": _tensor(
            "deconv_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "relu_nchw": _tensor(
            "relu_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
    }
    operators = [
        OperatorIR(
            op_type="TRANSPOSE_CONV",
            inputs=["x", "filter", "bias"],
            outputs=["deconv_nhwc"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["deconv_nhwc", "to_nchw"],
            outputs=["deconv_nchw"],
        ),
        OperatorIR(op_type="RELU", inputs=["deconv_nchw"], outputs=["relu_nchw"]),
    ]
    last_name = "relu_nchw"
    if binary_before is not None:
        side = np.linspace(0.5, 1.25, _W, dtype=np.float32).reshape(_SIDE_NCHW)
        model_ir.tensors["side"] = _tensor("side", _SIDE_NCHW, data=side)
        model_ir.tensors["binary_nchw"] = _tensor(
            "binary_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        )
        inputs = [last_name, "side"]
        if int(main_input_index) == 1:
            inputs.reverse()
        operators.append(
            OperatorIR(
                op_type=binary_before,
                inputs=inputs,
                outputs=["binary_nchw"],
            )
        )
        last_name = "binary_nchw"
        if shared_constant:
            model_ir.inputs.append("legacy_nchw")
            model_ir.outputs.append("legacy_keep")
            model_ir.tensors["legacy_nchw"] = _tensor(
                "legacy_nchw",
                _NCHW,
                layout=LOGICAL_LAYOUT_NCHW,
            )
            model_ir.tensors["legacy_keep"] = _tensor(
                "legacy_keep",
                _NCHW,
                layout=LOGICAL_LAYOUT_NCHW,
            )

    squeeze_output = "z" if unary_after is None and not scalar_after else "squeezed"
    model_ir.tensors[squeeze_output] = _tensor(
        squeeze_output,
        _OUTPUT,
        signature=output_signature,
    )
    operators.append(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[last_name],
            outputs=[squeeze_output],
            options={"squeezeDims": list(axes)},
        )
    )
    last_name = squeeze_output
    if unary_after is not None:
        model_ir.tensors["z"] = _tensor("z", _OUTPUT, signature=output_signature)
        operators.append(
            OperatorIR(op_type=unary_after, inputs=[last_name], outputs=["z"])
        )
        last_name = "z"
    elif scalar_after:
        model_ir.tensors["scalar"] = _tensor(
            "scalar",
            [1],
            data=np.asarray([0.5], dtype=np.float32),
        )
        model_ir.tensors["z"] = _tensor("z", _OUTPUT, signature=output_signature)
        operators.append(
            OperatorIR(op_type="ADD", inputs=[last_name, "scalar"], outputs=["z"])
        )
        last_name = "z"
    if shared_constant:
        operators.append(
            OperatorIR(
                op_type="ADD",
                inputs=["legacy_nchw", "side"],
                outputs=["legacy_keep"],
            )
        )
    assert last_name == "z"
    model_ir.operators = operators
    return model_ir


def _binary(operator_type: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return {
        "ADD": np.add,
        "SUB": np.subtract,
        "MUL": np.multiply,
        "DIV": np.divide,
        "MAXIMUM": np.maximum,
        "MINIMUM": np.minimum,
    }[operator_type](left, right)


def _snapshot(model_ir: ModelIR) -> bytes:
    return pickle.dumps(model_ir, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("axes", [(1,), (-3,), ()], ids=["explicit", "negative", "implicit"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_indexed_transposeconv_terminal_remaps_channel_squeeze(
    axes: tuple[int, ...],
    dynamic: bool,
) -> None:
    model_ir = _make_model(axes=axes, dynamic=dynamic)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains": 1
    }
    assert "TRANSPOSE" not in {str(operator.op_type) for operator in model_ir.operators}
    squeeze = next(operator for operator in model_ir.operators if str(operator.op_type) == "SQUEEZE")
    assert list(squeeze.inputs) == ["relu_nchw"]
    assert list(squeeze.options["squeezeDims"]) == [3]
    assert model_ir.tensors["z"].shape == _OUTPUT
    assert model_ir.tensors["z"].shape_signature == (
        [_N, -1, -1] if dynamic else _OUTPUT
    )
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("unary_type", _TERMINAL_UNARIES)
def test_indexed_transposeconv_terminal_supports_unary_after_squeeze(
    unary_type: str,
) -> None:
    model_ir = _make_model(unary_after=unary_type)

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(model_ir)

    assert stats[
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains"
    ] == 1
    unary = model_ir.operators[-1]
    assert str(unary.op_type) == unary_type
    assert list(unary.inputs) == ["squeezed"]
    assert model_ir.tensors["squeezed"].shape == _OUTPUT
    assert model_ir.tensors["z"].shape == _OUTPUT


@pytest.mark.parametrize("binary_type", _BINARIES)
@pytest.mark.parametrize("main_input_index", [0, 1])
def test_indexed_transposeconv_terminal_preserves_binary_semantics(
    binary_type: str,
    main_input_index: int,
) -> None:
    model_ir = _make_model(
        binary_before=binary_type,
        main_input_index=main_input_index,
    )
    rng = np.random.default_rng(37)
    deconv_nhwc = rng.uniform(0.25, 2.0, size=_NHWC).astype(np.float32)
    main_nchw = np.maximum(np.transpose(deconv_nhwc, (0, 3, 1, 2)), 0.0)
    side_nchw = np.asarray(model_ir.tensors["side"].data)
    operands = [main_nchw, side_nchw]
    if int(main_input_index) == 1:
        operands.reverse()
    expected = np.squeeze(_binary(binary_type, operands[0], operands[1]), axis=1)

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(model_ir)

    assert stats[
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains"
    ] == 1
    binary = next(operator for operator in model_ir.operators if str(operator.op_type) == binary_type)
    side_name = next(name for name in binary.inputs if str(name) != "relu_nchw")
    assert model_ir.tensors[side_name].shape == _SIDE_NHWC
    new_operands = [np.maximum(deconv_nhwc, 0.0), np.asarray(model_ir.tensors[side_name].data)]
    if int(main_input_index) == 1:
        new_operands.reverse()
    actual = np.squeeze(_binary(binary_type, new_operands[0], new_operands[1]), axis=3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_indexed_transposeconv_terminal_supports_scalar_binary_after_squeeze() -> None:
    model_ir = _make_model(scalar_after=True)

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(model_ir)

    assert stats[
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains"
    ] == 1
    add = model_ir.operators[-1]
    assert list(add.inputs) == ["squeezed", "scalar"]
    assert model_ir.tensors["scalar"].shape == [1]


def test_indexed_transposeconv_terminal_clones_shared_rank4_constant() -> None:
    model_ir = _make_model(binary_before="MUL", shared_constant=True)
    original = np.asarray(model_ir.tensors["side"].data).copy()

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(model_ir)

    assert stats[
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains"
    ] == 1
    np.testing.assert_array_equal(model_ir.tensors["side"].data, original)
    assert model_ir.tensors["side"].shape == _SIDE_NCHW
    assert model_ir.tensors["side_nhwc"].shape == _SIDE_NHWC
    mul = next(operator for operator in model_ir.operators if str(operator.op_type) == "MUL")
    assert "side_nhwc" in mul.inputs
    legacy = next(
        operator
        for operator in model_ir.operators
        if "legacy_nchw" in {str(value) for value in operator.inputs}
    )
    assert list(legacy.inputs) == ["legacy_nchw", "side"]


def test_indexed_transposeconv_terminal_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    candidate = model_ir.operators[1]
    index = ModelIRGraphIndex(model_ir)

    assert optimize_transposeconv_output_channel1_terminal_transpose_chains(
        model_ir,
        graph_index=index,
        candidate=candidate,
        max_rewrites=0,
    ) == {"rewritten_transposeconv_output_channel1_terminal_transpose_chains": 0}
    assert optimize_transposeconv_output_channel1_terminal_transpose_chains(
        model_ir,
        graph_index=index,
        candidate=candidate,
    ) == {"rewritten_transposeconv_output_channel1_terminal_transpose_chains": 1}
    assert optimize_transposeconv_output_channel1_terminal_transpose_chains(
        model_ir,
        graph_index=index,
    ) == {"rewritten_transposeconv_output_channel1_terminal_transpose_chains": 0}
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []


UnsafeMutation = Callable[[ModelIR], None]


def _unsupported_source(model_ir: ModelIR) -> None:
    model_ir.operators[0].op_type = "CONV_2D"


def _non_singleton_channel(model_ir: ModelIR) -> None:
    model_ir.tensors["deconv_nhwc"].shape[-1] = 2


def _dynamic_channel(model_ir: ModelIR) -> None:
    model_ir.tensors["deconv_nhwc"].shape_signature[-1] = -1


def _wrong_pre_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["deconv_nchw"].shape = [_N, _H, 1, _W]


def _wrong_perm(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int64)


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["fanout"] = _tensor("fanout", _NCHW)
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["deconv_nchw"], outputs=["fanout"])
    )


def _remove_squeeze(model_ir: ModelIR) -> None:
    squeeze = next(operator for operator in model_ir.operators if str(operator.op_type) == "SQUEEZE")
    squeeze.op_type = "RELU"
    squeeze.options = {}


def _second_squeeze(model_ir: ModelIR) -> None:
    model_ir.outputs = ["z2"]
    model_ir.tensors["z2"] = _tensor("z2", [_N, _H])
    model_ir.operators.append(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["z"],
            outputs=["z2"],
            options={"squeezeDims": [2]},
        )
    )


def _invalid_squeeze_axis(model_ir: ModelIR) -> None:
    squeeze = next(operator for operator in model_ir.operators if str(operator.op_type) == "SQUEEZE")
    squeeze.options["squeezeDims"] = [2]


def _unknown_squeezed_axis(model_ir: ModelIR) -> None:
    model_ir.tensors["relu_nchw"].shape_signature[1] = -1


def _spatial_only_semantic_reorder(model_ir: ModelIR) -> None:
    for name in ("deconv_nhwc", "deconv_nchw", "relu_nchw"):
        tensor = model_ir.tensors[name]
        if name == "deconv_nhwc":
            tensor.shape = [_N, 1, _W, 1]
            tensor.shape_signature = [_N, 1, _W, 1]
        else:
            tensor.shape = [_N, 1, 1, _W]
            tensor.shape_signature = [_N, 1, 1, _W]
    model_ir.tensors["z"].shape = [_N, 1, _W]
    model_ir.tensors["z"].shape_signature = [_N, 1, _W]
    squeeze = next(operator for operator in model_ir.operators if str(operator.op_type) == "SQUEEZE")
    squeeze.options["squeezeDims"] = [2]


def _output_not_public(model_ir: ModelIR) -> None:
    model_ir.outputs = ["missing_public"]
    model_ir.tensors["missing_public"] = _tensor("missing_public", [1])


def _public_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("relu_nchw")


def _output_consumer(model_ir: ModelIR) -> None:
    model_ir.tensors["after"] = _tensor("after", _OUTPUT)
    model_ir.operators.append(OperatorIR(op_type="RELU", inputs=["z"], outputs=["after"]))


def _runtime_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].data = None
    model_ir.inputs.append("side")


def _variable_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].is_variable = True


def _per_axis_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _wrong_squeeze_output(model_ir: ModelIR) -> None:
    model_ir.tensors["z"].shape = [_N, _H, _W + 1]


def _missing_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["relu_nchw"]


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["x"], outputs=["deconv_nhwc"]),
    )


def _late_source(model_ir: ModelIR) -> None:
    model_ir.operators.insert(2, model_ir.operators.pop(0))


def _invalid_options(model_ir: ModelIR) -> None:
    squeeze = next(operator for operator in model_ir.operators if str(operator.op_type) == "SQUEEZE")
    squeeze.options = None  # type: ignore[assignment]


@pytest.mark.parametrize(
    "mutation",
    [
        _unsupported_source,
        _non_singleton_channel,
        _dynamic_channel,
        _wrong_pre_shape,
        _wrong_perm,
        _pre_fanout,
        _remove_squeeze,
        _second_squeeze,
        _invalid_squeeze_axis,
        _unknown_squeezed_axis,
        _spatial_only_semantic_reorder,
        _output_not_public,
        _public_intermediate,
        _output_consumer,
        _runtime_side,
        _variable_side,
        _per_axis_side,
        _wrong_squeeze_output,
        _missing_tensor,
        _duplicate_producer,
        _late_source,
        _invalid_options,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_transposeconv_terminal_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model(binary_before="ADD")
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))

    stats = optimize_transposeconv_output_channel1_terminal_transpose_chains(model_ir)

    assert stats == {
        "rewritten_transposeconv_output_channel1_terminal_transpose_chains": 0
    }
    assert _snapshot(model_ir) == before
