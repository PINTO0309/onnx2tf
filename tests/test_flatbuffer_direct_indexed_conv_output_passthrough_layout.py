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
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.conv_output_passthrough_layout import (
    optimize_transposeconv_output_nhwc_passthrough_chains,
)


_H = 3
_W = 4
_C = 2
_NHWC = [1, _H, _W, _C]
_NCHW = [1, _C, _H, _W]
_SIDE_NCHW = [1, _C, 1, 1]
_SIDE_NHWC = [1, 1, 1, _C]
_UNARY_TYPES = (
    "QUANTIZE",
    "DEQUANTIZE",
    "CAST",
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "HARD_SWISH",
)
_BINARY_TYPES = ("ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM")


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    dtype: str = "FLOAT32",
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
    layout: str = "UNKNOWN",
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
    producer_type: str = "CONV_2D",
    unary_type: str = "RELU",
    binary_type: str | None = None,
    main_input_index: int = 0,
    dynamic: bool = False,
    shared_constant: bool = False,
    repeated_binary: bool = False,
) -> ModelIR:
    model_ir = ModelIR("indexed_conv_output_passthrough")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    nhwc_signature = [1, -1, -1, _C] if dynamic else list(_NHWC)
    nchw_signature = [1, _C, -1, -1] if dynamic else list(_NCHW)

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
        "conv_nhwc": _tensor(
            "conv_nhwc",
            _NHWC,
            signature=nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            dtype="INT32",
        ),
        "conv_nchw": _tensor(
            "conv_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "unary_nchw": _tensor(
            "unary_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int64),
            dtype="INT64",
        ),
        "tail_nhwc": _tensor(
            "tail_nhwc",
            _NHWC,
            signature=nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "z": _tensor(
            "z",
            _NHWC,
            signature=nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
    }
    operators = [
        OperatorIR(
            op_type=producer_type,
            inputs=["x", "filter", "bias"],
            outputs=["conv_nhwc"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["conv_nhwc", "to_nchw"],
            outputs=["conv_nchw"],
        ),
        OperatorIR(
            op_type=unary_type,
            inputs=["conv_nchw"],
            outputs=["unary_nchw"],
        ),
    ]
    last_name = "unary_nchw"
    if binary_type is not None:
        side = np.asarray([0.5, 1.5], dtype=np.float32).reshape(_SIDE_NCHW)
        model_ir.tensors["side"] = _tensor("side", _SIDE_NCHW, data=side)
        model_ir.tensors["binary_nchw"] = _tensor(
            "binary_nchw",
            _NCHW,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        )
        binary_inputs = [last_name, "side"]
        if int(main_input_index) == 1:
            binary_inputs.reverse()
        operators.append(
            OperatorIR(
                op_type=binary_type,
                inputs=binary_inputs,
                outputs=["binary_nchw"],
            )
        )
        last_name = "binary_nchw"
        if repeated_binary:
            model_ir.tensors["binary2_nchw"] = _tensor(
                "binary2_nchw",
                _NCHW,
                signature=nchw_signature,
                layout=LOGICAL_LAYOUT_NCHW,
            )
            operators.append(
                OperatorIR(
                    op_type="ADD",
                    inputs=[last_name, "side"],
                    outputs=["binary2_nchw"],
                )
            )
            last_name = "binary2_nchw"
        if shared_constant:
            model_ir.inputs.append("legacy_nchw")
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
    operators.extend(
        [
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[last_name, "to_nhwc"],
                outputs=["tail_nhwc"],
            ),
            OperatorIR(op_type="RELU", inputs=["tail_nhwc"], outputs=["z"]),
        ]
    )
    if shared_constant:
        operators.append(
            OperatorIR(
                op_type="ADD",
                inputs=["legacy_nchw", "side"],
                outputs=["legacy_keep"],
            )
        )
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


@pytest.mark.parametrize("producer_type", sorted(("CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV")))
@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_indexed_conv_output_passthrough_supports_all_producers_and_unaries(
    producer_type: str,
    unary_type: str,
) -> None:
    model_ir = _make_model(producer_type=producer_type, unary_type=unary_type)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transposeconv_output_nhwc_passthrough_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 1}
    assert [str(operator.op_type) for operator in model_ir.operators].count("TRANSPOSE") == 0
    unary = next(operator for operator in model_ir.operators if str(operator.op_type) == unary_type)
    assert list(unary.inputs) == ["conv_nhwc"]
    assert list(unary.outputs) == ["tail_nhwc"]
    assert model_ir.tensors["tail_nhwc"].shape == _NHWC
    assert model_ir.tensors["tail_nhwc"].shape_signature == _NHWC
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("binary_type", _BINARY_TYPES)
@pytest.mark.parametrize("main_input_index", [0, 1])
def test_indexed_conv_output_passthrough_preserves_binary_semantics(
    binary_type: str,
    main_input_index: int,
) -> None:
    model_ir = _make_model(
        binary_type=binary_type,
        main_input_index=main_input_index,
    )
    rng = np.random.default_rng(31)
    conv_nhwc = rng.uniform(0.25, 2.0, size=_NHWC).astype(np.float32)
    side_nchw = np.asarray(model_ir.tensors["side"].data)
    main_nchw = np.maximum(np.transpose(conv_nhwc, (0, 3, 1, 2)), 0.0)
    operands = [main_nchw, side_nchw]
    if int(main_input_index) == 1:
        operands.reverse()
    expected = np.transpose(
        _binary(binary_type, operands[0], operands[1]),
        (0, 2, 3, 1),
    )

    stats = optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)

    assert stats["rewritten_transposeconv_output_nhwc_passthrough_chains"] == 1
    binary = next(operator for operator in model_ir.operators if str(operator.op_type) == binary_type)
    side_name = next(name for name in binary.inputs if str(name) != "unary_nchw")
    assert side_name == "side"
    assert model_ir.tensors[side_name].shape == _SIDE_NHWC
    side_nhwc = np.asarray(model_ir.tensors[side_name].data)
    new_operands = [np.maximum(conv_nhwc, 0.0), side_nhwc]
    if int(main_input_index) == 1:
        new_operands.reverse()
    actual = _binary(binary_type, new_operands[0], new_operands[1])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert model_ir.tensors["tail_nhwc"].shape == _NHWC


def test_indexed_conv_output_passthrough_groups_shared_constant_updates() -> None:
    model_ir = _make_model(
        binary_type="MUL",
        shared_constant=True,
        repeated_binary=True,
    )
    original = np.asarray(model_ir.tensors["side"].data).copy()

    stats = optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)

    assert stats["rewritten_transposeconv_output_nhwc_passthrough_chains"] == 1
    assert model_ir.tensors["side"].shape == _SIDE_NCHW
    np.testing.assert_array_equal(model_ir.tensors["side"].data, original)
    assert model_ir.tensors["side_nhwc"].shape == _SIDE_NHWC
    chain_binary = [
        operator
        for operator in model_ir.operators
        if str(operator.op_type) in {"MUL", "ADD"}
        and "legacy_nchw" not in {str(value) for value in operator.inputs}
    ]
    assert len(chain_binary) == 2
    assert all("side_nhwc" in operator.inputs for operator in chain_binary)
    legacy = next(
        operator
        for operator in model_ir.operators
        if "legacy_nchw" in {str(value) for value in operator.inputs}
    )
    assert list(legacy.inputs) == ["legacy_nchw", "side"]


def test_indexed_conv_output_passthrough_supports_dynamic_signatures() -> None:
    model_ir = _make_model(binary_type="ADD", dynamic=True)

    stats = optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)

    assert stats["rewritten_transposeconv_output_nhwc_passthrough_chains"] == 1
    assert model_ir.tensors["tail_nhwc"].shape_signature == [1, -1, -1, _C]
    assert model_ir.tensors["side"].shape_signature == _SIDE_NHWC


def test_indexed_conv_output_passthrough_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    candidate = model_ir.operators[1]
    index = ModelIRGraphIndex(model_ir)

    assert optimize_transposeconv_output_nhwc_passthrough_chains(
        model_ir,
        graph_index=index,
        candidate=candidate,
        max_rewrites=0,
    ) == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 0}
    assert optimize_transposeconv_output_nhwc_passthrough_chains(
        model_ir,
        graph_index=index,
        candidate=model_ir.operators[-1],
    ) == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 0}
    assert optimize_transposeconv_output_nhwc_passthrough_chains(
        model_ir,
        graph_index=index,
        candidate=candidate,
    ) == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 1}
    assert optimize_transposeconv_output_nhwc_passthrough_chains(
        model_ir,
        graph_index=index,
    ) == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 0}
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []


UnsafeMutation = Callable[[ModelIR], None]


def _wrong_pre_perm(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _untyped_pre_perm(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].dtype = "FLOAT32"


def _public_pre_tensor(model_ir: ModelIR) -> None:
    model_ir.outputs = ["conv_nchw"]


def _unsupported_source(model_ir: ModelIR) -> None:
    model_ir.operators[0].op_type = "AVERAGE_POOL_2D"


def _late_source(model_ir: ModelIR) -> None:
    model_ir.operators.insert(2, model_ir.operators.pop(0))


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["fanout"] = _tensor("fanout", _NCHW)
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["conv_nchw"], outputs=["fanout"])
    )


def _missing_pre_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["conv_nchw"]


def _public_chain_tensor(model_ir: ModelIR) -> None:
    model_ir.outputs = ["unary_nchw"]


def _per_axis_chain_tensor(model_ir: ModelIR) -> None:
    model_ir.tensors["unary_nchw"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _runtime_binary_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].data = None
    model_ir.inputs.append("side")


def _variable_binary_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].is_variable = True


def _producer_backed_binary_side(model_ir: ModelIR) -> None:
    model_ir.tensors["side_source"] = _tensor(
        "side_source",
        _SIDE_NCHW,
        data=np.ones(_SIDE_NCHW, dtype=np.float32),
    )
    model_ir.operators.insert(
        3,
        OperatorIR(op_type="RELU", inputs=["side_source"], outputs=["side"]),
    )


def _invalid_binary_data_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].data = np.ones([1, 1, _C, 1], dtype=np.float32)


def _invalid_binary_broadcast(model_ir: ModelIR) -> None:
    model_ir.tensors["side"].shape = [1, 3, 1, 1]
    model_ir.tensors["side"].shape_signature = [1, 3, 1, 1]
    model_ir.tensors["side"].data = np.ones([1, 3, 1, 1], dtype=np.float32)


def _wrong_binary_output_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["binary_nchw"].shape = [1, _C, _H, _W + 1]
    model_ir.tensors["binary_nchw"].shape_signature = [1, _C, _H, _W + 1]


def _wrong_post_perm(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nhwc"].data = np.asarray([0, 3, 1, 2], dtype=np.int64)


def _public_post_output(model_ir: ModelIR) -> None:
    model_ir.outputs = ["tail_nhwc"]


def _missing_post_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["tail_nhwc"]


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["x"], outputs=["conv_nhwc"]),
    )


def _early_post_consumer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(2, model_ir.operators.pop(-1))


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_pre_perm,
        _untyped_pre_perm,
        _public_pre_tensor,
        _unsupported_source,
        _late_source,
        _pre_fanout,
        _missing_pre_tensor,
        _public_chain_tensor,
        _per_axis_chain_tensor,
        _runtime_binary_side,
        _variable_binary_side,
        _producer_backed_binary_side,
        _invalid_binary_data_shape,
        _invalid_binary_broadcast,
        _wrong_binary_output_shape,
        _wrong_post_perm,
        _public_post_output,
        _missing_post_tensor,
        _duplicate_producer,
        _early_post_consumer,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_conv_output_passthrough_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model(binary_type="ADD")
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))

    stats = optimize_transposeconv_output_nhwc_passthrough_chains(model_ir)

    assert stats == {"rewritten_transposeconv_output_nhwc_passthrough_chains": 0}
    assert _snapshot(model_ir) == before
