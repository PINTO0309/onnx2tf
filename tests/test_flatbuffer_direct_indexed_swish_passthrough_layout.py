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
    LOGICAL_LAYOUT_NCW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_NWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.swish_passthrough_layout import (
    optimize_swish_transpose_passthrough_chains,
)


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
    post_count: int = 2,
    reverse_multiply: bool = False,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    rank3: bool = False,
    legacy_consumer: bool = False,
    multiply_output_public: bool = False,
    public_post_index: int | None = None,
    shared_post_permutation: bool = False,
    duplicate_alias_consumer: bool = False,
) -> ModelIR:
    numpy_integer_dtype = np.int64 if integer_dtype == "INT64" else np.int32
    if rank3:
        source_shape = [1, 5, 4]
        transposed_shape = [1, 4, 5]
        source_signature = [1, -1, 4] if dynamic else list(source_shape)
        transposed_signature = [1, 4, -1] if dynamic else list(transposed_shape)
        pre_permutation = [0, 2, 1]
        post_permutation = [0, 2, 1]
        source_layout = LOGICAL_LAYOUT_NWC
        transposed_layout = LOGICAL_LAYOUT_NCW
    else:
        source_shape = [1, 3, 5, 4]
        transposed_shape = [1, 4, 3, 5]
        source_signature = [1, -1, -1, 4] if dynamic else list(source_shape)
        transposed_signature = [1, 4, -1, -1] if dynamic else list(transposed_shape)
        pre_permutation = [0, 3, 1, 2]
        post_permutation = [0, 2, 3, 1]
        source_layout = LOGICAL_LAYOUT_NHWC
        transposed_layout = LOGICAL_LAYOUT_NCHW

    model_ir = ModelIR("indexed_swish_passthrough")
    model_ir.inputs = ["source"]
    model_ir.tensors = {
        "source": _tensor(
            "source",
            source_shape,
            signature=source_signature,
            layout=source_layout,
        ),
        "to_transposed": _tensor(
            "to_transposed",
            [len(pre_permutation)],
            data=np.asarray(pre_permutation, dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "to_source": _tensor(
            "to_source",
            [len(post_permutation)],
            data=np.asarray(post_permutation, dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "pre_output": _tensor(
            "pre_output",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        ),
        "logistic_output": _tensor(
            "logistic_output",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        ),
        "multiply_output": _tensor(
            "multiply_output",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        ),
    }
    multiply_inputs = ["pre_output", "logistic_output"]
    if reverse_multiply:
        multiply_inputs.reverse()
    operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["source", "to_transposed"],
            outputs=["pre_output"],
        ),
        OperatorIR(
            op_type="LOGISTIC",
            inputs=["pre_output"],
            outputs=["logistic_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=multiply_inputs,
            outputs=["multiply_output"],
            options={"fusedActivationFunction": "NONE"},
        ),
    ]
    for index in range(post_count):
        post_name = f"post_{index}"
        model_ir.tensors[post_name] = _tensor(
            post_name,
            source_shape,
            signature=source_signature,
            layout=source_layout,
        )
        operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["multiply_output", "to_source"],
                outputs=[post_name],
                onnx_node_name=f"post_{index}_adapter",
            )
        )
        if public_post_index == index:
            model_ir.outputs.append(post_name)
            continue
        sink_name = f"sink_{index}"
        model_ir.tensors[sink_name] = _tensor(
            sink_name,
            source_shape,
            signature=source_signature,
            layout=source_layout,
        )
        model_ir.outputs.append(sink_name)
        sink_inputs = [post_name, post_name] if duplicate_alias_consumer else [post_name]
        operators.append(
            OperatorIR(
                op_type="ADD" if duplicate_alias_consumer else "RELU",
                inputs=sink_inputs,
                outputs=[sink_name],
            )
        )
    if legacy_consumer:
        model_ir.tensors["legacy_sink"] = _tensor(
            "legacy_sink",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        )
        model_ir.outputs.append("legacy_sink")
        operators.append(
            OperatorIR(
                op_type="RELU6",
                inputs=["multiply_output"],
                outputs=["legacy_sink"],
            )
        )
    if multiply_output_public:
        model_ir.outputs.append("multiply_output")
    if shared_post_permutation:
        model_ir.inputs.append("legacy_transposed")
        model_ir.tensors["legacy_transposed"] = _tensor(
            "legacy_transposed",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        )
        model_ir.tensors["legacy_source"] = _tensor(
            "legacy_source",
            source_shape,
            signature=source_signature,
            layout=source_layout,
        )
        model_ir.outputs.append("legacy_source")
        operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["legacy_transposed", "to_source"],
                outputs=["legacy_source"],
            )
        )
    model_ir.operators = operators
    return model_ir


def _snapshot(model_ir: ModelIR) -> bytes:
    return pickle.dumps(model_ir, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("post_count", [1, 2])
@pytest.mark.parametrize("reverse_multiply", [False, True])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("integer_dtype", ["INT32", "INT64"])
def test_indexed_swish_passthrough_preserves_source_layout_boundary(
    post_count: int,
    reverse_multiply: bool,
    dynamic: bool,
    integer_dtype: str,
) -> None:
    model_ir = _make_model(
        post_count=post_count,
        reverse_multiply=reverse_multiply,
        dynamic=dynamic,
        integer_dtype=integer_dtype,
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_swish_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_swish_transpose_passthrough_chains": 1}
    assert all(str(operator.op_type) != "TRANSPOSE" for operator in model_ir.operators)
    logistic = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "LOGISTIC"
    )
    multiply = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "MUL"
    )
    assert list(logistic.inputs) == ["source"]
    assert "source" in list(multiply.inputs)
    assert list(multiply.outputs) == ["post_0"]
    for operator in model_ir.operators:
        if list(operator.outputs) == ["sink_1"]:
            assert list(operator.inputs) == ["post_0"]
    assert model_ir.tensors["logistic_output"].shape == model_ir.tensors["source"].shape
    assert model_ir.tensors["logistic_output"].shape_signature == model_ir.tensors[
        "source"
    ].shape_signature
    assert model_ir.tensors["logistic_output"].physical_layout == LOGICAL_LAYOUT_NHWC
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_swish_passthrough_supports_rank3_channel_permutation() -> None:
    model_ir = _make_model(rank3=True, dynamic=True)

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    assert model_ir.tensors["logistic_output"].shape == [1, 5, 4]
    assert model_ir.tensors["logistic_output"].shape_signature == [1, -1, 4]
    assert model_ir.tensors["logistic_output"].physical_layout == LOGICAL_LAYOUT_NWC


def test_indexed_swish_passthrough_accepts_immutable_constant_source() -> None:
    model_ir = _make_model()
    model_ir.inputs.remove("source")
    model_ir.tensors["source"].data = np.arange(
        1 * 3 * 5 * 4,
        dtype=np.float32,
    ).reshape(1, 3, 5, 4)

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    logistic = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "LOGISTIC"
    )
    assert list(logistic.inputs) == ["source"]
    np.testing.assert_array_equal(
        model_ir.tensors["source"].data,
        np.arange(1 * 3 * 5 * 4, dtype=np.float32).reshape(1, 3, 5, 4),
    )


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((1, 3, 5, 3), dtype=np.float32),
        np.zeros((1, 3, 5, 4), dtype=np.float16),
    ],
    ids=["shape_mismatch", "dtype_mismatch"],
)
def test_indexed_swish_passthrough_rejects_inconsistent_constant_source(
    data: np.ndarray,
) -> None:
    model_ir = _make_model()
    model_ir.inputs.remove("source")
    model_ir.tensors["source"].data = data
    before = _snapshot(copy.deepcopy(model_ir))

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats == {"rewritten_swish_transpose_passthrough_chains": 0}
    assert _snapshot(model_ir) == before


def test_indexed_swish_passthrough_accepts_view_proven_unknown_source_layout() -> None:
    model_ir = _make_model()
    model_ir.tensors["source"].logical_layout = "UNKNOWN"
    model_ir.tensors["source"].physical_layout = "UNKNOWN"
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_swish_transpose_passthrough_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    assert model_ir.tensors["logistic_output"].logical_layout == "UNKNOWN"
    assert model_ir.tensors["post_0"].physical_layout == "UNKNOWN"
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize(
    ("legacy_consumer", "multiply_output_public"),
    [(True, False), (False, True), (True, True)],
)
def test_indexed_swish_passthrough_preserves_legacy_transposed_boundary(
    legacy_consumer: bool,
    multiply_output_public: bool,
) -> None:
    model_ir = _make_model(
        legacy_consumer=legacy_consumer,
        multiply_output_public=multiply_output_public,
    )

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    transposes = [
        operator for operator in model_ir.operators if str(operator.op_type) == "TRANSPOSE"
    ]
    assert len(transposes) == 1
    assert list(transposes[0].inputs) == ["post_0", "to_transposed"]
    assert list(transposes[0].outputs) == ["multiply_output"]
    multiply_index = model_ir.operators.index(
        next(operator for operator in model_ir.operators if str(operator.op_type) == "MUL")
    )
    assert model_ir.operators[multiply_index + 1] is transposes[0]
    if legacy_consumer:
        legacy = next(
            operator for operator in model_ir.operators if list(operator.outputs) == ["legacy_sink"]
        )
        assert list(legacy.inputs) == ["multiply_output"]


def test_indexed_swish_passthrough_preserves_shared_post_permutation() -> None:
    model_ir = _make_model(
        legacy_consumer=True,
        shared_post_permutation=True,
    )
    before = np.asarray(model_ir.tensors["to_source"].data).copy()

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    np.testing.assert_array_equal(model_ir.tensors["to_source"].data, before)
    unrelated = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["legacy_source"]
    )
    assert list(unrelated.inputs) == ["legacy_transposed", "to_source"]


def test_indexed_swish_passthrough_selects_public_post_alias() -> None:
    model_ir = _make_model(public_post_index=1)

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    multiply = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "MUL"
    )
    assert list(multiply.outputs) == ["post_1"]
    sink = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["sink_0"]
    )
    assert list(sink.inputs) == ["post_1"]
    assert "post_1" in model_ir.outputs


def test_indexed_swish_passthrough_groups_repeated_alias_slots() -> None:
    model_ir = _make_model(duplicate_alias_consumer=True)

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    sink = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["sink_1"]
    )
    assert list(sink.inputs) == ["post_0", "post_0"]


def test_indexed_swish_passthrough_preserves_numerical_semantics() -> None:
    model_ir = _make_model(reverse_multiply=True)
    rng = np.random.default_rng(97)
    source = rng.normal(size=(1, 3, 5, 4)).astype(np.float32)
    transposed = np.transpose(source, (0, 3, 1, 2))
    expected = np.transpose(
        transposed / (1.0 + np.exp(-transposed)),
        (0, 2, 3, 1),
    )

    stats = optimize_swish_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_swish_transpose_passthrough_chains"] == 1
    actual = source / (1.0 + np.exp(-source))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_indexed_swish_passthrough_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    candidate = model_ir.operators[0]
    graph_index = ModelIRGraphIndex(model_ir)

    assert optimize_swish_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        candidate=candidate,
        max_rewrites=0,
    ) == {"rewritten_swish_transpose_passthrough_chains": 0}
    assert optimize_swish_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        candidate=candidate,
    ) == {"rewritten_swish_transpose_passthrough_chains": 1}
    assert optimize_swish_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
    ) == {"rewritten_swish_transpose_passthrough_chains": 0}


UnsafeMutation = Callable[[ModelIR], None]


def _invalid_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_transposed"].data = np.asarray([0, 3, 1, 1], dtype=np.int32)


def _variable_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_transposed"].is_variable = True


def _public_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.inputs.append("to_transposed")


def _unresolved_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("source")


def _variable_source(model_ir: ModelIR) -> None:
    model_ir.tensors["source"].is_variable = True


def _public_pre_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("pre_output")


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_side"] = _tensor("pre_side", [1, 4, 3, 5])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["pre_output"], outputs=["pre_side"])
    )


def _pre_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_output"].shape[1] -= 1


def _pre_quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_output"].quantization = QuantParamIR(
        scale=[0.5], zero_point=[0]
    )


def _public_logistic_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("logistic_output")


def _logistic_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["logistic_side"] = _tensor("logistic_side", [1, 4, 3, 5])
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["logistic_output"],
            outputs=["logistic_side"],
        )
    )


def _logistic_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["logistic_output"].shape[2] -= 1


def _wrong_multiply_inputs(model_ir: ModelIR) -> None:
    model_ir.operators[2].inputs[1] = "source"


def _multiply_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["multiply_output"].shape[3] -= 1


def _multiply_per_axis_quantization(model_ir: ModelIR) -> None:
    model_ir.tensors["multiply_output"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _no_inverse_post(model_ir: ModelIR) -> None:
    for operator in model_ir.operators:
        if str(operator.op_type) == "TRANSPOSE" and str(operator.inputs[0]) == "multiply_output":
            operator.op_type = "RESHAPE"


def _wrong_post_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_source"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _post_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].shape[-1] -= 1


def _post_dtype_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].dtype = "FLOAT16"


def _post_quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].quantization = QuantParamIR(
        scale=[0.5], zero_point=[0]
    )


def _multiple_public_post_aliases(model_ir: ModelIR) -> None:
    model_ir.outputs.extend(["post_0", "post_1"])


def _public_post_input(model_ir: ModelIR) -> None:
    model_ir.inputs.append("post_0")


def _missing_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["logistic_output"]


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["source"], outputs=["pre_output"]),
    )


def _variable_post_output(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].is_variable = True


def _contradictory_post_layout(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].logical_layout = LOGICAL_LAYOUT_NCHW
    model_ir.tensors["post_0"].physical_layout = LOGICAL_LAYOUT_NCHW


def _stale_post_consumer_order(model_ir: ModelIR) -> None:
    sink = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["sink_0"]
    )
    model_ir.operators.remove(sink)
    model_ir.operators.insert(3, sink)


@pytest.mark.parametrize(
    "mutation",
    [
        _invalid_pre_permutation,
        _variable_pre_permutation,
        _public_pre_permutation,
        _unresolved_source,
        _variable_source,
        _public_pre_output,
        _pre_fanout,
        _pre_shape_mismatch,
        _pre_quantization_mismatch,
        _public_logistic_output,
        _logistic_fanout,
        _logistic_shape_mismatch,
        _wrong_multiply_inputs,
        _multiply_shape_mismatch,
        _multiply_per_axis_quantization,
        _no_inverse_post,
        _wrong_post_permutation,
        _post_shape_mismatch,
        _post_dtype_mismatch,
        _post_quantization_mismatch,
        _multiple_public_post_aliases,
        _public_post_input,
        _missing_tensor,
        _duplicate_producer,
        _variable_post_output,
        _contradictory_post_layout,
        _stale_post_consumer_order,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_swish_passthrough_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_swish_transpose_passthrough_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_swish_transpose_passthrough_chains": 0}
    assert _snapshot(model_ir) == before
    assert layout_state.validate_against_model_ir(model_ir) == []
