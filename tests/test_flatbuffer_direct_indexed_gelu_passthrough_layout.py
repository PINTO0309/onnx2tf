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
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_gelu_tanh_transpose_passthrough_chains,
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
    reverse_binary: bool = False,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    rank3: bool = False,
    legacy_consumer: bool = False,
    final_output_public: bool = False,
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

    model_ir = ModelIR("indexed_gelu_passthrough")
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
    }
    constant_values = {
        "cubic_constant": np.float32(0.044715),
        "scale_constant": np.float32(0.7978845834732056),
        "one_constant": np.float32(1.0),
        "half_constant": np.float32(0.5),
    }
    for name, value in constant_values.items():
        model_ir.tensors[name] = _tensor(
            name,
            [1],
            data=np.asarray([value], dtype=np.float32),
        )
    chain_outputs = [
        "square_output",
        "cube_output",
        "multiply_cubic_output",
        "add_residual_output",
        "multiply_scale_output",
        "tanh_output",
        "add_one_output",
        "multiply_residual_output",
        "final_output",
    ]
    for name in chain_outputs:
        model_ir.tensors[name] = _tensor(
            name,
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        )

    def _ordered(left: str, right: str) -> list[str]:
        return [right, left] if reverse_binary else [left, right]

    operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["source", "to_transposed"],
            outputs=["pre_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["pre_output", "pre_output"],
            outputs=["square_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=_ordered("square_output", "pre_output"),
            outputs=["cube_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=_ordered("cube_output", "cubic_constant"),
            outputs=["multiply_cubic_output"],
        ),
        OperatorIR(
            op_type="ADD",
            inputs=_ordered("pre_output", "multiply_cubic_output"),
            outputs=["add_residual_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=_ordered("add_residual_output", "scale_constant"),
            outputs=["multiply_scale_output"],
        ),
        OperatorIR(
            op_type="TANH",
            inputs=["multiply_scale_output"],
            outputs=["tanh_output"],
        ),
        OperatorIR(
            op_type="ADD",
            inputs=_ordered("tanh_output", "one_constant"),
            outputs=["add_one_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=_ordered("pre_output", "add_one_output"),
            outputs=["multiply_residual_output"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=_ordered("multiply_residual_output", "half_constant"),
            outputs=["final_output"],
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
                inputs=["final_output", "to_source"],
                outputs=[post_name],
                onnx_node_name=f"gelu_post_{index}",
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
                inputs=["final_output"],
                outputs=["legacy_sink"],
            )
        )
    if final_output_public:
        model_ir.outputs.append("final_output")
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
@pytest.mark.parametrize("reverse_binary", [False, True])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("integer_dtype", ["INT32", "INT64"])
def test_indexed_gelu_passthrough_preserves_source_layout_boundary(
    post_count: int,
    reverse_binary: bool,
    dynamic: bool,
    integer_dtype: str,
) -> None:
    model_ir = _make_model(
        post_count=post_count,
        reverse_binary=reverse_binary,
        dynamic=dynamic,
        integer_dtype=integer_dtype,
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_gelu_tanh_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_gelu_tanh_transpose_passthrough_chains": 1}
    assert all(str(operator.op_type) != "TRANSPOSE" for operator in model_ir.operators)
    final = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["post_0"]
    )
    assert str(final.op_type) == "MUL"
    for operator in model_ir.operators:
        if str(operator.op_type) in {"MUL", "ADD"}:
            assert "pre_output" not in list(operator.inputs)
    for name in [
        "square_output",
        "cube_output",
        "multiply_cubic_output",
        "add_residual_output",
        "multiply_scale_output",
        "tanh_output",
        "add_one_output",
        "multiply_residual_output",
        "post_0",
    ]:
        tensor = model_ir.tensors[name]
        assert tensor.shape == model_ir.tensors["source"].shape
        assert tensor.shape_signature == model_ir.tensors["source"].shape_signature
        assert tensor.physical_layout == LOGICAL_LAYOUT_NHWC
    if post_count == 2:
        sink = next(
            operator for operator in model_ir.operators if list(operator.outputs) == ["sink_1"]
        )
        assert list(sink.inputs) == ["post_0"]
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_gelu_passthrough_supports_rank3_dynamic_view() -> None:
    model_ir = _make_model(rank3=True, dynamic=True)

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    assert model_ir.tensors["tanh_output"].shape == [1, 5, 4]
    assert model_ir.tensors["tanh_output"].shape_signature == [1, -1, 4]
    assert model_ir.tensors["tanh_output"].physical_layout == LOGICAL_LAYOUT_NWC


def test_indexed_gelu_passthrough_accepts_immutable_constant_source() -> None:
    model_ir = _make_model()
    source = np.linspace(-2.0, 2.0, 1 * 3 * 5 * 4, dtype=np.float32).reshape(
        1, 3, 5, 4
    )
    model_ir.inputs.remove("source")
    model_ir.tensors["source"].data = source.copy()

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    np.testing.assert_array_equal(model_ir.tensors["source"].data, source)


@pytest.mark.parametrize(
    ("legacy_consumer", "final_output_public"),
    [(True, False), (False, True), (True, True)],
)
def test_indexed_gelu_passthrough_preserves_legacy_transposed_boundary(
    legacy_consumer: bool,
    final_output_public: bool,
) -> None:
    model_ir = _make_model(
        legacy_consumer=legacy_consumer,
        final_output_public=final_output_public,
    )

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    transposes = [
        operator for operator in model_ir.operators if str(operator.op_type) == "TRANSPOSE"
    ]
    assert len(transposes) == 1
    assert list(transposes[0].inputs) == ["post_0", "to_transposed"]
    assert list(transposes[0].outputs) == ["final_output"]
    final = next(operator for operator in model_ir.operators if list(operator.outputs) == ["post_0"])
    final_index = model_ir.operators.index(final)
    assert model_ir.operators[final_index + 1] is transposes[0]


def test_indexed_gelu_passthrough_preserves_shared_post_permutation() -> None:
    model_ir = _make_model(
        legacy_consumer=True,
        shared_post_permutation=True,
    )
    before = np.asarray(model_ir.tensors["to_source"].data).copy()

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    np.testing.assert_array_equal(model_ir.tensors["to_source"].data, before)
    unrelated = next(
        operator for operator in model_ir.operators if list(operator.outputs) == ["legacy_source"]
    )
    assert list(unrelated.inputs) == ["legacy_transposed", "to_source"]


def test_indexed_gelu_passthrough_selects_public_post_alias() -> None:
    model_ir = _make_model(public_post_index=1)

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    final = next(operator for operator in model_ir.operators if str(operator.op_type) == "MUL" and "half_constant" in operator.inputs)
    assert list(final.outputs) == ["post_1"]
    assert "post_1" in model_ir.outputs


def test_indexed_gelu_passthrough_groups_repeated_alias_slots() -> None:
    model_ir = _make_model(duplicate_alias_consumer=True)

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    sink = next(operator for operator in model_ir.operators if list(operator.outputs) == ["sink_1"])
    assert list(sink.inputs) == ["post_0", "post_0"]


def test_indexed_gelu_passthrough_preserves_numerical_semantics() -> None:
    model_ir = _make_model(reverse_binary=True)
    rng = np.random.default_rng(101)
    source = rng.normal(size=(1, 3, 5, 4)).astype(np.float32)
    transposed = np.transpose(source, (0, 3, 1, 2))

    def _gelu(value: np.ndarray) -> np.ndarray:
        return np.float32(0.5) * value * (
            np.float32(1.0)
            + np.tanh(
                np.float32(0.7978845834732056)
                * (value + np.float32(0.044715) * value * value * value)
            )
        )

    expected = np.transpose(_gelu(transposed), (0, 2, 3, 1))

    stats = optimize_gelu_tanh_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_gelu_tanh_transpose_passthrough_chains"] == 1
    np.testing.assert_allclose(_gelu(source), expected, rtol=0.0, atol=0.0)


def test_indexed_gelu_passthrough_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    candidate = model_ir.operators[0]
    graph_index = ModelIRGraphIndex(model_ir)

    assert optimize_gelu_tanh_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        candidate=candidate,
        max_rewrites=0,
    ) == {"rewritten_gelu_tanh_transpose_passthrough_chains": 0}
    assert optimize_gelu_tanh_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        candidate=candidate,
    ) == {"rewritten_gelu_tanh_transpose_passthrough_chains": 1}
    assert optimize_gelu_tanh_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
    ) == {"rewritten_gelu_tanh_transpose_passthrough_chains": 0}


UnsafeMutation = Callable[[ModelIR], None]


def _invalid_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_transposed"].data = np.asarray([0, 3, 1, 1], dtype=np.int32)


def _variable_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_transposed"].is_variable = True


def _unresolved_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("source")


def _inconsistent_constant_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("source")
    model_ir.tensors["source"].data = np.zeros((1, 3, 5, 3), dtype=np.float32)


def _public_pre_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("pre_output")


def _extra_pre_consumer(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_side"] = _tensor("pre_side", [1, 4, 3, 5])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["pre_output"], outputs=["pre_side"])
    )


def _wrong_square_inputs(model_ir: ModelIR) -> None:
    model_ir.operators[1].inputs[1] = "cubic_constant"


def _square_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["square_side"] = _tensor("square_side", [1, 4, 3, 5])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["square_output"], outputs=["square_side"])
    )


def _wrong_cube_inputs(model_ir: ModelIR) -> None:
    model_ir.operators[2].inputs[1] = "cubic_constant"


def _nonsingleton_cubic_constant(model_ir: ModelIR) -> None:
    tensor = model_ir.tensors["cubic_constant"]
    tensor.data = np.asarray([0.044715, 0.044715], dtype=np.float32)
    tensor.shape = [2]
    tensor.shape_signature = [2]


def _wrong_add_residual_inputs(model_ir: ModelIR) -> None:
    model_ir.operators[4].inputs[0] = "scale_constant"


def _variable_scale_constant(model_ir: ModelIR) -> None:
    model_ir.tensors["scale_constant"].is_variable = True


def _wrong_tanh_type(model_ir: ModelIR) -> None:
    model_ir.operators[6].op_type = "LOGISTIC"


def _tanh_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["tanh_side"] = _tensor("tanh_side", [1, 4, 3, 5])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["tanh_output"], outputs=["tanh_side"])
    )


def _one_constant_dtype_mismatch(model_ir: ModelIR) -> None:
    tensor = model_ir.tensors["one_constant"]
    tensor.dtype = "FLOAT16"
    tensor.data = np.asarray([1.0], dtype=np.float16)


def _wrong_multiply_residual_inputs(model_ir: ModelIR) -> None:
    model_ir.operators[8].inputs[0] = "half_constant"


def _produced_half_constant(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["source"], outputs=["half_constant"]),
    )


def _public_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("multiply_scale_output")


def _intermediate_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["add_one_output"].shape[1] -= 1


def _intermediate_per_axis_quantization(model_ir: ModelIR) -> None:
    model_ir.tensors["multiply_residual_output"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _no_inverse_post(model_ir: ModelIR) -> None:
    for operator in model_ir.operators:
        if str(operator.op_type) == "TRANSPOSE" and str(operator.inputs[0]) == "final_output":
            operator.op_type = "RESHAPE"


def _wrong_post_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_source"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _post_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].shape[-1] -= 1


def _post_quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].quantization = QuantParamIR(
        scale=[0.5], zero_point=[0]
    )


def _multiple_public_post_aliases(model_ir: ModelIR) -> None:
    model_ir.outputs.extend(["post_0", "post_1"])


def _public_post_input(model_ir: ModelIR) -> None:
    model_ir.inputs.append("post_0")


def _missing_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["tanh_output"]


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["source"], outputs=["pre_output"]),
    )


def _stale_post_consumer_order(model_ir: ModelIR) -> None:
    sink = next(operator for operator in model_ir.operators if list(operator.outputs) == ["sink_0"])
    model_ir.operators.remove(sink)
    model_ir.operators.insert(10, sink)


def _contradictory_post_layout(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].logical_layout = LOGICAL_LAYOUT_NCHW
    model_ir.tensors["post_0"].physical_layout = LOGICAL_LAYOUT_NCHW


@pytest.mark.parametrize(
    "mutation",
    [
        _invalid_pre_permutation,
        _variable_pre_permutation,
        _unresolved_source,
        _inconsistent_constant_source,
        _public_pre_output,
        _extra_pre_consumer,
        _wrong_square_inputs,
        _square_fanout,
        _wrong_cube_inputs,
        _nonsingleton_cubic_constant,
        _wrong_add_residual_inputs,
        _variable_scale_constant,
        _wrong_tanh_type,
        _tanh_fanout,
        _one_constant_dtype_mismatch,
        _wrong_multiply_residual_inputs,
        _produced_half_constant,
        _public_intermediate,
        _intermediate_shape_mismatch,
        _intermediate_per_axis_quantization,
        _no_inverse_post,
        _wrong_post_permutation,
        _post_shape_mismatch,
        _post_quantization_mismatch,
        _multiple_public_post_aliases,
        _public_post_input,
        _missing_tensor,
        _duplicate_producer,
        _stale_post_consumer_order,
        _contradictory_post_layout,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_gelu_passthrough_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_gelu_tanh_transpose_passthrough_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_gelu_tanh_transpose_passthrough_chains": 0}
    assert _snapshot(model_ir) == before
    assert layout_state.validate_against_model_ir(model_ir) == []
