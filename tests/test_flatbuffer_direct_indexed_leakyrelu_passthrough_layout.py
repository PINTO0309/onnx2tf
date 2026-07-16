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
from onnx2tf.tflite_builder.passes.leakyrelu_passthrough_layout import (
    optimize_leakyrelu_transpose_passthrough,
    optimize_leakyrelu_transpose_passthrough_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=False,
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
    final_output_public: bool = False,
    public_post_index: int | None = None,
    shared_post_permutation: bool = False,
    repeated_alias_slots: bool = False,
    per_tensor_quantized: bool = False,
) -> ModelIR:
    integer_numpy_dtype = np.int64 if integer_dtype == "INT64" else np.int32
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
    quantization = (
        QuantParamIR(scale=[0.125], zero_point=[0])
        if per_tensor_quantized
        else None
    )
    model_ir = ModelIR("indexed_leakyrelu_passthrough")
    model_ir.inputs = ["source"]
    model_ir.tensors = {
        "source": _tensor(
            "source",
            source_shape,
            signature=source_signature,
            layout=source_layout,
            quantization=copy.deepcopy(quantization),
        ),
        "to_transposed": _tensor(
            "to_transposed",
            [len(pre_permutation)],
            dtype=integer_dtype,
            data=np.asarray(pre_permutation, dtype=integer_numpy_dtype),
        ),
        "to_source": _tensor(
            "to_source",
            [len(post_permutation)],
            dtype=integer_dtype,
            data=np.asarray(post_permutation, dtype=integer_numpy_dtype),
        ),
        "pre_output": _tensor(
            "pre_output",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
            quantization=copy.deepcopy(quantization),
        ),
        "alpha": _tensor(
            "alpha",
            [1],
            data=np.asarray([0.2], dtype=np.float32),
            quantization=copy.deepcopy(quantization),
        ),
    }
    for name in (
        "negative_output",
        "negative_relu_output",
        "multiply_output",
        "positive_output",
        "subtract_output",
    ):
        model_ir.tensors[name] = _tensor(
            name,
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
            quantization=copy.deepcopy(quantization),
        )
    multiply_inputs = (
        ["alpha", "negative_relu_output"]
        if reverse_multiply
        else ["negative_relu_output", "alpha"]
    )
    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["source", "to_transposed"],
            ["pre_output"],
        ),
        OperatorIR("NEG", ["pre_output"], ["negative_output"]),
        OperatorIR("RELU", ["negative_output"], ["negative_relu_output"]),
        OperatorIR("MUL", multiply_inputs, ["multiply_output"]),
        OperatorIR("RELU", ["pre_output"], ["positive_output"]),
        OperatorIR(
            "SUB",
            ["positive_output", "multiply_output"],
            ["subtract_output"],
        ),
    ]
    for index in range(post_count):
        post_name = f"post_{index}"
        model_ir.tensors[post_name] = _tensor(
            post_name,
            source_shape,
            signature=source_signature,
            layout=source_layout,
            quantization=copy.deepcopy(quantization),
        )
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE",
                ["subtract_output", "to_source"],
                [post_name],
            )
        )
    if repeated_alias_slots and post_count > 1:
        model_ir.tensors["combined_sink"] = _tensor(
            "combined_sink",
            source_shape,
            signature=source_signature,
            layout=source_layout,
            quantization=copy.deepcopy(quantization),
        )
        model_ir.operators.append(
            OperatorIR("ADD", ["post_1", "post_1"], ["combined_sink"])
        )
        model_ir.outputs.append("combined_sink")
    for index in range(post_count):
        post_name = f"post_{index}"
        if public_post_index == index:
            model_ir.outputs.append(post_name)
            continue
        if repeated_alias_slots and index == 1:
            continue
        sink_name = f"sink_{index}"
        model_ir.tensors[sink_name] = _tensor(
            sink_name,
            source_shape,
            signature=source_signature,
            layout=source_layout,
            quantization=copy.deepcopy(quantization),
        )
        model_ir.operators.append(OperatorIR("ABS", [post_name], [sink_name]))
        model_ir.outputs.append(sink_name)
    if legacy_consumer:
        model_ir.tensors["legacy_sink"] = _tensor(
            "legacy_sink",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
            quantization=copy.deepcopy(quantization),
        )
        model_ir.operators.append(
            OperatorIR("ABS", ["subtract_output"], ["legacy_sink"])
        )
        model_ir.outputs.append("legacy_sink")
    if final_output_public:
        model_ir.outputs.append("subtract_output")
    if shared_post_permutation:
        model_ir.inputs.append("peer_transposed")
        model_ir.tensors["peer_transposed"] = _tensor(
            "peer_transposed",
            transposed_shape,
            signature=transposed_signature,
            layout=transposed_layout,
        )
        model_ir.tensors["peer_source"] = _tensor(
            "peer_source",
            source_shape,
            signature=source_signature,
            layout=source_layout,
        )
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE",
                ["peer_transposed", "to_source"],
                ["peer_source"],
            )
        )
        model_ir.outputs.append("peer_source")
    return model_ir


@pytest.mark.parametrize("post_count", [1, 2])
@pytest.mark.parametrize("reverse_multiply", [False, True], ids=["data_first", "const_first"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("integer_dtype", ["INT32", "INT64"])
def test_indexed_leakyrelu_passthrough_and_fusion_preserve_source_contract(
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
        per_tensor_quantized=True,
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_leakyrelu_transpose_passthrough_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "rewritten_leakyrelu_transpose_passthrough_chains": 1,
        "fused_pseudo_leakyrelu_chains": 1,
    }
    leaky = next(op for op in model_ir.operators if op.op_type == "LEAKY_RELU")
    assert leaky.inputs == ["source"]
    assert leaky.outputs == ["post_0"]
    assert leaky.options == {"alpha": pytest.approx(0.2)}
    for index in range(1, post_count):
        sink = next(op for op in model_ir.operators if op.outputs == [f"sink_{index}"])
        assert sink.inputs == ["post_0"]
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_passthrough_supports_rank3_and_immutable_constant_source() -> None:
    model_ir = _make_model(rank3=True, dynamic=True)
    model_ir.inputs.remove("source")
    model_ir.tensors["source"].data = np.arange(
        20,
        dtype=np.float32,
    ).reshape(1, 5, 4)
    original = np.asarray(model_ir.tensors["source"].data).copy()

    stats = optimize_leakyrelu_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_leakyrelu_transpose_passthrough_chains"] == 1
    assert stats["fused_pseudo_leakyrelu_chains"] == 1
    np.testing.assert_array_equal(model_ir.tensors["source"].data, original)
    assert model_ir.tensors["post_0"].shape == [1, 5, 4]
    assert model_ir.tensors["post_0"].physical_layout == LOGICAL_LAYOUT_NWC


@pytest.mark.parametrize(
    ("legacy_consumer", "final_output_public"),
    [(True, False), (False, True), (True, True)],
)
def test_legacy_or_public_transposed_boundary_gets_local_adapter(
    legacy_consumer: bool,
    final_output_public: bool,
) -> None:
    model_ir = _make_model(
        legacy_consumer=legacy_consumer,
        final_output_public=final_output_public,
    )

    stats = optimize_leakyrelu_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_leakyrelu_transpose_passthrough_chains"] == 1
    leaky = next(op for op in model_ir.operators if op.op_type == "LEAKY_RELU")
    adapter = next(
        op
        for op in model_ir.operators
        if op.op_type == "TRANSPOSE" and op.outputs == ["subtract_output"]
    )
    assert model_ir.operators.index(adapter) == model_ir.operators.index(leaky) + 1
    assert adapter.inputs == ["post_0", "to_transposed"]
    if legacy_consumer:
        legacy = next(op for op in model_ir.operators if op.outputs == ["legacy_sink"])
        assert legacy.inputs == ["subtract_output"]


def test_public_post_is_representative_and_repeated_alias_slots_are_exact() -> None:
    model_ir = _make_model(
        public_post_index=1,
        repeated_alias_slots=True,
    )

    stats = optimize_leakyrelu_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_leakyrelu_transpose_passthrough_chains"] == 1
    leaky = next(op for op in model_ir.operators if op.op_type == "LEAKY_RELU")
    combined = next(op for op in model_ir.operators if op.outputs == ["combined_sink"])
    assert leaky.outputs == ["post_1"]
    assert combined.inputs == ["post_1", "post_1"]
    assert "post_1" in model_ir.outputs


def test_shared_post_permutation_is_never_mutated() -> None:
    model_ir = _make_model(
        shared_post_permutation=True,
        legacy_consumer=True,
    )
    original = np.asarray(model_ir.tensors["to_source"].data).copy()

    stats = optimize_leakyrelu_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_leakyrelu_transpose_passthrough_chains"] == 1
    np.testing.assert_array_equal(model_ir.tensors["to_source"].data, original)
    peer = next(op for op in model_ir.operators if op.outputs == ["peer_source"])
    assert peer.inputs == ["peer_transposed", "to_source"]
    adapter = next(op for op in model_ir.operators if op.outputs == ["subtract_output"])
    assert adapter.inputs[1] == "to_transposed"


def test_passthrough_only_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    graph_index = ModelIRGraphIndex(model_ir)
    pre = model_ir.operators[0]
    wrong_candidate = next(op for op in model_ir.operators if op.outputs == ["post_0"])

    assert optimize_leakyrelu_transpose_passthrough(
        model_ir,
        graph_index=graph_index,
        candidate=wrong_candidate,
    ) == {"rewritten_leakyrelu_transpose_passthrough_chains": 0}
    assert optimize_leakyrelu_transpose_passthrough(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=0,
    ) == {"rewritten_leakyrelu_transpose_passthrough_chains": 0}
    assert optimize_leakyrelu_transpose_passthrough(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=1,
    ) == {"rewritten_leakyrelu_transpose_passthrough_chains": 1}
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert optimize_leakyrelu_transpose_passthrough(
        model_ir,
        graph_index=graph_index,
    ) == {"rewritten_leakyrelu_transpose_passthrough_chains": 0}


def test_layout_passthrough_is_numerically_equivalent() -> None:
    rng = np.random.default_rng(20260715)
    source = rng.normal(size=[1, 3, 5, 4]).astype(np.float32)
    alpha = np.float32(0.2)
    transposed = np.transpose(source, [0, 3, 1, 2])
    old = np.transpose(
        np.maximum(transposed, 0.0)
        - np.maximum(-transposed, 0.0) * alpha,
        [0, 2, 3, 1],
    )
    new = np.maximum(source, 0.0) - np.maximum(-source, 0.0) * alpha
    np.testing.assert_allclose(new, old, rtol=0.0, atol=0.0)


Mutation = Callable[[ModelIR], None]


def _add_consumer(model_ir: ModelIR, source: str, suffix: str) -> None:
    source_tensor = model_ir.tensors[source]
    output = f"{source}_{suffix}"
    model_ir.tensors[output] = copy.deepcopy(source_tensor)
    model_ir.tensors[output].name = output
    model_ir.operators.append(OperatorIR("ABS", [source], [output]))


def _wrong_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_transposed"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _public_pre_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("pre_output")


def _pre_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "pre_output", "fanout")


def _wrong_negative_root(model_ir: ModelIR) -> None:
    next(op for op in model_ir.operators if op.outputs == ["negative_output"]).op_type = "ABS"


def _wrong_positive_root(model_ir: ModelIR) -> None:
    next(op for op in model_ir.operators if op.outputs == ["positive_output"]).op_type = "RELU6"


def _negative_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "negative_output", "fanout")


def _negative_relu_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "negative_relu_output", "fanout")


def _positive_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "positive_output", "fanout")


def _multiply_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "multiply_output", "fanout")


def _mutable_alpha(model_ir: ModelIR) -> None:
    model_ir.tensors["alpha"].is_variable = True


def _wrong_alpha_dtype(model_ir: ModelIR) -> None:
    model_ir.tensors["alpha"].dtype = "FLOAT16"
    model_ir.tensors["alpha"].data = np.asarray([0.2], dtype=np.float16)


def _per_axis_alpha(model_ir: ModelIR) -> None:
    model_ir.tensors["alpha"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=0,
    )


def _reverse_subtract(model_ir: ModelIR) -> None:
    subtract = next(op for op in model_ir.operators if op.outputs == ["subtract_output"])
    subtract.inputs.reverse()


def _public_negative_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("negative_relu_output")


def _duplicate_chain_output(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        5,
        OperatorIR("ABS", ["pre_output"], ["multiply_output"]),
    )


def _subtract_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["subtract_output"].shape = [1, 3, 5, 4]


def _wrong_post_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_source"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _multiple_public_posts(model_ir: ModelIR) -> None:
    model_ir.outputs.extend(["post_0", "post_1"])


def _post_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].shape = [1, 4, 3, 5]


def _post_dtype_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].dtype = "FLOAT16"


def _post_quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].quantization = QuantParamIR(
        scale=[0.25],
        zero_point=[0],
    )


def _post_layout_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0"].physical_layout = LOGICAL_LAYOUT_NCHW


def _public_post_input(model_ir: ModelIR) -> None:
    model_ir.inputs.append("post_0")


def _post_consumer_before_producer(model_ir: ModelIR) -> None:
    sink = next(op for op in model_ir.operators if op.outputs == ["sink_0"])
    model_ir.operators.remove(sink)
    model_ir.operators.insert(6, sink)


def _unobservable_posts(model_ir: ModelIR) -> None:
    model_ir.operators = [
        op for op in model_ir.operators if not str(op.outputs[0]).startswith("sink_")
    ]
    model_ir.outputs = []


def _per_axis_source(model_ir: ModelIR) -> None:
    model_ir.tensors["source"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=3,
    )


def _stale_chain_order(model_ir: ModelIR) -> None:
    negate = next(op for op in model_ir.operators if op.outputs == ["negative_output"])
    model_ir.operators.remove(negate)
    model_ir.operators.insert(0, negate)


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_pre_permutation,
        _public_pre_output,
        _pre_fanout,
        _wrong_negative_root,
        _wrong_positive_root,
        _negative_fanout,
        _negative_relu_fanout,
        _positive_fanout,
        _multiply_fanout,
        _mutable_alpha,
        _wrong_alpha_dtype,
        _per_axis_alpha,
        _reverse_subtract,
        _public_negative_intermediate,
        _duplicate_chain_output,
        _subtract_shape_mismatch,
        _wrong_post_permutation,
        _multiple_public_posts,
        _post_shape_mismatch,
        _post_dtype_mismatch,
        _post_quantization_mismatch,
        _post_layout_mismatch,
        _public_post_input,
        _post_consumer_before_producer,
        _unobservable_posts,
        _per_axis_source,
        _stale_chain_order,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_unsafe_passthrough_candidate_is_transactional_noop(
    mutation: Mutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = pickle.dumps(model_ir)

    stats = optimize_leakyrelu_transpose_passthrough(model_ir)

    assert stats == {"rewritten_leakyrelu_transpose_passthrough_chains": 0}
    assert pickle.dumps(model_ir) == before
