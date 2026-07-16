from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.prelu_passthrough_layout import (
    optimize_prelu_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
    )


def _make_model(
    *,
    rank: int = 4,
    permutation_dtype: np.dtype = np.dtype(np.int32),
    dynamic: bool = False,
    alpha_data: np.ndarray | None = None,
    post_count: int = 1,
    public_post: bool = True,
    legacy: bool = False,
    shared_alpha: bool = False,
    shared_post_permutation: bool = False,
    pre_fanout: bool = False,
    repeated_alias_slots: bool = False,
) -> ModelIR:
    if rank == 3:
        source_shape = [1, 2, 4]
        pre_shape = [1, 4, 2]
        permutation = [0, 2, 1]
        inverse = [0, 2, 1]
        default_alpha = np.arange(4, dtype=np.float32).reshape(1, 4, 1) / 10
    else:
        source_shape = [1, 2, 3, 4]
        pre_shape = [1, 4, 2, 3]
        permutation = [0, 3, 1, 2]
        inverse = [0, 2, 3, 1]
        default_alpha = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1) / 10
    source_signature = list(source_shape)
    pre_signature = list(pre_shape)
    if dynamic:
        source_signature[1] = -1
        pre_signature[2 if rank == 4 else 2] = -1
    alpha = np.asarray(default_alpha if alpha_data is None else alpha_data)

    model = ModelIR("prelu_passthrough")
    model.inputs = ["x"]
    model.tensors = {
        "x": _tensor("x", source_shape, signature=source_signature),
        "pre_perm": _tensor(
            "pre_perm",
            [rank],
            dtype=str(permutation_dtype).upper(),
            data=np.asarray(permutation, dtype=permutation_dtype),
        ),
        "x_t": _tensor("x_t", pre_shape, signature=pre_signature),
        "alpha": _tensor(
            "alpha",
            list(alpha.shape),
            data=alpha.copy(),
        ),
        "prelu_t": _tensor("prelu_t", pre_shape, signature=pre_signature),
        "post_perm": _tensor(
            "post_perm",
            [rank],
            dtype=str(permutation_dtype).upper(),
            data=np.asarray(inverse, dtype=permutation_dtype),
        ),
    }
    pre = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["x", "pre_perm"],
        outputs=["x_t"],
    )
    prelu = OperatorIR(
        op_type="PRELU",
        inputs=["x_t", "alpha"],
        outputs=["prelu_t"],
        onnx_node_name="prelu",
        onnx_op_type="PRelu",
    )
    model.operators = [pre, prelu]
    for index in range(post_count):
        name = f"y{index}"
        model.tensors[name] = _tensor(
            name,
            source_shape,
            signature=source_signature,
        )
        model.operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["prelu_t", "post_perm"],
                outputs=[name],
            )
        )
    if legacy:
        model.tensors["legacy"] = _tensor(
            "legacy",
            pre_shape,
            signature=pre_signature,
        )
        model.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["prelu_t"],
                outputs=["legacy"],
            )
        )
    if shared_alpha:
        model.tensors["alpha_use"] = _tensor("alpha_use", list(alpha.shape))
        model.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["alpha"],
                outputs=["alpha_use"],
            )
        )
    if shared_post_permutation:
        model.tensors["other"] = _tensor("other", pre_shape)
        model.tensors["other_t"] = _tensor("other_t", source_shape)
        model.inputs.append("other")
        model.operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["other", "post_perm"],
                outputs=["other_t"],
            )
        )
    if pre_fanout:
        model.tensors["pre_fanout"] = _tensor(
            "pre_fanout",
            pre_shape,
            signature=pre_signature,
        )
        model.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["x_t"],
                outputs=["pre_fanout"],
            )
        )

    if repeated_alias_slots:
        model.tensors["sum"] = _tensor(
            "sum",
            source_shape,
            signature=source_signature,
        )
        model.operators.append(
            OperatorIR(op_type="ADD", inputs=["y1", "y1"], outputs=["sum"])
        )
        model.outputs = ["sum"]
    else:
        model.outputs = ["y0"] if public_post else ["legacy"]
    return model


def _snapshot(model: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model.inputs),
        tuple(model.outputs),
        tuple(
            (
                op.op_type,
                tuple(op.inputs),
                tuple(op.outputs),
                _freeze(op.options),
                _freeze(op.axis_semantics),
                op.version,
                op.onnx_node_name,
                op.onnx_op_type,
            )
            for op in model.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                tensor.is_variable,
                _freeze(tensor.quantization),
                tensor.logical_layout,
                tensor.physical_layout,
                tensor.onnx_tensor_name,
            )
            for name, tensor in sorted(model.tensors.items())
        ),
    )


@pytest.mark.parametrize(
    ("rank", "permutation_dtype", "dynamic"),
    [
        (3, np.dtype(np.int32), False),
        (3, np.dtype(np.int64), True),
        (4, np.dtype(np.int32), True),
        (4, np.dtype(np.int64), False),
    ],
)
def test_rewrites_typed_static_and_dynamic_views(
    rank: int,
    permutation_dtype: np.dtype,
    dynamic: bool,
) -> None:
    model = _make_model(
        rank=rank,
        permutation_dtype=permutation_dtype,
        dynamic=dynamic,
    )
    index = ModelIRGraphIndex(model)
    layout = LayoutState.from_model_ir(model)

    stats = optimize_prelu_transpose_passthrough_chains(
        model,
        graph_index=index,
        layout_state=layout,
    )

    assert stats == {"rewritten_prelu_transpose_passthrough_chains": 1}
    assert [op.op_type for op in model.operators] == ["PRELU"]
    assert model.operators[0].inputs[0] == "x"
    assert model.operators[0].outputs == ["y0"]
    expected = [1, 1, 4] if rank == 3 else [1, 1, 1, 4]
    assert list(np.asarray(model.tensors["alpha"].data).shape) == expected
    fresh = ModelIRGraphIndex(model)
    assert index.producers == fresh.producers
    assert index.consumers == fresh.consumers
    assert index.duplicate_producers == fresh.duplicate_producers
    assert layout.validate_against_model_ir(model) == []


def test_shared_alpha_uses_copy_on_write_clone() -> None:
    model = _make_model(shared_alpha=True)
    original = np.asarray(model.tensors["alpha"].data).copy()

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    prelu = next(op for op in model.operators if op.op_type == "PRELU")
    assert prelu.inputs[1] == "alpha_nhwc"
    assert np.array_equal(model.tensors["alpha"].data, original)
    assert np.array_equal(
        model.tensors["alpha_nhwc"].data,
        np.transpose(original, (0, 2, 3, 1)),
    )
    alpha_user = next(op for op in model.operators if op.outputs == ["alpha_use"])
    assert alpha_user.inputs == ["alpha"]


def test_legacy_adapter_does_not_mutate_shared_post_permutation() -> None:
    model = _make_model(
        legacy=True,
        public_post=False,
        shared_post_permutation=True,
    )
    post_permutation = np.asarray(model.tensors["post_perm"].data).copy()

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    assert np.array_equal(model.tensors["post_perm"].data, post_permutation)
    adapter = next(op for op in model.operators if op.outputs == ["prelu_t"])
    assert adapter.inputs == ["y0", "pre_perm"]
    legacy = next(op for op in model.operators if op.outputs == ["legacy"])
    assert legacy.inputs == ["prelu_t"]
    unrelated = next(op for op in model.operators if op.outputs == ["other_t"])
    assert unrelated.inputs == ["other", "post_perm"]


def test_other_pre_consumer_retains_pre_adapter() -> None:
    model = _make_model(pre_fanout=True)

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    pre = next(op for op in model.operators if op.outputs == ["x_t"])
    assert pre.op_type == "TRANSPOSE"
    prelu = next(op for op in model.operators if op.op_type == "PRELU")
    assert prelu.inputs[0] == "x"
    fanout = next(op for op in model.operators if op.outputs == ["pre_fanout"])
    assert fanout.inputs == ["x_t"]


def test_multiple_posts_and_repeated_slots_coalesce_exactly() -> None:
    model = _make_model(
        post_count=2,
        public_post=False,
        repeated_alias_slots=True,
    )

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model.operators)
    prelu = next(op for op in model.operators if op.op_type == "PRELU")
    add = next(op for op in model.operators if op.op_type == "ADD")
    assert prelu.outputs == ["y1"]
    assert add.inputs == ["y1", "y1"]


@pytest.mark.parametrize(
    "alpha_data",
    [
        np.asarray([0.25], dtype=np.float32),
        np.asarray(0.25, dtype=np.float32),
    ],
)
def test_scalar_alpha_is_reused(alpha_data: np.ndarray) -> None:
    model = _make_model(alpha_data=alpha_data)

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    prelu = next(op for op in model.operators if op.op_type == "PRELU")
    assert prelu.inputs[1] == "alpha"
    assert np.array_equal(model.tensors["alpha"].data, alpha_data)


def test_ambiguous_equal_shape_prefers_layout_remap() -> None:
    alpha = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    model = _make_model(alpha_data=alpha)
    for name in ("x", "x_t", "prelu_t", "y0"):
        model.tensors[name].shape = [1, 2, 2, 2]
        model.tensors[name].shape_signature = [1, 2, 2, 2]

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 1
    assert np.array_equal(
        model.tensors["alpha"].data,
        np.transpose(alpha, (0, 2, 3, 1)),
    )


def test_numerical_layout_equivalence() -> None:
    model = _make_model()
    alpha_nchw = np.asarray(model.tensors["alpha"].data)
    rng = np.random.default_rng(0)
    source = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    transposed = np.transpose(source, (0, 3, 1, 2))
    expected = np.transpose(
        np.where(transposed >= 0, transposed, transposed * alpha_nchw),
        (0, 2, 3, 1),
    )

    optimize_prelu_transpose_passthrough_chains(model)
    alpha_nhwc = np.asarray(model.tensors["alpha"].data)
    actual = np.where(source >= 0, source, source * alpha_nhwc)

    assert np.array_equal(actual, expected)


def test_candidate_limit_and_idempotence() -> None:
    model = _make_model()
    candidate = model.operators[0]

    assert optimize_prelu_transpose_passthrough_chains(
        model,
        max_rewrites=0,
        candidate=candidate,
    )["rewritten_prelu_transpose_passthrough_chains"] == 0
    assert optimize_prelu_transpose_passthrough_chains(
        model,
        max_rewrites=1,
        candidate=candidate,
    )["rewritten_prelu_transpose_passthrough_chains"] == 1
    first = _snapshot(model)
    assert optimize_prelu_transpose_passthrough_chains(model)[
        "rewritten_prelu_transpose_passthrough_chains"
    ] == 0
    assert _snapshot(model) == first


def _wrong_pre_permutation(model: ModelIR) -> None:
    model.tensors["pre_perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _wrong_post_permutation(model: ModelIR) -> None:
    model.tensors["post_perm"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _variable_alpha(model: ModelIR) -> None:
    model.tensors["alpha"].is_variable = True


def _wrong_alpha_dtype(model: ModelIR) -> None:
    model.tensors["alpha"].dtype = "FLOAT16"


def _wrong_alpha_numpy_dtype(model: ModelIR) -> None:
    model.tensors["alpha"].data = np.asarray(model.tensors["alpha"].data, dtype=np.float64)


def _per_axis_alpha(model: ModelIR) -> None:
    model.tensors["alpha"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _nonbroadcast_alpha(model: ModelIR) -> None:
    data = np.ones((2, 5), dtype=np.float32)
    model.tensors["alpha"].data = data
    model.tensors["alpha"].shape = [2, 5]
    model.tensors["alpha"].shape_signature = [2, 5]


def _wrong_pre_shape(model: ModelIR) -> None:
    model.tensors["x_t"].shape[-1] += 1


def _wrong_prelu_shape(model: ModelIR) -> None:
    model.tensors["prelu_t"].shape[-1] += 1


def _wrong_post_shape(model: ModelIR) -> None:
    model.tensors["y0"].shape[-1] += 1


def _public_pre_output(model: ModelIR) -> None:
    model.outputs.append("x_t")


def _duplicate_pre_output(model: ModelIR) -> None:
    model.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["x"], outputs=["x_t"]),
    )


def _produced_alpha(model: ModelIR) -> None:
    model.operators.insert(
        0,
        OperatorIR(op_type="RELU", inputs=["x"], outputs=["alpha"]),
    )


def _out_of_order_post(model: ModelIR) -> None:
    model.operators.insert(0, model.operators.pop(2))


def _known_layout_mismatch(model: ModelIR) -> None:
    model.tensors["x"].logical_layout = "NCHW"
    model.tensors["x_t"].logical_layout = "NCHW"


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_pre_permutation,
        _wrong_post_permutation,
        _variable_alpha,
        _wrong_alpha_dtype,
        _wrong_alpha_numpy_dtype,
        _per_axis_alpha,
        _nonbroadcast_alpha,
        _wrong_pre_shape,
        _wrong_prelu_shape,
        _wrong_post_shape,
        _public_pre_output,
        _duplicate_pre_output,
        _produced_alpha,
        _out_of_order_post,
        _known_layout_mismatch,
    ],
)
def test_unsafe_candidates_are_transactional_noops(
    mutate: Callable[[ModelIR], None],
) -> None:
    model = _make_model()
    mutate(model)
    before = _snapshot(copy.deepcopy(model))

    stats = optimize_prelu_transpose_passthrough_chains(model)

    assert stats["rewritten_prelu_transpose_passthrough_chains"] == 0
    assert _snapshot(model) == before
