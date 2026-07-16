from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_unary_reshape_suffix_layout import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains,
)


_STATS_KEY = "optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains"


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
        is_variable=False,
    )


def _make_swish_suffix_ir(*, shared_shape: bool = False) -> ModelIR:
    model_ir = ModelIR("indexed_pre_unary_swish_suffix")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4]),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 4, 2, 3]),
        "sigmoid": _tensor("sigmoid", [1, 4, 2, 3]),
        "swish": _tensor("swish", [1, 4, 2, 3]),
        "shape": _tensor(
            "shape",
            [3],
            dtype="INT64",
            data=np.asarray([1, 4, 6], dtype=np.int64),
        ),
        "view": _tensor("view", [1, 4, 6]),
        "post_perm": _tensor(
            "post_perm",
            [3],
            dtype="INT32",
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "z": _tensor("z", [1, 6, 4]),
    }
    if shared_shape:
        model_ir.tensors["shape_alias"] = _tensor(
            "shape_alias",
            [3],
            dtype="INT64",
        )
        model_ir.outputs.append("shape_alias")
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "pre_perm"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type="LOGISTIC",
            inputs=["x_nchw"],
            outputs=["sigmoid"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "sigmoid"],
            outputs=["swish"],
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["swish", "shape"],
            outputs=["view"],
            options={
                "newShape": [1, 4, 6],
                "onnxRawNewShape": [1, 4, -1],
            },
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["view", "post_perm"],
            outputs=["z"],
        ),
    ]
    if shared_shape:
        model_ir.operators.append(
            OperatorIR(
                op_type="IDENTITY",
                inputs=["shape"],
                outputs=["shape_alias"],
            )
        )
    return model_ir


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                op.op_type,
                tuple(op.inputs),
                tuple(op.outputs),
                repr(op.options),
            )
            for op in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                tensor.logical_layout,
                tensor.physical_layout,
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def test_indexed_swish_suffix_rewrite_preserves_index_layout_and_lineage() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS_KEY: 1}
    assert [op.op_type for op in model_ir.operators] == [
        "LOGISTIC",
        "MUL",
        "RESHAPE",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].inputs == ["x", "sigmoid"]
    assert model_ir.operators[2].outputs == ["z"]
    assert model_ir.operators[2].options == {
        "newShape": [1, 6, 4],
        "onnxRawNewShape": [1, -1, 4],
    }
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([1, 6, 4], dtype=np.int64),
    )
    assert model_ir.tensors["sigmoid"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["swish"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["z"].shape == [1, 6, 4]
    assert [
        event["source"] for event in model_ir.metadata["tensor_lineage_events"]
    ] == [
        "set_operator_inputs",
        "replace_operator_input_at",
        "set_operator_outputs",
    ]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_indexed_swish_suffix_rejects_shared_shape_atomically() -> None:
    model_ir = _make_swish_suffix_ir(shared_shape=True)
    before = _fingerprint(model_ir)

    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
    ) == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_indexed_swish_suffix_candidate_and_rewrite_bound_are_strict() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    mul = model_ir.operators[2]

    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=mul,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=mul,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}


def test_indexed_swish_suffix_stale_plan_is_rejected_before_mutation() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    mul = model_ir.operators[2]
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        mul,
        layout_state=layout_state,
    )
    assert plan is not None
    signature = _plan_signature(plan)

    model_ir.operators[3].options["newShape"] = [1, 4, -1]
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_swish_suffix_is_deterministic() -> None:
    first = _make_swish_suffix_ir()
    second = deepcopy(first)

    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        first,
    ) == {_STATS_KEY: 1}
    assert optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
        second,
    ) == {_STATS_KEY: 1}
    assert _fingerprint(first) == _fingerprint(second)


def test_indexed_swish_wrapper_retains_one_cleanup_and_layout_boundary() -> None:
    model_ir = _make_swish_suffix_ir()
    layout_state = LayoutState.from_model_ir(model_ir)

    assert _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains(
        model_ir,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}
    assert set(model_ir.tensors) == {"x", "sigmoid", "swish", "shape", "z"}
    assert layout_state.validate_against_model_ir(model_ir) == []
