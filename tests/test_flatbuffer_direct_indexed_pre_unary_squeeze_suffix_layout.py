from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_unary_squeeze_suffix_compat_layout import (
    optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.pre_unary_squeeze_suffix_layout import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains,
)


_STATS_KEY = "optimized_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains"


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


def _make_swish_suffix_ir(*, axis: int = 2) -> ModelIR:
    if axis == 2:
        source_shape = [1, 1, 3, 4]
        nchw_shape = [1, 4, 1, 3]
        squeezed_shape = [1, 4, 3]
        suffix_shape = [1, 3, 4]
    elif axis == 3:
        source_shape = [1, 2, 1, 4]
        nchw_shape = [1, 4, 2, 1]
        squeezed_shape = [1, 4, 2]
        suffix_shape = [1, 2, 4]
    else:
        raise ValueError(axis)

    model_ir = ModelIR("indexed_pre_unary_swish_squeeze_suffix")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x": _tensor("x", source_shape),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", nchw_shape),
        "sigmoid": _tensor("sigmoid", nchw_shape),
        "swish": _tensor("swish", nchw_shape),
        "squeezed": _tensor("squeezed", squeezed_shape),
        "post_perm": _tensor(
            "post_perm",
            [3],
            dtype="INT32",
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "z": _tensor("z", suffix_shape),
    }
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
            op_type="SQUEEZE",
            inputs=["swish"],
            outputs=["squeezed"],
            options={"squeezeDims": [axis]},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["squeezed", "post_perm"],
            outputs=["z"],
        ),
    ]
    return model_ir


def _make_plain_unary_suffix_ir() -> ModelIR:
    model_ir = _make_swish_suffix_ir()
    model_ir.tensors.pop("sigmoid")
    model_ir.operators[1:3] = [
        OperatorIR(
            op_type="RELU",
            inputs=["x_nchw"],
            outputs=["swish"],
        )
    ]
    return model_ir


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
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


def test_indexed_swish_squeeze_suffix_preserves_index_layout_and_lineage() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS_KEY: 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "LOGISTIC",
        "MUL",
        "SQUEEZE",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].inputs == ["x", "sigmoid"]
    assert model_ir.operators[2].outputs == ["z"]
    assert model_ir.operators[2].options == {"squeezeDims": [1]}
    assert model_ir.tensors["sigmoid"].shape == [1, 1, 3, 4]
    assert model_ir.tensors["swish"].shape == [1, 1, 3, 4]
    assert model_ir.tensors["z"].shape == [1, 3, 4]
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
    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_indexed_swish_squeeze_suffix_candidate_and_bound_are_strict() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    mul = model_ir.operators[2]

    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=mul,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=mul,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}


def test_indexed_swish_squeeze_suffix_stale_plan_is_atomic() -> None:
    model_ir = _make_swish_suffix_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        model_ir.operators[2],
        layout_state=layout_state,
    )
    assert plan is not None
    signature = _plan_signature(plan)

    model_ir.operators[3].options["compatibilityMarker"] = True
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_swish_squeeze_suffix_rejects_malformed_axes_atomically() -> None:
    model_ir = _make_swish_suffix_ir()
    model_ir.operators[3].options["squeezeDims"] = [object()]
    before = _fingerprint(model_ir)

    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
    ) == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_indexed_swish_squeeze_suffix_is_deterministic() -> None:
    first = _make_swish_suffix_ir()
    second = deepcopy(first)

    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        first,
    ) == {_STATS_KEY: 1}
    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        second,
    ) == {_STATS_KEY: 1}
    assert _fingerprint(first) == _fingerprint(second)


def test_indexed_swish_squeeze_wrapper_prunes_one_layout_boundary() -> None:
    model_ir = _make_swish_suffix_ir()
    wrapped_model_ir = deepcopy(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    wrapped_layout_state = LayoutState.from_model_ir(wrapped_model_ir)

    stats = optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat(
        model_ir,
        layout_state=layout_state,
    )
    wrapped_stats = _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains(
        wrapped_model_ir,
        layout_state=wrapped_layout_state,
    )

    assert stats == {_STATS_KEY: 1}
    assert wrapped_stats == stats
    assert _fingerprint(wrapped_model_ir) == _fingerprint(model_ir)
    assert set(model_ir.tensors) == {"x", "sigmoid", "swish", "z"}
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert wrapped_layout_state.validate_against_model_ir(wrapped_model_ir) == []


def test_plain_unary_and_axis3_stay_on_compatibility_fallback() -> None:
    plain = _make_plain_unary_suffix_ir()
    wrapped_plain = deepcopy(plain)
    axis3 = _make_swish_suffix_ir(axis=3)
    wrapped_axis3 = deepcopy(axis3)

    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        plain,
    ) == {_STATS_KEY: 0}
    plain_stats = (
        optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat(
            plain,
        )
    )
    wrapped_plain_stats = (
        _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains(
            wrapped_plain,
        )
    )
    assert plain_stats == {_STATS_KEY: 1}
    assert wrapped_plain_stats == plain_stats
    assert _fingerprint(wrapped_plain) == _fingerprint(plain)
    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        axis3,
    ) == {_STATS_KEY: 0}
    axis3_stats = (
        optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat(
            axis3,
        )
    )
    wrapped_axis3_stats = (
        _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains(
            wrapped_axis3,
        )
    )
    assert axis3_stats == {_STATS_KEY: 1}
    assert wrapped_axis3_stats == axis3_stats
    assert _fingerprint(wrapped_axis3) == _fingerprint(axis3)


def test_dynamic_signature_stays_on_compatibility_fallback() -> None:
    model_ir = _make_swish_suffix_ir()
    for name in ("x", "x_nchw", "sigmoid", "swish"):
        model_ir.tensors[name].shape_signature[0] = -1
    for name in ("squeezed", "z"):
        model_ir.tensors[name].shape_signature[0] = -1
    wrapped_model_ir = deepcopy(model_ir)

    assert optimize_transpose_pre_swish_squeeze_transpose_suffix_nhwc_chains(
        model_ir,
    ) == {_STATS_KEY: 0}
    stats = optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat(
        model_ir,
    )
    wrapped_stats = _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains(
        wrapped_model_ir,
    )
    assert stats == {_STATS_KEY: 1}
    assert wrapped_stats == stats
    assert _fingerprint(wrapped_model_ir) == _fingerprint(model_ir)
