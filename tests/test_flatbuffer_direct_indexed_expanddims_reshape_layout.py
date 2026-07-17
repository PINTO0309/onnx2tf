from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.expanddims_reshape_layout import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_transpose_factorized_expanddims_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.expanddims_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat,
)


_STATS_KEY = "optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains"


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


def _make_factorized_ir(
    *,
    anchors: int = 3,
    shared_shape: bool = False,
    shared_post_permutation: bool = False,
) -> ModelIR:
    values_per_anchor = 4
    channels = int(anchors) * int(values_per_anchor)
    model_ir = ModelIR("indexed_factorized_expanddims")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    reshape_shape = [1, anchors, values_per_anchor, 2, 3]
    post_shape = [1, anchors, 2, 3, values_per_anchor]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, channels]),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, channels, 2, 3]),
        "shape": _tensor(
            "shape",
            [5],
            dtype="INT64",
            data=np.asarray(reshape_shape, dtype=np.int64),
        ),
        "view": _tensor("view", reshape_shape),
        "post_perm": _tensor(
            "post_perm",
            [5],
            dtype="INT32",
            data=np.asarray([0, 1, 3, 4, 2], dtype=np.int32),
        ),
        "z": _tensor("z", post_shape),
    }
    if shared_shape:
        model_ir.tensors["shape_alias"] = _tensor(
            "shape_alias",
            [5],
            dtype="INT64",
        )
        model_ir.outputs.append("shape_alias")
    if shared_post_permutation:
        model_ir.tensors["post_perm_alias"] = _tensor(
            "post_perm_alias",
            [5],
            dtype="INT32",
        )
        model_ir.outputs.append("post_perm_alias")
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "pre_perm"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x_nchw", "shape"],
            outputs=["view"],
            options={
                "newShape": list(reshape_shape),
                "onnxRawNewShape": list(reshape_shape),
                "allowZero": False,
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
    if shared_post_permutation:
        model_ir.operators.append(
            OperatorIR(
                op_type="IDENTITY",
                inputs=["post_perm"],
                outputs=["post_perm_alias"],
            )
        )
    return model_ir


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (op.op_type, tuple(op.inputs), tuple(op.outputs), repr(op.options))
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


def test_indexed_factorized_expanddims_rewrites_differentially() -> None:
    model_ir = _make_factorized_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    assert optimize_transpose_factorized_expanddims_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE", "TRANSPOSE"]
    assert model_ir.operators[0].inputs == ["x", "shape"]
    assert model_ir.operators[0].options == {
        "newShape": [1, 2, 3, 3, 4],
        "onnxRawNewShape": [1, 2, 3, 3, 4],
        "allowZero": False,
    }
    assert model_ir.operators[1].inputs == ["view", "post_perm"]
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([1, 2, 3, 3, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["post_perm"].data,
        np.asarray([0, 3, 1, 2, 4], dtype=np.int32),
    )
    assert model_ir.tensors["view"].shape == [1, 2, 3, 3, 4]
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "replace_input",
            "src_name": "x_nchw",
            "dst_name": "x",
            "source": "set_operator_inputs",
            "event_index": 0,
        }
    ]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert optimize_transpose_factorized_expanddims_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_indexed_factorized_expanddims_rejects_singleton_case_a() -> None:
    model_ir = _make_factorized_ir(anchors=1)
    before = _fingerprint(model_ir)
    assert optimize_transpose_factorized_expanddims_nhwc_chains(model_ir) == {
        _STATS_KEY: 0
    }
    assert _fingerprint(model_ir) == before

    compat_ir = deepcopy(model_ir)
    wrapper_ir = deepcopy(model_ir)
    assert optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat(
        compat_ir
    ) == {_STATS_KEY: 1}
    assert _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(
        wrapper_ir
    ) == {_STATS_KEY: 1}
    assert _fingerprint(compat_ir) == _fingerprint(wrapper_ir)


def test_indexed_factorized_expanddims_rejects_shared_constants_atomically() -> None:
    for kwargs in ({"shared_shape": True}, {"shared_post_permutation": True}):
        model_ir = _make_factorized_ir(**kwargs)
        before = _fingerprint(model_ir)
        assert optimize_transpose_factorized_expanddims_nhwc_chains(model_ir) == {
            _STATS_KEY: 0
        }
        assert _fingerprint(model_ir) == before
        compat_ir = deepcopy(model_ir)
        wrapper_ir = deepcopy(model_ir)
        assert optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat(
            compat_ir
        ) == {_STATS_KEY: 0}
        assert _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(
            wrapper_ir
        ) == {_STATS_KEY: 0}
        assert _fingerprint(compat_ir) == before
        assert _fingerprint(wrapper_ir) == before


def test_indexed_factorized_expanddims_candidate_and_bound_are_strict() -> None:
    model_ir = _make_factorized_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    pre = model_ir.operators[0]

    assert optimize_transpose_factorized_expanddims_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_factorized_expanddims_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_factorized_expanddims_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}


def test_indexed_factorized_expanddims_rejects_stale_plan() -> None:
    model_ir = _make_factorized_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        model_ir.operators[0],
        layout_state=layout_state,
    )
    assert plan is not None
    signature = _plan_signature(plan)

    model_ir.operators[1].options["newShape"] = [1, 3, 4, 2, -1]
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_factorized_expanddims_is_deterministic() -> None:
    first = _make_factorized_ir()
    second = deepcopy(first)
    assert optimize_transpose_factorized_expanddims_nhwc_chains(first) == {
        _STATS_KEY: 1
    }
    assert optimize_transpose_factorized_expanddims_nhwc_chains(second) == {
        _STATS_KEY: 1
    }
    assert _fingerprint(first) == _fingerprint(second)


def test_indexed_factorized_expanddims_wrapper_cleanup_tracks_layout() -> None:
    compat_ir = _make_factorized_ir()
    wrapper_ir = deepcopy(compat_ir)
    compat_layout_state = LayoutState.from_model_ir(compat_ir)
    wrapper_layout_state = LayoutState.from_model_ir(wrapper_ir)
    assert optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat(
        compat_ir,
        layout_state=compat_layout_state,
    ) == {_STATS_KEY: 1}
    assert _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(
        wrapper_ir,
        layout_state=wrapper_layout_state,
    ) == {_STATS_KEY: 1}
    assert _fingerprint(compat_ir) == _fingerprint(wrapper_ir)
    assert compat_layout_state.logical == wrapper_layout_state.logical
    assert compat_layout_state.physical == wrapper_layout_state.physical
    assert set(wrapper_ir.tensors) == {"x", "shape", "view", "post_perm", "z"}
    assert wrapper_layout_state.validate_against_model_ir(wrapper_ir) == []
