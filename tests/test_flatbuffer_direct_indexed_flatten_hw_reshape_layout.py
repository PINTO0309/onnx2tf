from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.flatten_hw_reshape_layout import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_transpose_flatten_hw_reshape_nhwc_chains,
)


_STATS_KEY = "optimized_transpose_reshape_transpose_to_flatten_hw_nhwc_chains"


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
        is_variable=False,
    )


def _make_ir(
    *,
    shared_shape: bool = False,
    boundary_shape: bool = False,
    dynamic_signature: bool = False,
    shape_dtype: str = "INT32",
) -> ModelIR:
    model_ir = ModelIR("indexed_flatten_hw")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    integer_dtype = np.int32 if shape_dtype == "INT32" else np.int64
    model_ir.tensors = {
        "x": _tensor(
            "x",
            [1, 2, 3, 4],
            signature=[-1, 2, 3, 4] if dynamic_signature else None,
        ),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor(
            "x_nchw",
            [1, 4, 2, 3],
            signature=[-1, 4, 2, 3] if dynamic_signature else None,
        ),
        "shape": _tensor(
            "shape",
            [3],
            dtype=shape_dtype,
            data=np.asarray([1, 4, 6], dtype=integer_dtype),
        ),
        "view": _tensor(
            "view",
            [1, 4, 6],
            signature=[-1, 4, 6] if dynamic_signature else None,
        ),
        "post_perm": _tensor(
            "post_perm",
            [3],
            dtype="INT64",
            data=np.asarray([0, 2, 1], dtype=np.int64),
        ),
        "z": _tensor(
            "z",
            [1, 6, 4],
            signature=[-1, 6, 4] if dynamic_signature else None,
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "RESHAPE",
            ["x_nchw", "shape"],
            ["view"],
            options={
                "newShape": [1, 4, 6],
                "onnxRawNewShape": [1, 4, 6],
                "allowZero": True,
            },
        ),
        OperatorIR("TRANSPOSE", ["view", "post_perm"], ["z"]),
    ]
    if boundary_shape:
        model_ir.outputs.append("shape")
    if shared_shape:
        model_ir.outputs.append("shape_alias")
        model_ir.tensors["shape_alias"] = _tensor(
            "shape_alias",
            [3],
            dtype=shape_dtype,
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["shape"], ["shape_alias"])
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


def test_indexed_flatten_hw_rewrites_differentially() -> None:
    model_ir = _make_ir(shape_dtype="INT64")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}
    assert [operator.op_type for operator in model_ir.operators] == ["RESHAPE"]
    assert model_ir.operators[0].inputs == ["x", "shape"]
    assert model_ir.operators[0].outputs == ["z"]
    assert model_ir.operators[0].options == {
        "newShape": [1, 6, 4],
        "onnxRawNewShape": [1, 6, 4],
        "allowZero": True,
    }
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([1, 6, 4], dtype=np.int64),
    )
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "replace_input",
            "src_name": "x_nchw",
            "dst_name": "x",
            "source": "set_operator_inputs",
            "event_index": 0,
        },
        {
            "kind": "rename_tensor",
            "old_name": "view",
            "new_name": "z",
            "source": "set_operator_outputs",
            "event_index": 1,
        },
    ]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_indexed_flatten_hw_wrapper_prunes_and_tracks_layout() -> None:
    model_ir = _make_ir()
    layout_state = LayoutState.from_model_ir(model_ir)
    assert _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains(
        model_ir,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}
    assert set(model_ir.tensors) == {"x", "shape", "z"}
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_flatten_hw_rejects_shared_shape_atomically() -> None:
    for kwargs in ({"shared_shape": True}, {"boundary_shape": True}):
        model_ir = _make_ir(**kwargs)
        before = _fingerprint(model_ir)
        assert optimize_transpose_flatten_hw_reshape_nhwc_chains(model_ir) == {
            _STATS_KEY: 0
        }
        assert _fingerprint(model_ir) == before
        assert _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains(
            model_ir
        ) == {_STATS_KEY: 0}
        assert _fingerprint(model_ir) == before


def test_flatten_hw_wrapper_rejects_produced_or_variable_shape_atomically() -> None:
    produced = _make_ir()
    produced.tensors["shape_seed"] = _tensor(
        "shape_seed",
        [3],
        dtype="INT32",
        data=np.asarray([1, 4, 6], dtype=np.int32),
    )
    produced.operators.insert(
        0,
        OperatorIR("IDENTITY", ["shape_seed"], ["shape"]),
    )
    variable = _make_ir()
    variable.tensors["shape"].is_variable = True

    for model_ir in (produced, variable):
        before = _fingerprint(model_ir)
        assert optimize_transpose_flatten_hw_reshape_nhwc_chains(model_ir) == {
            _STATS_KEY: 0
        }
        assert _fingerprint(model_ir) == before
        assert _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains(
            model_ir
        ) == {_STATS_KEY: 0}
        assert _fingerprint(model_ir) == before


def test_indexed_flatten_hw_preserves_dynamic_compatibility_fallback() -> None:
    model_ir = _make_ir(dynamic_signature=True)
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(model_ir) == {
        _STATS_KEY: 0
    }
    assert _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains(
        model_ir
    ) == {_STATS_KEY: 1}
    assert [operator.op_type for operator in model_ir.operators] == ["RESHAPE"]


def test_indexed_flatten_hw_candidate_and_bound_are_strict() -> None:
    model_ir = _make_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    pre = model_ir.operators[0]
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=pre,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}


def test_indexed_flatten_hw_rejects_stale_plan() -> None:
    model_ir = _make_ir()
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

    model_ir.operators[1].options["newShape"] = [1, 4, -1]
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_flatten_hw_is_deterministic() -> None:
    first = _make_ir()
    second = deepcopy(first)
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(first) == {
        _STATS_KEY: 1
    }
    assert optimize_transpose_flatten_hw_reshape_nhwc_chains(second) == {
        _STATS_KEY: 1
    }
    assert _fingerprint(first) == _fingerprint(second)
