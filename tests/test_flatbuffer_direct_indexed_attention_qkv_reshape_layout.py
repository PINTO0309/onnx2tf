from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains,
)
from onnx2tf.tflite_builder.passes.attention_qkv_reshape_layout import (
    _apply_plan,
    _plan_signature,
    _resolve_candidate,
    optimize_attention_qkv_had_reshape_transpose_chains,
)


_STATS_KEY = (
    "optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
)


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
    permutation: tuple[int, ...] = (1, 0, 2),
    shared_shape: bool = False,
    shared_permutation: bool = False,
    dynamic_signature: bool = False,
) -> ModelIR:
    transpose_shape = [2, 3, 4] if permutation == (1, 0, 2) else [2, 4, 3]
    output_shape = [1, *transpose_shape]
    model_ir = ModelIR("indexed_attention_qkv_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x": _tensor(
            "x",
            [3, 1, 8],
            signature=[-1, 1, 8] if dynamic_signature else None,
        ),
        "shape": _tensor(
            "shape",
            [3],
            dtype="INT64",
            data=np.asarray([3, 2, 4], dtype=np.int64),
        ),
        "r": _tensor(
            "r",
            [3, 2, 4],
            signature=[-1, 2, 4] if dynamic_signature else None,
        ),
        "perm": _tensor(
            "perm",
            [3],
            dtype="INT32",
            data=np.asarray(permutation, dtype=np.int32),
        ),
        "t": _tensor(
            "t",
            transpose_shape,
            signature=(
                [2, -1, 4]
                if dynamic_signature and permutation == (1, 0, 2)
                else None
            ),
        ),
        "tail_shape": _tensor(
            "tail_shape",
            [4],
            dtype="INT32",
            data=np.asarray(output_shape, dtype=np.int32),
        ),
        "z": _tensor(
            "z",
            output_shape,
            signature=(
                [1, 2, -1, 4]
                if dynamic_signature and permutation == (1, 0, 2)
                else None
            ),
        ),
    }
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["x", "shape"],
            ["r"],
            options={
                "newShape": [3, 2, 4],
                "onnxRawNewShape": [3, -1, 4],
                "allowZero": False,
            },
        ),
        OperatorIR("TRANSPOSE", ["r", "perm"], ["t"]),
        OperatorIR(
            "RESHAPE",
            ["t", "tail_shape"],
            ["z"],
            options={
                "newShape": list(output_shape),
                "onnxRawNewShape": list(output_shape),
                "allowZero": False,
            },
        ),
    ]
    if shared_shape:
        model_ir.outputs.append("shape_alias")
        model_ir.tensors["shape_alias"] = _tensor(
            "shape_alias",
            [3],
            dtype="INT64",
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["shape"], ["shape_alias"])
        )
    if shared_permutation:
        model_ir.outputs.append("perm_alias")
        model_ir.tensors["perm_alias"] = _tensor(
            "perm_alias",
            [3],
            dtype="INT32",
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["perm"], ["perm_alias"])
        )
    return model_ir


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (operator.op_type, tuple(operator.inputs), tuple(operator.outputs), repr(operator.options))
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
                bool(tensor.is_variable),
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


def test_indexed_attention_qkv_had_rewrites_differentially() -> None:
    model_ir = _make_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    assert optimize_attention_qkv_had_reshape_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "RESHAPE",
        "TRANSPOSE",
    ]
    assert model_ir.operators[0].outputs == ["r"]
    assert model_ir.operators[0].options == {
        "newShape": [1, 3, 2, 4],
        "onnxRawNewShape": [1, 3, 2, 4],
        "allowZero": False,
    }
    assert model_ir.operators[1].outputs == ["z"]
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([1, 3, 2, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["perm"].data,
        np.asarray([0, 2, 1, 3], dtype=np.int32),
    )
    assert model_ir.tensors["r"].shape == [1, 3, 2, 4]
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "rename_tensor",
            "old_name": "t",
            "new_name": "z",
            "source": "set_operator_outputs",
            "event_index": 0,
        }
    ]

    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert optimize_attention_qkv_had_reshape_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_indexed_attention_qkv_had_wrapper_prunes_and_tracks_layout() -> None:
    model_ir = _make_ir()
    layout_state = LayoutState.from_model_ir(model_ir)
    assert (
        _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains(
            model_ir,
            layout_state=layout_state,
        )
        == {_STATS_KEY: 1}
    )
    assert set(model_ir.tensors) == {"x", "shape", "r", "perm", "z"}
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_attention_qkv_preserves_hda_compatibility_fallback() -> None:
    model_ir = _make_ir(permutation=(1, 2, 0))
    assert optimize_attention_qkv_had_reshape_transpose_chains(model_ir) == {
        _STATS_KEY: 0
    }
    assert (
        _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains(
            model_ir
        )
        == {_STATS_KEY: 1}
    )
    np.testing.assert_array_equal(
        model_ir.tensors["perm"].data,
        np.asarray([0, 2, 3, 1], dtype=np.int32),
    )


def test_indexed_attention_qkv_preserves_shared_constant_copy_on_write() -> None:
    model_ir = _make_ir(shared_shape=True, shared_permutation=True)
    before = _fingerprint(model_ir)
    assert optimize_attention_qkv_had_reshape_transpose_chains(model_ir) == {
        _STATS_KEY: 0
    }
    assert _fingerprint(model_ir) == before
    assert (
        _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains(
            model_ir
        )
        == {_STATS_KEY: 1}
    )
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([3, 2, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["perm"].data,
        np.asarray([1, 0, 2], dtype=np.int32),
    )
    reshape, transpose = model_ir.operators[:2]
    assert reshape.inputs[1] != "shape"
    assert transpose.inputs[1] != "perm"
    np.testing.assert_array_equal(
        model_ir.tensors[reshape.inputs[1]].data,
        np.asarray([1, 3, 2, 4], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors[transpose.inputs[1]].data,
        np.asarray([0, 2, 1, 3], dtype=np.int32),
    )


def test_indexed_attention_qkv_preserves_dynamic_compatibility_fallback() -> None:
    model_ir = _make_ir(dynamic_signature=True)
    assert optimize_attention_qkv_had_reshape_transpose_chains(model_ir) == {
        _STATS_KEY: 0
    }
    assert (
        _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains(
            model_ir
        )
        == {_STATS_KEY: 1}
    )


def test_indexed_attention_qkv_candidate_and_bound_are_strict() -> None:
    model_ir = _make_ir()
    graph_index = ModelIRGraphIndex(model_ir)
    reshape = model_ir.operators[0]
    assert optimize_attention_qkv_had_reshape_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=model_ir.operators[1],
    ) == {_STATS_KEY: 0}
    assert optimize_attention_qkv_had_reshape_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=reshape,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_attention_qkv_had_reshape_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=reshape,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}


def test_indexed_attention_qkv_rejects_stale_plan() -> None:
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

    model_ir.operators[0].options["newShape"] = [3, -1, 4]
    before_apply = _fingerprint(model_ir)
    assert _plan_signature(plan) == signature
    assert not _apply_plan(
        model_ir,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _fingerprint(model_ir) == before_apply


def test_indexed_attention_qkv_is_deterministic() -> None:
    first = _make_ir()
    second = deepcopy(first)
    assert optimize_attention_qkv_had_reshape_transpose_chains(first) == {
        _STATS_KEY: 1
    }
    assert optimize_attention_qkv_had_reshape_transpose_chains(second) == {
        _STATS_KEY: 1
    }
    assert _fingerprint(first) == _fingerprint(second)
