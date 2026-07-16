from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    ModelIRPassStateScope,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    _optimize_maximum_minimum_relu0to1_chains,
    _optimize_squeeze_reshape_identity_chains,
    _optimize_duplicate_reshape_fanout,
    _optimize_duplicate_transpose_fanout,
    prune_dead_operators,
    run_clamp_cleanup,
    run_consecutive_mul_constants_cleanup,
    run_duplicate_fanout_cleanup,
    run_maximum_zero_relu_cleanup,
    run_squeeze_reshape_identity_cleanup,
)


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
        is_variable=data is None,
    )


def test_dead_operator_pruning_uses_one_batch_index_compaction(monkeypatch) -> None:
    model_ir = ModelIR("dead_operator_pruning")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        name: _tensor(name, [1, 3])
        for name in ("x", "y", "dead0", "mid", "dead1", "out")
    }
    for tensor in model_ir.tensors.values():
        tensor.is_variable = False
    model_ir.operators = [
        OperatorIR("IDENTITY", ["x"], ["dead0"]),
        OperatorIR("ADD", ["x", "y"], ["mid"]),
        OperatorIR("MUL", ["dead0", "y"], ["dead1"]),
        OperatorIR("MUL", ["mid", "y"], ["out"]),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = prune_dead_operators(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"removed_dead_operators": 2}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["ADD", "MUL"]
    assert graph_index.operator_indices("IDENTITY") == []
    assert graph_index.operator_indices("ADD") == [0]
    assert graph_index.operator_indices("MUL") == [1]
    assert "dead0" not in model_ir.tensors
    assert "dead1" not in model_ir.tensors
    assert layout_state.validate_against_model_ir(model_ir) == []


def _consecutive_mul_model(
    first_constant: np.ndarray,
    second_constant: np.ndarray,
    *,
    extra_mid_consumer: bool = False,
) -> ModelIR:
    model_ir = ModelIR("consecutive_mul_constants")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"] + (["side"] if extra_mid_consumer else [])
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "c0": _tensor(
            "c0",
            list(first_constant.shape),
            data=first_constant,
        ),
        "mid": _tensor("mid", [1, 3]),
        "c1": _tensor(
            "c1",
            list(second_constant.shape),
            data=second_constant,
        ),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.tensors["c0"].quantization = {
        "scale": [0.25],
        "zero_point": [0],
    }
    model_ir.operators = [
        OperatorIR("MUL", ["c0", "x"], ["mid"]),
        OperatorIR("MUL", ["mid", "c1"], ["out"]),
    ]
    if extra_mid_consumer:
        model_ir.tensors["side"] = _tensor("side", [1, 3])
        model_ir.operators.append(OperatorIR("IDENTITY", ["mid"], ["side"]))
    return model_ir


def _squeeze_unary_reshape_model(
    *,
    fanout: bool = False,
    squeeze_axes: list[int] | None = None,
) -> ModelIR:
    model_ir = ModelIR("squeeze_unary_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4]),
        "squeezed": _tensor("squeezed", [2, 3, 4]),
        "unary": _tensor("unary", [2, 3, 4]),
        "shape": _tensor(
            "shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 2, 3, 4], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 3, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            "SQUEEZE",
            ["x"],
            ["squeezed"],
            options={"squeezeDims": list(squeeze_axes or [0])},
        ),
        OperatorIR("RELU", ["squeezed"], ["unary"]),
        OperatorIR("RESHAPE", ["unary", "shape"], ["y"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [2, 3, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", ["unary"], ["side"]))
    return model_ir


def test_duplicate_transpose_cleanup_uses_one_incremental_index_refresh(
    monkeypatch,
) -> None:
    model_ir = ModelIR("duplicate_transpose_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm0": _tensor(
            "perm0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "perm1": _tensor(
            "perm1",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y0": _tensor("y0", [1, 3, 2, 2]),
        "y1": _tensor("y1", [1, 3, 2, 2]),
        "out": _tensor("out", [1, 3, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm0"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["y1"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_duplicate_transpose_fanout(model_ir)

    assert stats == {"removed_duplicate_transpose_fanout": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["TRANSPOSE", "IDENTITY"]
    assert model_ir.operators[1].inputs == ["y0"]
    assert "y1" not in model_ir.tensors


def test_duplicate_cleanup_ordered_group_shares_one_index(monkeypatch) -> None:
    model_ir = ModelIR("ordered_duplicate_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm0": _tensor(
            "perm0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "perm1": _tensor(
            "perm1",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y0": _tensor("y0", [1, 3, 2, 2]),
        "y1": _tensor("y1", [1, 3, 2, 2]),
        "out": _tensor("out", [1, 3, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm0"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["y1"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    layout_state = LayoutState.from_model_ir(model_ir)
    stats = run_duplicate_fanout_cleanup(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "removed_duplicate_transpose_fanout": 1,
        "removed_duplicate_reshape_fanout": 0,
    }
    assert refresh_count == 1
    assert model_ir.operators[1].inputs == ["y0"]
    assert "y1" not in layout_state.logical
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_duplicate_cleanup_ordered_group_rolls_back_invalid_state() -> None:
    model_ir = ModelIR("invalid_ordered_duplicate_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm0": _tensor(
            "perm0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "perm1": _tensor(
            "perm1",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y0": _tensor("y0", [1, 3, 2, 2]),
        "y1": _tensor("y1", [1, 3, 2, 2]),
        "out": _tensor("out", [1]),
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm0"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["missing"], outputs=["out"]),
    ]

    with pytest.raises(RuntimeError, match="missing_input_tensor"):
        run_duplicate_fanout_cleanup(model_ir)

    assert len(model_ir.operators) == 3
    assert model_ir.operators[2].inputs == ["missing"]
    assert model_ir.operators[2].outputs == ["out"]


def test_duplicate_reshape_cleanup_uses_one_incremental_index_refresh(
    monkeypatch,
) -> None:
    model_ir = ModelIR("duplicate_reshape_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 6]),
        "shape0": _tensor(
            "shape0",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "shape1": _tensor(
            "shape1",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "y0": _tensor("y0", [2, 3]),
        "y1": _tensor("y1", [2, 3]),
        "out": _tensor("out", [2, 3]),
    }
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape0"], outputs=["y0"]),
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["y1"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_duplicate_reshape_fanout(model_ir)

    assert stats == {"removed_duplicate_reshape_fanout": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE", "IDENTITY"]
    assert model_ir.operators[1].inputs == ["y0"]
    assert "y1" not in model_ir.tensors


def test_clamp_cleanup_uses_one_incremental_index_refresh(monkeypatch) -> None:
    model_ir = ModelIR("clamp_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "zero": _tensor(
            "zero",
            [],
            data=np.asarray(0.0, dtype=np.float32),
        ),
        "maximum": _tensor("maximum", [1, 3]),
        "one": _tensor(
            "one",
            [],
            data=np.asarray(1.0, dtype=np.float32),
        ),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="MAXIMUM",
            inputs=["x", "zero"],
            outputs=["maximum"],
        ),
        OperatorIR(
            op_type="MINIMUM",
            inputs=["maximum", "one"],
            outputs=["out"],
        ),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_maximum_minimum_relu0to1_chains(model_ir)

    assert stats == {"rewritten_maximum_minimum_relu0to1_chains": 1}
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "RELU_0_TO_1"
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[0].outputs == ["out"]
    assert "maximum" not in model_ir.tensors


def test_ordered_clamp_cleanup_updates_layout_state_and_uses_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("ordered_clamp_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "zero": _tensor("zero", [], data=np.asarray(0.0, dtype=np.float32)),
        "maximum": _tensor("maximum", [1, 3]),
        "one": _tensor("one", [], data=np.asarray(1.0, dtype=np.float32)),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [
        OperatorIR("MAXIMUM", ["x", "zero"], ["maximum"]),
        OperatorIR("MINIMUM", ["maximum", "one"], ["out"]),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(
        model_ir,
        layout_state=layout_state,
    )

    stats = run_clamp_cleanup(
        model_ir,
        layout_state=layout_state,
        state_scope=state_scope,
    )
    pass_state, state_built = state_scope.acquire(
        model_ir=model_ir,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_maximum_minimum_relu0to1_chains": 1}
    assert refresh_count == 1
    assert state_built is False
    assert pass_state.graph_index.operator_indices("MAXIMUM") == []
    assert pass_state.graph_index.operator_indices("MINIMUM") == []
    assert pass_state.graph_index.operator_indices("RELU_0_TO_1") == [0]
    assert [op.op_type for op in model_ir.operators] == ["RELU_0_TO_1"]
    assert "maximum" not in model_ir.tensors
    assert "maximum" not in layout_state.logical
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_ordered_clamp_cleanup_skips_snapshot_without_chain(monkeypatch) -> None:
    model_ir = ModelIR("no_clamp_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "zero": _tensor("zero", [], data=np.asarray(0.0, dtype=np.float32)),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [OperatorIR("MAXIMUM", ["x", "zero"], ["out"])]
    snapshot_count = 0
    original_snapshot = ModelIRPassState.snapshot

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(state)

    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)

    stats = run_clamp_cleanup(model_ir)

    assert stats == {"rewritten_maximum_minimum_relu0to1_chains": 0}
    assert snapshot_count == 0
    assert [op.op_type for op in model_ir.operators] == ["MAXIMUM"]


def test_ordered_maximum_zero_relu_cleanup_updates_layout_and_diagnostics(
    monkeypatch,
) -> None:
    model_ir = ModelIR("ordered_maximum_zero_relu")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "zero": _tensor("zero", [], data=np.asarray(0.0, dtype=np.float32)),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [OperatorIR("MAXIMUM", ["x", "zero"], ["out"])]
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(
        model_ir,
        layout_state=layout_state,
    )

    stats = run_maximum_zero_relu_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    pass_state, state_built = state_scope.acquire(
        model_ir=model_ir,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_maximum_with_zero_input2_to_relu": 1}
    assert refresh_count == 1
    assert state_built is False
    assert pass_state.graph_index.operator_indices("MAXIMUM") == []
    assert pass_state.graph_index.operator_indices("RELU") == [0]
    assert model_ir.operators[0].op_type == "RELU"
    assert model_ir.operators[0].inputs == ["x"]
    assert "zero" not in model_ir.tensors
    assert "zero" not in layout_state.logical
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert diagnostics[0]["code"] == "canonicalize.maximum_zero_relu"
    assert diagnostics[0]["status"] == "changed"


def test_ordered_maximum_zero_relu_cleanup_skips_snapshot_without_maximum(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_maximum")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "y": _tensor("y", [1, 3]),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [OperatorIR("ADD", ["x", "y"], ["out"])]
    snapshot_count = 0
    original_snapshot = ModelIRPassState.snapshot

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(state)

    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)

    stats = run_maximum_zero_relu_cleanup(model_ir)

    assert stats == {"rewritten_maximum_with_zero_input2_to_relu": 0}
    assert snapshot_count == 0
    assert model_ir.operators[0].op_type == "ADD"


def test_ordered_consecutive_mul_constants_cleanup_folds_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _consecutive_mul_model(
        np.asarray([2.0, 3.0, 4.0], dtype=np.float32),
        np.asarray(0.5, dtype=np.float32),
    )
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_consecutive_mul_constants_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_fold_consecutive_mul_constants_chains": 1}
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "MUL"
    assert model_ir.operators[0].inputs[0] == "x"
    fused_name = model_ir.operators[0].inputs[1]
    np.testing.assert_array_equal(
        model_ir.tensors[fused_name].data,
        np.asarray([1.0, 1.5, 2.0], dtype=np.float32),
    )
    assert model_ir.tensors[fused_name].quantization == {
        "scale": [0.25],
        "zero_point": [0],
    }
    assert {"c0", "c1", "mid"}.isdisjoint(model_ir.tensors)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert diagnostics[0]["code"] == "canonicalize.fold_consecutive_mul_constants"
    assert diagnostics[0]["status"] == "changed"


def test_ordered_consecutive_mul_constants_cleanup_preserves_fanout() -> None:
    model_ir = _consecutive_mul_model(
        np.asarray(2.0, dtype=np.float32),
        np.asarray(3.0, dtype=np.float32),
        extra_mid_consumer=True,
    )

    stats = run_consecutive_mul_constants_cleanup(model_ir)

    assert stats == {"optimized_fold_consecutive_mul_constants_chains": 0}
    assert [op.op_type for op in model_ir.operators] == ["MUL", "MUL", "IDENTITY"]
    assert model_ir.operators[1].inputs == ["mid", "c1"]


@pytest.mark.parametrize(
    ("first_constant", "second_constant"),
    [
        (
            np.asarray(2, dtype=np.int32),
            np.asarray(3.0, dtype=np.float32),
        ),
        (
            np.asarray(np.inf, dtype=np.float32),
            np.asarray(3.0, dtype=np.float32),
        ),
    ],
)
def test_ordered_consecutive_mul_constants_cleanup_rejects_unsafe_constants(
    first_constant: np.ndarray,
    second_constant: np.ndarray,
) -> None:
    model_ir = _consecutive_mul_model(first_constant, second_constant)

    stats = run_consecutive_mul_constants_cleanup(model_ir)

    assert stats == {"optimized_fold_consecutive_mul_constants_chains": 0}
    assert [op.op_type for op in model_ir.operators] == ["MUL", "MUL"]


def test_squeeze_reshape_cleanup_uses_one_incremental_index_refresh(
    monkeypatch,
) -> None:
    model_ir = ModelIR("squeeze_reshape_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "squeezed": _tensor("squeezed", [3]),
        "shape": _tensor(
            "shape",
            [2],
            dtype="INT32",
            data=np.asarray([1, 3], dtype=np.int32),
        ),
        "reshaped": _tensor("reshaped", [1, 3]),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["x"],
            outputs=["squeezed"],
            options={"squeezeDims": [0]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["squeezed", "shape"],
            outputs=["reshaped"],
        ),
        OperatorIR(op_type="IDENTITY", inputs=["reshaped"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_squeeze_reshape_identity_chains(model_ir)

    assert stats == {"optimized_squeeze_reshape_identity_chains": 1}
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "IDENTITY"
    assert model_ir.operators[0].inputs == ["x"]
    assert "squeezed" not in model_ir.tensors
    assert "reshaped" not in model_ir.tensors


def test_ordered_squeeze_group_folds_unary_passthrough_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _squeeze_unary_reshape_model()
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_squeeze_reshape_identity_cleanup(
        model_ir,
        include_unary_passthrough=True,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert stats == {
        "optimized_squeeze_reshape_identity_chains": 0,
        "optimized_squeeze_unary_reshape_passthrough_chains": 1,
    }
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[0].outputs == ["y"]
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert [event["code"] for event in diagnostics] == [
        "cleanup.squeeze_unary_reshape_passthrough",
        "cleanup.squeeze_reshape_identity",
    ]
    assert [event["status"] for event in diagnostics] == ["changed", "skipped"]


def test_ordered_squeeze_group_preserves_unary_fanout_via_reordered_squeeze() -> None:
    model_ir = _squeeze_unary_reshape_model(fanout=True)

    stats = run_squeeze_reshape_identity_cleanup(
        model_ir,
        include_unary_passthrough=True,
    )

    assert stats["optimized_squeeze_unary_reshape_passthrough_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == [
        "RELU",
        "SQUEEZE",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[0].outputs == ["y"]
    assert model_ir.operators[1].inputs == ["y"]
    assert model_ir.operators[1].outputs == ["unary"]
    assert model_ir.operators[2].inputs == ["unary"]


def test_ordered_squeeze_group_rejects_nonzero_squeeze_axis() -> None:
    model_ir = _squeeze_unary_reshape_model(squeeze_axes=[1])

    stats = run_squeeze_reshape_identity_cleanup(
        model_ir,
        include_unary_passthrough=True,
    )

    assert stats == {
        "optimized_squeeze_reshape_identity_chains": 0,
        "optimized_squeeze_unary_reshape_passthrough_chains": 0,
    }
    assert [op.op_type for op in model_ir.operators] == [
        "SQUEEZE",
        "RELU",
        "RESHAPE",
    ]


def test_ordered_squeeze_reshape_cleanup_updates_layout_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("ordered_squeeze_reshape_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "squeezed": _tensor("squeezed", [3]),
        "shape": _tensor(
            "shape",
            [2],
            dtype="INT32",
            data=np.asarray([1, 3], dtype=np.int32),
        ),
        "reshaped": _tensor("reshaped", [1, 3]),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            "SQUEEZE",
            ["x"],
            ["squeezed"],
            options={"squeezeDims": [0]},
        ),
        OperatorIR("RESHAPE", ["squeezed", "shape"], ["reshaped"]),
        OperatorIR("IDENTITY", ["reshaped"], ["out"]),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_squeeze_reshape_identity_cleanup(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"optimized_squeeze_reshape_identity_chains": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["IDENTITY"]
    assert model_ir.operators[0].inputs == ["x"]
    assert "squeezed" not in layout_state.logical
    assert "reshaped" not in layout_state.logical
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_ordered_squeeze_reshape_cleanup_skips_snapshot_without_chain(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_squeeze_reshape_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "out": _tensor("out", [3]),
    }
    model_ir.operators = [
        OperatorIR(
            "SQUEEZE",
            ["x"],
            ["out"],
            options={"squeezeDims": [0]},
        )
    ]
    snapshot_count = 0
    original_snapshot = ModelIRPassState.snapshot

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(state)

    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)

    stats = run_squeeze_reshape_identity_cleanup(model_ir)

    assert stats == {"optimized_squeeze_reshape_identity_chains": 0}
    assert snapshot_count == 0
    assert [op.op_type for op in model_ir.operators] == ["SQUEEZE"]
