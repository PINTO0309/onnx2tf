from __future__ import annotations

import copy

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _infer_batch_matmul_output_shape_and_signature,
    _prune_dead_operators,
    _reconcile_static_tensor_shapes,
    _replace_expand_dims_and_squeeze_with_reshape,
    _restore_placeholder_matmul_flattened_inputs,
    _resolve_dynamic_reshape_shapes,
    _run_indexed_shape_convergence_cleanup,
    _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape_resolution import (
    resolve_dynamic_reshape_shapes,
)


def test_final_reshape_preserves_raw_minus_one_when_static_metadata_is_stale() -> None:
    model_ir = ModelIR("stale_static_high_rank_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 10, 4096, 1],
        shape_signature=[1, 10, 4096, 1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 5, 64, 64, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 64, 64, 2],
        shape_signature=[1, 5, 64, 64, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={
                "newShape": [1, 5, 64, 64, 2],
                "onnxRawNewShape": [1, -1, 64, 64, 2],
                "allowZero": False,
            },
        )
    ]

    module_ir = copy.deepcopy(model_ir)
    module_stats = resolve_dynamic_reshape_shapes(
        module_ir,
        prefer_runtime_inferable_from_onnx_raw=True,
        graph_index=ModelIRGraphIndex(module_ir),
    )
    graph_index = ModelIRGraphIndex(model_ir)

    class _NoFullOperatorIteration(list):
        def __iter__(self):
            raise AssertionError("unexpected full operator scan")

    model_ir.operators = _NoFullOperatorIteration(model_ir.operators)
    stats = _resolve_dynamic_reshape_shapes(
        model_ir,
        prefer_runtime_inferable_from_onnx_raw=True,
        graph_index=graph_index,
    )

    assert stats == {"resolved_dynamic_reshape_shapes": 1}
    assert module_stats == stats
    assert module_ir.operators[0].options == model_ir.operators[0].options
    np.testing.assert_array_equal(
        module_ir.tensors["shape"].data,
        model_ir.tensors["shape"].data,
    )
    assert module_ir.tensors["y"].shape == model_ir.tensors["y"].shape
    assert (
        module_ir.tensors["y"].shape_signature
        == model_ir.tensors["y"].shape_signature
    )
    assert model_ir.operators[0].options["newShape"] == [1, -1, 64, 64, 2]
    assert np.asarray(model_ir.tensors["shape"].data).tolist() == [1, -1, 64, 64, 2]
    assert model_ir.tensors["y"].shape == [1, 1, 64, 64, 2]
    assert model_ir.tensors["y"].shape_signature == [1, -1, 64, 64, 2]


def test_indexed_shape_convergence_matches_legacy_sequence(monkeypatch) -> None:
    model_ir = ModelIR("indexed_shape_convergence")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 10, 4096, 1],
        shape_signature=[1, 10, 4096, 1],
    )
    model_ir.tensors["dead"] = TensorIR(
        name="dead",
        dtype="FLOAT32",
        shape=[1, 10, 4096, 1],
        shape_signature=[1, 10, 4096, 1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[5],
        shape_signature=[5],
        data=np.asarray([1, 5, 64, 64, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 5, 64, 64, 2],
        shape_signature=[1, 5, 64, 64, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RELU",
            inputs=["x"],
            outputs=["dead"],
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={
                "newShape": [1, 5, 64, 64, 2],
                "onnxRawNewShape": [1, -1, 64, 64, 2],
                "allowZero": False,
            },
        ),
    ]
    legacy_model_ir = copy.deepcopy(model_ir)

    legacy_prune_stats = _prune_dead_operators(legacy_model_ir)
    legacy_first_reconcile_stats = _reconcile_static_tensor_shapes(
        legacy_model_ir
    )
    legacy_reshape_stats = _resolve_dynamic_reshape_shapes(legacy_model_ir)
    legacy_final_reconcile_stats = _reconcile_static_tensor_shapes(
        legacy_model_ir
    )
    legacy_stats = {
        "removed_dead_operators": legacy_prune_stats[
            "removed_dead_operators"
        ],
        "resolved_dynamic_reshape_shapes": legacy_reshape_stats[
            "resolved_dynamic_reshape_shapes"
        ],
        "reconciled_static_tensor_shapes": (
            legacy_first_reconcile_stats["reconciled_static_tensor_shapes"]
            + legacy_final_reconcile_stats["reconciled_static_tensor_shapes"]
        ),
    }

    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    indexed_stats = _run_indexed_shape_convergence_cleanup(model_ir)

    assert refresh_count == 1
    assert indexed_stats == legacy_stats
    assert [str(op.op_type) for op in model_ir.operators] == [
        str(op.op_type) for op in legacy_model_ir.operators
    ]
    for indexed_op, legacy_op in zip(
        model_ir.operators,
        legacy_model_ir.operators,
    ):
        assert indexed_op.inputs == legacy_op.inputs
        assert indexed_op.outputs == legacy_op.outputs
        assert indexed_op.options == legacy_op.options
    assert set(model_ir.tensors) == set(legacy_model_ir.tensors)
    for tensor_name, indexed_tensor in model_ir.tensors.items():
        legacy_tensor = legacy_model_ir.tensors[tensor_name]
        assert indexed_tensor.dtype == legacy_tensor.dtype
        assert indexed_tensor.shape == legacy_tensor.shape
        assert indexed_tensor.shape_signature == legacy_tensor.shape_signature
        if indexed_tensor.data is None or legacy_tensor.data is None:
            assert indexed_tensor.data is legacy_tensor.data
        else:
            np.testing.assert_array_equal(
                indexed_tensor.data,
                legacy_tensor.data,
            )


def test_shape_reconciliation_repairs_stale_flatten_shape_constant() -> None:
    model_ir = ModelIR("stale_flatten_shape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1024, 7, 7],
        shape_signature=[-1, 1024, 7, 7],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[1, 1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={"newShape": [1, 1], "onnxFlattenAxis": 1},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)

    assert stats == {"reconciled_static_tensor_shapes": 1}
    assert model_ir.operators[0].options["newShape"] == [-1, 50176]
    assert np.asarray(model_ir.tensors["shape"].data).tolist() == [-1, 50176]
    assert model_ir.tensors["y"].shape == [1, 50176]
    assert model_ir.tensors["y"].shape_signature == [-1, 50176]


def test_dynamic_unsqueeze_contract_survives_folded_higher_rank_input() -> None:
    model_ir = ModelIR("folded_squeeze_unsqueeze")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["scores"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 1, 1],
        shape_signature=[-1, -1, -1, -1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 1], dtype=np.int32),
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["scores"],
            options={"newShape": [1, 1]},
        )
    ]

    stats = _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(model_ir)

    assert stats == {"rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs": 1}
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE"]
    assert model_ir.operators[0].options["newShape"] == []
    assert np.asarray(model_ir.tensors["shape"].data).tolist() == [-1, 1]


def test_dynamic_rank1_unsqueeze_inserts_indexed_runtime_shape_pipeline(
    monkeypatch,
) -> None:
    model_ir = ModelIR("dynamic_rank1_unsqueeze")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1],
        shape_signature=[-1, 1],
    )
    reshape_op = OperatorIR(
        op_type="RESHAPE",
        inputs=["x", "shape"],
        outputs=["y"],
        options={"newShape": [1, 1]},
    )
    model_ir.operators = [reshape_op]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs": 1,
    }
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == [
        "SHAPE",
        "CONCATENATION",
        "RESHAPE",
    ]
    assert graph_index.operator_index(reshape_op) == 2
    assert graph_index.operator_indices("SHAPE") == [0]
    assert graph_index.operator_indices("CONCATENATION") == [1]
    assert graph_index.operator_indices("RESHAPE") == [2]
    assert reshape_op.inputs[0] == "x"
    assert reshape_op.inputs[1].endswith("_unsqueeze_runtime_shape")
    assert reshape_op.options["newShape"] == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_dynamic_squeeze_inserts_pre_ops_without_rebuilding_operator_list(
    monkeypatch,
) -> None:
    model_ir = ModelIR("dynamic_squeeze")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 288, 1, 144],
        shape_signature=[-1, 288, 1, -1],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 288, 144],
        shape_signature=[-1, 288, -1],
    )
    squeeze_op = OperatorIR(
        op_type="SQUEEZE",
        inputs=["x"],
        outputs=["y"],
        options={"squeezeDims": [2]},
    )
    model_ir.operators = [squeeze_op]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _replace_expand_dims_and_squeeze_with_reshape(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "replaced_expand_dims_and_squeeze_with_reshape": 1,
        "expand_dims_squeeze_rewrite_shape_tensors": 1,
    }
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == [
        "SHAPE",
        "GATHER",
        "RESHAPE",
    ]
    assert model_ir.operators[-1] is squeeze_op
    assert squeeze_op.inputs[0] == "x"
    assert squeeze_op.outputs == ["y"]
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_batch_matmul_shape_inference_accepts_unbatched_rhs() -> None:
    output_shape, output_signature = _infer_batch_matmul_output_shape_and_signature(
        shape_a=[1, 56, 56, 96],
        shape_b=[96, 384],
        signature_a=[-1, 56, 56, 96],
        signature_b=[96, 384],
        adj_x=False,
        adj_y=False,
    )

    assert output_shape == [1, 56, 56, 384]
    assert output_signature == [-1, 56, 56, 384]


def test_restores_placeholder_matmul_flatten_after_rank_recovery(monkeypatch) -> None:
    model_ir = ModelIR("placeholder_matmul_flatten")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 56, 56, 384],
        shape_signature=[-1, 56, 56, 384],
    )
    model_ir.tensors["flatten_shape"] = TensorIR(
        name="flatten_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([-1, 384], dtype=np.int32),
    )
    model_ir.tensors["x_flat"] = TensorIR(
        name="x_flat",
        dtype="FLOAT32",
        shape=[3136, 384],
        shape_signature=[-1, 384],
    )
    model_ir.tensors["weights"] = TensorIR(
        name="weights",
        dtype="FLOAT32",
        shape=[384, 96],
        shape_signature=[384, 96],
        data=np.ones((384, 96), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[3136, 96],
        shape_signature=[-1, 96],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "flatten_shape"],
            outputs=["x_flat"],
            options={
                "newShape": [-1, 384],
                "preserveDynamicShape": True,
                "onnxMatMulFlattenedPlaceholder": True,
            },
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["x_flat", "weights"],
            outputs=["y"],
            options={"adjX": False, "adjY": False},
        ),
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

    stats = _restore_placeholder_matmul_flattened_inputs(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    _reconcile_static_tensor_shapes(model_ir)

    assert stats == {"restored_placeholder_matmul_flattened_inputs": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["BATCH_MATMUL"]
    assert model_ir.operators[0].inputs == ["x", "weights"]
    assert graph_index.operator_indices("RESHAPE") == []
    assert graph_index.operator_indices("BATCH_MATMUL") == [0]
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert model_ir.tensors["y"].shape == [1, 56, 56, 96]
    assert model_ir.tensors["y"].shape_signature == [-1, 56, 56, 96]


def test_flatten_reconciliation_preserves_semantic_axis_after_layout_change() -> None:
    model_ir = ModelIR("semantic_flatten_axis")
    model_ir.inputs = ["x_nchw"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 192, 320],
        shape_signature=[1, 1, 192, 320],
    )
    model_ir.tensors["flatten_shape"] = TensorIR(
        name="flatten_shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([192, 320], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[192, 320],
        shape_signature=[192, 320],
    )
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["x_nchw", "flatten_shape"],
            ["y"],
            {
                "newShape": [192, 320],
                "onnxFlattenAxis": 3,
                "onnxFlattenInputShape": [1, 192, 320, 1],
            },
        )
    ]

    _reconcile_static_tensor_shapes(model_ir)

    assert model_ir.tensors["y"].shape == [61440, 1]
    assert model_ir.tensors["y"].shape_signature == [61440, 1]
    assert np.asarray(model_ir.tensors["flatten_shape"].data).tolist() == [
        61440,
        1,
    ]
