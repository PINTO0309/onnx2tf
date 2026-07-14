from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.window_partition_layout as window_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.window_partition_layout import (
    _optimize_window_partition_reshape_transpose_to_space_to_depth_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _quantization() -> QuantParamIR:
    return QuantParamIR(scale=[0.25], zero_point=[3])


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    produced_source: bool = False,
    quantized: bool = False,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "upstream",
            "input",
            "shape1",
            "r1",
            "perm",
            "t1",
            "shape2",
            "output",
        )
    }
    dtype = "INT8" if quantized else "FLOAT32"
    qparams = _quantization() if quantized else None
    model_ir.tensors.update(
        {
            names["input"]: _tensor(
                names["input"],
                [1, 4, 6, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
            names["shape1"]: _tensor(
                names["shape1"],
                [6],
                dtype="INT64",
                data=np.asarray([1, 2, 2, 3, 2, 3], dtype=np.int64),
            ),
            names["r1"]: _tensor(
                names["r1"],
                [1, 2, 2, 3, 2, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
            names["perm"]: _tensor(
                names["perm"],
                [6],
                dtype="INT32",
                data=np.asarray([0, 1, 3, 2, 4, 5], dtype=np.int32),
            ),
            names["t1"]: _tensor(
                names["t1"],
                [1, 2, 3, 2, 2, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
            names["shape2"]: _tensor(
                names["shape2"],
                [3],
                dtype="INT32",
                data=np.asarray([6, 4, 3], dtype=np.int32),
            ),
            names["output"]: _tensor(
                names["output"],
                [6, 4, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
        }
    )
    if produced_source:
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            [1, 4, 6, 3],
            dtype=dtype,
            quantization=copy.deepcopy(qparams),
        )
        model_ir.inputs.append(names["upstream"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["input"]])
        )
    else:
        model_ir.inputs.append(names["input"])
    model_ir.outputs.append(names["output"])
    model_ir.operators.extend(
        [
            OperatorIR(
                "RESHAPE",
                [names["input"], names["shape1"]],
                [names["r1"]],
                {"newShape": [1, 2, 2, 3, 2, 3]},
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["r1"], names["perm"]],
                [names["t1"]],
                {"legacy": True},
                axis_semantics={"perm": "layout"},
                version=3,
                onnx_node_name=f"{prefix}_transpose",
                onnx_op_type="Transpose",
            ),
            OperatorIR(
                "RESHAPE",
                [names["t1"], names["shape2"]],
                [names["output"]],
                {"newShape": [6, 4, 3], "onnxRawNewShape": [6, 4, 3]},
            ),
        ]
    )
    return names


def _operators(model_ir: ModelIR, names: dict[str, str]):
    reshape1 = next(op for op in model_ir.operators if op.outputs == [names["r1"]])
    transpose = next(op for op in model_ir.operators if op.outputs == [names["t1"]])
    reshape2 = next(op for op in model_ir.operators if op.outputs == [names["output"]])
    return reshape1, transpose, reshape2


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_window_partition_rewrites_multiple_chains_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("indexed_window_partition")
    branches = [
        _add_branch(model_ir, "plain"),
        _add_branch(model_ir, "produced", produced_source=True),
        _add_branch(model_ir, "quantized", quantized=True),
    ]
    original_transposes = [
        copy.deepcopy(_operators(model_ir, names)[1]) for names in branches
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
        model_ir
    )

    assert stats == {
        "optimized_window_partition_reshape_transpose_to_space_to_depth_chains": 3
    }
    assert refreshes == 1
    assert [op.op_type for op in model_ir.operators].count("RESHAPE") == 3
    for names, original in zip(branches, original_transposes):
        space_to_depth = next(
            op for op in model_ir.operators if op.outputs == [names["t1"]]
        )
        assert space_to_depth.op_type == "SPACE_TO_DEPTH"
        assert space_to_depth.inputs == [names["input"]]
        assert space_to_depth.options == {"blockSize": 2}
        assert space_to_depth.axis_semantics == original.axis_semantics
        assert space_to_depth.version == original.version
        assert space_to_depth.onnx_node_name == original.onnx_node_name
        assert space_to_depth.onnx_op_type == original.onnx_op_type
        assert model_ir.tensors[names["t1"]].shape == [1, 2, 3, 12]
        assert model_ir.tensors[names["output"]].shape == [6, 4, 3]


def test_window_partition_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_window_partition")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_window_partition_reshape_transpose_to_space_to_depth_chains": 1
    }
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize(
    ("input_signature", "r1_signature", "t1_signature", "output_signature"),
    [
        ([-1, 4, 6, 3], [-1, 2, 2, 3, 2, 3], [-1, 2, 3, 2, 2, 3], [-1, 4, 3]),
        ([1, -1, 6, 3], [1, -1, 2, 3, 2, 3], [1, -1, 3, 2, 2, 3], [-1, 4, 3]),
        ([1, 4, -1, 3], [1, 2, 2, -1, 2, 3], [1, 2, -1, 2, 2, 3], [-1, 4, 3]),
        ([1, 4, 6, -1], [1, 2, 2, 3, 2, -1], [1, 2, 3, 2, 2, -1], [6, 4, -1]),
    ],
)
def test_window_partition_preserves_one_dynamic_output_dimension(
    input_signature: list[int],
    r1_signature: list[int],
    t1_signature: list[int],
    output_signature: list[int],
) -> None:
    from onnx2tf.tflite_builder.lower_from_onnx2tf import (
        _reconcile_static_tensor_shapes,
        _resolve_dynamic_reshape_shapes,
    )

    model_ir = ModelIR("dynamic_window_partition")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["input"]].shape_signature = input_signature
    model_ir.tensors[names["r1"]].shape_signature = r1_signature
    model_ir.tensors[names["t1"]].shape_signature = t1_signature
    model_ir.tensors[names["output"]].shape_signature = output_signature

    stats = _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
        model_ir
    )
    _resolve_dynamic_reshape_shapes(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    assert stats == {
        "optimized_window_partition_reshape_transpose_to_space_to_depth_chains": 1
    }
    space_to_depth, reshape2 = model_ir.operators
    expected_s2d_signature = [
        input_signature[0],
        -1 if input_signature[1] < 0 else 2,
        -1 if input_signature[2] < 0 else 3,
        -1 if input_signature[3] < 0 else 12,
    ]
    assert model_ir.tensors[names["t1"]].shape_signature == expected_s2d_signature
    assert model_ir.tensors[names["output"]].shape_signature == output_signature
    assert reshape2.options["newShape"] == output_signature
    assert reshape2.options["onnxRawNewShape"] == output_signature
    assert reshape2.options["preserveDynamicShape"] is True
    assert (
        np.asarray(model_ir.tensors[names["shape2"]].data).tolist() == output_signature
    )
    assert space_to_depth.op_type == "SPACE_TO_DEPTH"


@pytest.mark.parametrize(
    "case",
    [
        "reshape1_arity",
        "reshape1_public",
        "reshape1_input_boundary",
        "duplicate_reshape1_producer",
        "input_missing",
        "input_unbound",
        "input_late_producer",
        "input_boundary_and_producer",
        "input_rank",
        "input_signature",
        "shape1_dtype",
        "shape1_metadata",
        "shape1_values",
        "shape1_produced",
        "shape1_input",
        "block_size_one",
        "block_size_rectangular",
        "spatial_equation",
        "reshape1_signature",
        "transpose_type",
        "transpose_arity",
        "transpose_perm",
        "perm_dtype",
        "perm_produced",
        "perm_input",
        "transpose_fanout",
        "transpose_public",
        "transpose_signature",
        "reshape2_type",
        "reshape2_arity",
        "reshape2_output_missing",
        "reshape2_output_input",
        "duplicate_reshape2_producer",
        "reshape2_shape",
        "reshape2_signature",
        "shape2_values",
        "shape2_produced",
        "shape2_input",
        "dtype_mismatch",
        "quantization_mismatch",
        "per_axis_quantization",
        "two_dynamic_dimensions",
        "dynamic_shape_shared",
        "dynamic_shape_public",
    ],
)
def test_window_partition_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = ModelIR("rejected_window_partition")
    names = _add_branch(model_ir, "branch")
    reshape1, transpose, reshape2 = _operators(model_ir, names)
    if case == "reshape1_arity":
        reshape1.inputs.append("extra")
    elif case == "reshape1_public":
        model_ir.outputs.append(names["r1"])
    elif case == "reshape1_input_boundary":
        model_ir.inputs.append(names["r1"])
    elif case == "duplicate_reshape1_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["input"]], [names["r1"]])
        )
    elif case == "input_missing":
        del model_ir.tensors[names["input"]]
    elif case == "input_unbound":
        model_ir.inputs.remove(names["input"])
    elif case == "input_late_producer":
        model_ir.inputs.remove(names["input"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["output"]], [names["input"]])
        )
    elif case == "input_boundary_and_producer":
        model_ir.operators.insert(
            0, OperatorIR("IDENTITY", [names["output"]], [names["input"]])
        )
    elif case == "input_rank":
        model_ir.tensors[names["input"]].shape = [1, 24, 3]
    elif case == "input_signature":
        model_ir.tensors[names["input"]].shape_signature[1] = 5
    elif case == "shape1_dtype":
        model_ir.tensors[names["shape1"]].dtype = "FLOAT32"
    elif case == "shape1_metadata":
        model_ir.tensors[names["shape1"]].shape = [2, 3]
    elif case == "shape1_values":
        model_ir.tensors[names["shape1"]].data[1] = 3
    elif case == "shape1_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["perm"]], [names["shape1"]])
        )
    elif case == "shape1_input":
        model_ir.inputs.append(names["shape1"])
    elif case == "block_size_one":
        model_ir.tensors[names["r1"]].shape = [1, 4, 1, 6, 1, 3]
        model_ir.tensors[names["r1"]].shape_signature = [1, 4, 1, 6, 1, 3]
        model_ir.tensors[names["shape1"]].data = np.asarray(
            [1, 4, 1, 6, 1, 3], dtype=np.int64
        )
    elif case == "block_size_rectangular":
        model_ir.tensors[names["r1"]].shape[4] = 3
        model_ir.tensors[names["r1"]].shape_signature[4] = 3
        model_ir.tensors[names["shape1"]].data[4] = 3
    elif case == "spatial_equation":
        model_ir.tensors[names["input"]].shape[1] = 5
        model_ir.tensors[names["input"]].shape_signature[1] = 5
    elif case == "reshape1_signature":
        model_ir.tensors[names["r1"]].shape_signature[1] = -1
    elif case == "transpose_type":
        transpose.op_type = "GATHER"
    elif case == "transpose_arity":
        transpose.inputs.append("extra")
    elif case == "transpose_perm":
        model_ir.tensors[names["perm"]].data[2] = 2
    elif case == "perm_dtype":
        model_ir.tensors[names["perm"]].data = np.asarray(
            [0, 1, 3, 2, 4, 5], dtype=np.int64
        )
    elif case == "perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["shape2"]], [names["perm"]])
        )
    elif case == "perm_input":
        model_ir.inputs.append(names["perm"])
    elif case == "transpose_fanout":
        model_ir.tensors["side"] = _tensor("side", [1])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["t1"]], ["side"]))
    elif case == "transpose_public":
        model_ir.outputs.append(names["t1"])
    elif case == "transpose_signature":
        model_ir.tensors[names["t1"]].shape_signature[1] = -1
    elif case == "reshape2_type":
        reshape2.op_type = "SQUEEZE"
    elif case == "reshape2_arity":
        reshape2.inputs.append("extra")
    elif case == "reshape2_output_missing":
        del model_ir.tensors[names["output"]]
    elif case == "reshape2_output_input":
        model_ir.inputs.append(names["output"])
    elif case == "duplicate_reshape2_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["input"]], [names["output"]])
        )
    elif case == "reshape2_shape":
        model_ir.tensors[names["output"]].shape[0] = 5
    elif case == "reshape2_signature":
        model_ir.tensors[names["output"]].shape_signature[0] = -1
    elif case == "shape2_values":
        model_ir.tensors[names["shape2"]].data[0] = 5
    elif case == "shape2_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["perm"]], [names["shape2"]])
        )
    elif case == "shape2_input":
        model_ir.inputs.append(names["shape2"])
    elif case == "dtype_mismatch":
        model_ir.tensors[names["t1"]].dtype = "FLOAT16"
    elif case == "quantization_mismatch":
        for key in ("input", "r1", "t1", "output"):
            model_ir.tensors[names[key]].quantization = _quantization()
        model_ir.tensors[names["output"]].quantization.scale = [0.5]
    elif case == "per_axis_quantization":
        for key in ("input", "r1", "t1", "output"):
            model_ir.tensors[names[key]].quantization = QuantParamIR(
                scale=[0.25, 0.5], zero_point=[0, 0], quantized_dimension=3
            )
    elif case in {
        "two_dynamic_dimensions",
        "dynamic_shape_shared",
        "dynamic_shape_public",
    }:
        model_ir.tensors[names["input"]].shape_signature = [-1, 4, 6, 3]
        model_ir.tensors[names["r1"]].shape_signature = [-1, 2, 2, 3, 2, 3]
        model_ir.tensors[names["t1"]].shape_signature = [-1, 2, 3, 2, 2, 3]
        model_ir.tensors[names["output"]].shape_signature = [-1, 4, 3]
        if case == "two_dynamic_dimensions":
            model_ir.tensors[names["input"]].shape_signature[3] = -1
            model_ir.tensors[names["r1"]].shape_signature[5] = -1
            model_ir.tensors[names["t1"]].shape_signature[5] = -1
            model_ir.tensors[names["output"]].shape_signature[2] = -1
        elif case == "dynamic_shape_shared":
            model_ir.tensors["shape_side"] = _tensor("shape_side", [1])
            model_ir.operators.append(
                OperatorIR("IDENTITY", [names["shape2"]], ["shape_side"])
            )
        elif case == "dynamic_shape_public":
            model_ir.outputs.append(names["shape2"])

    before = repr(model_ir)
    stats = _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
        model_ir
    )

    assert stats == {
        "optimized_window_partition_reshape_transpose_to_space_to_depth_chains": 0
    }
    assert repr(model_ir) == before


def test_window_partition_preflight_preserves_pruning_without_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_window_partition")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(window_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
        model_ir
    ) == {"optimized_window_partition_reshape_transpose_to_space_to_depth_chains": 0}
    assert "unused" not in model_ir.tensors
