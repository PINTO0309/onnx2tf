from __future__ import annotations

import copy

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

import onnx2tf.tflite_builder.passes.window_partition_layout as window_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.window_partition_layout import (
    _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains,
)


_STATS = "optimized_window_reverse_reshape_transpose_to_depth_to_space_chains"


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
    shared_shape1: str | None = None,
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
    if shared_shape1 is not None:
        names["shape1"] = shared_shape1
    dtype = "INT8" if quantized else "FLOAT32"
    qparams = _quantization() if quantized else None
    model_ir.tensors.update(
        {
            names["input"]: _tensor(
                names["input"],
                [6, 4, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
            names["r1"]: _tensor(
                names["r1"],
                [1, 2, 3, 2, 2, 3],
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
                [1, 2, 2, 3, 2, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
            names["shape2"]: _tensor(
                names["shape2"],
                [4],
                dtype="INT64",
                data=np.asarray([1, 4, 6, 3], dtype=np.int64),
            ),
            names["output"]: _tensor(
                names["output"],
                [1, 4, 6, 3],
                dtype=dtype,
                quantization=copy.deepcopy(qparams),
            ),
        }
    )
    if names["shape1"] not in model_ir.tensors:
        model_ir.tensors[names["shape1"]] = _tensor(
            names["shape1"],
            [6],
            dtype="INT64",
            data=np.asarray([1, 2, 3, 2, 2, 3], dtype=np.int64),
        )
    if produced_source:
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            [6, 4, 3],
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
                {
                    "newShape": [1, 2, 3, 2, 2, 3],
                    "onnxRawNewShape": [1, 2, 3, 2, 2, 3],
                    "legacy": True,
                },
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
                {"newShape": [1, 4, 6, 3]},
            ),
        ]
    )
    return names


def _operators(model_ir: ModelIR, names: dict[str, str]):
    reshape1 = next(op for op in model_ir.operators if op.outputs == [names["r1"]])
    transpose = next(op for op in model_ir.operators if op.outputs == [names["t1"]])
    reshape2 = next(
        op for op in model_ir.operators if op.outputs == [names["output"]]
    )
    return reshape1, transpose, reshape2


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_window_reverse_rewrites_multiple_chains_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("indexed_window_reverse")
    branches = [
        _add_branch(model_ir, "plain"),
        _add_branch(model_ir, "produced", produced_source=True),
        _add_branch(model_ir, "quantized", quantized=True),
        _add_branch(model_ir, "shared0", shared_shape1="shared_shape1"),
        _add_branch(model_ir, "shared1", shared_shape1="shared_shape1"),
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

    stats = _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir
    )

    assert stats == {_STATS: 5}
    assert refreshes == 1
    assert [op.op_type for op in model_ir.operators].count("RESHAPE") == 5
    for names, original in zip(branches, original_transposes):
        reshape = next(op for op in model_ir.operators if op.outputs == [names["r1"]])
        depth_to_space = next(
            op for op in model_ir.operators if op.outputs == [names["output"]]
        )
        assert reshape.options == {
            "newShape": [1, 2, 3, 12],
            "onnxRawNewShape": [1, 2, 3, 12],
            "legacy": True,
        }
        assert model_ir.tensors[names["r1"]].shape == [1, 2, 3, 12]
        assert depth_to_space.op_type == "DEPTH_TO_SPACE"
        assert depth_to_space.inputs == [names["r1"]]
        assert depth_to_space.options == {"blockSize": 2}
        assert depth_to_space.axis_semantics == original.axis_semantics
        assert depth_to_space.version == original.version
        assert depth_to_space.onnx_node_name == original.onnx_node_name
        assert depth_to_space.onnx_op_type == original.onnx_op_type

    first_shared_reshape = next(
        op for op in model_ir.operators if op.outputs == [branches[3]["r1"]]
    )
    second_shared_reshape = next(
        op for op in model_ir.operators if op.outputs == [branches[4]["r1"]]
    )
    assert first_shared_reshape.inputs[1] == "shared_shape1_d2s_shape"
    assert second_shared_reshape.inputs[1] == "shared_shape1"
    assert np.asarray(model_ir.tensors["shared_shape1"].data).tolist() == [
        1,
        2,
        3,
        12,
    ]


def test_window_reverse_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_window_reverse")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize(
    ("input_signature", "r1_signature", "t1_signature", "output_signature"),
    [
        ([-1, 4, 3], [-1, 2, 3, 2, 2, 3], [-1, 2, 2, 3, 2, 3], [-1, 4, 6, 3]),
        ([-1, 4, 3], [1, -1, 3, 2, 2, 3], [1, -1, 2, 3, 2, 3], [1, -1, 6, 3]),
        ([-1, 4, 3], [1, 2, -1, 2, 2, 3], [1, 2, 2, -1, 2, 3], [1, 4, -1, 3]),
        ([6, 4, -1], [1, 2, 3, 2, 2, -1], [1, 2, 2, 3, 2, -1], [1, 4, 6, -1]),
    ],
)
def test_window_reverse_preserves_one_dynamic_reshape_dimension(
    input_signature: list[int],
    r1_signature: list[int],
    t1_signature: list[int],
    output_signature: list[int],
) -> None:
    from onnx2tf.tflite_builder.lower_from_onnx2tf import (
        _reconcile_static_tensor_shapes,
        _resolve_dynamic_reshape_shapes,
    )

    model_ir = ModelIR("dynamic_window_reverse")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["input"]].shape_signature = input_signature
    model_ir.tensors[names["r1"]].shape_signature = r1_signature
    model_ir.tensors[names["t1"]].shape_signature = t1_signature
    model_ir.tensors[names["output"]].shape_signature = output_signature

    stats = _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir
    )
    _resolve_dynamic_reshape_shapes(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    expected_reshape_signature = [
        r1_signature[0],
        r1_signature[1],
        r1_signature[2],
        -1 if r1_signature[5] < 0 else 12,
    ]
    assert stats == {_STATS: 1}
    reshape, depth_to_space = model_ir.operators
    assert reshape.options["newShape"] == expected_reshape_signature
    assert reshape.options["onnxRawNewShape"] == expected_reshape_signature
    assert reshape.options["preserveDynamicShape"] is True
    assert model_ir.tensors[names["r1"]].shape_signature == expected_reshape_signature
    assert np.asarray(model_ir.tensors[names["shape1"]].data).tolist() == (
        expected_reshape_signature
    )
    assert depth_to_space.op_type == "DEPTH_TO_SPACE"
    assert model_ir.tensors[names["output"]].shape_signature == output_signature


@pytest.mark.parametrize(
    "case",
    [
        "reshape1_arity",
        "reshape1_public",
        "duplicate_reshape1_producer",
        "input_missing",
        "input_unbound",
        "input_late_producer",
        "input_boundary_and_producer",
        "input_signature",
        "shape1_dtype",
        "shape1_values",
        "shape1_produced",
        "shape1_input",
        "shape1_public",
        "block_size_one",
        "block_size_rectangular",
        "flatten_equation",
        "reshape1_signature",
        "transpose_type",
        "transpose_arity",
        "transpose_perm",
        "perm_dtype",
        "perm_produced",
        "transpose_fanout",
        "transpose_public",
        "transpose_signature",
        "reshape2_type",
        "reshape2_arity",
        "output_missing",
        "output_input",
        "duplicate_output_producer",
        "output_shape",
        "output_signature",
        "shape2_values",
        "shape2_dtype",
        "shape2_produced",
        "dtype_mismatch",
        "quantization_mismatch",
        "per_axis_quantization",
        "two_dynamic_dimensions",
    ],
)
def test_window_reverse_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = ModelIR("rejected_window_reverse")
    names = _add_branch(model_ir, "branch")
    reshape1, transpose, reshape2 = _operators(model_ir, names)
    if case == "reshape1_arity":
        reshape1.inputs.append("extra")
    elif case == "reshape1_public":
        model_ir.outputs.append(names["r1"])
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
    elif case == "input_signature":
        model_ir.tensors[names["input"]].shape_signature[0] = -1
    elif case == "shape1_dtype":
        model_ir.tensors[names["shape1"]].dtype = "FLOAT32"
    elif case == "shape1_values":
        model_ir.tensors[names["shape1"]].data[1] = 3
    elif case == "shape1_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["perm"]], [names["shape1"]])
        )
    elif case == "shape1_input":
        model_ir.inputs.append(names["shape1"])
    elif case == "shape1_public":
        model_ir.outputs.append(names["shape1"])
    elif case == "block_size_one":
        model_ir.tensors[names["input"]].shape[1] = 1
        model_ir.tensors[names["input"]].shape_signature[1] = 1
        model_ir.tensors[names["r1"]].shape = [1, 2, 3, 1, 1, 3]
        model_ir.tensors[names["r1"]].shape_signature = [1, 2, 3, 1, 1, 3]
        model_ir.tensors[names["shape1"]].data = np.asarray(
            [1, 2, 3, 1, 1, 3], dtype=np.int64
        )
    elif case == "block_size_rectangular":
        model_ir.tensors[names["r1"]].shape[4] = 3
        model_ir.tensors[names["r1"]].shape_signature[4] = 3
        model_ir.tensors[names["shape1"]].data[4] = 3
    elif case == "flatten_equation":
        model_ir.tensors[names["input"]].shape[0] = 5
        model_ir.tensors[names["input"]].shape_signature[0] = 5
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
    elif case == "transpose_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 2, 2, 3, 2, 3])
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [names["t1"]], ["side"]))
    elif case == "transpose_public":
        model_ir.outputs.append(names["t1"])
    elif case == "transpose_signature":
        model_ir.tensors[names["t1"]].shape_signature[1] = -1
    elif case == "reshape2_type":
        reshape2.op_type = "SQUEEZE"
    elif case == "reshape2_arity":
        reshape2.inputs.append("extra")
    elif case == "output_missing":
        del model_ir.tensors[names["output"]]
    elif case == "output_input":
        model_ir.inputs.append(names["output"])
    elif case == "duplicate_output_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["input"]], [names["output"]])
        )
    elif case == "output_shape":
        model_ir.tensors[names["output"]].shape[1] = 5
    elif case == "output_signature":
        model_ir.tensors[names["output"]].shape_signature[1] = -1
    elif case == "shape2_values":
        model_ir.tensors[names["shape2"]].data[1] = 5
    elif case == "shape2_dtype":
        model_ir.tensors[names["shape2"]].dtype = "FLOAT32"
    elif case == "shape2_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["perm"]], [names["shape2"]])
        )
    elif case == "dtype_mismatch":
        model_ir.tensors[names["t1"]].dtype = "FLOAT16"
    elif case == "quantization_mismatch":
        for key in ("input", "r1", "t1", "output"):
            model_ir.tensors[names[key]].quantization = _quantization()
        model_ir.tensors[names["output"]].quantization.scale = [0.5]
    elif case == "per_axis_quantization":
        for key in ("input", "r1", "t1", "output"):
            model_ir.tensors[names[key]].quantization = QuantParamIR(
                scale=[0.25, 0.5], zero_point=[0, 0], quantized_dimension=2
            )
    elif case == "two_dynamic_dimensions":
        model_ir.tensors[names["input"]].shape_signature = [-1, 4, -1]
        model_ir.tensors[names["r1"]].shape_signature = [-1, 2, 3, 2, 2, -1]
        model_ir.tensors[names["t1"]].shape_signature = [-1, 2, 2, 3, 2, -1]
        model_ir.tensors[names["output"]].shape_signature = [-1, 4, 6, -1]

    before = repr(model_ir)
    stats = _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_window_reverse_quantization_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_window_reverse")
    names = _add_branch(model_ir, "branch")
    shape1 = model_ir.tensors[names["shape1"]]
    shape1.quantization = Unclonable()
    model_ir.tensors["shape_side"] = _tensor("shape_side", [6])
    model_ir.outputs.append("shape_side")
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["shape1"]], ["shape_side"])
    )
    before = repr(model_ir)

    stats = _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_window_reverse_preflight_preserves_pruning_without_index(monkeypatch) -> None:
    model_ir = ModelIR("no_window_reverse")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(window_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
        model_ir
    ) == {_STATS: 0}
    assert "unused" not in model_ir.tensors


def test_window_reverse_real_onnx_uses_indexed_production_owner() -> None:
    from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir

    input_value = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, [6, 4, 3]
    )
    output_value = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, [1, 4, 6, 3]
    )
    initializers = [
        numpy_helper.from_array(
            np.asarray([1, 2, 3, 2, 2, 3], dtype=np.int64), name="shape1"
        ),
        numpy_helper.from_array(
            np.asarray([1, 4, 6, 3], dtype=np.int64), name="shape2"
        ),
    ]
    nodes = [
        helper.make_node("Reshape", ["x", "shape1"], ["r1"], name="reshape1"),
        helper.make_node(
            "Transpose",
            ["r1"],
            ["t1"],
            name="transpose",
            perm=[0, 1, 3, 2, 4, 5],
        ),
        helper.make_node("Reshape", ["t1", "shape2"], ["y"], name="reshape2"),
    ]
    graph = helper.make_graph(
        nodes,
        "window_reverse",
        [input_value],
        [output_value],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )

    model_ir = lower_onnx_to_ir(model, "window_reverse")

    assert [operator.op_type for operator in model_ir.operators] == [
        "RESHAPE",
        "DEPTH_TO_SPACE",
    ]
    reshape, depth_to_space = model_ir.operators
    assert reshape.options["newShape"] == [1, 2, 3, 12]
    assert model_ir.tensors["r1"].shape == [1, 2, 3, 12]
    assert depth_to_space.inputs == ["r1"]
    assert depth_to_space.outputs == ["y"]
    assert depth_to_space.options == {"blockSize": 2}
