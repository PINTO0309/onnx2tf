from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.quantized_hardsigmoid as hardsigmoid_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.quantized_hardsigmoid import (
    _optimize_dequant_hardsigmoid_quantize_chains,
)


def _tensor(
    name: str,
    *,
    dtype: str,
    shape: list[int],
    signature: list[int],
    data: np.ndarray | None = None,
    quantization: QuantParamIR | dict | None = None,
    is_variable: bool = True,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
    )


def _grid(dtype: str) -> QuantParamIR:
    return QuantParamIR(
        scale=[0.1],
        zero_point=[0 if dtype == "INT8" else 10],
        quantized_dimension=0,
    )


def _constant(name: str, value: float) -> TensorIR:
    return _tensor(
        name,
        dtype="FLOAT32",
        shape=[1],
        signature=[1],
        data=np.asarray([value], dtype=np.float32),
        is_variable=False,
    )


def _expanded_hardsigmoid_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_quantized_expanded_hardsigmoid")
    scalar_values = {
        "alpha": 0.2,
        "beta": 0.5,
        "low": 0.0,
        "high": 1.0,
    }
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        dtype = "INT8" if branch_index % 2 == 0 else "UINT8"
        names = {
            "quantized_input": f"{prefix}_quantized_input",
            "float_input": f"{prefix}_float_input",
            "multiply_output": f"{prefix}_multiply_output",
            "add_output": f"{prefix}_add_output",
            "maximum_output": f"{prefix}_maximum_output",
            "float_output": f"{prefix}_float_output",
            "quantized_output": f"{prefix}_quantized_output",
        }
        constant_names = {key: f"{prefix}_{key}" for key in scalar_values}
        model_ir.inputs.append(names["quantized_input"])
        model_ir.outputs.append(names["quantized_output"])
        for key, name in names.items():
            is_quantized = key in {"quantized_input", "quantized_output"}
            model_ir.tensors[name] = _tensor(
                name,
                dtype=dtype if is_quantized else "FLOAT32",
                shape=[1, 4, 4, 2],
                signature=[-1, 4, 4, 2],
                quantization=_grid(dtype) if is_quantized else None,
            )
        for key, value in scalar_values.items():
            model_ir.tensors[constant_names[key]] = _constant(
                constant_names[key],
                value,
            )

        multiply_inputs = (
            [names["float_input"], constant_names["alpha"]]
            if branch_index == 0
            else [constant_names["alpha"], names["float_input"]]
        )
        add_inputs = (
            [constant_names["beta"], names["multiply_output"]]
            if branch_index == 0
            else [names["multiply_output"], constant_names["beta"]]
        )
        maximum_inputs = (
            [names["add_output"], constant_names["low"]]
            if branch_index == 0
            else [constant_names["low"], names["add_output"]]
        )
        minimum_inputs = (
            [constant_names["high"], names["maximum_output"]]
            if branch_index == 0
            else [names["maximum_output"], constant_names["high"]]
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "DEQUANTIZE",
                    [names["quantized_input"]],
                    [names["float_input"]],
                ),
                OperatorIR(
                    "MUL",
                    multiply_inputs,
                    [names["multiply_output"]],
                    options={"fusedActivationFunction": "NONE"},
                    version=2,
                    onnx_node_name=f"{prefix}_alpha",
                    onnx_op_type="Mul",
                ),
                OperatorIR(
                    "ADD",
                    add_inputs,
                    [names["add_output"]],
                    options={"fusedActivationFunction": "NONE"},
                    version=3,
                    onnx_node_name=f"{prefix}_beta",
                    onnx_op_type="Add",
                ),
                OperatorIR(
                    "MAXIMUM",
                    maximum_inputs,
                    [names["maximum_output"]],
                    version=4,
                    onnx_node_name=f"{prefix}_low",
                    onnx_op_type="Max",
                ),
                OperatorIR(
                    "MINIMUM",
                    minimum_inputs,
                    [names["float_output"]],
                    version=5,
                    onnx_node_name=f"{prefix}_high",
                    onnx_op_type="Min",
                ),
                OperatorIR(
                    "QUANTIZE",
                    [names["float_output"]],
                    [names["quantized_output"]],
                ),
            ]
        )
    return model_ir


def _assert_index_current(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def _add_shared_constant_consumers(model_ir: ModelIR) -> None:
    for constant_kind in ("alpha", "beta", "low", "high"):
        source = f"branch0_{constant_kind}"
        output = f"{source}_side"
        model_ir.tensors[output] = _constant(output, 0.0)
        model_ir.outputs.append(output)
        model_ir.operators.append(OperatorIR("IDENTITY", [source], [output]))


def test_quantized_hardsigmoid_folds_multiple_dtypes_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _expanded_hardsigmoid_model()
    input_quantizations = [
        model_ir.tensors[f"branch{index}_quantized_input"].quantization
        for index in range(2)
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_dequant_hardsigmoid_quantize_chains(model_ir)

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 2}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "ADD",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "ADD",
        "MAXIMUM",
        "MINIMUM",
    ]
    expected_values = {
        0: {"alpha": 2, "beta": 5, "low": 0, "high": 10},
        1: {"alpha": 12, "beta": 15, "low": 10, "high": 20},
    }
    for branch_index in range(2):
        prefix = f"branch{branch_index}"
        dtype = "INT8" if branch_index == 0 else "UINT8"
        branch_ops = model_ir.operators[branch_index * 4 : branch_index * 4 + 4]
        assert branch_ops[0].onnx_node_name == f"{prefix}_alpha"
        assert branch_ops[0].options == {"fusedActivationFunction": "NONE"}
        assert branch_ops[0].version == 2
        assert branch_ops[1].onnx_node_name == f"{prefix}_beta"
        assert branch_ops[1].version == 3
        assert branch_ops[2].onnx_node_name == f"{prefix}_low"
        assert branch_ops[2].version == 4
        assert branch_ops[3].onnx_node_name == f"{prefix}_high"
        assert branch_ops[3].version == 5
        assert f"{prefix}_quantized_input" in branch_ops[0].inputs
        assert branch_ops[3].outputs == [f"{prefix}_quantized_output"]
        for tensor_kind in (
            "multiply_output",
            "add_output",
            "maximum_output",
            "quantized_output",
        ):
            tensor = model_ir.tensors[f"{prefix}_{tensor_kind}"]
            assert tensor.dtype == dtype
            assert tensor.quantization == input_quantizations[branch_index]
            assert tensor.quantization is not input_quantizations[branch_index]
        for constant_kind, expected in expected_values[branch_index].items():
            tensor = model_ir.tensors[f"{prefix}_{constant_kind}"]
            assert tensor.dtype == dtype
            assert np.asarray(tensor.data).reshape(-1).tolist() == [expected]
            assert tensor.quantization == input_quantizations[branch_index]
    assert "branch0_float_input" not in model_ir.tensors
    assert "branch0_float_output" not in model_ir.tensors
    assert "branch1_float_input" not in model_ir.tensors
    assert "branch1_float_output" not in model_ir.tensors


def test_quantized_hardsigmoid_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _expanded_hardsigmoid_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_dequant_hardsigmoid_quantize_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_quantized_hardsigmoid_preserves_shared_constants_with_clones() -> None:
    model_ir = _expanded_hardsigmoid_model(branches=1)
    _add_shared_constant_consumers(model_ir)
    original_constants = {
        kind: copy.deepcopy(model_ir.tensors[f"branch0_{kind}"])
        for kind in ("alpha", "beta", "low", "high")
    }

    stats = _optimize_dequant_hardsigmoid_quantize_chains(model_ir)

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 1}
    for kind, original in original_constants.items():
        retained = model_ir.tensors[f"branch0_{kind}"]
        assert retained.dtype == original.dtype
        np.testing.assert_array_equal(retained.data, original.data)
        clone = model_ir.tensors[f"branch0_{kind}_q"]
        assert clone.dtype == "INT8"
        assert (
            clone.quantization
            == model_ir.tensors["branch0_quantized_input"].quantization
        )
    assert model_ir.operators[0].inputs[1] == "branch0_alpha_q"
    assert model_ir.operators[1].inputs[0] == "branch0_beta_q"
    assert model_ir.operators[2].inputs[1] == "branch0_low_q"
    assert model_ir.operators[3].inputs[0] == "branch0_high_q"


def test_quantized_hardsigmoid_clones_public_constants() -> None:
    model_ir = _expanded_hardsigmoid_model(branches=1)
    model_ir.outputs.extend(
        ["branch0_alpha", "branch0_beta", "branch0_low", "branch0_high"]
    )

    stats = _optimize_dequant_hardsigmoid_quantize_chains(model_ir)

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 1}
    for kind, value in {
        "alpha": 0.2,
        "beta": 0.5,
        "low": 0.0,
        "high": 1.0,
    }.items():
        retained = model_ir.tensors[f"branch0_{kind}"]
        assert retained.dtype == "FLOAT32"
        np.testing.assert_allclose(retained.data, [value], rtol=0.0, atol=1e-7)
        assert f"branch0_{kind}_q" in model_ir.tensors


def test_quantized_hardsigmoid_clone_failure_is_transactional(
    monkeypatch,
) -> None:
    model_ir = _expanded_hardsigmoid_model(branches=1)
    before = repr(model_ir)
    clone_count = 0
    original_clone = hardsigmoid_module._clone_quantization

    def fail_second_clone(quantization):
        nonlocal clone_count
        clone_count += 1
        if clone_count == 2:
            raise RuntimeError("injected clone failure")
        return original_clone(quantization)

    monkeypatch.setattr(
        hardsigmoid_module,
        "_clone_quantization",
        fail_second_clone,
    )

    stats = _optimize_dequant_hardsigmoid_quantize_chains(model_ir)

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 0}
    assert clone_count == 2
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    [
        "float_input_public_output",
        "float_input_public_input",
        "multiply_output_public_output",
        "multiply_output_public_input",
        "add_output_public_output",
        "add_output_public_input",
        "maximum_output_public_output",
        "maximum_output_public_input",
        "float_output_public_output",
        "float_output_public_input",
        "quantized_output_public_input",
        "float_input_fanout",
        "multiply_output_fanout",
        "add_output_fanout",
        "maximum_output_fanout",
        "float_output_fanout",
        "duplicate_float_input_producer",
        "duplicate_multiply_output_producer",
        "duplicate_add_output_producer",
        "duplicate_maximum_output_producer",
        "duplicate_float_output_producer",
        "duplicate_quantized_output_producer",
        "reverse_dequantize_multiply_order",
        "reverse_minimum_quantize_order",
        "wrong_dequantize_type",
        "wrong_multiply_type",
        "wrong_add_type",
        "wrong_maximum_type",
        "wrong_minimum_type",
        "wrong_quantize_type",
        "wrong_dequantize_arity",
        "wrong_multiply_arity",
        "wrong_add_arity",
        "wrong_maximum_arity",
        "wrong_minimum_arity",
        "wrong_quantize_arity",
        "missing_quantized_input",
        "missing_float_input",
        "missing_multiply_output",
        "missing_add_output",
        "missing_maximum_output",
        "missing_float_output",
        "missing_quantized_output",
        "unsupported_quantized_dtype",
        "quantized_dtype_mismatch",
        "missing_input_quantization",
        "missing_output_quantization",
        "per_axis_quantization",
        "scale_mismatch",
        "near_scale_mismatch",
        "zero_point_mismatch",
        "negative_scale",
        "nonfinite_scale",
        "zero_point_out_of_range",
        "nonfloat_intermediate",
        "float_dtype_mismatch",
        "shape_mismatch",
        "signature_mismatch",
        "invalid_shape",
        "invalid_signature",
        "missing_constant",
        "missing_constant_data",
        "nonsingleton_constant",
        "nonfinite_constant",
        "produced_constant",
        "invalid_constant_signature",
        "unrepresentable_constant",
        "wrong_multiply_data_edge",
        "wrong_add_data_edge",
        "wrong_maximum_data_edge",
        "wrong_minimum_data_edge",
    ],
)
def test_quantized_hardsigmoid_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _expanded_hardsigmoid_model(branches=1)
    dequantize, multiply, add, maximum, minimum, quantize = model_ir.operators
    bridge_names = {
        "float_input": "branch0_float_input",
        "multiply_output": "branch0_multiply_output",
        "add_output": "branch0_add_output",
        "maximum_output": "branch0_maximum_output",
        "float_output": "branch0_float_output",
    }

    if case.endswith("_public_output"):
        key = case.removesuffix("_public_output")
        model_ir.outputs.append(bridge_names[key])
    elif case.endswith("_public_input") and case != "quantized_output_public_input":
        key = case.removesuffix("_public_input")
        model_ir.inputs.append(bridge_names[key])
    elif case == "quantized_output_public_input":
        model_ir.inputs.append("branch0_quantized_output")
    elif case.endswith("_fanout"):
        key = case.removesuffix("_fanout")
        source = bridge_names[key]
        model_ir.tensors["side"] = _tensor(
            "side",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            signature=[-1, 4, 4, 2],
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    elif case.startswith("duplicate_"):
        key = case.removeprefix("duplicate_").removesuffix("_producer")
        output_name = (
            "branch0_quantized_output"
            if key == "quantized_output"
            else bridge_names[key]
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["branch0_quantized_input"], [output_name])
        )
    elif case == "reverse_dequantize_multiply_order":
        model_ir.operators = [multiply, dequantize, add, maximum, minimum, quantize]
    elif case == "reverse_minimum_quantize_order":
        model_ir.operators = [dequantize, multiply, add, maximum, quantize, minimum]
    elif case.startswith("wrong_") and case.endswith("_type"):
        operator_name = case.removeprefix("wrong_").removesuffix("_type")
        operators = {
            "dequantize": dequantize,
            "multiply": multiply,
            "add": add,
            "maximum": maximum,
            "minimum": minimum,
            "quantize": quantize,
        }
        operators[operator_name].op_type = "IDENTITY"
    elif case.startswith("wrong_") and case.endswith("_arity"):
        operator_name = case.removeprefix("wrong_").removesuffix("_arity")
        operators = {
            "dequantize": dequantize,
            "multiply": multiply,
            "add": add,
            "maximum": maximum,
            "minimum": minimum,
            "quantize": quantize,
        }
        operators[operator_name].inputs.append(operators[operator_name].inputs[0])
    elif case.startswith("missing_") and case not in {
        "missing_input_quantization",
        "missing_output_quantization",
        "missing_constant",
        "missing_constant_data",
    }:
        key = case.removeprefix("missing_")
        tensor_name = {
            "quantized_input": "branch0_quantized_input",
            "float_input": "branch0_float_input",
            "multiply_output": "branch0_multiply_output",
            "add_output": "branch0_add_output",
            "maximum_output": "branch0_maximum_output",
            "float_output": "branch0_float_output",
            "quantized_output": "branch0_quantized_output",
        }[key]
        del model_ir.tensors[tensor_name]
    elif case == "unsupported_quantized_dtype":
        model_ir.tensors["branch0_quantized_input"].dtype = "INT16"
        model_ir.tensors["branch0_quantized_output"].dtype = "INT16"
    elif case == "quantized_dtype_mismatch":
        model_ir.tensors["branch0_quantized_output"].dtype = "UINT8"
    elif case == "missing_input_quantization":
        model_ir.tensors["branch0_quantized_input"].quantization = None
    elif case == "missing_output_quantization":
        model_ir.tensors["branch0_quantized_output"].quantization = None
    elif case == "per_axis_quantization":
        model_ir.tensors["branch0_quantized_input"].quantization = QuantParamIR(
            [0.1, 0.2], [0, 0], 3
        )
    elif case == "scale_mismatch":
        model_ir.tensors["branch0_quantized_output"].quantization.scale = [0.2]
    elif case == "near_scale_mismatch":
        model_ir.tensors["branch0_quantized_output"].quantization.scale = [0.10000001]
    elif case == "zero_point_mismatch":
        model_ir.tensors["branch0_quantized_output"].quantization.zero_point = [1]
    elif case == "negative_scale":
        model_ir.tensors["branch0_quantized_input"].quantization.scale = [-0.1]
        model_ir.tensors["branch0_quantized_output"].quantization.scale = [-0.1]
    elif case == "nonfinite_scale":
        model_ir.tensors["branch0_quantized_input"].quantization.scale = [np.inf]
        model_ir.tensors["branch0_quantized_output"].quantization.scale = [np.inf]
    elif case == "zero_point_out_of_range":
        model_ir.tensors["branch0_quantized_input"].quantization.zero_point = [-129]
        model_ir.tensors["branch0_quantized_output"].quantization.zero_point = [-129]
    elif case == "nonfloat_intermediate":
        for name in bridge_names.values():
            model_ir.tensors[name].dtype = "INT32"
    elif case == "float_dtype_mismatch":
        model_ir.tensors["branch0_add_output"].dtype = "FLOAT16"
    elif case == "shape_mismatch":
        model_ir.tensors["branch0_maximum_output"].shape = [1, 4, 2, 4]
    elif case == "signature_mismatch":
        model_ir.tensors["branch0_float_output"].shape_signature = [1, 4, 4, 2]
    elif case == "invalid_shape":
        model_ir.tensors["branch0_multiply_output"].shape = [1, None, 4, 2]
    elif case == "invalid_signature":
        model_ir.tensors["branch0_add_output"].shape_signature = [
            -1,
            None,
            4,
            2,
        ]
    elif case == "missing_constant":
        del model_ir.tensors["branch0_high"]
    elif case == "missing_constant_data":
        model_ir.tensors["branch0_high"].data = None
    elif case == "nonsingleton_constant":
        model_ir.tensors["branch0_high"].data = np.asarray([1.0, 1.0], dtype=np.float32)
    elif case == "nonfinite_constant":
        model_ir.tensors["branch0_high"].data = np.asarray([np.inf], dtype=np.float32)
    elif case == "produced_constant":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_input"],
                ["branch0_high"],
            )
        )
    elif case == "invalid_constant_signature":
        model_ir.tensors["branch0_high"].shape_signature = [None]
    elif case == "unrepresentable_constant":
        model_ir.tensors["branch0_high"].data = np.asarray([1.06], dtype=np.float32)
    elif case == "wrong_multiply_data_edge":
        multiply.inputs[0] = "branch0_alpha"
    elif case == "wrong_add_data_edge":
        add.inputs[1] = "branch0_beta"
    elif case == "wrong_maximum_data_edge":
        maximum.inputs[0] = "branch0_low"
    elif case == "wrong_minimum_data_edge":
        minimum.inputs[1] = "branch0_high"

    before = repr(model_ir)
    stats = _optimize_dequant_hardsigmoid_quantize_chains(model_ir)

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 0}
    assert repr(model_ir) == before


def test_quantized_hardsigmoid_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_hardsigmoid_without_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor(
            "x",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
        "y": _tensor(
            "y",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
        "unused": _tensor(
            "unused",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    layout_state = LayoutState.from_model_ir(model_ir)

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(
        hardsigmoid_module,
        "ModelIRGraphIndex",
        unexpected_index,
    )

    stats = _optimize_dequant_hardsigmoid_quantize_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
