from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.affine_chain_fold as affine_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.affine_chain_fold import (
    _apply_plan,
    _resolve_candidate,
    optimize_fold_mul_add_mul_affine_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}


def _tensor(
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
    signature: tuple[int, ...] | None = None,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=str(name),
        dtype=str(dtype),
        shape=[int(value) for value in shape],
        shape_signature=[
            int(value)
            for value in (shape if signature is None else signature)
        ],
        data=data,
        is_variable=False,
    )


def _broadcast(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    result: tuple[int, ...] = ()
    for shape in shapes:
        result = tuple(int(value) for value in np.broadcast_shapes(result, shape))
    return result


def _constant_values(
    shape: tuple[int, ...],
    dtype: str,
    offset: float,
) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    values = np.linspace(
        0.5 + float(offset),
        1.0 + float(offset),
        num=size,
        dtype=np.float64,
    )
    return np.asarray(values.reshape(shape), dtype=_NP_DTYPES[dtype])


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    commuted: tuple[bool, bool, bool] = (False, False, False),
    source_shape: tuple[int, ...] = (1, 4, 3, 2),
    constant_shapes: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ] = ((1, 4, 1, 1), (4, 1, 1), (1, 4, 1, 1)),
    source_signature: tuple[int, ...] | None = None,
) -> dict[str, str]:
    names = {
        key: f"{prefix}{key}"
        for key in ("x", "c1", "m1", "c2", "a1", "c3", "y")
    }
    c1_shape, c2_shape, c3_shape = constant_shapes
    mul1_shape = _broadcast(source_shape, c1_shape)
    add_shape = _broadcast(mul1_shape, c2_shape)
    final_shape = _broadcast(add_shape, c3_shape)
    source_signature = (
        source_shape if source_signature is None else source_signature
    )
    mul1_signature = affine_module._broadcast_signature(
        source_signature,
        c1_shape,
    )
    assert mul1_signature is not None
    add_signature = affine_module._broadcast_signature(
        mul1_signature,
        c2_shape,
    )
    assert add_signature is not None
    final_signature = affine_module._broadcast_signature(
        add_signature,
        c3_shape,
    )
    assert final_signature is not None

    model_ir.inputs.append(names["x"])
    model_ir.outputs.append(names["y"])
    model_ir.tensors[names["x"]] = _tensor(
        names["x"],
        dtype=dtype,
        shape=source_shape,
        signature=source_signature,
    )
    for index, (key, shape) in enumerate(
        (("c1", c1_shape), ("c2", c2_shape), ("c3", c3_shape))
    ):
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            dtype=dtype,
            shape=shape,
            data=_constant_values(shape, dtype, float(index) * 0.25),
        )
    for key, shape, signature in (
        ("m1", mul1_shape, mul1_signature),
        ("a1", add_shape, add_signature),
        ("y", final_shape, final_signature),
    ):
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            dtype=dtype,
            shape=shape,
            signature=signature,
        )

    def _inputs(data_name: str, constant_name: str, swap: bool) -> list[str]:
        return (
            [constant_name, data_name]
            if swap
            else [data_name, constant_name]
        )

    model_ir.operators.extend(
        [
            OperatorIR(
                "MUL",
                _inputs(names["x"], names["c1"], commuted[0]),
                [names["m1"]],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                "ADD",
                _inputs(names["m1"], names["c2"], commuted[1]),
                [names["a1"]],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                "MUL",
                _inputs(names["a1"], names["c3"], commuted[2]),
                [names["y"]],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, str]]:
    model_ir = ModelIR("indexed_affine_chain_fold")
    return model_ir, _add_chain(model_ir, **kwargs)


def _evaluate(model_ir: ModelIR, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {
        str(name): np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({str(name): np.asarray(value) for name, value in inputs.items()})
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        operator_inputs = [values[str(name)] for name in operator.inputs]
        if op_type == "MUL":
            output = np.multiply(operator_inputs[0], operator_inputs[1])
        elif op_type == "ADD":
            output = np.add(operator_inputs[0], operator_inputs[1])
        elif op_type == "IDENTITY":
            output = np.asarray(operator_inputs[0])
        else:
            raise AssertionError(f"unsupported test operator: {op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    tensors = []
    for name, tensor in model_ir.tensors.items():
        data = None
        if tensor.data is not None:
            array = np.asarray(tensor.data)
            data = (array.dtype.str, array.shape, array.tobytes())
        tensors.append(
            (
                str(name),
                str(tensor.dtype),
                tuple(tensor.shape),
                None
                if tensor.shape_signature is None
                else tuple(tensor.shape_signature),
                bool(tensor.is_variable),
                repr(tensor.quantization),
                str(tensor.logical_layout),
                str(tensor.physical_layout),
                tensor.onnx_tensor_name,
                data,
            )
        )
    operators = tuple(
        (
            str(operator.op_type),
            tuple(str(name) for name in operator.inputs),
            tuple(str(name) for name in operator.outputs),
            repr(operator.options),
            repr(operator.axis_semantics),
            int(operator.version),
            operator.onnx_node_name,
            operator.onnx_op_type,
        )
        for operator in model_ir.operators
    )
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(tensors),
        operators,
        repr(model_ir.metadata),
    )


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


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32", "FLOAT64"))
@pytest.mark.parametrize(
    "commuted",
    tuple(
        (bool(mask & 1), bool(mask & 2), bool(mask & 4))
        for mask in range(8)
    ),
)
@pytest.mark.parametrize(
    "constant_shapes",
    (
        ((), (), ()),
        ((1, 4, 1, 1), (4, 1, 1), (1, 4, 1, 1)),
    ),
)
def test_affine_chain_fold_is_indexed_and_numerically_equivalent(
    dtype: str,
    commuted: tuple[bool, bool, bool],
    constant_shapes: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ],
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        commuted=commuted,
        constant_shapes=constant_shapes,
        source_signature=(-1, 4, -1, 2),
    )
    original = copy.deepcopy(model_ir)
    input_value = np.asarray(
        np.linspace(-0.75, 0.9, num=24).reshape(1, 4, 3, 2),
        dtype=_NP_DTYPES[dtype],
    )
    expected = _evaluate(original, {names["x"]: input_value})[names["y"]]
    final_tensor = copy.deepcopy(model_ir.tensors[names["y"]])
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_fold_mul_add_mul_affine_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == ["MUL", "ADD"]
    assert model_ir.operators[-1].outputs == [names["y"]]
    assert names["a1"] not in model_ir.tensors
    actual = _evaluate(model_ir, {names["x"]: input_value})[names["y"]]
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )
    assert model_ir.tensors[names["y"]] == final_tensor
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_affine_chain_fold_updates_expanded_intermediate_metadata() -> None:
    model_ir, names = _model(
        source_shape=(1, 1),
        constant_shapes=((), (), (2, 3)),
    )
    assert model_ir.tensors[names["m1"]].shape == [1, 1]

    stats = optimize_fold_mul_add_mul_affine_chains(model_ir)

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    assert model_ir.tensors[names["m1"]].shape == [2, 3]
    assert model_ir.tensors[names["m1"]].shape_signature == [2, 3]
    assert model_ir.tensors[names["y"]].shape == [2, 3]


def _share_constant(
    model_ir: ModelIR,
    names: dict[str, str],
    target_roles: tuple[str, ...],
) -> None:
    source_name = names[target_roles[0]]
    for role in target_roles[1:]:
        old_name = names[role]
        for operator in model_ir.operators:
            operator.inputs = [
                source_name if str(name) == old_name else str(name)
                for name in operator.inputs
            ]
        model_ir.tensors.pop(old_name)
        names[role] = source_name


@pytest.mark.parametrize(
    "shared_roles",
    (("c1", "c2"), ("c1", "c3"), ("c1", "c2", "c3")),
)
def test_affine_chain_fold_groups_constants_shared_inside_chain(
    shared_roles: tuple[str, ...],
) -> None:
    model_ir, names = _model(constant_shapes=((), (), ()))
    _share_constant(model_ir, names, shared_roles)
    original = copy.deepcopy(model_ir)
    input_value = np.full((1, 4, 3, 2), 0.75, dtype=np.float32)
    expected = _evaluate(original, {names["x"]: input_value})[names["y"]]

    stats = optimize_fold_mul_add_mul_affine_chains(model_ir)

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    actual = _evaluate(model_ir, {names["x"]: input_value})[names["y"]]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert not any("_folded" in name for name in model_ir.tensors)


def test_affine_chain_fold_clones_constants_with_unrelated_consumers() -> None:
    model_ir, names = _model()
    original_c1 = np.asarray(model_ir.tensors[names["c1"]].data).copy()
    original_c2 = np.asarray(model_ir.tensors[names["c2"]].data).copy()
    for constant_name in (names["c1"], names["c2"]):
        side_name = f"{constant_name}_side"
        model_ir.tensors[side_name] = _tensor(
            side_name,
            dtype="FLOAT32",
            shape=tuple(model_ir.tensors[constant_name].shape),
        )
        model_ir.outputs.append(side_name)
        model_ir.operators.append(
            OperatorIR("IDENTITY", [constant_name], [side_name])
        )

    stats = optimize_fold_mul_add_mul_affine_chains(model_ir)

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    np.testing.assert_array_equal(model_ir.tensors[names["c1"]].data, original_c1)
    np.testing.assert_array_equal(model_ir.tensors[names["c2"]].data, original_c2)
    assert model_ir.operators[0].inputs[1] == f"{names['c1']}_folded"
    assert model_ir.operators[1].inputs[1] == f"{names['c2']}_folded"


def test_affine_chain_fold_preserves_downstream_fanout_and_repeated_slots() -> None:
    model_ir, names = _model()
    model_ir.outputs = ["fanout"]
    model_ir.tensors["fanout"] = _tensor(
        "fanout",
        dtype="FLOAT32",
        shape=tuple(model_ir.tensors[names["y"]].shape),
        signature=tuple(model_ir.tensors[names["y"]].shape_signature or ()),
    )
    model_ir.operators.append(
        OperatorIR("ADD", [names["y"], names["y"]], ["fanout"])
    )

    stats = optimize_fold_mul_add_mul_affine_chains(model_ir)

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    assert model_ir.operators[-1].inputs == [names["y"], names["y"]]
    index = ModelIRGraphIndex(model_ir)
    assert index.consumer_indices(names["y"]) == [2, 2]


def test_affine_chain_fold_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_affine_chain_fold")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    third = _add_chain(model_ir, prefix="third_")
    second_candidate = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [second["y"]]
    )

    candidate_stats = optimize_fold_mul_add_mul_affine_chains(
        model_ir,
        candidate=second_candidate,
    )
    capped_stats = optimize_fold_mul_add_mul_affine_chains(
        model_ir,
        max_rewrites=1,
    )

    assert candidate_stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    assert capped_stats == {"optimized_fold_mul_add_mul_affine_chains": 1}
    remaining_final_mul_outputs = {
        str(operator.outputs[0])
        for operator in model_ir.operators
        if str(operator.op_type) == "MUL"
        and operator.outputs
        and str(operator.outputs[0]) in {first["y"], second["y"], third["y"]}
    }
    assert remaining_final_mul_outputs == {third["y"]}


def _mutate_fused(model_ir: ModelIR, names: dict[str, str], index: int) -> None:
    del names
    model_ir.operators[index].options["fusedActivationFunction"] = "RELU"


def _mutate_public(model_ir: ModelIR, names: dict[str, str], key: str) -> None:
    model_ir.outputs.append(names[key])


def _mutate_quantized(model_ir: ModelIR, names: dict[str, str], key: str) -> None:
    model_ir.tensors[names[key]].quantization = {"scale": [0.1], "zero_point": [0]}


def _mutate_dtype(model_ir: ModelIR, names: dict[str, str], key: str) -> None:
    model_ir.tensors[names[key]].dtype = "FLOAT16"


def _mutate_data_dtype(model_ir: ModelIR, names: dict[str, str], key: str) -> None:
    tensor = model_ir.tensors[names[key]]
    tensor.data = np.asarray(tensor.data, dtype=np.float16)


def _mutate_nonfinite(
    model_ir: ModelIR,
    names: dict[str, str],
    key: str,
) -> None:
    tensor = model_ir.tensors[names[key]]
    data = np.asarray(tensor.data).copy()
    data.reshape(-1)[0] = np.inf
    tensor.data = data


def _mutate_fanout(model_ir: ModelIR, names: dict[str, str], key: str) -> None:
    output_name = f"{key}_side"
    model_ir.tensors[output_name] = _tensor(
        output_name,
        dtype="FLOAT32",
        shape=tuple(model_ir.tensors[names[key]].shape),
    )
    model_ir.operators.append(OperatorIR("IDENTITY", [names[key]], [output_name]))


def _mutate_duplicate_producer(
    model_ir: ModelIR,
    names: dict[str, str],
    key: str,
) -> None:
    model_ir.operators.append(OperatorIR("IDENTITY", [names["x"]], [names[key]]))


def _mutate_wrong_order(model_ir: ModelIR, names: dict[str, str]) -> None:
    del names
    model_ir.operators[0], model_ir.operators[1] = (
        model_ir.operators[1],
        model_ir.operators[0],
    )


def _mutate_constant_shape(
    model_ir: ModelIR,
    names: dict[str, str],
) -> None:
    model_ir.tensors[names["c1"]].shape = [4]


def _mutate_constant_signature(
    model_ir: ModelIR,
    names: dict[str, str],
) -> None:
    model_ir.tensors[names["c1"]].shape_signature = [1, 4, -1, 1]


def _mutate_output_shape(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.tensors[names["a1"]].shape = [1, 4, 2, 2]


def _mutate_output_signature(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.tensors[names["y"]].shape_signature = [-1, 4, 3, -1]


def _mutate_unresolved_source(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.inputs.remove(names["x"])


def _mutate_source_constant(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.tensors[names["x"]].data = np.ones(
        model_ir.tensors[names["x"]].shape,
        dtype=np.float32,
    )


def _mutate_missing_constant(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.tensors[names["c1"]].data = None


def _mutate_variable_constant(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.tensors[names["c1"]].is_variable = True


def _mutate_produced_constant(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.operators.append(OperatorIR("IDENTITY", [names["c2"]], [names["c1"]]))


def _mutate_arity(model_ir: ModelIR, names: dict[str, str]) -> None:
    model_ir.operators[0].inputs.append(names["c2"])


_UnsafeMutation = Callable[[ModelIR, dict[str, str]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: _mutate_fused(model, names, 0),
        lambda model, names: _mutate_fused(model, names, 1),
        lambda model, names: _mutate_fused(model, names, 2),
        lambda model, names: _mutate_public(model, names, "m1"),
        lambda model, names: _mutate_public(model, names, "a1"),
        lambda model, names: _mutate_public(model, names, "c1"),
        lambda model, names: _mutate_public(model, names, "c2"),
        lambda model, names: _mutate_public(model, names, "c3"),
        lambda model, names: _mutate_quantized(model, names, "x"),
        lambda model, names: _mutate_quantized(model, names, "m1"),
        lambda model, names: _mutate_quantized(model, names, "a1"),
        lambda model, names: _mutate_quantized(model, names, "y"),
        lambda model, names: _mutate_quantized(model, names, "c1"),
        lambda model, names: _mutate_dtype(model, names, "m1"),
        lambda model, names: _mutate_dtype(model, names, "c1"),
        lambda model, names: _mutate_data_dtype(model, names, "c1"),
        lambda model, names: _mutate_nonfinite(model, names, "c1"),
        lambda model, names: _mutate_nonfinite(model, names, "c2"),
        lambda model, names: _mutate_nonfinite(model, names, "c3"),
        lambda model, names: _mutate_fanout(model, names, "m1"),
        lambda model, names: _mutate_fanout(model, names, "a1"),
        lambda model, names: _mutate_duplicate_producer(model, names, "m1"),
        _mutate_wrong_order,
        _mutate_constant_shape,
        _mutate_constant_signature,
        _mutate_output_shape,
        _mutate_output_signature,
        _mutate_unresolved_source,
        _mutate_source_constant,
        _mutate_missing_constant,
        _mutate_variable_constant,
        _mutate_produced_constant,
        _mutate_arity,
    ),
)
def test_affine_chain_fold_rejects_unsafe_contracts_transactionally(
    mutation: _UnsafeMutation,
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_fold_mul_add_mul_affine_chains(model_ir)

    assert stats == {"optimized_fold_mul_add_mul_affine_chains": 0}
    assert _fingerprint(model_ir) == before


def test_affine_chain_fold_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _model()
    _mutate_fanout(model_ir, names, "c1")
    graph_index = ModelIRGraphIndex(model_ir)
    candidate = next(
        operator for operator in model_ir.operators if operator.outputs == [names["y"]]
    )
    plan = _resolve_candidate(model_ir, graph_index, candidate)
    assert plan is not None
    clone_name = next(
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    )
    assert clone_name is not None
    model_ir.tensors[clone_name] = _tensor(
        clone_name,
        dtype="FLOAT32",
        shape=(),
        data=np.asarray(0.0, dtype=np.float32),
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_affine_chain_fold_preflight_avoids_graph_index(monkeypatch) -> None:
    model_ir = ModelIR("no_affine_chain")
    model_ir.operators = [OperatorIR("MUL", ["x", "c"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(affine_module, "ModelIRGraphIndex", fail_index)

    assert optimize_fold_mul_add_mul_affine_chains(model_ir) == {
        "optimized_fold_mul_add_mul_affine_chains": 0
    }
