from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.affine_prepost_layout as prepost_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_PRE_PERM = (0, 3, 1, 2)
_POST_PERM = (0, 2, 3, 1)


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


def _constant_data(
    shape: tuple[int, ...],
    dtype: str,
    offset: float,
) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.linspace(
            0.5 + float(offset),
            1.0 + float(offset),
            num=size,
            dtype=np.float64,
        ).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _constant_shape(mode: str, nchw_shape: tuple[int, ...]) -> tuple[int, ...]:
    _, channels, height, width = nchw_shape
    return {
        "scalar": (),
        "channel": (1, channels, 1, 1),
        "spatial": (1, 1, height, width),
        "full": nchw_shape,
        "nhwc": (1, 1, 1, channels),
    }[str(mode)]


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    commuted: tuple[bool, bool] = (False, False),
    constant_mode: str = "channel",
    source_shape: tuple[int, ...] = (1, 3, 2, 4),
    source_signature: tuple[int, ...] | None = None,
    posts: int = 1,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "x",
            "pre_perm",
            "x_nchw",
            "mul_const",
            "mul_out",
            "add_const",
            "add_out",
            "post_perm",
            "z",
        )
    }
    nchw_shape = tuple(source_shape[index] for index in _PRE_PERM)
    source_signature = (
        source_shape if source_signature is None else source_signature
    )
    nchw_signature = tuple(source_signature[index] for index in _PRE_PERM)
    constant_shape = _constant_shape(constant_mode, nchw_shape)

    model_ir.inputs.append(str(names["x"]))
    model_ir.outputs.append(str(names["z"]))
    model_ir.tensors[str(names["x"])] = _tensor(
        str(names["x"]),
        dtype=dtype,
        shape=source_shape,
        signature=source_signature,
    )
    for perm_key, values in (
        ("pre_perm", _PRE_PERM),
        ("post_perm", _POST_PERM),
    ):
        model_ir.tensors[str(names[perm_key])] = _tensor(
            str(names[perm_key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    model_ir.tensors[str(names["x_nchw"])] = _tensor(
        str(names["x_nchw"]),
        dtype=dtype,
        shape=nchw_shape,
        signature=nchw_signature,
    )
    for index, key in enumerate(("mul_const", "add_const")):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=constant_shape,
            data=_constant_data(constant_shape, dtype, float(index) * 0.25),
        )
    for key in ("mul_out", "add_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
        )

    post_output_names = [f"{prefix}y{index}" for index in range(int(posts))]
    for output_name in post_output_names:
        model_ir.tensors[output_name] = _tensor(
            output_name,
            dtype=dtype,
            shape=source_shape,
            signature=source_signature,
        )
    model_ir.tensors[str(names["z"])] = _tensor(
        str(names["z"]),
        dtype=dtype,
        shape=source_shape,
        signature=source_signature,
    )

    def _inputs(data_name: str, constant_name: str, swap: bool) -> list[str]:
        return (
            [constant_name, data_name]
            if swap
            else [data_name, constant_name]
        )

    operators = [
        OperatorIR(
            "TRANSPOSE",
            [str(names["x"]), str(names["pre_perm"])],
            [str(names["x_nchw"])],
        ),
        OperatorIR(
            "MUL",
            _inputs(
                str(names["x_nchw"]),
                str(names["mul_const"]),
                commuted[0],
            ),
            [str(names["mul_out"])],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "ADD",
            _inputs(
                str(names["mul_out"]),
                str(names["add_const"]),
                commuted[1],
            ),
            [str(names["add_out"])],
            options={"fusedActivationFunction": "NONE"},
        ),
    ]
    operators.extend(
        OperatorIR(
            "TRANSPOSE",
            [str(names["add_out"]), str(names["post_perm"])],
            [output_name],
        )
        for output_name in post_output_names
    )
    if len(post_output_names) == 1:
        operators.append(
            OperatorIR("IDENTITY", [post_output_names[0]], [str(names["z"])])
        )
    else:
        operators.append(
            OperatorIR(
                "ADD",
                [post_output_names[0], post_output_names[1]],
                [str(names["z"])],
            )
        )
    model_ir.operators.extend(operators)
    names["post_outputs"] = tuple(post_output_names)
    names["mul"] = operators[1]
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_affine_prepost_layout")
    return model_ir, _add_chain(model_ir, **kwargs)


def _evaluate(model_ir: ModelIR, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {
        str(name): np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({str(name): np.asarray(value) for name, value in inputs.items()})
    for operator in model_ir.operators:
        operator_inputs = [values[str(name)] for name in operator.inputs]
        if str(operator.op_type) == "TRANSPOSE":
            output = np.transpose(
                operator_inputs[0],
                tuple(int(value) for value in operator_inputs[1].reshape(-1)),
            )
        elif str(operator.op_type) == "MUL":
            output = np.multiply(operator_inputs[0], operator_inputs[1])
        elif str(operator.op_type) == "ADD":
            output = np.add(operator_inputs[0], operator_inputs[1])
        elif str(operator.op_type) == "IDENTITY":
            output = np.asarray(operator_inputs[0])
        else:
            raise AssertionError(f"unsupported test operator: {operator.op_type}")
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
            tuple(operator.inputs),
            tuple(operator.outputs),
            repr(operator.options),
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
    ((False, False), (True, False), (False, True), (True, True)),
)
@pytest.mark.parametrize(
    "constant_mode",
    ("scalar", "channel", "spatial", "full"),
)
def test_affine_prepost_layout_is_indexed_and_numerically_equivalent(
    dtype: str,
    commuted: tuple[bool, bool],
    constant_mode: str,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        commuted=commuted,
        constant_mode=constant_mode,
        source_signature=(-1, 3, -1, 4),
    )
    original = copy.deepcopy(model_ir)
    input_value = np.asarray(
        np.linspace(-0.75, 0.9, num=24).reshape(1, 3, 2, 4),
        dtype=_NP_DTYPES[dtype],
    )
    expected = _evaluate(original, {str(names["x"]): input_value})
    canonical_name = str(tuple(names["post_outputs"])[0])
    canonical_tensor = copy.deepcopy(model_ir.tensors[canonical_name])
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "ADD",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs.count(str(names["x"])) == 1
    assert str(names["x_nchw"]) not in model_ir.operators[0].inputs
    assert model_ir.operators[1].outputs == [canonical_name]
    actual = _evaluate(model_ir, {str(names["x"]): input_value})
    tolerance = 2e-3 if dtype == "FLOAT16" else 1e-7
    np.testing.assert_allclose(
        actual[str(names["z"])],
        expected[str(names["z"])],
        rtol=tolerance,
        atol=tolerance,
    )
    assert model_ir.tensors[canonical_name] == canonical_tensor
    assert model_ir.tensors[str(names["mul_out"])].shape == [1, 3, 2, 4]
    assert model_ir.tensors[str(names["mul_out"])].shape_signature == [
        -1,
        3,
        -1,
        4,
    ]
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_affine_prepost_layout_accepts_already_nhwc_constants_idempotently() -> None:
    model_ir, names = _model(constant_mode="nhwc")
    original_mul = np.asarray(
        model_ir.tensors[str(names["mul_const"])].data
    ).copy()
    original_add = np.asarray(
        model_ir.tensors[str(names["add_const"])].data
    ).copy()

    first = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    second = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert first == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    assert second == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 0}
    np.testing.assert_array_equal(
        model_ir.tensors[str(names["mul_const"])].data,
        original_mul,
    )
    np.testing.assert_array_equal(
        model_ir.tensors[str(names["add_const"])].data,
        original_add,
    )


def test_affine_prepost_layout_uses_canonical_output_layout_for_survivors() -> None:
    model_ir, names = _model()
    model_ir.tensors[str(names["x_nchw"])].logical_layout = "NCHW"
    model_ir.tensors[str(names["x_nchw"])].physical_layout = "NCHW"
    for key in ("mul_out", "add_out"):
        model_ir.tensors[str(names[key])].logical_layout = "NCHW"
        model_ir.tensors[str(names[key])].physical_layout = "NCHW"
    canonical_name = str(tuple(names["post_outputs"])[0])
    model_ir.tensors[canonical_name].logical_layout = "NHWC"
    model_ir.tensors[canonical_name].physical_layout = "NHWC"
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    for key in ("mul_out",):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_affine_prepost_layout_preserves_pre_transpose_fanout() -> None:
    model_ir, names = _model()
    legacy_name = "legacy_nchw"
    model_ir.tensors[legacy_name] = _tensor(
        legacy_name,
        dtype="FLOAT32",
        shape=(1, 4, 3, 2),
    )
    model_ir.outputs.append(legacy_name)
    model_ir.operators.append(
        OperatorIR("IDENTITY", [str(names["x_nchw"])], [legacy_name])
    )

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    assert any(
        operator.op_type == "TRANSPOSE"
        and operator.outputs == [names["x_nchw"]]
        for operator in model_ir.operators
    )
    assert next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "MUL"
    ).inputs[0] == names["x"]


def test_affine_prepost_layout_merges_post_alias_fanout_and_repeated_slots() -> None:
    model_ir, names = _model(posts=2)
    post_outputs = tuple(str(name) for name in names["post_outputs"])

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "ADD",
        "ADD",
    ]
    assert model_ir.operators[-1].inputs == [post_outputs[0], post_outputs[0]]
    assert post_outputs[1] not in model_ir.tensors
    assert ModelIRGraphIndex(model_ir).consumer_indices(post_outputs[0]) == [2, 2]


def _share_affine_constants(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    mul_name = str(names["mul_const"])
    add_name = str(names["add_const"])
    add = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "ADD" and operator.outputs == [names["add_out"]]
    )
    add.inputs = [mul_name if str(name) == add_name else str(name) for name in add.inputs]
    model_ir.tensors.pop(add_name)
    names["add_const"] = mul_name


def test_affine_prepost_layout_groups_shared_constants() -> None:
    model_ir, names = _model()
    _share_affine_constants(model_ir, names)

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    constant_name = str(names["mul_const"])
    assert model_ir.tensors[constant_name].shape == [1, 1, 1, 4]
    mul, add = model_ir.operators[:2]
    assert constant_name in mul.inputs
    assert constant_name in add.inputs


def test_affine_prepost_layout_clones_shared_constants_transactionally() -> None:
    model_ir, names = _model()
    constant_name = str(names["mul_const"])
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    side_name = "constant_side"
    model_ir.tensors[side_name] = _tensor(
        side_name,
        dtype="FLOAT32",
        shape=tuple(model_ir.tensors[constant_name].shape),
    )
    model_ir.outputs.append(side_name)
    model_ir.operators.append(OperatorIR("IDENTITY", [constant_name], [side_name]))

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 1}
    np.testing.assert_array_equal(model_ir.tensors[constant_name].data, original)
    mul = next(operator for operator in model_ir.operators if operator.op_type == "MUL")
    assert f"{constant_name}_nhwc" in mul.inputs
    assert model_ir.tensors[f"{constant_name}_nhwc"].shape == [1, 1, 1, 4]


def test_affine_prepost_layout_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_affine_prepost")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    third = _add_chain(model_ir, prefix="third_")

    candidate_stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(
        model_ir,
        candidate=second["mul"],
    )
    capped_stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(
        model_ir,
        max_rewrites=1,
    )

    assert candidate_stats == {
        "optimized_transpose_mul_add_const_prepost_nhwc_chains": 1
    }
    assert capped_stats == {
        "optimized_transpose_mul_add_const_prepost_nhwc_chains": 1
    }
    remaining_post_outputs = {
        str(operator.outputs[0])
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs
        and str(operator.outputs[0])
        in {
            *tuple(first["post_outputs"]),
            *tuple(second["post_outputs"]),
            *tuple(third["post_outputs"]),
        }
    }
    assert remaining_post_outputs == set(third["post_outputs"])


def _mutate_fused(model: ModelIR, names: dict[str, object], op_type: str) -> None:
    operator = next(
        operator
        for operator in model.operators
        if operator.op_type == op_type and operator in model.operators[1:3]
    )
    operator.options["fusedActivationFunction"] = "RELU"


def _mutate_public(model: ModelIR, names: dict[str, object], key: str) -> None:
    if key == "post":
        model.outputs.append(str(tuple(names["post_outputs"])[0]))
    else:
        model.outputs.append(str(names[key]))


def _mutate_quantized(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].quantization = {
        "scale": [0.1],
        "zero_point": [0],
    }


def _mutate_dtype(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].dtype = "FLOAT16"


def _mutate_data_dtype(model: ModelIR, names: dict[str, object]) -> None:
    tensor = model.tensors[str(names["mul_const"])]
    tensor.data = np.asarray(tensor.data, dtype=np.float16)


def _mutate_nonfinite(model: ModelIR, names: dict[str, object], key: str) -> None:
    tensor = model.tensors[str(names[key])]
    data = np.asarray(tensor.data).copy()
    data.reshape(-1)[0] = np.inf
    tensor.data = data


def _mutate_fanout(model: ModelIR, names: dict[str, object], key: str) -> None:
    output_name = f"{key}_fanout"
    source = model.tensors[str(names[key])]
    model.tensors[output_name] = _tensor(
        output_name,
        dtype=str(source.dtype),
        shape=tuple(source.shape),
        signature=tuple(source.shape_signature or source.shape),
    )
    model.operators.append(OperatorIR("IDENTITY", [str(names[key])], [output_name]))


def _mutate_duplicate(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.operators.append(
        OperatorIR("IDENTITY", [str(names["x"])], [str(names[key])])
    )


def _mutate_wrong_perm(
    model: ModelIR,
    names: dict[str, object],
    key: str,
) -> None:
    model.tensors[str(names[key])].data = np.asarray(
        [0, 1, 2, 3],
        dtype=np.int32,
    )


def _mutate_perm_dtype(model: ModelIR, names: dict[str, object]) -> None:
    tensor = model.tensors[str(names["pre_perm"])]
    tensor.dtype = "FLOAT32"
    tensor.data = np.asarray(tensor.data, dtype=np.float32)


def _mutate_wrong_order(model: ModelIR, names: dict[str, object]) -> None:
    del names
    model.operators[0], model.operators[1] = model.operators[1], model.operators[0]


def _mutate_source_unresolved(model: ModelIR, names: dict[str, object]) -> None:
    model.inputs.remove(str(names["x"]))


def _mutate_source_constant(model: ModelIR, names: dict[str, object]) -> None:
    tensor = model.tensors[str(names["x"])]
    tensor.data = np.ones(tensor.shape, dtype=np.float32)


def _mutate_missing_constant(model: ModelIR, names: dict[str, object]) -> None:
    model.tensors[str(names["mul_const"])].data = None


def _mutate_variable_constant(model: ModelIR, names: dict[str, object]) -> None:
    model.tensors[str(names["mul_const"])].is_variable = True


def _mutate_produced_constant(model: ModelIR, names: dict[str, object]) -> None:
    model.operators.append(
        OperatorIR(
            "IDENTITY",
            [str(names["add_const"])],
            [str(names["mul_const"])],
        )
    )


def _mutate_constant_rank(model: ModelIR, names: dict[str, object]) -> None:
    tensor = model.tensors[str(names["mul_const"])]
    tensor.shape = [4, 1]
    tensor.shape_signature = [4, 1]
    tensor.data = np.asarray(tensor.data).reshape(4, 1)


def _mutate_constant_signature(model: ModelIR, names: dict[str, object]) -> None:
    model.tensors[str(names["mul_const"])].shape_signature = [1, 4, -1, 1]


def _mutate_tensor_shape(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].shape = [1, 4, 2, 3]


def _mutate_tensor_signature(
    model: ModelIR,
    names: dict[str, object],
    key: str,
) -> None:
    model.tensors[str(names[key])].shape_signature = [-1, 4, 2, 3]


def _mutate_arity(model: ModelIR, names: dict[str, object], index: int) -> None:
    model.operators[index].inputs.append(str(names["add_const"]))


def _mutate_both_constant(model: ModelIR, names: dict[str, object]) -> None:
    tensor = model.tensors[str(names["x_nchw"])]
    tensor.data = np.ones(tensor.shape, dtype=np.float32)


def _mutate_post_consumer_order(model: ModelIR, names: dict[str, object]) -> None:
    consumer = model.operators.pop()
    model.operators.insert(3, consumer)


_UnsafeMutation = Callable[[ModelIR, dict[str, object]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: _mutate_fused(model, names, "MUL"),
        lambda model, names: _mutate_fused(model, names, "ADD"),
        lambda model, names: _mutate_public(model, names, "x"),
        lambda model, names: _mutate_public(model, names, "x_nchw"),
        lambda model, names: _mutate_public(model, names, "mul_out"),
        lambda model, names: _mutate_public(model, names, "add_out"),
        lambda model, names: _mutate_public(model, names, "post"),
        lambda model, names: _mutate_public(model, names, "mul_const"),
        lambda model, names: _mutate_quantized(model, names, "x"),
        lambda model, names: _mutate_quantized(model, names, "x_nchw"),
        lambda model, names: _mutate_quantized(model, names, "mul_out"),
        lambda model, names: _mutate_quantized(model, names, "add_out"),
        lambda model, names: _mutate_quantized(model, names, "mul_const"),
        lambda model, names: _mutate_dtype(model, names, "mul_out"),
        lambda model, names: _mutate_dtype(model, names, "mul_const"),
        _mutate_data_dtype,
        lambda model, names: _mutate_nonfinite(model, names, "mul_const"),
        lambda model, names: _mutate_nonfinite(model, names, "add_const"),
        lambda model, names: _mutate_fanout(model, names, "mul_out"),
        lambda model, names: _mutate_fanout(model, names, "add_out"),
        lambda model, names: _mutate_duplicate(model, names, "x_nchw"),
        lambda model, names: _mutate_duplicate(model, names, "mul_out"),
        lambda model, names: _mutate_wrong_perm(model, names, "pre_perm"),
        lambda model, names: _mutate_wrong_perm(model, names, "post_perm"),
        _mutate_perm_dtype,
        _mutate_wrong_order,
        _mutate_source_unresolved,
        _mutate_source_constant,
        _mutate_missing_constant,
        _mutate_variable_constant,
        _mutate_produced_constant,
        _mutate_constant_rank,
        _mutate_constant_signature,
        lambda model, names: _mutate_tensor_shape(model, names, "x_nchw"),
        lambda model, names: _mutate_tensor_shape(model, names, "mul_out"),
        lambda model, names: _mutate_tensor_shape(model, names, "add_out"),
        lambda model, names: _mutate_tensor_signature(model, names, "x_nchw"),
        lambda model, names: _mutate_tensor_signature(model, names, "mul_out"),
        lambda model, names: _mutate_tensor_signature(model, names, "add_out"),
        lambda model, names: _mutate_arity(model, names, 0),
        lambda model, names: _mutate_arity(model, names, 1),
        lambda model, names: _mutate_arity(model, names, 2),
        _mutate_both_constant,
        _mutate_post_consumer_order,
    ),
)
def test_affine_prepost_layout_rejects_unsafe_contracts_transactionally(
    mutation: _UnsafeMutation,
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 0}
    assert _fingerprint(model_ir) == before


def test_affine_prepost_layout_rejects_ambiguous_equal_axis_constant() -> None:
    model_ir, _ = _model(
        source_shape=(1, 4, 4, 4),
        constant_mode="full",
    )
    before = _fingerprint(model_ir)

    stats = optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 0}
    assert _fingerprint(model_ir) == before


def test_affine_prepost_layout_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _model()
    _mutate_fanout(model_ir, names, "mul_const")
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["mul"])
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


def test_affine_prepost_layout_preflight_avoids_graph_index(monkeypatch) -> None:
    model_ir = ModelIR("no_affine_prepost")
    model_ir.operators = [OperatorIR("MUL", ["x", "c"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(prepost_module, "ModelIRGraphIndex", fail_index)

    assert optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir) == {
        "optimized_transpose_mul_add_const_prepost_nhwc_chains": 0
    }
