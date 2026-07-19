from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.affine_post_add_layout as post_add_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_mul_posttranspose_add_nhwc_chains,
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
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=False,
    )


def _constant_data(shape: tuple[int, ...], dtype: str) -> np.ndarray:
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.linspace(0.5, 1.25, num=size, dtype=np.float64).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _mul_shape(mode: str) -> tuple[int, ...]:
    return {
        "scalar": (),
        "channel": (1, 4, 1, 1),
        "spatial": (1, 1, 2, 3),
        "full": (1, 4, 2, 3),
        "nhwc": (1, 1, 1, 4),
        "vector": (4,),
    }[mode]


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    mul_mode: str = "channel",
    add_mode: str = "channel",
    commuted: tuple[bool, bool] = (False, False),
    add_count: int = 1,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "x",
            "pre_perm",
            "x_nchw",
            "mul_const",
            "mul_out",
            "post_perm",
            "post_out",
        )
    }
    source_shape = (1, 2, 3, 4)
    source_signature = (-1, -1, 3, 4)
    nchw_shape = tuple(source_shape[index] for index in _PRE_PERM)
    nchw_signature = tuple(source_signature[index] for index in _PRE_PERM)
    model_ir.inputs.append(str(names["x"]))
    model_ir.tensors[str(names["x"])] = _tensor(
        str(names["x"]),
        dtype=dtype,
        shape=source_shape,
        signature=source_signature,
    )
    for key, values in (
        ("pre_perm", _PRE_PERM),
        ("post_perm", _POST_PERM),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    for key in ("x_nchw", "mul_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
        )
    mul_shape = _mul_shape(mul_mode)
    model_ir.tensors[str(names["mul_const"])] = _tensor(
        str(names["mul_const"]),
        dtype=dtype,
        shape=mul_shape,
        data=_constant_data(mul_shape, dtype),
    )
    model_ir.tensors[str(names["post_out"])] = _tensor(
        str(names["post_out"]),
        dtype=dtype,
        shape=source_shape,
        signature=source_signature,
    )

    def binary_inputs(data: str, constant: str, swap: bool) -> list[str]:
        return [constant, data] if swap else [data, constant]

    operators = [
        OperatorIR(
            "TRANSPOSE",
            [str(names["x"]), str(names["pre_perm"])],
            [str(names["x_nchw"])],
        ),
        OperatorIR(
            "MUL",
            binary_inputs(
                str(names["x_nchw"]),
                str(names["mul_const"]),
                commuted[0],
            ),
            [str(names["mul_out"])],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "TRANSPOSE",
            [str(names["mul_out"]), str(names["post_perm"])],
            [str(names["post_out"])],
        ),
    ]
    add_shape = () if add_mode == "scalar" else (1, 1, 1, 4)
    add_names = []
    for index in range(add_count):
        constant_name = f"{prefix}add_const_{index}"
        output_name = f"{prefix}y_{index}"
        model_ir.tensors[constant_name] = _tensor(
            constant_name,
            dtype=dtype,
            shape=add_shape,
            data=np.asarray(
                _constant_data(add_shape, dtype) + index,
                dtype=_NP_DTYPES[dtype],
            ),
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            dtype=dtype,
            shape=source_shape,
            signature=source_signature,
        )
        model_ir.outputs.append(output_name)
        operators.append(
            OperatorIR(
                "ADD",
                binary_inputs(
                    str(names["post_out"]),
                    constant_name,
                    commuted[1],
                ),
                [output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        add_names.append((constant_name, output_name))
    model_ir.operators.extend(operators)
    names["mul"] = operators[1]
    names["adds"] = tuple(add_names)
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_affine_post_add_layout")
    return model_ir, _add_chain(model_ir, **kwargs)


def _evaluate(model_ir: ModelIR, input_value: np.ndarray) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values[model_ir.inputs[0]] = input_value
    for operator in model_ir.operators:
        inputs = [values[name] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            value = np.transpose(inputs[0], tuple(inputs[1].reshape(-1)))
        elif operator.op_type == "MUL":
            value = np.multiply(inputs[0], inputs[1])
        elif operator.op_type == "ADD":
            value = np.add(inputs[0], inputs[1])
        elif operator.op_type == "IDENTITY":
            value = inputs[0]
        else:
            raise AssertionError(operator.op_type)
        values[operator.outputs[0]] = np.asarray(value)
    return {name: values[name] for name in model_ir.outputs}


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    tensors = tuple(
        (
            name,
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.shape_signature or ()),
            tensor.is_variable,
            repr(tensor.quantization),
            tensor.logical_layout,
            tensor.physical_layout,
            None
            if tensor.data is None
            else (
                np.asarray(tensor.data).dtype.str,
                np.asarray(tensor.data).shape,
                np.asarray(tensor.data).tobytes(),
            ),
        )
        for name, tensor in model_ir.tensors.items()
    )
    operators = tuple(
        (op.op_type, tuple(op.inputs), tuple(op.outputs), repr(op.options))
        for op in model_ir.operators
    )
    return tuple(model_ir.inputs), tuple(model_ir.outputs), tensors, operators


def _assert_index_current(model_ir: ModelIR, index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert index.producers == fresh.producers
    assert index.consumers == fresh.consumers
    assert index.duplicate_producers == fresh.duplicate_producers
    assert index._operator_indices_by_id == fresh._operator_indices_by_id
    assert index._operator_indices_by_type == fresh._operator_indices_by_type


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32", "FLOAT64"))
@pytest.mark.parametrize(
    "mul_mode",
    ("scalar", "channel", "spatial", "full"),
)
@pytest.mark.parametrize("commuted", ((False, False), (True, True)))
def test_affine_post_add_is_indexed_and_numerically_equivalent(
    dtype: str,
    mul_mode: str,
    commuted: tuple[bool, bool],
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        mul_mode=mul_mode,
        commuted=commuted,
        add_count=2,
    )
    original = copy.deepcopy(model_ir)
    input_value = np.asarray(
        np.linspace(-0.75, 0.9, num=24).reshape(1, 2, 3, 4),
        dtype=_NP_DTYPES[dtype],
    )
    expected = _evaluate(original, input_value)
    post_tensor = copy.deepcopy(model_ir.tensors[str(names["post_out"])])
    post_tensor.logical_layout = "NHWC"
    post_tensor.physical_layout = "NHWC"
    model_ir.tensors[str(names["post_out"])] = post_tensor
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "ADD",
        "ADD",
    ]
    mul = model_ir.operators[0]
    assert str(names["x"]) in mul.inputs
    assert str(names["x_nchw"]) not in mul.inputs
    assert all(str(names["mul_out"]) in op.inputs for op in model_ir.operators[1:])
    actual = _evaluate(model_ir, input_value)
    tolerance = 2e-3 if dtype == "FLOAT16" else 1e-7
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=tolerance,
            atol=tolerance,
        )
    surviving = model_ir.tensors[str(names["mul_out"])]
    assert surviving.shape == post_tensor.shape
    assert surviving.shape_signature == post_tensor.shape_signature
    assert surviving.logical_layout == "NHWC"
    assert surviving.physical_layout == "NHWC"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("mul_mode", ("nhwc", "vector"))
def test_affine_post_add_preserves_legacy_direct_constant_modes(
    mul_mode: str,
) -> None:
    model_ir, names = _model(mul_mode=mul_mode)
    original = np.asarray(model_ir.tensors[str(names["mul_const"])].data).copy()

    first = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)
    second = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)

    assert first == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    assert second == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0}
    np.testing.assert_array_equal(
        model_ir.tensors[str(names["mul_const"])].data,
        original,
    )


def test_affine_post_add_preserves_pre_fanout_and_clones_shared_constant() -> None:
    model_ir, names = _model()
    constant_name = str(names["mul_const"])
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    for source_name, output_name, shape in (
        (str(names["x_nchw"]), "pre_side", (1, 4, 2, 3)),
        (constant_name, "constant_side", (1, 4, 1, 1)),
    ):
        model_ir.tensors[output_name] = _tensor(
            output_name,
            dtype="FLOAT32",
            shape=shape,
        )
        model_ir.outputs.append(output_name)
        model_ir.operators.append(OperatorIR("IDENTITY", [source_name], [output_name]))

    stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    assert any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    np.testing.assert_array_equal(model_ir.tensors[constant_name].data, original)
    clone_name = f"{constant_name}_nhwc"
    assert clone_name in next(op for op in model_ir.operators if op.op_type == "MUL").inputs
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]


def test_affine_post_add_honors_candidate_and_cap() -> None:
    model_ir = ModelIR("bounded_affine_post_add")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    third = _add_chain(model_ir, prefix="third_")

    candidate_stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(
        model_ir,
        candidate=second["mul"],
    )
    capped_stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(
        model_ir,
        max_rewrites=1,
    )

    assert candidate_stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    assert capped_stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    remaining = {
        op.outputs[0]
        for op in model_ir.operators
        if op.op_type == "TRANSPOSE"
        and op.outputs[0] in {
            str(first["post_out"]),
            str(second["post_out"]),
            str(third["post_out"]),
        }
    }
    assert remaining == {str(third["post_out"])}


def _public(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.outputs.append(str(names[key]))


def _quantized(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].quantization = {"scale": [0.1], "zero_point": [0]}


def _fanout(model: ModelIR, names: dict[str, object], key: str) -> None:
    source = model.tensors[str(names[key])]
    output_name = f"{key}_side"
    model.tensors[output_name] = _tensor(
        output_name,
        dtype=source.dtype,
        shape=tuple(source.shape),
        signature=tuple(source.shape_signature or source.shape),
    )
    model.operators.append(OperatorIR("IDENTITY", [str(names[key])], [output_name]))


def _wrong_order(model: ModelIR, names: dict[str, object]) -> None:
    del names
    model.operators[0], model.operators[1] = model.operators[1], model.operators[0]


def _nonfinite(model: ModelIR, names: dict[str, object], key: str) -> None:
    tensor = model.tensors[str(names[key] if key in names else key)]
    tensor.data = np.asarray(tensor.data).copy()
    tensor.data.reshape(-1)[0] = np.inf


_Mutation = Callable[[ModelIR, dict[str, object]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.operators[1].options.update(fusedActivationFunction="RELU"),
        lambda model, names: model.operators[3].options.update(fusedActivationFunction="RELU"),
        lambda model, names: _public(model, names, "x_nchw"),
        lambda model, names: _public(model, names, "mul_out"),
        lambda model, names: _public(model, names, "post_out"),
        lambda model, names: _public(model, names, "mul_const"),
        lambda model, names: _quantized(model, names, "x"),
        lambda model, names: _quantized(model, names, "post_out"),
        lambda model, names: _quantized(model, names, "mul_const"),
        lambda model, names: setattr(model.tensors[str(names["mul_out"])], "dtype", "FLOAT16"),
        lambda model, names: setattr(
            model.tensors[str(names["mul_const"])],
            "data",
            np.asarray(model.tensors[str(names["mul_const"])].data, dtype=np.float16),
        ),
        lambda model, names: _nonfinite(model, names, "mul_const"),
        lambda model, names: _nonfinite(model, names, str(tuple(names["adds"])[0][0])),
        lambda model, names: _fanout(model, names, "mul_out"),
        lambda model, names: _fanout(model, names, "post_out"),
        lambda model, names: model.tensors[str(names["pre_perm"])].data.__setitem__(slice(None), [0, 1, 2, 3]),
        lambda model, names: model.inputs.remove(str(names["x"])),
        lambda model, names: setattr(model.tensors[str(names["mul_const"])], "is_variable", True),
        lambda model, names: model.operators.append(
            OperatorIR("IDENTITY", [str(tuple(names["adds"])[0][0])], [str(names["mul_const"])])
        ),
        lambda model, names: setattr(
            model.tensors[str(tuple(names["adds"])[0][0])],
            "shape_signature",
            [1, 1, -1, 4],
        ),
        lambda model, names: setattr(
            model.tensors[str(tuple(names["adds"])[0][0])],
            "shape",
            [1, 1, 4, 1],
        ),
        lambda model, names: model.operators.append(
            OperatorIR("IDENTITY", [str(names["x"])], [str(names["mul_out"])])
        ),
        _wrong_order,
    ),
)
def test_affine_post_add_rejects_unsafe_contracts_transactionally(
    mutation: _Mutation,
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0}
    assert _fingerprint(model_ir) == before


def test_affine_post_add_rejects_ambiguous_equal_axis_constant() -> None:
    model_ir, _ = _model(mul_mode="full")
    for key in ("x", "x_nchw", "mul_out", "post_out", "y_0"):
        tensor = model_ir.tensors[key]
        tensor.shape = [1, 4, 4, 4]
        tensor.shape_signature = [1, 4, 4, 4]
    tensor = model_ir.tensors["mul_const"]
    tensor.shape = [1, 4, 4, 4]
    tensor.shape_signature = [1, 4, 4, 4]
    tensor.data = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)
    before = _fingerprint(model_ir)

    stats = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0}
    assert _fingerprint(model_ir) == before


def test_affine_post_add_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _model()
    _fanout(model_ir, names, "mul_const")
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["mul"])
    assert plan is not None
    clone_name = plan.constant_update.clone_name
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


def test_affine_post_add_preflight_avoids_graph_index(monkeypatch: pytest.MonkeyPatch) -> None:
    model_ir = ModelIR("no_affine_post_add")
    model_ir.operators = [OperatorIR("MUL", ["x", "c"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(post_add_module, "ModelIRGraphIndex", fail_index)

    assert optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir) == {
        "optimized_transpose_mul_posttranspose_add_nhwc_chains": 0
    }


def test_affine_post_add_counter_is_complete_mutation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()
    prune_calls: list[tuple[ModelIR, LayoutState | None]] = []

    def record_prune(
        active_model_ir: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> None:
        prune_calls.append((active_model_ir, layout_state))

    monkeypatch.setattr(post_add_module, "_prune_unused_tensors", record_prune)

    first = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)
    second = optimize_transpose_mul_posttranspose_add_nhwc_chains(model_ir)

    assert first == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 1}
    assert second == {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0}
    assert prune_calls == [(model_ir, None)]
