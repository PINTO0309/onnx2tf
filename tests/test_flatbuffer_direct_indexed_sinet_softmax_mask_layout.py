from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_softmax_mask_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_softmax_mask_residual_nhwc_tail_chains,
)


_STATS_KEY = "optimized_sinet_softmax_mask_residual_nhwc_tail_chains"
_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = np.asarray([0, 3, 1, 2], dtype=np.int32)
_NCHW_TO_NHWC = np.asarray([0, 2, 3, 1], dtype=np.int32)
_NCHW_TO_NWHC = np.asarray([0, 3, 2, 1], dtype=np.int32)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    shape: tuple[int, ...],
    signature: tuple[int, ...] | None = None,
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=False,
        quantization=None,
        logical_layout=layout,
        physical_layout=layout,
    )


def _add_constant(
    model_ir: ModelIR,
    name: str,
    data: np.ndarray,
    *,
    dtype: str,
    declared_shape: tuple[int, ...] | None = None,
) -> str:
    array = np.asarray(data)
    shape = tuple(array.shape) if declared_shape is None else declared_shape
    model_ir.tensors[name] = _tensor(
        name,
        dtype=dtype,
        shape=shape,
        data=array,
    )
    return name


def _add_value(
    model_ir: ModelIR,
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
    signature: tuple[int, ...],
    layout: str,
) -> str:
    model_ir.tensors[name] = _tensor(
        name,
        dtype=dtype,
        shape=shape,
        signature=signature,
        layout=layout,
    )
    return name


def _binary_inputs(
    data_name: str,
    constant_name: str,
    reversed_inputs: bool,
) -> list[str]:
    return (
        [constant_name, data_name]
        if reversed_inputs
        else [data_name, constant_name]
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    constant_mode: str = "channel",
    reversed_inputs: bool = False,
    alias: bool = False,
    legacy: bool = True,
    repeated_slots: bool = False,
    external_constant_use: bool = False,
    dynamic_signature: bool = False,
) -> dict[str, object]:
    np_dtype = _NP_DTYPES[dtype]
    nhwc_shape = (1, 2, 3, 4)
    nhwc_signature = (
        (-1, 2, -1, 4) if dynamic_signature else nhwc_shape
    )
    nchw_shape = (1, 4, 2, 3)
    nchw_signature = (
        (-1, 4, 2, -1) if dynamic_signature else nchw_shape
    )
    nwhc_shape = (1, 3, 2, 4)
    nwhc_signature = (
        (-1, -1, 2, 4) if dynamic_signature else nwhc_shape
    )
    reduce_shape = (1, 2, 3)
    reduce_signature = (
        (-1, 2, -1) if dynamic_signature else reduce_shape
    )
    reshape_shape = (1, 1, 2, 3)
    reshape_signature = (
        (-1, 1, 2, -1) if dynamic_signature else reshape_shape
    )

    def name(value: str) -> str:
        return f"{prefix}{value}"

    main = _add_value(
        model_ir,
        name("main"),
        dtype=dtype,
        shape=nhwc_shape,
        signature=nhwc_signature,
        layout="NHWC",
    )
    side = _add_value(
        model_ir,
        name("side"),
        dtype=dtype,
        shape=nhwc_shape,
        signature=nhwc_signature,
        layout="NHWC",
    )
    model_ir.inputs.extend([main, side])

    constants: dict[str, str] = {}
    for key, values in (
        ("side_perm", _NHWC_TO_NCHW),
        ("main_perm", _NHWC_TO_NCHW),
        ("soft_pre_perm", _NCHW_TO_NWHC),
        ("soft_post_perm", _NCHW_TO_NWHC),
        ("post_perm", _NCHW_TO_NHWC),
    ):
        constants[key] = _add_constant(
            model_ir,
            name(key),
            np.asarray(values, dtype=np.int32).copy(),
            dtype="INT32",
        )
    constants["reduce_axis"] = _add_constant(
        model_ir,
        name("reduce_axis"),
        np.asarray([1], dtype=np.int32),
        dtype="INT32",
    )
    constants["reshape_shape"] = _add_constant(
        model_ir,
        name("reshape_shape"),
        np.asarray(reshape_shape, dtype=np.int32),
        dtype="INT32",
    )
    constants["sub_one"] = _add_constant(
        model_ir,
        name("sub_one"),
        np.asarray(1.0, dtype=np_dtype),
        dtype=dtype,
        declared_shape=(1,),
    )
    if constant_mode == "scalar":
        main_mul_data = np.asarray(0.75, dtype=np_dtype)
        main_add_data = np.asarray(0.1, dtype=np_dtype)
        side_alpha_data = np.asarray(0.2, dtype=np_dtype)
    elif constant_mode == "channel":
        main_mul_data = np.linspace(0.5, 0.8, 4, dtype=np_dtype).reshape(
            1, 4, 1, 1
        )
        main_add_data = np.linspace(-0.2, 0.1, 4, dtype=np_dtype).reshape(
            1, 4, 1, 1
        )
        side_alpha_data = np.linspace(0.1, 0.4, 4, dtype=np_dtype).reshape(
            1, 4, 1, 1
        )
    elif constant_mode == "full":
        main_mul_data = np.linspace(
            0.5, 0.8, np.prod(nchw_shape), dtype=np_dtype
        ).reshape(nchw_shape)
        main_add_data = np.linspace(
            -0.2, 0.1, np.prod(nchw_shape), dtype=np_dtype
        ).reshape(nchw_shape)
        side_alpha_data = np.linspace(
            0.1, 0.4, np.prod(nchw_shape), dtype=np_dtype
        ).reshape(nchw_shape)
    else:
        raise AssertionError(constant_mode)
    for key, data in (
        ("main_mul_const", main_mul_data),
        ("main_add_const", main_add_data),
        ("side_alpha", side_alpha_data),
        (
            "expand_const",
            np.ones(nchw_shape, dtype=np_dtype),
        ),
    ):
        constants[key] = _add_constant(
            model_ir,
            name(key),
            np.asarray(data),
            dtype=dtype,
        )

    shapes = {
        "side_nchw": (nchw_shape, nchw_signature, "NCHW"),
        "side_prelu": (nchw_shape, nchw_signature, "NCHW"),
        "main_nchw": (nchw_shape, nchw_signature, "NCHW"),
        "main_mul": (nchw_shape, nchw_signature, "NCHW"),
        "main_add": (nchw_shape, nchw_signature, "NCHW"),
        "soft_pre": (nwhc_shape, nwhc_signature, "UNKNOWN"),
        "softmax": (nwhc_shape, nwhc_signature, "UNKNOWN"),
        "soft_back": (nchw_shape, nchw_signature, "NCHW"),
        "reduce": (reduce_shape, reduce_signature, "UNKNOWN"),
        "sub": (reduce_shape, reduce_signature, "UNKNOWN"),
        "reshape": (reshape_shape, reshape_signature, "NCHW"),
        "expand": (nchw_shape, nchw_signature, "NCHW"),
        "side_mask": (nchw_shape, nchw_signature, "NCHW"),
        "residual": (nchw_shape, nchw_signature, "NCHW"),
        "post": (nhwc_shape, nhwc_signature, "NHWC"),
        "downstream": (nhwc_shape, nhwc_signature, "NHWC"),
    }
    values = {
        key: _add_value(
            model_ir,
            name(key),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        for key, (shape, signature, layout) in shapes.items()
    }

    side_pre = OperatorIR(
        "TRANSPOSE",
        [side, constants["side_perm"]],
        [values["side_nchw"]],
    )
    side_prelu = OperatorIR(
        "PRELU",
        [values["side_nchw"], constants["side_alpha"]],
        [values["side_prelu"]],
        {"fusedActivationFunction": "NONE"},
    )
    main_pre = OperatorIR(
        "TRANSPOSE",
        [main, constants["main_perm"]],
        [values["main_nchw"]],
    )
    main_mul = OperatorIR(
        "MUL",
        _binary_inputs(
            values["main_nchw"],
            constants["main_mul_const"],
            reversed_inputs,
        ),
        [values["main_mul"]],
        {"fusedActivationFunction": "NONE"},
    )
    main_add = OperatorIR(
        "ADD",
        _binary_inputs(
            values["main_mul"],
            constants["main_add_const"],
            reversed_inputs,
        ),
        [values["main_add"]],
        {"fusedActivationFunction": "NONE"},
    )
    soft_pre = OperatorIR(
        "TRANSPOSE",
        [values["main_add"], constants["soft_pre_perm"]],
        [values["soft_pre"]],
    )
    softmax = OperatorIR(
        "SOFTMAX",
        [values["soft_pre"]],
        [values["softmax"]],
        {"axis": 3, "beta": 1.0},
    )
    soft_post = OperatorIR(
        "TRANSPOSE",
        [values["softmax"], constants["soft_post_perm"]],
        [values["soft_back"]],
    )
    reduce_max = OperatorIR(
        "REDUCE_MAX",
        [values["soft_back"], constants["reduce_axis"]],
        [values["reduce"]],
        {"keepDims": False},
    )
    sub_inputs = [constants["sub_one"], values["reduce"]]
    if reversed_inputs:
        sub_inputs.reverse()
    sub = OperatorIR(
        "SUB",
        sub_inputs,
        [values["sub"]],
        {"fusedActivationFunction": "NONE"},
    )
    reshape = OperatorIR(
        "RESHAPE",
        [values["sub"], constants["reshape_shape"]],
        [values["reshape"]],
        {"newShape": list(reshape_shape)},
    )
    expand = OperatorIR(
        "MUL",
        _binary_inputs(
            values["reshape"],
            constants["expand_const"],
            reversed_inputs,
        ),
        [values["expand"]],
        {"fusedActivationFunction": "NONE"},
    )
    side_mask_inputs = [values["side_prelu"], values["expand"]]
    residual_inputs = [values["side_mask"], values["main_add"]]
    if reversed_inputs:
        side_mask_inputs.reverse()
        residual_inputs.reverse()
    side_mask = OperatorIR(
        "MUL",
        side_mask_inputs,
        [values["side_mask"]],
        {"fusedActivationFunction": "NONE"},
    )
    residual = OperatorIR(
        "ADD",
        residual_inputs,
        [values["residual"]],
        {"fusedActivationFunction": "NONE"},
    )
    root = OperatorIR(
        "TRANSPOSE",
        [values["residual"], constants["post_perm"]],
        [values["post"]],
    )
    downstream = OperatorIR(
        "IDENTITY",
        [values["post"]],
        [values["downstream"]],
    )
    operators = [
        side_pre,
        side_prelu,
        main_pre,
        main_mul,
        main_add,
        soft_pre,
        softmax,
        soft_post,
        reduce_max,
        sub,
        reshape,
        expand,
        side_mask,
        residual,
        root,
        downstream,
    ]
    model_ir.outputs.append(values["downstream"])

    alias_post = None
    alias_consumer = None
    if alias:
        alias_perm = _add_constant(
            model_ir,
            name("alias_perm"),
            _NCHW_TO_NHWC.copy(),
            dtype="INT32",
        )
        alias_output = _add_value(
            model_ir,
            name("alias_post"),
            dtype=dtype,
            shape=nhwc_shape,
            signature=nhwc_signature,
            layout="NHWC",
        )
        alias_result = _add_value(
            model_ir,
            name("alias_result"),
            dtype=dtype,
            shape=nhwc_shape,
            signature=nhwc_signature,
            layout="NHWC",
        )
        alias_post = OperatorIR(
            "TRANSPOSE",
            [values["residual"], alias_perm],
            [alias_output],
        )
        alias_inputs = (
            [alias_output, alias_output]
            if repeated_slots
            else [alias_output]
        )
        alias_consumer = OperatorIR(
            "ADD" if repeated_slots else "IDENTITY",
            alias_inputs,
            [alias_result],
            {"fusedActivationFunction": "NONE"}
            if repeated_slots
            else {},
        )
        operators.extend([alias_post, alias_consumer])
        model_ir.outputs.append(alias_result)

    legacy_op = None
    if legacy:
        legacy_output = _add_value(
            model_ir,
            name("legacy_output"),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
            layout="NCHW",
        )
        legacy_inputs = (
            [values["residual"], values["residual"]]
            if repeated_slots
            else [values["residual"]]
        )
        legacy_op = OperatorIR(
            "ADD" if repeated_slots else "IDENTITY",
            legacy_inputs,
            [legacy_output],
            {"fusedActivationFunction": "NONE"}
            if repeated_slots
            else {},
        )
        operators.append(legacy_op)
        model_ir.outputs.append(legacy_output)

    constant_side = None
    if external_constant_use:
        source = constants["main_mul_const"]
        tensor = model_ir.tensors[source]
        constant_side_output = _add_value(
            model_ir,
            name("constant_side_output"),
            dtype=dtype,
            shape=tuple(tensor.shape),
            signature=tuple(tensor.shape_signature or tensor.shape),
            layout="UNKNOWN",
        )
        constant_side = OperatorIR(
            "IDENTITY",
            [source],
            [constant_side_output],
        )
        operators.append(constant_side)
        model_ir.outputs.append(constant_side_output)

    model_ir.operators.extend(operators)
    result: dict[str, object] = {
        "main": main,
        "side": side,
        **constants,
        **values,
        "side_pre_op": side_pre,
        "side_prelu_op": side_prelu,
        "main_pre_op": main_pre,
        "main_mul_op": main_mul,
        "main_add_op": main_add,
        "soft_pre_op": soft_pre,
        "softmax_op": softmax,
        "soft_post_op": soft_post,
        "reduce_op": reduce_max,
        "sub_op": sub,
        "reshape_op": reshape,
        "expand_op": expand,
        "side_mask_op": side_mask,
        "residual_op": residual,
        "root": root,
        "downstream_op": downstream,
        "alias_post_op": alias_post,
        "alias_consumer_op": alias_consumer,
        "legacy_op": legacy_op,
        "constant_side_op": constant_side,
    }
    return result


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_softmax_mask_layout")
    return model_ir, _add_chain(model_ir, **kwargs)


def _evaluate(
    model_ir: ModelIR,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = {
        str(name): np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({str(name): np.asarray(value) for name, value in inputs.items()})
    for operator in model_ir.operators:
        args = [values[str(name)] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            output = np.transpose(
                args[0],
                tuple(int(value) for value in args[1].reshape(-1)),
            )
        elif operator.op_type == "PRELU":
            output = np.where(args[0] >= 0, args[0], args[0] * args[1])
        elif operator.op_type == "MUL":
            output = np.multiply(args[0], args[1])
        elif operator.op_type == "ADD":
            output = np.add(args[0], args[1])
        elif operator.op_type == "SUB":
            output = np.subtract(args[0], args[1])
        elif operator.op_type == "SOFTMAX":
            axis = int(operator.options["axis"])
            beta = float(operator.options.get("beta", 1.0))
            shifted = beta * args[0]
            shifted = shifted - np.max(shifted, axis=axis, keepdims=True)
            exponential = np.exp(shifted)
            output = exponential / np.sum(
                exponential,
                axis=axis,
                keepdims=True,
            )
        elif operator.op_type == "REDUCE_MAX":
            output = np.max(
                args[0],
                axis=tuple(int(value) for value in args[1].reshape(-1)),
                keepdims=bool(operator.options.get("keepDims", False)),
            )
        elif operator.op_type == "RESHAPE":
            output = np.reshape(
                args[0],
                tuple(int(value) for value in args[1].reshape(-1)),
            )
        elif operator.op_type == "IDENTITY":
            output = np.asarray(args[0])
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _inputs(names: dict[str, object], dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(952)
    return {
        str(names["main"]): rng.normal(size=(1, 2, 3, 4)).astype(
            _NP_DTYPES[dtype]
        ),
        str(names["side"]): rng.normal(size=(1, 2, 3, 4)).astype(
            _NP_DTYPES[dtype]
        ),
    }


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
                tuple(tensor.shape_signature or ()),
                bool(tensor.is_variable),
                repr(tensor.quantization),
                str(tensor.logical_layout),
                str(tensor.physical_layout),
                data,
            )
        )
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(tensors),
        tuple(
            (
                str(operator.op_type),
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
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
@pytest.mark.parametrize("constant_mode", ("scalar", "channel", "full"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_sinet_softmax_mask_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        dynamic_signature=True,
    )
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, dtype)
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_STATS_KEY: 1}
    assert second == {_STATS_KEY: 0}
    for key in (
        "side_pre_op",
        "main_pre_op",
        "soft_pre_op",
        "soft_post_op",
        "root",
    ):
        assert names[key] not in model_ir.operators
    assert names["side_prelu_op"].inputs[0] == str(names["side"])
    assert str(names["main"]) in names["main_mul_op"].inputs
    assert names["softmax_op"].inputs == [str(names["main_add"])]
    assert names["softmax_op"].outputs == [str(names["soft_back"])]
    assert names["residual_op"].outputs == [str(names["post"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["reduce_axis"])].data),
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["reshape_shape"])].data),
        np.asarray([1, 2, 3, 1], dtype=np.int32),
    )
    assert names["reshape_op"].options["newShape"] == [1, 2, 3, 1]
    actual = _evaluate(model_ir, inputs)
    tolerance = 5e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=tolerance,
            atol=tolerance,
        )
    for key in (
        "main_mul",
        "main_add",
        "soft_back",
        "side_prelu",
        "expand",
        "side_mask",
    ):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 4]
        assert tensor.shape_signature == [-1, 2, -1, 4]
        assert tensor.logical_layout == "NHWC"
    reshape_tensor = model_ir.tensors[str(names["reshape"])]
    assert reshape_tensor.shape == [1, 2, 3, 1]
    assert reshape_tensor.shape_signature == [-1, 2, -1, 1]
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_softmax_mask_merges_aliases_and_preserves_repeated_slots() -> None:
    model_ir, names = _model(alias=True, repeated_slots=True)
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, "FLOAT32")
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert names["alias_post_op"] not in model_ir.operators
    assert names["alias_consumer_op"].inputs == [
        str(names["post"]),
        str(names["post"]),
    ]
    assert names["legacy_op"].inputs == [
        str(names["residual"]),
        str(names["residual"]),
    ]
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(actual[output_name], expected[output_name])


def test_sinet_softmax_mask_supports_no_legacy_and_constant_clones() -> None:
    model_ir, names = _model(
        legacy=False,
        external_constant_use=True,
        constant_mode="full",
    )
    original = np.asarray(
        model_ir.tensors[str(names["main_mul_const"])].data
    ).copy()

    stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert names["constant_side_op"].inputs == [str(names["main_mul_const"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["main_mul_const"])].data),
        original,
    )
    assert any(
        str(input_name).endswith("_nhwc")
        for input_name in names["main_mul_op"].inputs
    )


def test_sinet_softmax_mask_clones_externally_used_axis_constant() -> None:
    model_ir, names = _model()
    axis_name = str(names["reduce_axis"])
    axis_side_output = "axis_side_output"
    model_ir.tensors[axis_side_output] = _tensor(
        axis_side_output,
        dtype="INT32",
        shape=(1,),
    )
    axis_side = OperatorIR("IDENTITY", [axis_name], [axis_side_output])
    model_ir.operators.append(axis_side)
    model_ir.outputs.append(axis_side_output)

    stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[axis_name].data),
        np.asarray([1], dtype=np.int32),
    )
    assert axis_side.inputs == [axis_name]
    rewritten_axis = str(names["reduce_op"].inputs[1])
    assert rewritten_axis != axis_name
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[rewritten_axis].data),
        np.asarray([3], dtype=np.int32),
    )


def test_sinet_softmax_mask_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_softmax_mask")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
        model_ir,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert candidate_stats == {_STATS_KEY: 1}
    assert capped_stats == {_STATS_KEY: 1}
    assert first["root"] not in model_ir.operators
    assert second["root"] not in model_ir.operators
    _assert_index_current(model_ir, graph_index)


def _append_fanout(
    model_ir: ModelIR,
    names: dict[str, object],
    key: str,
) -> None:
    source_name = str(names[key])
    source = model_ir.tensors[source_name]
    output_name = f"{source_name}_fanout"
    model_ir.tensors[output_name] = _tensor(
        output_name,
        dtype=str(source.dtype),
        shape=tuple(source.shape),
        signature=tuple(source.shape_signature or source.shape),
        layout=str(source.logical_layout),
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", [source_name], [output_name])
    )
    model_ir.outputs.append(output_name)


def _duplicate_producer(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.insert(
        0,
        OperatorIR(
            "IDENTITY",
            [str(names["main"])],
            [str(names["main_add"])],
        ),
    )


def _move_root_before_residual(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.remove(names["root"])
    model_ir.operators.insert(
        model_ir.operators.index(names["residual_op"]),
        names["root"],
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.tensors[
            str(names["main_perm"])
        ].data.__setitem__(slice(None), _NCHW_TO_NHWC),
        lambda model, names: model.tensors[
            str(names["soft_pre_perm"])
        ].data.__setitem__(slice(None), _NCHW_TO_NHWC),
        lambda model, names: model.tensors[
            str(names["post_perm"])
        ].data.__setitem__(slice(None), _NHWC_TO_NCHW),
        lambda model, names: names["softmax_op"].options.__setitem__("axis", 1),
        lambda model, names: names["softmax_op"].options.__setitem__(
            "beta", np.nan
        ),
        lambda model, names: names["reduce_op"].options.__setitem__(
            "keepDims", True
        ),
        lambda model, names: model.tensors[
            str(names["reduce_axis"])
        ].data.__setitem__(0, 2),
        lambda model, names: model.tensors[
            str(names["reshape_shape"])
        ].data.__setitem__(1, 2),
        lambda model, names: names["reshape_op"].options.__setitem__(
            "newShape", [1, 2, 3, 1]
        ),
        lambda model, names: names["main_mul_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["side_mask_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: _append_fanout(model, names, "main_nchw"),
        lambda model, names: _append_fanout(model, names, "soft_pre"),
        lambda model, names: _append_fanout(model, names, "soft_back"),
        lambda model, names: _append_fanout(model, names, "reshape"),
        lambda model, names: _append_fanout(model, names, "side_prelu"),
        lambda model, names: model.outputs.append(str(names["residual"])),
        lambda model, names: model.outputs.append(str(names["post"])),
        lambda model, names: setattr(
            model.tensors[str(names["side"])], "dtype", "FLOAT16"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["main"])], "shape", [1, 3, 2, 4]
        ),
        lambda model, names: setattr(
            model.tensors[str(names["side_alpha"])], "is_variable", True
        ),
        lambda model, names: model.tensors[
            str(names["main_add_const"])
        ].data.__setitem__((0, 0, 0, 0), np.nan),
        lambda model, names: setattr(
            model.tensors[str(names["expand_const"])],
            "quantization",
            {"scale": [1.0]},
        ),
        _duplicate_producer,
        _move_root_before_residual,
    ),
)
def test_sinet_softmax_mask_rejects_unsafe_variants_transactionally(
    mutation: Callable[[ModelIR, dict[str, object]], None],
) -> None:
    model_ir, names = _model(constant_mode="channel")
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_softmax_mask_residual_nhwc_tail_chains(model_ir)

    assert stats == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_softmax_mask_rejects_stale_plan_transactionally() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["softmax_op"].options["axis"] = 1
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_softmax_mask_rejects_clone_collision_transactionally() -> None:
    model_ir, names = _model(external_constant_use=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    clone_name = next(
        constant.clone_name
        for constant in plan.constant_plans
        if constant.clone_name is not None
    )
    assert clone_name is not None
    source = model_ir.tensors[str(names["main_mul_const"])]
    model_ir.tensors[clone_name] = copy.deepcopy(source)
    model_ir.tensors[clone_name].name = clone_name
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_softmax_mask_no_index_preflight_avoids_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight should return before index construction")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.passes.sinet_softmax_mask_layout.ModelIRGraphIndex",
        fail_index,
    )

    assert optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
        model_ir,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
