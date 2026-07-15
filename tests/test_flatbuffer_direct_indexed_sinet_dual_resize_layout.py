from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.sinet_dual_resize_layout as dual_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_dual_resize_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_deep_skip_dual_resize_affine_transpose_chains,
    optimize_sinet_dual_resize_affine_transpose_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = (0, 3, 1, 2)
_NCHW_TO_NHWC = (0, 2, 3, 1)
_DIRECT_KEY = "optimized_sinet_dual_resize_affine_transpose_chains"
_SIBLING_KEY = (
    "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains"
)


def _tensor(
    name: str,
    *,
    dtype: str,
    shape: tuple[int, ...],
    signature: tuple[int, ...] | None = None,
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=str(name),
        dtype=str(dtype),
        shape=[int(value) for value in shape],
        shape_signature=[
            int(value) for value in (shape if signature is None else signature)
        ],
        data=data,
        is_variable=False,
        logical_layout=str(layout),
        physical_layout=str(layout),
    )


def _constant(
    *,
    channels: int,
    dtype: str,
    mode: str,
    offset: float,
) -> np.ndarray:
    shape = () if mode == "scalar" else (1, int(channels), 1, 1)
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.linspace(
            0.2 + float(offset),
            0.75 + float(offset),
            num=size,
            dtype=np.float64,
        ).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "dual_",
    mode: str = "direct",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    resize_type: str = "RESIZE_BILINEAR",
    legacy: bool = False,
    legacy_before_root: bool = False,
    post_alias: bool = False,
    external_constant_use: bool = False,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "residual_nhwc",
            "residual_nchw",
            "residual_perm",
            "residual_side",
            "r0",
            "r1",
            "resize0_size",
            "resize1_size",
            "resize0",
            "resize1",
            "pre0_perm",
            "pre1_perm",
            "pre0",
            "pre1",
            "b0_mul_const",
            "b0_mul",
            "b0_add_const",
            "b0_add",
            "b1_mul_const",
            "b1_mul",
            "b1_add_const",
            "b1_add",
            "concat",
            "add0",
            "tail_mul_const",
            "tail_mul",
            "tail_add_const",
            "tail_add",
            "alpha",
            "prelu",
            "post_perm",
            "post",
            "final",
            "alias_perm",
            "alias",
            "alias_final",
            "legacy",
            "constant_side",
        )
    }
    branch0_nhwc = (1, 2, 3, 2)
    branch0_nhwc_sig = (-1, 2, -1, 2)
    branch0_nchw = (1, 2, 2, 3)
    branch0_nchw_sig = (-1, 2, 2, -1)
    branch1_nhwc = (1, 2, 3, 3)
    branch1_nhwc_sig = (-1, 2, -1, 3)
    branch1_nchw = (1, 3, 2, 3)
    branch1_nchw_sig = (-1, 3, 2, -1)
    target_nhwc = (1, 2, 3, 5)
    target_nhwc_sig = (-1, 2, -1, 5)
    target_nchw = (1, 5, 2, 3)
    target_nchw_sig = (-1, 5, 2, -1)

    residual_input = (
        str(names["residual_nhwc"])
        if mode == "direct"
        else str(names["residual_nchw"])
    )
    model_ir.inputs.extend(
        [residual_input, str(names["r0"]), str(names["r1"])]
    )
    model_ir.tensors[str(names["residual_nhwc"])] = _tensor(
        str(names["residual_nhwc"]),
        dtype=dtype,
        shape=target_nhwc,
        signature=target_nhwc_sig,
        layout="NHWC",
    )
    model_ir.tensors[str(names["residual_nchw"])] = _tensor(
        str(names["residual_nchw"]),
        dtype=dtype,
        shape=target_nchw,
        signature=target_nchw_sig,
        layout="NCHW",
    )
    for key, shape, signature in (
        ("r0", branch0_nhwc, branch0_nhwc_sig),
        ("resize0", branch0_nhwc, branch0_nhwc_sig),
        ("r1", branch1_nhwc, branch1_nhwc_sig),
        ("resize1", branch1_nhwc, branch1_nhwc_sig),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout="NHWC",
        )
    for key, shape, signature in (
        ("pre0", branch0_nchw, branch0_nchw_sig),
        ("b0_mul", branch0_nchw, branch0_nchw_sig),
        ("b0_add", branch0_nchw, branch0_nchw_sig),
        ("pre1", branch1_nchw, branch1_nchw_sig),
        ("b1_mul", branch1_nchw, branch1_nchw_sig),
        ("b1_add", branch1_nchw, branch1_nchw_sig),
        ("concat", target_nchw, target_nchw_sig),
        ("add0", target_nchw, target_nchw_sig),
        ("tail_mul", target_nchw, target_nchw_sig),
        ("tail_add", target_nchw, target_nchw_sig),
        ("prelu", target_nchw, target_nchw_sig),
        ("legacy", target_nchw, target_nchw_sig),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout="NCHW",
        )
    for key in ("post", "final", "alias", "alias_final"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=target_nhwc,
            signature=target_nhwc_sig,
            layout="NHWC",
        )
    if mode == "sibling":
        model_ir.tensors[str(names["residual_side"])] = _tensor(
            str(names["residual_side"]),
            dtype=dtype,
            shape=target_nhwc,
            signature=target_nhwc_sig,
            layout="NHWC",
        )

    for key, values in (
        ("residual_perm", _NHWC_TO_NCHW if mode == "direct" else _NCHW_TO_NHWC),
        ("pre0_perm", _NHWC_TO_NCHW),
        ("pre1_perm", _NHWC_TO_NCHW),
        ("post_perm", _NCHW_TO_NHWC),
        ("alias_perm", _NCHW_TO_NHWC),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    for key in ("resize0_size", "resize1_size"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(2,),
            data=np.asarray([2, 3], dtype=np.int32),
        )
    constant_specs = (
        ("b0_mul_const", 2),
        ("b0_add_const", 2),
        ("b1_mul_const", 3),
        ("b1_add_const", 3),
        ("tail_mul_const", 5),
        ("tail_add_const", 5),
        ("alpha", 5),
    )
    for index, (key, channels) in enumerate(constant_specs):
        data = _constant(
            channels=channels,
            dtype=dtype,
            mode=constant_mode,
            offset=float(index) * 0.03,
        )
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=tuple(data.shape),
            data=data,
        )

    def binary(data: str, constant: str) -> list[str]:
        return [constant, data] if reversed_inputs else [data, constant]

    residual_adapter = OperatorIR(
        "TRANSPOSE",
        [residual_input, str(names["residual_perm"])],
        [
            str(names["residual_nchw"])
            if mode == "direct"
            else str(names["residual_nhwc"])
        ],
    )
    residual_side = None
    if mode == "sibling":
        residual_side = OperatorIR(
            "IDENTITY",
            [str(names["residual_nhwc"])],
            [str(names["residual_side"])],
        )
        model_ir.outputs.append(str(names["residual_side"]))
    resize0 = OperatorIR(
        str(resize_type),
        [str(names["r0"]), str(names["resize0_size"])],
        [str(names["resize0"])],
    )
    resize1 = OperatorIR(
        str(resize_type),
        [str(names["r1"]), str(names["resize1_size"])],
        [str(names["resize1"])],
    )
    pre0 = OperatorIR(
        "TRANSPOSE",
        [str(names["resize0"]), str(names["pre0_perm"])],
        [str(names["pre0"])],
    )
    pre1 = OperatorIR(
        "TRANSPOSE",
        [str(names["resize1"]), str(names["pre1_perm"])],
        [str(names["pre1"])],
    )
    b0_mul = OperatorIR(
        "MUL",
        binary(str(names["pre0"]), str(names["b0_mul_const"])),
        [str(names["b0_mul"])],
        options={"fusedActivationFunction": "NONE"},
    )
    b0_add = OperatorIR(
        "ADD",
        binary(str(names["b0_mul"]), str(names["b0_add_const"])),
        [str(names["b0_add"])],
        options={"fusedActivationFunction": "NONE"},
    )
    b1_mul = OperatorIR(
        "MUL",
        binary(str(names["pre1"]), str(names["b1_mul_const"])),
        [str(names["b1_mul"])],
        options={"fusedActivationFunction": "NONE"},
    )
    b1_add = OperatorIR(
        "ADD",
        binary(str(names["b1_mul"]), str(names["b1_add_const"])),
        [str(names["b1_add"])],
        options={"fusedActivationFunction": "NONE"},
    )
    concat_inputs = [str(names["b0_add"]), str(names["b1_add"])]
    add0_inputs = [str(names["residual_nchw"]), str(names["concat"])]
    if reversed_inputs:
        concat_inputs.reverse()
        add0_inputs.reverse()
    concat = OperatorIR(
        "CONCATENATION",
        concat_inputs,
        [str(names["concat"])],
        options={"axis": 1, "fusedActivationFunction": "NONE"},
    )
    add0 = OperatorIR(
        "ADD",
        add0_inputs,
        [str(names["add0"])],
        options={"fusedActivationFunction": "NONE"},
    )
    tail_mul = OperatorIR(
        "MUL",
        binary(str(names["add0"]), str(names["tail_mul_const"])),
        [str(names["tail_mul"])],
        options={"fusedActivationFunction": "NONE"},
    )
    tail_add = OperatorIR(
        "ADD",
        binary(str(names["tail_mul"]), str(names["tail_add_const"])),
        [str(names["tail_add"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu = OperatorIR(
        "PRELU",
        [str(names["tail_add"]), str(names["alpha"])],
        [str(names["prelu"])],
    )
    root = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu"]), str(names["post_perm"])],
        [str(names["post"])],
    )
    final = OperatorIR("IDENTITY", [str(names["post"])], [str(names["final"])])
    alias = None
    alias_final = None
    if post_alias:
        alias = OperatorIR(
            "TRANSPOSE",
            [str(names["prelu"]), str(names["alias_perm"])],
            [str(names["alias"])],
        )
        alias_final = OperatorIR(
            "IDENTITY",
            [str(names["alias"])],
            [str(names["alias_final"])],
        )
        model_ir.outputs.append(str(names["alias_final"]))
    legacy_op = None
    if legacy:
        legacy_op = OperatorIR(
            "ADD",
            [str(names["prelu"]), str(names["prelu"])],
            [str(names["legacy"])],
        )
        model_ir.outputs.append(str(names["legacy"]))

    tail = [root]
    if alias is not None:
        tail.append(alias)
    tail.append(final)
    if alias_final is not None:
        tail.append(alias_final)
    if legacy_op is not None:
        if legacy_before_root:
            tail.insert(0, legacy_op)
        else:
            tail.append(legacy_op)
    operators = [residual_adapter]
    if residual_side is not None:
        operators.append(residual_side)
    operators.extend(
        [
            resize0,
            pre0,
            b0_mul,
            b0_add,
            resize1,
            pre1,
            b1_mul,
            b1_add,
            concat,
            add0,
            tail_mul,
            tail_add,
            prelu,
            *tail,
        ]
    )
    model_ir.outputs.append(str(names["final"]))

    constant_side = None
    if external_constant_use:
        source = model_ir.tensors[str(names["b0_mul_const"])]
        model_ir.tensors[str(names["constant_side"])] = _tensor(
            str(names["constant_side"]),
            dtype=dtype,
            shape=tuple(source.shape),
        )
        constant_side = OperatorIR(
            "IDENTITY",
            [str(names["b0_mul_const"])],
            [str(names["constant_side"])],
        )
        operators.append(constant_side)
        model_ir.outputs.append(str(names["constant_side"]))

    model_ir.operators.extend(operators)
    names.update(
        {
            "mode": mode,
            "residual_adapter_op": residual_adapter,
            "residual_side_op": residual_side,
            "resize0_op": resize0,
            "resize1_op": resize1,
            "pre0_op": pre0,
            "pre1_op": pre1,
            "b0_mul_op": b0_mul,
            "b0_add_op": b0_add,
            "b1_mul_op": b1_mul,
            "b1_add_op": b1_add,
            "concat_op": concat,
            "add0_op": add0,
            "tail_mul_op": tail_mul,
            "tail_add_op": tail_add,
            "prelu_op": prelu,
            "root": root,
            "final_op": final,
            "alias_op": alias,
            "alias_final_op": alias_final,
            "legacy_op": legacy_op,
            "constant_side_op": constant_side,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_dual_resize_layout")
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
        operator_inputs = [values[str(name)] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            output = np.transpose(
                operator_inputs[0],
                tuple(int(value) for value in operator_inputs[1].reshape(-1)),
            )
        elif operator.op_type in {
            "RESIZE_BILINEAR",
            "RESIZE_NEAREST_NEIGHBOR",
            "IDENTITY",
        }:
            output = np.asarray(operator_inputs[0])
        elif operator.op_type == "MUL":
            output = np.multiply(operator_inputs[0], operator_inputs[1])
        elif operator.op_type == "ADD":
            output = np.add(operator_inputs[0], operator_inputs[1])
        elif operator.op_type == "PRELU":
            output = np.where(
                operator_inputs[0] >= 0,
                operator_inputs[0],
                operator_inputs[0] * operator_inputs[1],
            )
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(
                operator_inputs,
                axis=int(operator.options["axis"]),
            )
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
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


def _optimizer(mode: str) -> Callable[..., dict[str, int]]:
    return (
        optimize_sinet_dual_resize_affine_transpose_chains
        if mode == "direct"
        else optimize_sinet_deep_skip_dual_resize_affine_transpose_chains
    )


def _stats_key(mode: str) -> str:
    return _DIRECT_KEY if mode == "direct" else _SIBLING_KEY


@pytest.mark.parametrize("mode", ("direct", "sibling"))
@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32", "FLOAT64"))
@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize(
    "resize_type", ("RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR")
)
def test_sinet_dual_resize_is_indexed_and_numerically_equivalent(
    mode: str,
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    resize_type: str,
) -> None:
    model_ir, names = _model(
        mode=mode,
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        resize_type=resize_type,
    )
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(952)
    inputs = {
        str(name): rng.normal(size=model_ir.tensors[str(name)].shape).astype(
            _NP_DTYPES[dtype]
        )
        for name in model_ir.inputs
    }
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    optimizer = _optimizer(mode)

    first = optimizer(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimizer(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_stats_key(mode): 1}
    assert second == {_stats_key(mode): 0}
    assert names["pre0_op"] not in model_ir.operators
    assert names["pre1_op"] not in model_ir.operators
    assert names["root"] not in model_ir.operators
    assert (names["residual_adapter_op"] in model_ir.operators) == (
        mode == "sibling"
    )
    assert int(names["concat_op"].options["axis"]) == 3
    assert set(names["add0_op"].inputs) == {
        str(names["residual_nhwc"]),
        str(names["concat"]),
    }
    actual = _evaluate(model_ir, inputs)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=tolerance,
            atol=tolerance,
        )
    for key, shape, signature in (
        ("b0_mul", [1, 2, 3, 2], [-1, 2, -1, 2]),
        ("b0_add", [1, 2, 3, 2], [-1, 2, -1, 2]),
        ("b1_mul", [1, 2, 3, 3], [-1, 2, -1, 3]),
        ("b1_add", [1, 2, 3, 3], [-1, 2, -1, 3]),
        ("concat", [1, 2, 3, 5], [-1, 2, -1, 5]),
        ("add0", [1, 2, 3, 5], [-1, 2, -1, 5]),
        ("tail_mul", [1, 2, 3, 5], [-1, 2, -1, 5]),
        ("tail_add", [1, 2, 3, 5], [-1, 2, -1, 5]),
    ):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == shape
        assert tensor.shape_signature == signature
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_direct_dual_resize_inserts_one_legacy_inverse_adapter() -> None:
    model_ir, names = _model(mode="direct", legacy=True)
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(953)
    inputs = {
        str(name): rng.normal(size=model_ir.tensors[str(name)].shape).astype(
            np.float32
        )
        for name in model_ir.inputs
    }
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_dual_resize_affine_transpose_chains(model_ir)

    assert stats == {_DIRECT_KEY: 1}
    adapters = [
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == [str(names["prelu"])]
    ]
    assert len(adapters) == 1
    assert adapters[0].inputs == [
        str(names["post"]),
        str(names["residual_perm"]),
    ]
    assert model_ir.operators.index(adapters[0]) < model_ir.operators.index(
        names["legacy_op"]
    )
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_sibling_dual_resize_rewires_repeated_legacy_slots() -> None:
    model_ir, names = _model(mode="sibling", legacy=True)

    stats = optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
        model_ir
    )

    assert stats == {_SIBLING_KEY: 1}
    assert names["legacy_op"].inputs == [
        str(names["post"]),
        str(names["post"]),
    ]
    assert not any(
        operator.op_type == "TRANSPOSE"
        and operator.outputs == [str(names["prelu"])]
        for operator in model_ir.operators
    )


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_merges_post_aliases(mode: str) -> None:
    model_ir, names = _model(mode=mode, post_alias=True)
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(954)
    inputs = {
        str(name): rng.normal(size=model_ir.tensors[str(name)].shape).astype(
            np.float32
        )
        for name in model_ir.inputs
    }
    expected = _evaluate(original, inputs)

    stats = _optimizer(mode)(model_ir)

    assert stats == {_stats_key(mode): 1}
    assert names["root"] not in model_ir.operators
    assert names["alias_op"] not in model_ir.operators
    assert names["alias_final_op"].inputs == [str(names["post"])]
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_dual_resize_clones_externally_used_constant() -> None:
    model_ir, names = _model(external_constant_use=True)
    original = np.asarray(
        model_ir.tensors[str(names["b0_mul_const"])].data
    ).copy()

    stats = optimize_sinet_dual_resize_affine_transpose_chains(model_ir)

    assert stats == {_DIRECT_KEY: 1}
    assert names["constant_side_op"].inputs == [
        str(names["b0_mul_const"])
    ]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["b0_mul_const"])].data),
        original,
    )
    clone_name = next(
        name
        for name in names["b0_mul_op"].inputs
        if str(name).endswith("_nhwc")
    )
    assert model_ir.tensors[str(clone_name)].shape == [1, 1, 1, 2]


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_honors_candidate_and_total_cap(mode: str) -> None:
    model_ir = ModelIR(f"bounded_sinet_dual_resize_{mode}")
    first = _add_chain(model_ir, prefix="first_", mode=mode)
    second = _add_chain(model_ir, prefix="second_", mode=mode)
    graph_index = ModelIRGraphIndex(model_ir)
    optimizer = _optimizer(mode)

    candidate_stats = optimizer(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimizer(
        model_ir,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert candidate_stats == {_stats_key(mode): 1}
    assert capped_stats == {_stats_key(mode): 1}
    assert first["root"] not in model_ir.operators
    assert second["root"] not in model_ir.operators
    _assert_index_current(model_ir, graph_index)


def test_sinet_dual_resize_owners_do_not_cross_residual_modes() -> None:
    direct, _ = _model(mode="direct")
    sibling, _ = _model(mode="sibling")
    direct_before = _fingerprint(direct)
    sibling_before = _fingerprint(sibling)

    direct_stats = optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
        direct
    )
    sibling_stats = optimize_sinet_dual_resize_affine_transpose_chains(sibling)

    assert direct_stats == {_SIBLING_KEY: 0}
    assert sibling_stats == {_DIRECT_KEY: 0}
    assert _fingerprint(direct) == direct_before
    assert _fingerprint(sibling) == sibling_before


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


def _move_residual_adapter_after_add(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    adapter = names["residual_adapter_op"]
    model_ir.operators.remove(adapter)
    index = model_ir.operators.index(names["add0_op"])
    model_ir.operators.insert(index + 1, adapter)


def _duplicate_add_output(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [str(names["r0"])],
            [str(names["add0"])],
        )
    )


_Mutation = Callable[[ModelIR, dict[str, object]], None]


@pytest.mark.parametrize("mode", ("direct", "sibling"))
@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: setattr(
            model.tensors[str(names["pre0_perm"])],
            "data",
            np.asarray([0, 1, 3, 2], dtype=np.int32),
        ),
        lambda model, names: setattr(
            model.tensors[str(names["post_perm"])],
            "data",
            np.asarray([0, 1, 3, 2], dtype=np.int32),
        ),
        lambda model, names: names["concat_op"].options.update({"axis": 3}),
        lambda model, names: _append_fanout(model, names, "pre0"),
        lambda model, names: _append_fanout(model, names, "b0_add"),
        lambda model, names: setattr(
            names["resize0_op"], "op_type", "IDENTITY"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["resize0"])],
            "shape",
            [1, 2, 4, 2],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["pre1"])],
            "shape_signature",
            [-1, 3, 3, -1],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["add0"])],
            "dtype",
            "FLOAT16",
        ),
        lambda model, names: setattr(
            model.tensors[str(names["tail_mul"])],
            "quantization",
            object(),
        ),
        lambda model, names: model.outputs.append(str(names["concat"])),
        lambda model, names: model.outputs.append(str(names["post"])),
        lambda model, names: names["tail_add_op"].options.update(
            {"fusedActivationFunction": "RELU"}
        ),
        lambda model, names: setattr(
            model.tensors[str(names["b0_mul_const"])],
            "data",
            np.full((1, 2, 1, 1), np.nan, dtype=np.float32),
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha"])],
            "is_variable",
            True,
        ),
        lambda model, names: setattr(
            model.tensors[str(names["tail_add_const"])],
            "data",
            None,
        ),
        lambda model, names: _duplicate_add_output(model, names),
        lambda model, names: model.operators.remove(names["final_op"]),
    ),
)
def test_sinet_dual_resize_rejects_unsafe_contracts_transactionally(
    mode: str,
    mutation: _Mutation,
) -> None:
    model_ir, names = _model(mode=mode)
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = _optimizer(mode)(model_ir)

    assert stats == {_stats_key(mode): 0}
    assert _fingerprint(model_ir) == before


def test_sinet_sibling_dual_resize_rejects_extra_residual_consumer() -> None:
    model_ir, names = _model(mode="sibling")
    _append_fanout(model_ir, names, "residual_nchw")
    before = _fingerprint(model_ir)

    stats = optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
        model_ir
    )

    assert stats == {_SIBLING_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_sibling_dual_resize_rejects_late_residual_adapter() -> None:
    model_ir, names = _model(mode="sibling")
    _move_residual_adapter_after_add(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
        model_ir
    )

    assert stats == {_SIBLING_KEY: 0}
    assert _fingerprint(model_ir) == before


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_rejects_legacy_consumer_before_root(
    mode: str,
) -> None:
    model_ir, _ = _model(
        mode=mode,
        legacy=True,
        legacy_before_root=True,
    )
    before = _fingerprint(model_ir)

    stats = _optimizer(mode)(model_ir)

    assert stats == {_stats_key(mode): 0}
    assert _fingerprint(model_ir) == before


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_apply_revalidates_stale_plan(mode: str) -> None:
    model_ir, names = _model(mode=mode)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        names["root"],
        residual_mode=mode,
    )
    assert plan is not None
    model_ir.tensors[str(names["post"])].shape = [1, 2, 3, 6]
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_apply_preflight_rejects_clone_collision(
    mode: str,
) -> None:
    model_ir, names = _model(mode=mode, external_constant_use=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        names["root"],
        residual_mode=mode,
    )
    assert plan is not None
    clone_name = next(
        constant.clone_name
        for constant in plan.constant_plans
        if constant.clone_name is not None
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


@pytest.mark.parametrize("mode", ("direct", "sibling"))
def test_sinet_dual_resize_preflight_avoids_graph_index(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR(f"no_sinet_dual_resize_{mode}")
    model_ir.operators = [OperatorIR("TRANSPOSE", ["x", "p"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(dual_module, "ModelIRGraphIndex", fail_index)

    assert _optimizer(mode)(model_ir) == {_stats_key(mode): 0}
