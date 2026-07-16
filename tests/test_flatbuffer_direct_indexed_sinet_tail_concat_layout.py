from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_tail_concat_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_concat_resize_affine_tail_concat_transpose_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = (0, 3, 1, 2)
_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = (
    "optimized_sinet_concat_resize_affine_tail_concat_transpose_chains"
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
            0.8 + float(offset),
            num=size,
            dtype=np.float64,
        ).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "tail_concat_",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    resize_type: str = "RESIZE_BILINEAR",
    legacy: bool = True,
    repeated_legacy: bool = False,
    alias: bool = False,
    external_constant_use: bool = False,
) -> dict[str, object]:
    keys = (
        "skip",
        "residual",
        "plain",
        "resize_input",
        "resize_size",
        "resize_out",
        "skip_perm",
        "residual_perm",
        "plain_perm",
        "affine_perm",
        "root_perm",
        "alias_perm",
        "skip_nchw",
        "residual_nchw",
        "plain_nchw",
        "affine_nchw",
        "branch_mul_const",
        "branch_mul_out",
        "branch_add_const",
        "branch_add_out",
        "concat1_out",
        "add0_out",
        "mul1_const",
        "mul1_out",
        "add1_const",
        "add1_out",
        "alpha1",
        "prelu1_out",
        "concat2_out",
        "mul2_const",
        "mul2_out",
        "add2_const",
        "add2_out",
        "alpha2",
        "prelu2_out",
        "post_out",
        "conv_out",
        "legacy_out",
        "alias_out",
        "alias_consumer_out",
        "constant_side",
    )
    names = {key: f"{prefix}{key}" for key in keys}
    nhwc_shapes = {
        "skip": (1, 2, 3, 3),
        "residual": (1, 2, 3, 6),
        "plain": (1, 2, 3, 2),
        "resize_input": (1, 2, 3, 4),
        "resize_out": (1, 2, 3, 4),
        "post_out": (1, 2, 3, 9),
        "conv_out": (1, 2, 3, 9),
    }
    nhwc_signatures = {
        key: (-1, shape[1], -1, shape[3]) for key, shape in nhwc_shapes.items()
    }
    nchw_shapes = {
        "skip_nchw": (1, 3, 2, 3),
        "residual_nchw": (1, 6, 2, 3),
        "plain_nchw": (1, 2, 2, 3),
        "affine_nchw": (1, 4, 2, 3),
        "branch_mul_out": (1, 4, 2, 3),
        "branch_add_out": (1, 4, 2, 3),
        "concat1_out": (1, 6, 2, 3),
        "add0_out": (1, 6, 2, 3),
        "mul1_out": (1, 6, 2, 3),
        "add1_out": (1, 6, 2, 3),
        "prelu1_out": (1, 6, 2, 3),
        "concat2_out": (1, 9, 2, 3),
        "mul2_out": (1, 9, 2, 3),
        "add2_out": (1, 9, 2, 3),
        "prelu2_out": (1, 9, 2, 3),
        "legacy_out": (1, 9, 2, 3),
    }
    nchw_signatures = {
        key: (-1, shape[1], shape[2], -1) for key, shape in nchw_shapes.items()
    }
    model_ir.inputs.extend(
        str(names[key]) for key in ("skip", "residual", "plain", "resize_input")
    )
    for key, shape in nhwc_shapes.items():
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=nhwc_signatures[key],
            layout="NHWC",
        )
    for key, shape in nchw_shapes.items():
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=nchw_signatures[key],
            layout="NCHW",
        )
    if alias:
        for key in ("alias_out", "alias_consumer_out"):
            model_ir.tensors[str(names[key])] = _tensor(
                str(names[key]),
                dtype=dtype,
                shape=(1, 2, 3, 9),
                signature=(-1, 2, -1, 9),
                layout="NHWC",
            )
    model_ir.tensors[str(names["resize_size"])] = _tensor(
        str(names["resize_size"]),
        dtype="INT32",
        shape=(2,),
        data=np.asarray([2, 3], dtype=np.int32),
    )
    for key, values in (
        ("skip_perm", _NHWC_TO_NCHW),
        ("residual_perm", _NHWC_TO_NCHW),
        ("plain_perm", _NHWC_TO_NCHW),
        ("affine_perm", _NHWC_TO_NCHW),
        ("root_perm", _NCHW_TO_NHWC),
        ("alias_perm", _NCHW_TO_NHWC),
    ):
        if key == "alias_perm" and not alias:
            continue
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    constant_specs = (
        ("branch_mul_const", 4),
        ("branch_add_const", 4),
        ("mul1_const", 6),
        ("add1_const", 6),
        ("alpha1", 6),
        ("mul2_const", 9),
        ("add2_const", 9),
        ("alpha2", 9),
    )
    for index, (key, channels) in enumerate(constant_specs):
        data = _constant(
            channels=channels,
            dtype=dtype,
            mode=constant_mode,
            offset=float(index) * 0.04,
        )
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=tuple(data.shape),
            data=data,
        )

    def binary(data: str, constant: str) -> list[str]:
        return [constant, data] if reversed_inputs else [data, constant]

    resize = OperatorIR(
        str(resize_type),
        [str(names["resize_input"]), str(names["resize_size"])],
        [str(names["resize_out"])],
    )
    adapters = {
        key: OperatorIR(
            "TRANSPOSE",
            [str(names[source]), str(names[perm])],
            [str(names[output])],
        )
        for key, source, perm, output in (
            ("skip_pre_op", "skip", "skip_perm", "skip_nchw"),
            (
                "residual_pre_op",
                "residual",
                "residual_perm",
                "residual_nchw",
            ),
            ("plain_pre_op", "plain", "plain_perm", "plain_nchw"),
            (
                "affine_pre_op",
                "resize_out",
                "affine_perm",
                "affine_nchw",
            ),
        )
    }
    branch_mul = OperatorIR(
        "MUL",
        binary(str(names["affine_nchw"]), str(names["branch_mul_const"])),
        [str(names["branch_mul_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    branch_add = OperatorIR(
        "ADD",
        binary(str(names["branch_mul_out"]), str(names["branch_add_const"])),
        [str(names["branch_add_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    concat1_inputs = [str(names["plain_nchw"]), str(names["branch_add_out"])]
    add0_inputs = [str(names["residual_nchw"]), str(names["concat1_out"])]
    concat2_inputs = [str(names["skip_nchw"]), str(names["prelu1_out"])]
    if reversed_inputs:
        concat1_inputs.reverse()
        add0_inputs.reverse()
        concat2_inputs.reverse()
    concat1 = OperatorIR(
        "CONCATENATION",
        concat1_inputs,
        [str(names["concat1_out"])],
        options={"axis": 1, "fusedActivationFunction": "NONE"},
    )
    add0 = OperatorIR(
        "ADD",
        add0_inputs,
        [str(names["add0_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    mul1 = OperatorIR(
        "MUL",
        binary(str(names["add0_out"]), str(names["mul1_const"])),
        [str(names["mul1_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    add1 = OperatorIR(
        "ADD",
        binary(str(names["mul1_out"]), str(names["add1_const"])),
        [str(names["add1_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu1 = OperatorIR(
        "PRELU",
        [str(names["add1_out"]), str(names["alpha1"])],
        [str(names["prelu1_out"])],
    )
    concat2 = OperatorIR(
        "CONCATENATION",
        concat2_inputs,
        [str(names["concat2_out"])],
        options={"axis": 1, "fusedActivationFunction": "NONE"},
    )
    mul2 = OperatorIR(
        "MUL",
        binary(str(names["concat2_out"]), str(names["mul2_const"])),
        [str(names["mul2_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    add2 = OperatorIR(
        "ADD",
        binary(str(names["mul2_out"]), str(names["add2_const"])),
        [str(names["add2_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu2 = OperatorIR(
        "PRELU",
        [str(names["add2_out"]), str(names["alpha2"])],
        [str(names["prelu2_out"])],
    )
    root = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu2_out"]), str(names["root_perm"])],
        [str(names["post_out"])],
    )
    downstream = OperatorIR(
        "CONV_2D",
        [str(names["post_out"])],
        [str(names["conv_out"])],
    )
    operators = [
        resize,
        adapters["skip_pre_op"],
        adapters["residual_pre_op"],
        adapters["plain_pre_op"],
        adapters["affine_pre_op"],
        branch_mul,
        branch_add,
        concat1,
        add0,
        mul1,
        add1,
        prelu1,
        concat2,
        mul2,
        add2,
        prelu2,
        root,
    ]
    alias_post = None
    alias_consumer = None
    if alias:
        alias_post = OperatorIR(
            "TRANSPOSE",
            [str(names["prelu2_out"]), str(names["alias_perm"])],
            [str(names["alias_out"])],
        )
        alias_consumer = OperatorIR(
            "ADD",
            [str(names["alias_out"]), str(names["alias_out"])],
            [str(names["alias_consumer_out"])],
            options={"fusedActivationFunction": "NONE"},
        )
        operators.append(alias_post)
    operators.append(downstream)
    if alias_consumer is not None:
        operators.append(alias_consumer)
    legacy_op = None
    if legacy:
        legacy_inputs = (
            [str(names["prelu2_out"]), str(names["prelu2_out"])]
            if repeated_legacy
            else [str(names["prelu2_out"])]
        )
        legacy_op = OperatorIR(
            "ADD" if repeated_legacy else "IDENTITY",
            legacy_inputs,
            [str(names["legacy_out"])],
            options={"fusedActivationFunction": "NONE"}
            if repeated_legacy
            else {},
        )
        operators.append(legacy_op)
    model_ir.outputs.append(str(names["conv_out"]))
    if alias:
        model_ir.outputs.append(str(names["alias_consumer_out"]))
    if legacy:
        model_ir.outputs.append(str(names["legacy_out"]))
    constant_side = None
    if external_constant_use:
        constant_side = OperatorIR(
            "IDENTITY",
            [str(names["branch_mul_const"])],
            [str(names["constant_side"])],
        )
        source = model_ir.tensors[str(names["branch_mul_const"])]
        model_ir.tensors[str(names["constant_side"])] = _tensor(
            str(names["constant_side"]),
            dtype=dtype,
            shape=tuple(source.shape),
        )
        model_ir.outputs.append(str(names["constant_side"]))
        operators.append(constant_side)
    model_ir.operators.extend(operators)
    names.update(adapters)
    names.update(
        {
            "resize_op": resize,
            "branch_mul_op": branch_mul,
            "branch_add_op": branch_add,
            "concat1_op": concat1,
            "add0_op": add0,
            "mul1_op": mul1,
            "add1_op": add1,
            "prelu1_op": prelu1,
            "concat2_op": concat2,
            "mul2_op": mul2,
            "add2_op": add2,
            "prelu2_op": prelu2,
            "root": root,
            "alias_post_op": alias_post,
            "alias_consumer_op": alias_consumer,
            "downstream_op": downstream,
            "legacy_op": legacy_op,
            "constant_side_op": constant_side,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_tail_concat_layout")
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
        }:
            output = np.asarray(operator_inputs[0])
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(
                operator_inputs,
                axis=int(operator.options["axis"]),
            )
        elif operator.op_type == "ADD":
            output = np.add(operator_inputs[0], operator_inputs[1])
        elif operator.op_type == "MUL":
            output = np.multiply(operator_inputs[0], operator_inputs[1])
        elif operator.op_type == "PRELU":
            output = np.where(
                operator_inputs[0] >= 0,
                operator_inputs[0],
                operator_inputs[0] * operator_inputs[1],
            )
        elif operator.op_type in {"IDENTITY", "CONV_2D"}:
            output = np.asarray(operator_inputs[0])
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _inputs(names: dict[str, object], dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(951)
    return {
        str(names[key]): rng.normal(size=shape).astype(_NP_DTYPES[dtype])
        for key, shape in (
            ("skip", (1, 2, 3, 3)),
            ("residual", (1, 2, 3, 6)),
            ("plain", (1, 2, 3, 2)),
            ("resize_input", (1, 2, 3, 4)),
        )
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
@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize(
    "resize_type", ("RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR")
)
def test_sinet_tail_concat_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    resize_type: str,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        resize_type=resize_type,
    )
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, dtype)
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_STATS_KEY: 1}
    assert second == {_STATS_KEY: 0}
    for key in (
        "skip_pre_op",
        "residual_pre_op",
        "plain_pre_op",
        "affine_pre_op",
        "root",
    ):
        assert names[key] not in model_ir.operators
    assert names["concat1_op"].options["axis"] == 3
    assert names["concat2_op"].options["axis"] == 3
    assert set(names["concat1_op"].inputs) == {
        str(names["plain"]),
        str(names["branch_add_out"]),
    }
    assert set(names["concat2_op"].inputs) == {
        str(names["skip"]),
        str(names["prelu1_out"]),
    }
    assert names["prelu2_op"].outputs == [str(names["post_out"])]
    actual = _evaluate(model_ir, inputs)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=tolerance,
            atol=tolerance,
        )
    for key in ("concat1_out", "add0_out", "mul1_out", "add1_out", "prelu1_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 6]
        assert tensor.shape_signature == [-1, 2, -1, 6]
        assert tensor.logical_layout == "NHWC"
    for key in ("concat2_out", "mul2_out", "add2_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 9]
        assert tensor.shape_signature == [-1, 2, -1, 9]
        assert tensor.logical_layout == "NHWC"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_tail_concat_merges_aliases_and_preserves_repeated_slots() -> None:
    model_ir, names = _model(alias=True, repeated_legacy=True)
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, "FLOAT32")
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir
    )

    assert stats == {_STATS_KEY: 1}
    assert names["alias_post_op"] not in model_ir.operators
    assert names["alias_consumer_op"].inputs == [
        str(names["post_out"]),
        str(names["post_out"]),
    ]
    assert names["legacy_op"].inputs == [
        str(names["prelu2_out"]),
        str(names["prelu2_out"]),
    ]
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(actual[output_name], expected[output_name])


def test_sinet_tail_concat_supports_no_legacy_and_external_constant_clone() -> None:
    model_ir, names = _model(legacy=False, external_constant_use=True)
    original = np.asarray(
        model_ir.tensors[str(names["branch_mul_const"])].data
    ).copy()

    stats = optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir
    )

    assert stats == {_STATS_KEY: 1}
    assert names["constant_side_op"].inputs == [
        str(names["branch_mul_const"])
    ]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["branch_mul_const"])].data),
        original,
    )
    assert any(
        str(name).endswith("_nhwc")
        for name in names["branch_mul_op"].inputs
    )


def test_sinet_tail_concat_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_tail_concat")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = (
        optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
            model_ir,
            graph_index=graph_index,
            candidate=second["root"],
        )
    )
    capped_stats = (
        optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
            model_ir,
            graph_index=graph_index,
            max_rewrites=1,
        )
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
    model_ir.operators.append(OperatorIR("IDENTITY", [source_name], [output_name]))
    model_ir.outputs.append(output_name)


def _move_root_before_prelu(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.remove(names["root"])
    model_ir.operators.insert(model_ir.operators.index(names["prelu2_op"]), names["root"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.tensors[str(names["skip_perm"])].data.__setitem__(
            slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)
        ),
        lambda model, names: model.tensors[
            str(names["residual_perm"])
        ].data.__setitem__(slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)),
        lambda model, names: model.tensors[
            str(names["affine_perm"])
        ].data.__setitem__(slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)),
        lambda model, names: model.tensors[str(names["root_perm"])].data.__setitem__(
            slice(None), np.asarray(_NHWC_TO_NCHW, dtype=np.int32)
        ),
        lambda model, names: setattr(names["resize_op"], "op_type", "IDENTITY"),
        lambda model, names: names["concat1_op"].options.__setitem__("axis", 3),
        lambda model, names: names["concat2_op"].options.__setitem__("axis", 3),
        lambda model, names: names["concat1_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["mul1_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["add2_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: _append_fanout(model, names, "skip_nchw"),
        lambda model, names: _append_fanout(model, names, "residual_nchw"),
        lambda model, names: _append_fanout(model, names, "plain_nchw"),
        lambda model, names: _append_fanout(model, names, "affine_nchw"),
        lambda model, names: _append_fanout(model, names, "prelu1_out"),
        lambda model, names: _append_fanout(model, names, "concat2_out"),
        lambda model, names: model.outputs.append(str(names["concat1_out"])),
        lambda model, names: model.outputs.append(str(names["post_out"])),
        lambda model, names: setattr(
            model.tensors[str(names["skip"])], "dtype", "FLOAT16"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["resize_out"])], "shape", [1, 3, 2, 4]
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha2"])], "is_variable", True
        ),
        lambda model, names: model.tensors[str(names["alpha2"])].data.__setitem__(
            (0, 0, 0, 0), np.nan
        ),
        lambda model, names: _move_root_before_prelu(model, names),
    ),
)
def test_sinet_tail_concat_rejects_unsafe_variants_transactionally(
    mutation: Callable[[ModelIR, dict[str, object]], None],
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir
    )

    assert stats == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_tail_concat_rejects_stale_plan_transactionally() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["concat2_op"].inputs.reverse()
    graph_index.replace_operator_inputs(
        graph_index.operator_index(names["concat2_op"]),
        names["concat2_op"].inputs,
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_tail_concat_no_index_preflight_avoids_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight should return before index construction")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.passes.sinet_tail_concat_layout.ModelIRGraphIndex",
        fail_index,
    )

    assert optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
        model_ir,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
