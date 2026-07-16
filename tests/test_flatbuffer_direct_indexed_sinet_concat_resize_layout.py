from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_concat_resize_affine_transpose_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = (0, 3, 1, 2)
_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = "optimized_sinet_concat_resize_affine_transpose_chains"


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
    prefix: str = "concat_resize_",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    resize_type: str = "RESIZE_BILINEAR",
    downstream_type: str = "CONV_2D",
    legacy: bool = True,
    repeated_legacy: bool = False,
    alias: bool = False,
    external_constant_use: bool = False,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "residual",
            "plain",
            "resize_input",
            "resize_size",
            "resize_out",
            "residual_perm",
            "plain_perm",
            "affine_perm",
            "root_perm",
            "alias_perm",
            "residual_nchw",
            "plain_nchw",
            "affine_nchw",
            "branch_mul_const",
            "branch_mul_out",
            "branch_add_const",
            "branch_add_out",
            "concat_out",
            "add0_out",
            "tail_mul_const",
            "tail_mul_out",
            "tail_add_const",
            "tail_add_out",
            "alpha",
            "prelu_out",
            "post_out",
            "conv_out",
            "legacy_out",
            "alias_out",
            "alias_consumer_out",
            "constant_side",
        )
    }
    plain_nhwc_shape = (1, 2, 3, 2)
    plain_nhwc_signature = (-1, 2, -1, 2)
    affine_nhwc_shape = (1, 2, 3, 4)
    affine_nhwc_signature = (-1, 2, -1, 4)
    merged_nhwc_shape = (1, 2, 3, 6)
    merged_nhwc_signature = (-1, 2, -1, 6)
    plain_nchw_shape = (1, 2, 2, 3)
    plain_nchw_signature = (-1, 2, 2, -1)
    affine_nchw_shape = (1, 4, 2, 3)
    affine_nchw_signature = (-1, 4, 2, -1)
    merged_nchw_shape = (1, 6, 2, 3)
    merged_nchw_signature = (-1, 6, 2, -1)

    model_ir.inputs.extend(
        [
            str(names["residual"]),
            str(names["plain"]),
            str(names["resize_input"]),
        ]
    )
    for key, shape, signature in (
        ("residual", merged_nhwc_shape, merged_nhwc_signature),
        ("plain", plain_nhwc_shape, plain_nhwc_signature),
        ("resize_input", affine_nhwc_shape, affine_nhwc_signature),
        ("resize_out", affine_nhwc_shape, affine_nhwc_signature),
        ("post_out", merged_nhwc_shape, merged_nhwc_signature),
        ("conv_out", merged_nhwc_shape, merged_nhwc_signature),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout="NHWC",
        )
    for key, shape, signature in (
        ("residual_nchw", merged_nchw_shape, merged_nchw_signature),
        ("plain_nchw", plain_nchw_shape, plain_nchw_signature),
        ("affine_nchw", affine_nchw_shape, affine_nchw_signature),
        ("branch_mul_out", affine_nchw_shape, affine_nchw_signature),
        ("branch_add_out", affine_nchw_shape, affine_nchw_signature),
        ("concat_out", merged_nchw_shape, merged_nchw_signature),
        ("add0_out", merged_nchw_shape, merged_nchw_signature),
        ("tail_mul_out", merged_nchw_shape, merged_nchw_signature),
        ("tail_add_out", merged_nchw_shape, merged_nchw_signature),
        ("prelu_out", merged_nchw_shape, merged_nchw_signature),
        ("legacy_out", merged_nchw_shape, merged_nchw_signature),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout="NCHW",
        )
    if alias:
        for key in ("alias_out", "alias_consumer_out"):
            model_ir.tensors[str(names[key])] = _tensor(
                str(names[key]),
                dtype=dtype,
                shape=merged_nhwc_shape,
                signature=merged_nhwc_signature,
                layout="NHWC",
            )
    model_ir.tensors[str(names["resize_size"])] = _tensor(
        str(names["resize_size"]),
        dtype="INT32",
        shape=(2,),
        data=np.asarray([2, 3], dtype=np.int32),
    )
    for key, values in (
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
    for index, (key, channels) in enumerate(
        (
            ("branch_mul_const", 4),
            ("branch_add_const", 4),
            ("tail_mul_const", 6),
            ("tail_add_const", 6),
            ("alpha", 6),
        )
    ):
        data = _constant(
            channels=channels,
            dtype=dtype,
            mode=constant_mode,
            offset=float(index) * 0.05,
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
        options={"alignCorners": False, "halfPixelCenters": True},
    )
    residual_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["residual"]), str(names["residual_perm"])],
        [str(names["residual_nchw"])],
    )
    plain_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["plain"]), str(names["plain_perm"])],
        [str(names["plain_nchw"])],
    )
    affine_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["resize_out"]), str(names["affine_perm"])],
        [str(names["affine_nchw"])],
    )
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
    concat_inputs = [str(names["plain_nchw"]), str(names["branch_add_out"])]
    add0_inputs = [str(names["residual_nchw"]), str(names["concat_out"])]
    if reversed_inputs:
        concat_inputs.reverse()
        add0_inputs.reverse()
    concat = OperatorIR(
        "CONCATENATION",
        concat_inputs,
        [str(names["concat_out"])],
        options={"axis": 1, "fusedActivationFunction": "NONE"},
    )
    add0 = OperatorIR(
        "ADD",
        add0_inputs,
        [str(names["add0_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    tail_mul = OperatorIR(
        "MUL",
        binary(str(names["add0_out"]), str(names["tail_mul_const"])),
        [str(names["tail_mul_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    tail_add = OperatorIR(
        "ADD",
        binary(str(names["tail_mul_out"]), str(names["tail_add_const"])),
        [str(names["tail_add_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu = OperatorIR(
        "PRELU",
        [str(names["tail_add_out"]), str(names["alpha"])],
        [str(names["prelu_out"])],
    )
    root = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu_out"]), str(names["root_perm"])],
        [str(names["post_out"])],
    )
    downstream = OperatorIR(
        str(downstream_type),
        [str(names["post_out"])],
        [str(names["conv_out"])],
    )
    operators = [
        resize,
        residual_pre,
        plain_pre,
        affine_pre,
        branch_mul,
        branch_add,
        concat,
        add0,
        tail_mul,
        tail_add,
        prelu,
        root,
    ]

    alias_post = None
    alias_consumer = None
    if alias:
        alias_post = OperatorIR(
            "TRANSPOSE",
            [str(names["prelu_out"]), str(names["alias_perm"])],
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
            [str(names["prelu_out"]), str(names["prelu_out"])]
            if repeated_legacy
            else [str(names["prelu_out"])]
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
    names.update(
        {
            "resize_op": resize,
            "residual_pre_op": residual_pre,
            "plain_pre_op": plain_pre,
            "affine_pre_op": affine_pre,
            "branch_mul_op": branch_mul,
            "branch_add_op": branch_add,
            "concat_op": concat,
            "add0_op": add0,
            "tail_mul_op": tail_mul,
            "tail_add_op": tail_add,
            "prelu_op": prelu,
            "root": root,
            "alias_post_op": alias_post,
            "downstream_op": downstream,
            "alias_consumer_op": alias_consumer,
            "legacy_op": legacy_op,
            "constant_side_op": constant_side,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_concat_resize_layout")
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
        elif operator.op_type in {
            "IDENTITY",
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
        }:
            output = np.asarray(operator_inputs[0])
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _inputs(names: dict[str, object], dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(950)
    return {
        str(names["residual"]): rng.normal(size=(1, 2, 3, 6)).astype(
            _NP_DTYPES[dtype]
        ),
        str(names["plain"]): rng.normal(size=(1, 2, 3, 2)).astype(
            _NP_DTYPES[dtype]
        ),
        str(names["resize_input"]): rng.normal(size=(1, 2, 3, 4)).astype(
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
@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize(
    "resize_type", ("RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR")
)
def test_sinet_concat_resize_is_indexed_and_numerically_equivalent(
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

    first = optimize_sinet_concat_resize_affine_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_concat_resize_affine_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_STATS_KEY: 1}
    assert second == {_STATS_KEY: 0}
    for key in ("residual_pre_op", "plain_pre_op", "affine_pre_op", "root"):
        assert names[key] not in model_ir.operators
    assert names["branch_mul_op"].inputs[
        0 if not reversed_inputs else 1
    ] == str(names["resize_out"])
    assert set(names["concat_op"].inputs) == {
        str(names["plain"]),
        str(names["branch_add_out"]),
    }
    assert names["concat_op"].options["axis"] == 3
    assert set(names["add0_op"].inputs) == {
        str(names["residual"]),
        str(names["concat_out"]),
    }
    assert names["prelu_op"].outputs == [str(names["post_out"])]
    actual = _evaluate(model_ir, inputs)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=tolerance,
            atol=tolerance,
        )
    for key in ("branch_mul_out", "branch_add_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 4]
        assert tensor.shape_signature == [-1, 2, -1, 4]
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    for key in ("concat_out", "add0_out", "tail_mul_out", "tail_add_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 6]
        assert tensor.shape_signature == [-1, 2, -1, 6]
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_concat_resize_merges_aliases_and_preserves_repeated_slots() -> None:
    model_ir, names = _model(alias=True, repeated_legacy=True)
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, "FLOAT32")
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_concat_resize_affine_transpose_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert names["alias_post_op"] not in model_ir.operators
    assert names["alias_consumer_op"].inputs == [
        str(names["post_out"]),
        str(names["post_out"]),
    ]
    assert names["legacy_op"].inputs == [
        str(names["prelu_out"]),
        str(names["prelu_out"]),
    ]
    adapters = [
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == [str(names["prelu_out"])]
    ]
    assert len(adapters) == 1
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_concat_resize_supports_no_legacy_branch() -> None:
    model_ir, names = _model(legacy=False)

    stats = optimize_sinet_concat_resize_affine_transpose_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert all(
        str(names["prelu_out"]) not in operator.outputs
        for operator in model_ir.operators
    )


def test_sinet_concat_resize_clones_externally_used_constants() -> None:
    model_ir, names = _model(external_constant_use=True)
    original = np.asarray(
        model_ir.tensors[str(names["branch_mul_const"])].data
    ).copy()

    stats = optimize_sinet_concat_resize_affine_transpose_chains(model_ir)

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


def test_sinet_concat_resize_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_concat_resize")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_concat_resize_affine_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_concat_resize_affine_transpose_chains(
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
    model_ir.operators.append(OperatorIR("IDENTITY", [source_name], [output_name]))
    model_ir.outputs.append(output_name)


def _duplicate_source(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [str(names["resize_input"])],
            [str(names["resize_out"])],
        )
    )


def _move_root_before_prelu(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.remove(names["root"])
    model_ir.operators.insert(model_ir.operators.index(names["prelu_op"]), names["root"])


def _move_legacy_before_prelu(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.remove(names["legacy_op"])
    model_ir.operators.insert(model_ir.operators.index(names["prelu_op"]), names["legacy_op"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.tensors[
            str(names["residual_perm"])
        ].data.__setitem__(slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)),
        lambda model, names: model.tensors[str(names["plain_perm"])].data.__setitem__(
            slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)
        ),
        lambda model, names: model.tensors[
            str(names["affine_perm"])
        ].data.__setitem__(slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)),
        lambda model, names: model.tensors[str(names["root_perm"])].data.__setitem__(
            slice(None), np.asarray(_NHWC_TO_NCHW, dtype=np.int32)
        ),
        lambda model, names: setattr(names["resize_op"], "op_type", "IDENTITY"),
        lambda model, names: names["concat_op"].options.__setitem__("axis", 3),
        lambda model, names: names["concat_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["branch_mul_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["branch_add_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["add0_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["tail_mul_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["tail_add_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: _append_fanout(model, names, "residual_nchw"),
        lambda model, names: _append_fanout(model, names, "plain_nchw"),
        lambda model, names: _append_fanout(model, names, "affine_nchw"),
        lambda model, names: _append_fanout(model, names, "branch_mul_out"),
        lambda model, names: _append_fanout(model, names, "branch_add_out"),
        lambda model, names: _append_fanout(model, names, "add0_out"),
        lambda model, names: model.outputs.append(str(names["concat_out"])),
        lambda model, names: setattr(
            model.tensors[str(names["plain"])], "dtype", "FLOAT16"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["resize_out"])], "shape", [1, 3, 2, 4]
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha"])], "is_variable", True
        ),
        lambda model, names: model.tensors[str(names["alpha"])].data.__setitem__(
            (0, 0, 0, 0), np.nan
        ),
        lambda model, names: _duplicate_source(model, names),
        lambda model, names: _move_root_before_prelu(model, names),
        lambda model, names: _move_legacy_before_prelu(model, names),
        lambda model, names: model.outputs.append(str(names["post_out"])),
    ),
)
def test_sinet_concat_resize_rejects_unsafe_variants_transactionally(
    mutation: Callable[[ModelIR, dict[str, object]], None],
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_concat_resize_affine_transpose_chains(model_ir)

    assert stats == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_concat_resize_rejects_stale_plan_transactionally() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["concat_op"].inputs.reverse()
    graph_index.replace_operator_inputs(
        graph_index.operator_index(names["concat_op"]),
        names["concat_op"].inputs,
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_concat_resize_rejects_clone_collision_transactionally() -> None:
    model_ir, names = _model(external_constant_use=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    clone_name = next(
        constant.clone_name
        for constant in plan.constant_plans
        if constant.clone_name is not None
    )
    source = model_ir.tensors[str(names["branch_mul_const"])]
    model_ir.tensors[str(clone_name)] = _tensor(
        str(clone_name),
        dtype=str(source.dtype),
        shape=tuple(source.shape),
        data=np.asarray(source.data).copy(),
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_concat_resize_no_index_preflight_avoids_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight should return before index construction")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.passes.sinet_concat_resize_layout.ModelIRGraphIndex",
        fail_index,
    )

    assert optimize_sinet_concat_resize_affine_transpose_chains(
        model_ir,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
