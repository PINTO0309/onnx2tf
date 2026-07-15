from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_shared_post_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_shared_post_prelu_transpose_fanout_chains,
)


_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = (0, 3, 1, 2)
_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = "optimized_sinet_shared_post_prelu_transpose_fanout_chains"


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
    prefix: str = "shared_",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    downstream_type: str = "CONV_2D",
    external_constant_use: bool = False,
    repeated_post_add: bool = False,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "lhs",
            "rhs",
            "concat",
            "direct",
            "side",
            "concat_perm",
            "direct_perm",
            "concat_nchw",
            "direct_nchw",
            "add0_out",
            "mul_const",
            "mul_out",
            "add_const",
            "add_out",
            "alpha",
            "prelu_out",
            "root_perm",
            "post_out",
            "conv_out",
            "fanout_out",
            "constant_side",
        )
    }
    half_shape = (1, 2, 3, 2)
    half_signature = (-1, 2, -1, 2)
    nhwc_shape = (1, 2, 3, 4)
    nhwc_signature = (-1, 2, -1, 4)
    nchw_shape = (1, 4, 2, 3)
    nchw_signature = (-1, 4, 2, -1)

    model_ir.inputs.extend(
        [
            str(names["lhs"]),
            str(names["rhs"]),
            str(names["direct"]),
            str(names["side"]),
        ]
    )
    for key in ("lhs", "rhs"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=half_shape,
            signature=half_signature,
            layout="NHWC",
        )
    for key in ("concat", "direct", "side", "post_out", "conv_out", "fanout_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nhwc_shape,
            signature=nhwc_signature,
            layout="NHWC",
        )
    for key in (
        "concat_nchw",
        "direct_nchw",
        "add0_out",
        "mul_out",
        "add_out",
        "prelu_out",
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
            layout="NCHW",
        )
    for key, values in (
        ("concat_perm", _NHWC_TO_NCHW),
        ("direct_perm", _NHWC_TO_NCHW),
        ("root_perm", _NCHW_TO_NHWC),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    for index, key in enumerate(("mul_const", "add_const", "alpha")):
        data = _constant(
            channels=4,
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

    concat_inputs = [str(names["lhs"]), str(names["rhs"])]
    add0_inputs = [str(names["concat_nchw"]), str(names["direct_nchw"])]
    if reversed_inputs:
        concat_inputs.reverse()
        add0_inputs.reverse()
    concat = OperatorIR(
        "CONCATENATION",
        concat_inputs,
        [str(names["concat"])],
        options={"axis": 3, "fusedActivationFunction": "NONE"},
    )
    concat_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["concat"]), str(names["concat_perm"])],
        [str(names["concat_nchw"])],
    )
    direct_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["direct"]), str(names["direct_perm"])],
        [str(names["direct_nchw"])],
    )
    add0 = OperatorIR(
        "ADD",
        add0_inputs,
        [str(names["add0_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    mul = OperatorIR(
        "MUL",
        binary(str(names["add0_out"]), str(names["mul_const"])),
        [str(names["mul_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    add = OperatorIR(
        "ADD",
        binary(str(names["mul_out"]), str(names["add_const"])),
        [str(names["add_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu = OperatorIR(
        "PRELU",
        [str(names["add_out"]), str(names["alpha"])],
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
    fanout_inputs = (
        [str(names["post_out"]), str(names["post_out"])]
        if repeated_post_add
        else [str(names["post_out"]), str(names["side"])]
    )
    fanout = OperatorIR(
        "ADD",
        fanout_inputs,
        [str(names["fanout_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    operators = [
        concat,
        concat_pre,
        direct_pre,
        add0,
        mul,
        add,
        prelu,
        root,
        downstream,
        fanout,
    ]
    model_ir.outputs.extend([str(names["conv_out"]), str(names["fanout_out"])])

    constant_side = None
    if external_constant_use:
        constant_side = OperatorIR(
            "IDENTITY",
            [str(names["mul_const"])],
            [str(names["constant_side"])],
        )
        source = model_ir.tensors[str(names["mul_const"])]
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
            "concat_op": concat,
            "concat_pre_op": concat_pre,
            "direct_pre_op": direct_pre,
            "add0_op": add0,
            "mul_op": mul,
            "add_op": add,
            "prelu_op": prelu,
            "root": root,
            "downstream_op": downstream,
            "fanout_op": fanout,
            "constant_side_op": constant_side,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_shared_post_layout")
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
        str(names["lhs"]): rng.normal(size=(1, 2, 3, 2)).astype(_NP_DTYPES[dtype]),
        str(names["rhs"]): rng.normal(size=(1, 2, 3, 2)).astype(_NP_DTYPES[dtype]),
        str(names["direct"]): rng.normal(size=(1, 2, 3, 4)).astype(
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
@pytest.mark.parametrize("downstream_type", ("CONV_2D", "DEPTHWISE_CONV_2D"))
def test_sinet_shared_post_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    downstream_type: str,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        downstream_type=downstream_type,
    )
    original = copy.deepcopy(model_ir)
    expected = _evaluate(original, _inputs(names, dtype))
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_shared_post_prelu_transpose_fanout_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_shared_post_prelu_transpose_fanout_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_STATS_KEY: 1}
    assert second == {_STATS_KEY: 0}
    assert names["concat_pre_op"] not in model_ir.operators
    assert names["direct_pre_op"] not in model_ir.operators
    assert names["root"] not in model_ir.operators
    assert set(names["add0_op"].inputs) == {
        str(names["concat"]),
        str(names["direct"]),
    }
    assert names["prelu_op"].outputs == [str(names["post_out"])]
    actual = _evaluate(model_ir, _inputs(names, dtype))
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=tolerance,
            atol=tolerance,
        )
    for key in ("add0_out", "mul_out", "add_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 4]
        assert tensor.shape_signature == [-1, 2, -1, 4]
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_shared_post_supports_repeated_add_slots() -> None:
    model_ir, names = _model(repeated_post_add=True)
    original = copy.deepcopy(model_ir)
    inputs = _inputs(names, "FLOAT32")
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_shared_post_prelu_transpose_fanout_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert names["fanout_op"].inputs == [
        str(names["post_out"]),
        str(names["post_out"]),
    ]
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_shared_post_clones_externally_used_constants() -> None:
    model_ir, names = _model(external_constant_use=True)
    original = np.asarray(model_ir.tensors[str(names["mul_const"])].data).copy()

    stats = optimize_sinet_shared_post_prelu_transpose_fanout_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    assert names["constant_side_op"].inputs == [str(names["mul_const"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["mul_const"])].data),
        original,
    )
    assert any(str(name).endswith("_nhwc") for name in names["mul_op"].inputs)


def test_sinet_shared_post_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_shared_post")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_shared_post_prelu_transpose_fanout_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_shared_post_prelu_transpose_fanout_chains(
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
            [str(names["lhs"])],
            [str(names["concat"])],
        )
    )


def _make_both_concat_backed(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.inputs.remove(str(names["direct"]))
    direct_concat = OperatorIR(
        "CONCATENATION",
        [str(names["lhs"]), str(names["rhs"])],
        [str(names["direct"])],
        options={"axis": 3, "fusedActivationFunction": "NONE"},
    )
    model_ir.operators.insert(model_ir.operators.index(names["direct_pre_op"]), direct_concat)


def _move_root_before_prelu(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.remove(names["root"])
    model_ir.operators.insert(model_ir.operators.index(names["prelu_op"]), names["root"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.tensors[str(names["concat_perm"])].data.__setitem__(
            slice(None), np.asarray(_NCHW_TO_NHWC, dtype=np.int32)
        ),
        lambda model, names: model.tensors[str(names["root_perm"])].data.__setitem__(
            slice(None), np.asarray(_NHWC_TO_NCHW, dtype=np.int32)
        ),
        lambda model, names: names["concat_op"].options.__setitem__("axis", 1),
        lambda model, names: names["concat_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: _make_both_concat_backed(model, names),
        lambda model, names: _append_fanout(model, names, "concat_nchw"),
        lambda model, names: _append_fanout(model, names, "prelu_out"),
        lambda model, names: model.outputs.append(str(names["post_out"])),
        lambda model, names: names["fanout_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["downstream_op"].inputs.append(
            str(names["post_out"])
        ),
        lambda model, names: names["mul_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha"])], "is_variable", True
        ),
        lambda model, names: model.tensors[str(names["alpha"])].data.__setitem__(
            (0, 0, 0, 0), np.nan
        ),
        lambda model, names: setattr(
            model.tensors[str(names["direct"])], "dtype", "FLOAT16"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["direct"])], "shape", [1, 3, 2, 4]
        ),
        lambda model, names: _duplicate_source(model, names),
        lambda model, names: _move_root_before_prelu(model, names),
    ),
)
def test_sinet_shared_post_rejects_unsafe_variants_transactionally(
    mutation: Callable[[ModelIR, dict[str, object]], None],
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_shared_post_prelu_transpose_fanout_chains(model_ir)

    assert stats == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_shared_post_rejects_stale_plan_transactionally() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["add0_op"].inputs.reverse()
    graph_index.replace_operator_inputs(
        graph_index.operator_index(names["add0_op"]),
        names["add0_op"].inputs,
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_shared_post_no_index_preflight_avoids_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight should return before index construction")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.passes.sinet_shared_post_layout.ModelIRGraphIndex",
        fail_index,
    )

    assert optimize_sinet_shared_post_prelu_transpose_fanout_chains(
        model_ir,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
