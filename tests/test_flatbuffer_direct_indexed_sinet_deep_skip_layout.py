from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.sinet_deep_skip_layout as deep_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_deep_skip_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_deep_skip_concat_resize_affine_tail_chains,
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
    nhwc: bool = False,
) -> np.ndarray:
    if mode == "scalar":
        shape: tuple[int, ...] = ()
    elif nhwc:
        shape = (1, 1, 1, int(channels))
    else:
        shape = (1, int(channels), 1, 1)
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.linspace(
            0.2 + float(offset),
            0.7 + float(offset),
            num=size,
            dtype=np.float64,
        ).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "deep_",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    resize_type: str = "RESIZE_BILINEAR",
    direct_pre: bool = False,
    external_constant_use: bool = False,
) -> dict[str, object]:
    keys = (
        "skip",
        "skip_perm",
        "skip_nchw",
        "a",
        "a_perm",
        "a_nchw",
        "b_input",
        "b",
        "b_perm",
        "b_nchw",
        "mulb_const",
        "mulb_out",
        "addb_const",
        "addb_out",
        "concat1_out",
        "pre",
        "pre_post_perm",
        "pre_nhwc",
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
        "post_perm",
        "post_out",
        "add2_const",
        "add2_out",
        "alpha2",
        "prelu2_out",
        "final",
        "constant_side",
    )
    names: dict[str, object] = {key: f"{prefix}{key}" for key in keys}
    nhwc2 = (1, 2, 3, 2)
    nhwc2_sig = (-1, 2, -1, 2)
    nchw2 = (1, 2, 2, 3)
    nchw2_sig = (-1, 2, 2, -1)
    nhwc4 = (1, 2, 3, 4)
    nhwc4_sig = (-1, 2, -1, 4)
    nchw4 = (1, 4, 2, 3)
    nchw4_sig = (-1, 4, 2, -1)
    nhwc6 = (1, 2, 3, 6)
    nhwc6_sig = (-1, 2, -1, 6)
    nchw6 = (1, 6, 2, 3)
    nchw6_sig = (-1, 6, 2, -1)

    for key in ("skip", "a", "b_input"):
        model_ir.inputs.append(str(names[key]))
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nhwc2,
            signature=nhwc2_sig,
            layout="NHWC",
        )
    pre_shape = nhwc4 if direct_pre else nchw4
    pre_signature = nhwc4_sig if direct_pre else nchw4_sig
    pre_layout = "NHWC" if direct_pre else "NCHW"
    model_ir.inputs.append(str(names["pre"]))
    model_ir.tensors[str(names["pre"])] = _tensor(
        str(names["pre"]),
        dtype=dtype,
        shape=pre_shape,
        signature=pre_signature,
        layout=pre_layout,
    )

    for key in ("skip_perm", "a_perm", "b_perm"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(_PRE_PERM, dtype=np.int32),
        )
    for key in ("pre_post_perm", "post_perm"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(_POST_PERM, dtype=np.int32),
        )

    for key in ("b", "pre_nhwc"):
        shape = nhwc2 if key == "b" else nhwc4
        signature = nhwc2_sig if key == "b" else nhwc4_sig
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout="NHWC",
        )
    for key in ("skip_nchw", "a_nchw", "b_nchw", "mulb_out", "addb_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw2,
            signature=nchw2_sig,
            layout="NCHW",
        )
    for key in (
        "concat1_out",
        "add0_out",
        "mul1_out",
        "add1_out",
        "prelu1_out",
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw4,
            signature=nchw4_sig,
            layout="NCHW",
        )
    for key in ("concat2_out", "mul2_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw6,
            signature=nchw6_sig,
            layout="NCHW",
        )
    for key in ("post_out", "add2_out", "prelu2_out", "final"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nhwc6,
            signature=nhwc6_sig,
            layout="NHWC",
        )

    constant_specs = (
        ("mulb_const", 2, False),
        ("addb_const", 2, False),
        ("mul1_const", 4, False),
        ("add1_const", 4, False),
        ("alpha1", 4, False),
        ("mul2_const", 6, False),
        ("add2_const", 6, True),
        ("alpha2", 6, True),
    )
    for index, (key, channels, nhwc) in enumerate(constant_specs):
        data = _constant(
            channels=channels,
            dtype=dtype,
            mode=constant_mode,
            offset=float(index) * 0.03,
            nhwc=nhwc,
        )
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=tuple(data.shape),
            data=data,
            layout="NHWC" if nhwc else "NCHW",
        )

    def binary(data: str, constant: str) -> list[str]:
        return [constant, data] if reversed_inputs else [data, constant]

    resize = OperatorIR(
        resize_type,
        [str(names["b_input"])],
        [str(names["b"])],
    )
    pre_b = OperatorIR(
        "TRANSPOSE",
        [str(names["b"]), str(names["b_perm"])],
        [str(names["b_nchw"])],
    )
    mulb = OperatorIR(
        "MUL",
        binary(str(names["b_nchw"]), str(names["mulb_const"])),
        [str(names["mulb_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    addb = OperatorIR(
        "ADD",
        binary(str(names["mulb_out"]), str(names["addb_const"])),
        [str(names["addb_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    pre_a = OperatorIR(
        "TRANSPOSE",
        [str(names["a"]), str(names["a_perm"])],
        [str(names["a_nchw"])],
    )
    concat1_inputs = [str(names["a_nchw"]), str(names["addb_out"])]
    if reversed_inputs:
        concat1_inputs.reverse()
    concat1 = OperatorIR(
        "CONCATENATION",
        concat1_inputs,
        [str(names["concat1_out"])],
        options={"axis": 1, "fusedActivationFunction": "NONE"},
    )
    pre_post = None
    if not direct_pre:
        pre_post = OperatorIR(
            "TRANSPOSE",
            [str(names["pre"]), str(names["pre_post_perm"])],
            [str(names["pre_nhwc"])],
        )
    add0_inputs = [str(names["pre"]), str(names["concat1_out"])]
    if reversed_inputs:
        add0_inputs.reverse()
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
    skip_pre = OperatorIR(
        "TRANSPOSE",
        [str(names["skip"]), str(names["skip_perm"])],
        [str(names["skip_nchw"])],
    )
    concat2_inputs = [str(names["skip_nchw"]), str(names["prelu1_out"])]
    if reversed_inputs:
        concat2_inputs.reverse()
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
    root = OperatorIR(
        "TRANSPOSE",
        [str(names["mul2_out"]), str(names["post_perm"])],
        [str(names["post_out"])],
    )
    add2 = OperatorIR(
        "ADD",
        binary(str(names["post_out"]), str(names["add2_const"])),
        [str(names["add2_out"])],
        options={"fusedActivationFunction": "NONE"},
    )
    prelu2 = OperatorIR(
        "PRELU",
        [str(names["add2_out"]), str(names["alpha2"])],
        [str(names["prelu2_out"])],
    )
    final = OperatorIR(
        "IDENTITY",
        [str(names["prelu2_out"])],
        [str(names["final"])],
    )
    operators = [resize, pre_b, mulb, addb, pre_a, concat1]
    if pre_post is not None:
        operators.append(pre_post)
    operators.extend(
        [
            add0,
            mul1,
            add1,
            prelu1,
            skip_pre,
            concat2,
            mul2,
            root,
            add2,
            prelu2,
            final,
        ]
    )
    constant_side = None
    if external_constant_use:
        constant_side = OperatorIR(
            "IDENTITY",
            [str(names["mul1_const"])],
            [str(names["constant_side"])],
        )
        constant = model_ir.tensors[str(names["mul1_const"])]
        model_ir.tensors[str(names["constant_side"])] = _tensor(
            str(names["constant_side"]),
            dtype=dtype,
            shape=tuple(int(value) for value in constant.shape),
        )
        operators.append(constant_side)
        model_ir.outputs.append(str(names["constant_side"]))
    model_ir.operators.extend(operators)
    model_ir.outputs.append(str(names["final"]))
    names.update(
        {
            "root": root,
            "resize_op": resize,
            "pre_b_op": pre_b,
            "mulb_op": mulb,
            "addb_op": addb,
            "pre_a_op": pre_a,
            "concat1_op": concat1,
            "pre_post_op": pre_post,
            "add0_op": add0,
            "mul1_op": mul1,
            "add1_op": add1,
            "prelu1_op": prelu1,
            "skip_pre_op": skip_pre,
            "concat2_op": concat2,
            "mul2_op": mul2,
            "add2_op": add2,
            "prelu2_op": prelu2,
            "constant_side_op": constant_side,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_deep_skip_layout")
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
        if operator.op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}:
            output = np.asarray(args[0])
        elif operator.op_type == "TRANSPOSE":
            output = np.transpose(
                args[0],
                tuple(int(value) for value in args[1].reshape(-1)),
            )
        elif operator.op_type == "MUL":
            output = np.multiply(args[0], args[1])
        elif operator.op_type == "ADD":
            output = np.add(args[0], args[1])
        elif operator.op_type == "PRELU":
            output = np.where(args[0] >= 0, args[0], args[0] * args[1])
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(args, axis=int(operator.options["axis"]))
        elif operator.op_type == "IDENTITY":
            output = np.asarray(args[0])
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _inputs(
    names: dict[str, object],
    *,
    dtype: str,
    direct_pre: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(217)
    dtype_np = _NP_DTYPES[dtype]
    return {
        str(names["skip"]): rng.normal(size=(1, 2, 3, 2)).astype(dtype_np),
        str(names["a"]): rng.normal(size=(1, 2, 3, 2)).astype(dtype_np),
        str(names["b_input"]): rng.normal(size=(1, 2, 3, 2)).astype(dtype_np),
        str(names["pre"]): rng.normal(
            size=(1, 2, 3, 4) if direct_pre else (1, 4, 2, 3)
        ).astype(dtype_np),
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
def test_deep_skip_explicit_adapter_is_numerically_equivalent(
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
    inputs = _inputs(names, dtype=dtype, direct_pre=False)
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 1
    }
    assert second == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 0
    }
    for key in ("pre_b_op", "pre_a_op", "skip_pre_op", "root"):
        assert names[key] not in model_ir.operators
    assert names["pre_post_op"] in model_ir.operators
    assert names["concat1_op"].options["axis"] == 3
    assert names["concat2_op"].options["axis"] == 3
    assert names["add0_op"].inputs.count(str(names["pre_nhwc"])) == 1
    assert names["add2_op"].inputs.count(str(names["mul2_out"])) == 1
    actual = _evaluate(model_ir, inputs)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    np.testing.assert_allclose(
        actual[str(names["final"])],
        expected[str(names["final"])],
        rtol=tolerance,
        atol=tolerance,
    )
    assert model_ir.tensors[str(names["concat1_out"])].shape == [1, 2, 3, 4]
    assert model_ir.tensors[str(names["concat2_out"])].shape == [1, 2, 3, 6]
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_deep_skip_direct_nhwc_pre_matches_explicit_reference(
    constant_mode: str,
    reversed_inputs: bool,
) -> None:
    direct, direct_names = _model(
        direct_pre=True,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
    )
    reference, reference_names = _model(
        prefix="ref_",
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
    )
    direct_inputs = _inputs(direct_names, dtype="FLOAT32", direct_pre=True)
    reference_inputs = {
        str(reference_names["skip"]): direct_inputs[str(direct_names["skip"])],
        str(reference_names["a"]): direct_inputs[str(direct_names["a"])],
        str(reference_names["b_input"]): direct_inputs[
            str(direct_names["b_input"])
        ],
        str(reference_names["pre"]): np.transpose(
            direct_inputs[str(direct_names["pre"])],
            _PRE_PERM,
        ),
    }
    expected = _evaluate(reference, reference_inputs)[str(reference_names["final"])]

    stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(direct)

    assert stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 1
    }
    assert direct_names["add0_op"].inputs.count(str(direct_names["pre"])) == 1
    actual = _evaluate(direct, direct_inputs)[str(direct_names["final"])]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_deep_skip_clones_changed_constant_with_external_use() -> None:
    model_ir, names = _model(external_constant_use=True)
    original = np.asarray(model_ir.tensors[str(names["mul1_const"])].data).copy()

    stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(model_ir)

    assert stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 1
    }
    assert names["constant_side_op"].inputs == [str(names["mul1_const"])]
    assert names["mul1_op"].inputs.count(str(names["mul1_const"])) == 0
    clone_name = next(
        str(name)
        for name in names["mul1_op"].inputs
        if str(name).endswith("_nhwc")
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["mul1_const"])].data),
        original,
    )
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]


def test_deep_skip_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("two_deep_skip_chains")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
        model_ir,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert candidate_stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 1
    }
    assert capped_stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 1
    }
    assert first["root"] not in model_ir.operators
    assert second["root"] not in model_ir.operators
    _assert_index_current(model_ir, graph_index)


_Mutation = Callable[[ModelIR, dict[str, object]], None]


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
        shape=tuple(int(value) for value in source.shape),
        signature=tuple(int(value) for value in source.shape_signature),
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", [source_name], [output_name])
    )
    model_ir.outputs.append(output_name)


def _duplicate_output(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [str(names["skip"])],
            [str(names["add0_out"])],
        )
    )


def _erase_pre_post_layout(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    tensor = model_ir.tensors[str(names["pre_nhwc"])]
    tensor.logical_layout = "UNKNOWN"
    tensor.physical_layout = "UNKNOWN"


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.outputs.append(str(names["post_out"])),
        lambda model, names: _append_fanout(model, names, "post_out"),
        lambda model, names: model.outputs.append(str(names["add2_out"])),
        lambda model, names: setattr(names["prelu2_op"], "op_type", "RELU"),
        lambda model, names: _append_fanout(model, names, "mul2_out"),
        lambda model, names: names["concat2_op"].options.update({"axis": 3}),
        lambda model, names: _append_fanout(model, names, "skip_nchw"),
        lambda model, names: _append_fanout(model, names, "prelu1_out"),
        lambda model, names: names["mul1_op"].options.update(
            {"fusedActivationFunction": "RELU"}
        ),
        lambda model, names: _append_fanout(model, names, "concat1_out"),
        _erase_pre_post_layout,
        lambda model, names: _append_fanout(model, names, "a_nchw"),
        lambda model, names: setattr(
            names["resize_op"], "op_type", "AVERAGE_POOL_2D"
        ),
        lambda model, names: _append_fanout(model, names, "b_nchw"),
        lambda model, names: setattr(
            model.tensors[str(names["concat2_out"])],
            "shape",
            [1, 7, 2, 3],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["prelu1_out"])], "dtype", "FLOAT64"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["addb_out"])], "quantization", object()
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mulb_const"])],
            "data",
            np.full((1, 2, 1, 1), np.nan, dtype=np.float32),
        ),
        _duplicate_output,
        lambda model, names: model.operators.insert(
            model.operators.index(names["root"]),
            model.operators.pop(model.operators.index(names["add2_op"])),
        ),
    ),
)
def test_deep_skip_rejects_unsafe_contracts_transactionally(
    mutation: _Mutation,
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(model_ir)

    assert stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_deep_skip_direct_pre_requires_channel_last_contract() -> None:
    model_ir, names = _model(direct_pre=True)
    model_ir.tensors[str(names["pre"])].logical_layout = "UNKNOWN"
    model_ir.tensors[str(names["pre"])].physical_layout = "UNKNOWN"
    before = _fingerprint(model_ir)

    stats = optimize_sinet_deep_skip_concat_resize_affine_tail_chains(model_ir)

    assert stats == {
        "optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_deep_skip_apply_revalidates_stale_plan() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    model_ir.tensors[str(names["post_out"])].shape = [1, 2, 3, 7]
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_deep_skip_apply_preflight_rejects_clone_collision() -> None:
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
    model_ir.tensors[clone_name] = _tensor(
        clone_name,
        dtype="FLOAT32",
        shape=(),
        data=np.asarray(0.0, dtype=np.float32),
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_deep_skip_preflight_avoids_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("no_deep_skip")
    model_ir.operators = [OperatorIR("TRANSPOSE", ["x", "p"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(deep_module, "ModelIRGraphIndex", fail_index)

    assert optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
        model_ir
    ) == {"optimized_sinet_deep_skip_concat_resize_affine_tail_chains": 0}
