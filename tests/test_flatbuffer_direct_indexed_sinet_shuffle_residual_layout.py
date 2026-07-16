from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout as shuffle_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _apply_late_plan,
    _apply_plan,
    _apply_postmul_plan,
    _resolve_candidate,
    _resolve_late_candidate,
    _resolve_postmul_candidate,
    optimize_sinet_late_residual_pre_add_mul_add_prelu_chains,
    optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains,
    optimize_sinet_shuffle_residual_transpose_chains,
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
            int(value) for value in (shape if signature is None else signature)
        ],
        data=data,
        is_variable=False,
    )


def _constant(
    *,
    channels: int,
    dtype: str,
    mode: str,
    offset: float,
) -> np.ndarray:
    shape = {
        "scalar": (),
        "raw": (1, int(channels), 1, 1),
        "nhwc": (1, 1, 1, int(channels)),
    }[str(mode)]
    size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    return np.asarray(
        np.linspace(
            0.15 + float(offset),
            0.65 + float(offset),
            num=size,
            dtype=np.float64,
        ).reshape(shape),
        dtype=_NP_DTYPES[dtype],
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    post1_after_concat: bool = False,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "a",
            "x",
            "y",
            "pre_perm_a",
            "pre_perm_x",
            "pre_perm_y",
            "a_nchw",
            "x_nchw",
            "y_nchw",
            "add0_out",
            "mul1_const",
            "mul1_out",
            "add1_const",
            "add1_out",
            "alpha1",
            "prelu1_out",
            "post1_perm",
            "post1_out",
            "concat2_out",
            "mul2_const",
            "mul2_out",
            "add2_const",
            "add2_out",
            "alpha2",
            "prelu2_out",
            "post2_perm",
            "post2_out",
            "side1",
            "z",
        )
    }
    a_shape = (1, 2, 3, 2)
    a_signature = (-1, 2, -1, 2)
    xy_shape = (1, 2, 3, 4)
    xy_signature = (-1, 2, -1, 4)
    a_nchw_shape = tuple(a_shape[index] for index in _PRE_PERM)
    a_nchw_signature = tuple(a_signature[index] for index in _PRE_PERM)
    xy_nchw_shape = tuple(xy_shape[index] for index in _PRE_PERM)
    xy_nchw_signature = tuple(xy_signature[index] for index in _PRE_PERM)
    concat_nchw_shape = (1, 6, 2, 3)
    concat_nchw_signature = (-1, 6, 2, -1)
    concat_nhwc_shape = (1, 2, 3, 6)
    concat_nhwc_signature = (-1, 2, -1, 6)

    model_ir.inputs.extend([str(names[key]) for key in ("a", "x", "y")])
    for key, shape, signature in (
        ("a", a_shape, a_signature),
        ("x", xy_shape, xy_signature),
        ("y", xy_shape, xy_signature),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
        )
    for key, values in (
        ("pre_perm_a", _PRE_PERM),
        ("pre_perm_x", _PRE_PERM),
        ("pre_perm_y", _PRE_PERM),
        ("post1_perm", _POST_PERM),
        ("post2_perm", _POST_PERM),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(values, dtype=np.int32),
        )
    model_ir.tensors[str(names["a_nchw"])] = _tensor(
        str(names["a_nchw"]),
        dtype=dtype,
        shape=a_nchw_shape,
        signature=a_nchw_signature,
    )
    for key in (
        "x_nchw",
        "y_nchw",
        "add0_out",
        "mul1_out",
        "add1_out",
        "prelu1_out",
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=xy_nchw_shape,
            signature=xy_nchw_signature,
        )
        model_ir.tensors[str(names[key])].logical_layout = "NCHW"
        model_ir.tensors[str(names[key])].physical_layout = "NCHW"
    model_ir.tensors[str(names["post1_out"])] = _tensor(
        str(names["post1_out"]),
        dtype=dtype,
        shape=xy_shape,
        signature=xy_signature,
    )
    model_ir.tensors[str(names["post1_out"])].logical_layout = "NHWC"
    model_ir.tensors[str(names["post1_out"])].physical_layout = "NHWC"
    for key in ("concat2_out", "mul2_out", "add2_out", "prelu2_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=concat_nchw_shape,
            signature=concat_nchw_signature,
        )
        model_ir.tensors[str(names[key])].logical_layout = "NCHW"
        model_ir.tensors[str(names[key])].physical_layout = "NCHW"
    model_ir.tensors[str(names["post2_out"])] = _tensor(
        str(names["post2_out"]),
        dtype=dtype,
        shape=concat_nhwc_shape,
        signature=concat_nhwc_signature,
    )
    model_ir.tensors[str(names["post2_out"])].logical_layout = "NHWC"
    model_ir.tensors[str(names["post2_out"])].physical_layout = "NHWC"
    for key, shape, signature in (
        ("side1", xy_shape, xy_signature),
        ("z", concat_nhwc_shape, concat_nhwc_signature),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
        )
        model_ir.outputs.append(str(names[key]))

    for index, (key, channels) in enumerate(
        (
            ("mul1_const", 4),
            ("add1_const", 4),
            ("alpha1", 4),
            ("mul2_const", 6),
            ("add2_const", 6),
            ("alpha2", 6),
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

    pre_a = OperatorIR(
        "TRANSPOSE",
        [str(names["a"]), str(names["pre_perm_a"])],
        [str(names["a_nchw"])],
    )
    pre_x = OperatorIR(
        "TRANSPOSE",
        [str(names["x"]), str(names["pre_perm_x"])],
        [str(names["x_nchw"])],
    )
    pre_y = OperatorIR(
        "TRANSPOSE",
        [str(names["y"]), str(names["pre_perm_y"])],
        [str(names["y_nchw"])],
    )
    add0_inputs = [str(names["x_nchw"]), str(names["y_nchw"])]
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
    post1 = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu1_out"]), str(names["post1_perm"])],
        [str(names["post1_out"])],
    )
    concat_inputs = [str(names["a_nchw"]), str(names["prelu1_out"])]
    if reversed_inputs:
        concat_inputs.reverse()
    concat2 = OperatorIR(
        "CONCATENATION",
        concat_inputs,
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
    post2 = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu2_out"]), str(names["post2_perm"])],
        [str(names["post2_out"])],
    )
    side1 = OperatorIR("IDENTITY", [str(names["post1_out"])], [str(names["side1"])])
    final = OperatorIR("IDENTITY", [str(names["post2_out"])], [str(names["z"])])
    middle = [post1, concat2] if not post1_after_concat else [concat2, post1]
    operators = [
        pre_a,
        pre_x,
        pre_y,
        add0,
        mul1,
        add1,
        prelu1,
        *middle,
        mul2,
        add2,
        prelu2,
        post2,
        side1,
        final,
    ]
    model_ir.operators.extend(operators)
    names.update(
        {
            "root": post2,
            "pre_a_op": pre_a,
            "pre_x_op": pre_x,
            "pre_y_op": pre_y,
            "add0_op": add0,
            "mul1_op": mul1,
            "add1_op": add1,
            "prelu1_op": prelu1,
            "post1_op": post1,
            "concat2_op": concat2,
            "mul2_op": mul2,
            "add2_op": add2,
            "prelu2_op": prelu2,
            "post2_op": post2,
        }
    )
    return names


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_shuffle_residual_layout")
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
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(
                operator_inputs,
                axis=int(operator.options["axis"]),
            )
        elif operator.op_type == "IDENTITY":
            output = np.asarray(operator_inputs[0])
        elif operator.op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            output = np.asarray(operator_inputs[0])
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
@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize("post1_after_concat", (False, True))
def test_sinet_shuffle_residual_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    post1_after_concat: bool,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        post1_after_concat=post1_after_concat,
    )
    original = copy.deepcopy(model_ir)
    input_values = {
        str(names[key]): np.asarray(
            np.linspace(
                -0.8 + index * 0.1,
                0.9 + index * 0.1,
                num=int(np.prod(model_ir.tensors[str(names[key])].shape)),
                dtype=np.float64,
            ).reshape(model_ir.tensors[str(names[key])].shape),
            dtype=_NP_DTYPES[dtype],
        )
        for index, key in enumerate(("a", "x", "y"))
    }
    expected = _evaluate(original, input_values)
    post1 = copy.deepcopy(model_ir.tensors[str(names["post1_out"])])
    post2 = copy.deepcopy(model_ir.tensors[str(names["post2_out"])])
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_sinet_shuffle_residual_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert not any(operator.op_type == "TRANSPOSE" for operator in model_ir.operators)
    assert model_ir.operators[0].inputs == [
        str(names["y"] if reversed_inputs else names["x"]),
        str(names["x"] if reversed_inputs else names["y"]),
    ]
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )
    assert int(concat.options["axis"]) == 3
    assert str(names["a"]) in concat.inputs
    assert str(names["post1_out"]) in concat.inputs
    actual = _evaluate(model_ir, input_values)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-7
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=tolerance,
            atol=tolerance,
        )
    assert model_ir.tensors[str(names["post1_out"])] == post1
    assert model_ir.tensors[str(names["post2_out"])] == post2
    for key in ("add0_out", "mul1_out", "add1_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == post1.shape
        assert tensor.shape_signature == post1.shape_signature
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    for key in ("concat2_out", "mul2_out", "add2_out"):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == post2.shape
        assert tensor.shape_signature == post2.shape_signature
        assert tensor.logical_layout == "NHWC"
        assert tensor.physical_layout == "NHWC"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_sinet_shuffle_residual_preserves_legacy_nhwc_constants(
    reversed_inputs: bool,
) -> None:
    model_ir, names = _model(
        constant_mode="nhwc",
        reversed_inputs=reversed_inputs,
    )
    originals = {
        key: np.asarray(model_ir.tensors[str(names[key])].data).copy()
        for key in (
            "mul1_const",
            "add1_const",
            "alpha1",
            "mul2_const",
            "add2_const",
            "alpha2",
        )
    }

    first = optimize_sinet_shuffle_residual_transpose_chains(model_ir)
    second = optimize_sinet_shuffle_residual_transpose_chains(model_ir)

    assert first == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert second == {"optimized_sinet_shuffle_residual_transpose_chains": 0}
    for key, original in originals.items():
        np.testing.assert_array_equal(
            model_ir.tensors[str(names[key])].data,
            original,
        )


def test_sinet_shuffle_residual_groups_constants_and_clones_external_use() -> None:
    model_ir, names = _model()
    shared_name = str(names["mul1_const"])
    shared_tensor = model_ir.tensors[shared_name]
    shared_tensor.data = np.full((1, 4, 1, 1), 0.25, dtype=np.float32)
    for key in ("add1_const", "alpha1"):
        old_name = str(names[key])
        operator_key = "add1_op" if key == "add1_const" else "prelu1_op"
        operator = names[operator_key]
        operator.inputs = [
            shared_name if str(name) == old_name else str(name)
            for name in operator.inputs
        ]
        model_ir.tensors.pop(old_name)
    original = np.asarray(shared_tensor.data).copy()
    side_name = "constant_side"
    model_ir.tensors[side_name] = _tensor(
        side_name,
        dtype="FLOAT32",
        shape=(1, 4, 1, 1),
    )
    model_ir.outputs.append(side_name)
    model_ir.operators.append(OperatorIR("IDENTITY", [shared_name], [side_name]))

    stats = optimize_sinet_shuffle_residual_transpose_chains(model_ir)

    assert stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    np.testing.assert_array_equal(model_ir.tensors[shared_name].data, original)
    clone_name = f"{shared_name}_nhwc"
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]
    for key in ("mul1_op", "add1_op", "prelu1_op"):
        assert clone_name in names[key].inputs


def test_sinet_shuffle_residual_groups_one_scalar_across_both_stages() -> None:
    model_ir, names = _model(constant_mode="scalar")
    shared_name = str(names["mul1_const"])
    model_ir.tensors[shared_name].data = np.asarray(0.25, dtype=np.float32)
    for key, operator_key in (
        ("add1_const", "add1_op"),
        ("alpha1", "prelu1_op"),
        ("mul2_const", "mul2_op"),
        ("add2_const", "add2_op"),
        ("alpha2", "prelu2_op"),
    ):
        old_name = str(names[key])
        operator = names[operator_key]
        operator.inputs = [
            shared_name if str(name) == old_name else str(name)
            for name in operator.inputs
        ]
        model_ir.tensors.pop(old_name)

    stats = optimize_sinet_shuffle_residual_transpose_chains(model_ir)

    assert stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert not any(name.startswith(f"{shared_name}_nhwc") for name in model_ir.tensors)
    for key in (
        "mul1_op",
        "add1_op",
        "prelu1_op",
        "mul2_op",
        "add2_op",
        "prelu2_op",
    ):
        assert shared_name in names[key].inputs


def test_sinet_shuffle_residual_preserves_repeated_downstream_slots() -> None:
    model_ir, names = _model()
    side1 = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [str(names["side1"])]
    )
    final = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [str(names["z"])]
    )
    side1.op_type = "ADD"
    side1.inputs = [str(names["post1_out"]), str(names["post1_out"])]
    final.op_type = "ADD"
    final.inputs = [str(names["post2_out"]), str(names["post2_out"])]
    graph_index = ModelIRGraphIndex(model_ir)

    stats = optimize_sinet_shuffle_residual_transpose_chains(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert side1.inputs == [str(names["post1_out"])] * 2
    assert final.inputs == [str(names["post2_out"])] * 2
    _assert_index_current(model_ir, graph_index)


def test_sinet_shuffle_residual_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_shuffle")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    third = _add_chain(model_ir, prefix="third_")

    candidate_stats = optimize_sinet_shuffle_residual_transpose_chains(
        model_ir,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_shuffle_residual_transpose_chains(
        model_ir,
        max_rewrites=1,
    )

    assert candidate_stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert capped_stats == {"optimized_sinet_shuffle_residual_transpose_chains": 1}
    assert ModelIRGraphIndex(model_ir).operator_index(third["root"]) is not None
    assert ModelIRGraphIndex(model_ir).operator_index(first["root"]) is None
    assert ModelIRGraphIndex(model_ir).operator_index(second["root"]) is None


def _mutate_fused(model: ModelIR, names: dict[str, object], key: str) -> None:
    names[key].options["fusedActivationFunction"] = "RELU"


def _mutate_public(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.outputs.append(str(names[key]))


def _mutate_quantized(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].quantization = {
        "scale": [0.1],
        "zero_point": [0],
    }


def _mutate_fanout(model: ModelIR, names: dict[str, object], key: str) -> None:
    source_name = str(names[key])
    source = model.tensors[source_name]
    output_name = f"{key}_fanout"
    model.tensors[output_name] = _tensor(
        output_name,
        dtype=source.dtype,
        shape=tuple(source.shape),
        signature=tuple(source.shape_signature or source.shape),
    )
    model.operators.append(OperatorIR("IDENTITY", [source_name], [output_name]))


def _mutate_wrong_order(model: ModelIR, names: dict[str, object]) -> None:
    del names
    model.operators[2], model.operators[3] = model.operators[3], model.operators[2]


def _mutate_nonfinite(model: ModelIR, names: dict[str, object], key: str) -> None:
    tensor = model.tensors[str(names[key])]
    tensor.data = np.asarray(tensor.data).copy()
    tensor.data.reshape(-1)[0] = np.inf


def _mutate_wrong_perm(model: ModelIR, names: dict[str, object], key: str) -> None:
    model.tensors[str(names[key])].data = np.asarray(
        [0, 1, 2, 3],
        dtype=np.int32,
    )


def _mutate_downstream_order(model: ModelIR, names: dict[str, object]) -> None:
    side = next(
        operator
        for operator in model.operators
        if operator.outputs == [str(names["side1"])]
    )
    model.operators.remove(side)
    model.operators.insert(model.operators.index(names["post1_op"]), side)


_Mutation = Callable[[ModelIR, dict[str, object]], None]


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: _mutate_fused(model, names, "add0_op"),
        lambda model, names: _mutate_fused(model, names, "mul1_op"),
        lambda model, names: _mutate_fused(model, names, "add1_op"),
        lambda model, names: _mutate_fused(model, names, "concat2_op"),
        lambda model, names: _mutate_fused(model, names, "mul2_op"),
        lambda model, names: _mutate_fused(model, names, "add2_op"),
        lambda model, names: _mutate_fused(model, names, "prelu1_op"),
        lambda model, names: _mutate_fused(model, names, "prelu2_op"),
        lambda model, names: _mutate_public(model, names, "a_nchw"),
        lambda model, names: _mutate_public(model, names, "add0_out"),
        lambda model, names: _mutate_public(model, names, "prelu1_out"),
        lambda model, names: _mutate_public(model, names, "post1_out"),
        lambda model, names: _mutate_public(model, names, "concat2_out"),
        lambda model, names: _mutate_public(model, names, "prelu2_out"),
        lambda model, names: _mutate_public(model, names, "post2_out"),
        lambda model, names: _mutate_public(model, names, "mul1_const"),
        lambda model, names: _mutate_quantized(model, names, "x"),
        lambda model, names: _mutate_quantized(model, names, "mul1_out"),
        lambda model, names: _mutate_quantized(model, names, "post2_out"),
        lambda model, names: _mutate_quantized(model, names, "concat2_out"),
        lambda model, names: _mutate_quantized(model, names, "alpha1"),
        lambda model, names: setattr(
            model.tensors[str(names["mul2_out"])],
            "dtype",
            "FLOAT16",
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mul1_const"])],
            "data",
            np.asarray(
                model.tensors[str(names["mul1_const"])].data,
                dtype=np.float16,
            ),
        ),
        lambda model, names: _mutate_nonfinite(model, names, "mul1_const"),
        lambda model, names: _mutate_fanout(model, names, "x_nchw"),
        lambda model, names: _mutate_fanout(model, names, "add0_out"),
        lambda model, names: _mutate_fanout(model, names, "mul1_out"),
        lambda model, names: _mutate_fanout(model, names, "add1_out"),
        lambda model, names: _mutate_fanout(model, names, "prelu1_out"),
        lambda model, names: _mutate_fanout(model, names, "a_nchw"),
        lambda model, names: _mutate_fanout(model, names, "concat2_out"),
        lambda model, names: _mutate_fanout(model, names, "mul2_out"),
        lambda model, names: _mutate_fanout(model, names, "add2_out"),
        lambda model, names: _mutate_fanout(model, names, "prelu2_out"),
        lambda model, names: _mutate_wrong_perm(model, names, "pre_perm_a"),
        lambda model, names: _mutate_wrong_perm(model, names, "pre_perm_x"),
        lambda model, names: _mutate_wrong_perm(model, names, "pre_perm_y"),
        lambda model, names: _mutate_wrong_perm(model, names, "post1_perm"),
        lambda model, names: _mutate_wrong_perm(model, names, "post2_perm"),
        lambda model, names: model.inputs.remove(str(names["x"])),
        lambda model, names: model.inputs.remove(str(names["a"])),
        lambda model, names: model.inputs.remove(str(names["y"])),
        lambda model, names: setattr(
            model.tensors[str(names["x"])],
            "data",
            np.ones((1, 2, 3, 4), dtype=np.float32),
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mul1_const"])],
            "is_variable",
            True,
        ),
        lambda model, names: model.operators.append(
            OperatorIR(
                "IDENTITY",
                [str(names["add1_const"])],
                [str(names["mul1_const"])],
            )
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mul1_const"])],
            "shape_signature",
            [1, 4, -1, 1],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha2"])],
            "shape",
            [1, 1, 6, 1],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["x_nchw"])],
            "shape_signature",
            [-1, 4, -1, 2],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["post1_out"])],
            "shape",
            [1, 3, 2, 4],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["concat2_out"])],
            "shape_signature",
            [-1, 6, -1, 2],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["post2_out"])],
            "shape",
            [1, 3, 2, 6],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha1"])],
            "data",
            None,
        ),
        lambda model, names: names["concat2_op"].options.update(axis=2),
        lambda model, names: names["concat2_op"].inputs.append(str(names["a_nchw"])),
        lambda model, names: names["prelu1_op"].inputs.append(str(names["alpha1"])),
        lambda model, names: model.operators.append(
            OperatorIR("IDENTITY", [str(names["a"])], [str(names["add0_out"])])
        ),
        _mutate_downstream_order,
        _mutate_wrong_order,
    ),
)
def test_sinet_shuffle_residual_rejects_unsafe_contracts_transactionally(
    mutation: _Mutation,
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_shuffle_residual_transpose_chains(model_ir)

    assert stats == {"optimized_sinet_shuffle_residual_transpose_chains": 0}
    assert _fingerprint(model_ir) == before


def test_sinet_shuffle_residual_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _model()
    _mutate_fanout(model_ir, names, "mul1_const")
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


def test_sinet_shuffle_residual_preflight_avoids_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("no_sinet_shuffle_residual")
    model_ir.operators = [OperatorIR("TRANSPOSE", ["x", "p"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(shuffle_module, "ModelIRGraphIndex", fail_index)

    assert optimize_sinet_shuffle_residual_transpose_chains(model_ir) == {
        "optimized_sinet_shuffle_residual_transpose_chains": 0
    }


def _postmul_model(
    *,
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    post1_after_concat: bool = False,
) -> tuple[ModelIR, dict[str, object]]:
    model_ir, names = _model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        post1_after_concat=post1_after_concat,
    )
    post2 = names["post2_op"]
    mul2 = names["mul2_op"]
    add2 = names["add2_op"]
    model_ir.operators.remove(post2)
    model_ir.operators.insert(model_ir.operators.index(mul2) + 1, post2)
    post2.inputs[0] = str(names["mul2_out"])
    add2.inputs = [
        str(names["post2_out"])
        if str(name) == str(names["mul2_out"])
        else str(name)
        for name in add2.inputs
    ]
    post2_tensor = model_ir.tensors[str(names["post2_out"])]
    for key in ("add2_out", "prelu2_out"):
        tensor = model_ir.tensors[str(names[key])]
        tensor.shape = list(post2_tensor.shape)
        tensor.shape_signature = list(post2_tensor.shape_signature or post2_tensor.shape)
        tensor.logical_layout = "NHWC"
        tensor.physical_layout = "NHWC"
    if constant_mode == "raw":
        for key in ("add2_const", "alpha2"):
            tensor = model_ir.tensors[str(names[key])]
            tensor.data = np.transpose(
                np.asarray(tensor.data),
                _POST_PERM,
            ).astype(_NP_DTYPES[dtype], copy=False)
            tensor.shape = list(tensor.data.shape)
            tensor.shape_signature = list(tensor.data.shape)
    final = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [str(names["z"])]
    )
    final.inputs = [str(names["prelu2_out"])]
    names["root"] = post2
    return model_ir, names


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32", "FLOAT64"))
@pytest.mark.parametrize("constant_mode", ("scalar", "raw"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize("post1_after_concat", (False, True))
def test_sinet_shuffle_postmul_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    post1_after_concat: bool,
) -> None:
    model_ir, names = _postmul_model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        post1_after_concat=post1_after_concat,
    )
    original = copy.deepcopy(model_ir)
    input_values = {
        str(names[key]): np.asarray(
            np.linspace(
                -0.75 + index * 0.1,
                0.95 + index * 0.1,
                num=int(np.prod(model_ir.tensors[str(names[key])].shape)),
                dtype=np.float64,
            ).reshape(model_ir.tensors[str(names[key])].shape),
            dtype=_NP_DTYPES[dtype],
        )
        for index, key in enumerate(("a", "x", "y"))
    }
    expected = _evaluate(original, input_values)
    post1 = copy.deepcopy(model_ir.tensors[str(names["post1_out"])])
    post2 = copy.deepcopy(model_ir.tensors[str(names["post2_out"])])
    add2 = copy.deepcopy(model_ir.tensors[str(names["add2_out"])])
    prelu2 = copy.deepcopy(model_ir.tensors[str(names["prelu2_out"])])
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 1
    }
    assert not any(operator.op_type == "TRANSPOSE" for operator in model_ir.operators)
    assert names["mul2_op"].outputs == [str(names["post2_out"])]
    assert str(names["post2_out"]) in names["add2_op"].inputs
    assert names["prelu2_op"].outputs == [str(names["prelu2_out"])]
    concat = names["concat2_op"]
    assert int(concat.options["axis"]) == 3
    assert str(names["a"]) in concat.inputs
    assert str(names["post1_out"]) in concat.inputs
    actual = _evaluate(model_ir, input_values)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-7
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=tolerance,
            atol=tolerance,
        )
    assert model_ir.tensors[str(names["post1_out"])] == post1
    assert model_ir.tensors[str(names["post2_out"])] == post2
    assert model_ir.tensors[str(names["add2_out"])] == add2
    assert model_ir.tensors[str(names["prelu2_out"])] == prelu2
    for key in ("add0_out", "mul1_out", "add1_out"):
        assert model_ir.tensors[str(names[key])].shape == post1.shape
    assert model_ir.tensors[str(names["concat2_out"])].shape == post2.shape
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_sinet_shuffle_postmul_preserves_legacy_raw_nhwc_tail_constants(
    reversed_inputs: bool,
) -> None:
    model_ir, names = _postmul_model(
        constant_mode="raw",
        reversed_inputs=reversed_inputs,
    )
    for key in ("add2_const", "alpha2"):
        tensor = model_ir.tensors[str(names[key])]
        tensor.data = np.transpose(np.asarray(tensor.data), _PRE_PERM)
        tensor.shape = list(tensor.data.shape)
        tensor.shape_signature = list(tensor.data.shape)

    first = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(model_ir)
    second = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(model_ir)

    assert first == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 1
    }
    assert second == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 0
    }


def test_sinet_shuffle_postmul_clones_shared_mul_constant_once() -> None:
    model_ir, names = _postmul_model()
    constant_name = str(names["mul2_const"])
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    _mutate_fanout(model_ir, names, "mul2_const")

    stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(model_ir)

    assert stats == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 1
    }
    np.testing.assert_array_equal(model_ir.tensors[constant_name].data, original)
    clone_name = f"{constant_name}_nhwc"
    assert clone_name in names["mul2_op"].inputs
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 6]


def test_sinet_shuffle_postmul_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_shuffle_postmul")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_")
    for names in (first, second):
        post2 = names["post2_op"]
        mul2 = names["mul2_op"]
        add2 = names["add2_op"]
        model_ir.operators.remove(post2)
        model_ir.operators.insert(model_ir.operators.index(mul2) + 1, post2)
        post2.inputs[0] = str(names["mul2_out"])
        add2.inputs = [
            str(names["post2_out"])
            if str(name) == str(names["mul2_out"])
            else str(name)
            for name in add2.inputs
        ]
        post2_tensor = model_ir.tensors[str(names["post2_out"])]
        for key in ("add2_out", "prelu2_out"):
            tensor = model_ir.tensors[str(names[key])]
            tensor.shape = list(post2_tensor.shape)
            tensor.shape_signature = list(post2_tensor.shape_signature or post2_tensor.shape)
        for key in ("add2_const", "alpha2"):
            tensor = model_ir.tensors[str(names[key])]
            tensor.data = np.transpose(np.asarray(tensor.data), _POST_PERM)
            tensor.shape = list(tensor.data.shape)
            tensor.shape_signature = list(tensor.data.shape)
        final = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [str(names["z"])]
        )
        final.inputs = [str(names["prelu2_out"])]
        names["root"] = post2

    candidate_stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
        model_ir,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
        model_ir,
        max_rewrites=1,
    )

    assert candidate_stats == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 1
    }
    assert capped_stats == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 1
    }
    assert ModelIRGraphIndex(model_ir).operator_index(first["root"]) is None
    assert ModelIRGraphIndex(model_ir).operator_index(second["root"]) is None


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: _mutate_fused(model, names, "mul2_op"),
        lambda model, names: _mutate_fused(model, names, "add2_op"),
        lambda model, names: _mutate_fused(model, names, "prelu2_op"),
        lambda model, names: _mutate_public(model, names, "mul2_out"),
        lambda model, names: _mutate_public(model, names, "post2_out"),
        lambda model, names: _mutate_public(model, names, "add2_out"),
        lambda model, names: _mutate_public(model, names, "mul2_const"),
        lambda model, names: _mutate_quantized(model, names, "mul2_out"),
        lambda model, names: _mutate_quantized(model, names, "post2_out"),
        lambda model, names: _mutate_quantized(model, names, "add2_out"),
        lambda model, names: _mutate_quantized(model, names, "prelu2_out"),
        lambda model, names: _mutate_quantized(model, names, "add2_const"),
        lambda model, names: setattr(
            model.tensors[str(names["prelu2_out"])],
            "dtype",
            "FLOAT16",
        ),
        lambda model, names: _mutate_nonfinite(model, names, "mul2_const"),
        lambda model, names: _mutate_nonfinite(model, names, "add2_const"),
        lambda model, names: _mutate_fanout(model, names, "mul2_out"),
        lambda model, names: _mutate_fanout(model, names, "post2_out"),
        lambda model, names: _mutate_fanout(model, names, "add2_out"),
        lambda model, names: _mutate_wrong_perm(model, names, "post2_perm"),
        lambda model, names: setattr(
            model.tensors[str(names["mul2_const"])],
            "is_variable",
            True,
        ),
        lambda model, names: setattr(
            model.tensors[str(names["alpha2"])],
            "data",
            None,
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mul2_out"])],
            "shape_signature",
            [-1, 6, -1, 2],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["post2_out"])],
            "shape",
            [1, 3, 2, 6],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["add2_out"])],
            "shape_signature",
            [-1, 3, 2, 6],
        ),
        lambda model, names: names["prelu2_op"].inputs.append(
            str(names["alpha2"])
        ),
        lambda model, names: model.operators.append(
            OperatorIR(
                "IDENTITY",
                [str(names["concat2_out"])],
                [str(names["mul2_out"])],
            )
        ),
        lambda model, names: model.operators.insert(
            model.operators.index(names["post2_op"]),
            model.operators.pop(model.operators.index(names["add2_op"])),
        ),
    ),
)
def test_sinet_shuffle_postmul_rejects_unsafe_tail_transactionally(
    mutation: _Mutation,
) -> None:
    model_ir, names = _postmul_model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(model_ir)

    assert stats == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_sinet_shuffle_postmul_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _postmul_model()
    _mutate_fanout(model_ir, names, "mul2_const")
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_postmul_candidate(model_ir, graph_index, names["root"])
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

    assert not _apply_postmul_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_shuffle_postmul_preflight_avoids_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("no_sinet_shuffle_postmul")
    model_ir.operators = [OperatorIR("TRANSPOSE", ["x", "p"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(shuffle_module, "ModelIRGraphIndex", fail_index)

    assert optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
        model_ir
    ) == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 0
    }


def _add_late_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "late_",
    dtype: str = "FLOAT32",
    constant_mode: str = "raw",
    reversed_inputs: bool = False,
    downstream_type: str = "CONV_2D",
    legacy_before_root: bool = False,
    shared_constants: bool = False,
    external_constant_use: bool = False,
    shared_post_perm: bool = False,
    spatial_width: int = 3,
) -> dict[str, object]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "x0",
            "x1",
            "x",
            "y",
            "pre_x_perm",
            "pre_y_perm",
            "x_nchw",
            "y_nchw",
            "add0_out",
            "mul1_const",
            "mul1_out",
            "add1_const",
            "add1_out",
            "alpha1",
            "prelu1_out",
            "post1_perm",
            "post1_out",
            "conv_out",
            "legacy_out",
            "constant_side_out",
            "spare_nchw",
            "spare_nhwc",
        )
    }
    dtype_np = _NP_DTYPES[dtype]
    half_shape = (1, 2, int(spatial_width), 2)
    half_signature = (-1, 2, -1, 2)
    nhwc_shape = (1, 2, int(spatial_width), 4)
    nhwc_signature = (-1, 2, -1, 4)
    nchw_shape = tuple(nhwc_shape[index] for index in _PRE_PERM)
    nchw_signature = tuple(nhwc_signature[index] for index in _PRE_PERM)
    model_ir.inputs.extend(
        [str(names[key]) for key in ("x0", "x1", "y")]
    )
    for key, shape, signature in (
        ("x0", half_shape, half_signature),
        ("x1", half_shape, half_signature),
        ("x", nhwc_shape, nhwc_signature),
        ("y", nhwc_shape, nhwc_signature),
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=shape,
            signature=signature,
        )
    for key in ("pre_x_perm", "pre_y_perm"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype="INT32",
            shape=(4,),
            data=np.asarray(_PRE_PERM, dtype=np.int32),
        )
    model_ir.tensors[str(names["post1_perm"])] = _tensor(
        str(names["post1_perm"]),
        dtype="INT32",
        shape=(4,),
        data=np.asarray(_POST_PERM, dtype=np.int32),
    )
    for key in (
        "x_nchw",
        "y_nchw",
        "add0_out",
        "mul1_out",
        "add1_out",
        "prelu1_out",
        "legacy_out",
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
        )
        model_ir.tensors[str(names[key])].logical_layout = "NCHW"
        model_ir.tensors[str(names[key])].physical_layout = "NCHW"
    for key in ("post1_out", "conv_out"):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=nhwc_shape,
            signature=nhwc_signature,
        )
        model_ir.tensors[str(names[key])].logical_layout = "NHWC"
        model_ir.tensors[str(names[key])].physical_layout = "NHWC"

    constant_keys = ("mul1_const", "add1_const", "alpha1")
    if shared_constants:
        shared_name = str(names["mul1_const"])
        for key in constant_keys:
            names[key] = shared_name
        constant_keys = ("mul1_const",)
    for index, key in enumerate(constant_keys):
        data = _constant(
            channels=4,
            dtype=dtype,
            mode=constant_mode,
            offset=0.0 if shared_constants else float(index) * 0.05,
        )
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            dtype=dtype,
            shape=tuple(data.shape),
            data=np.asarray(data, dtype=dtype_np),
        )

    def binary(data: str, constant: str) -> list[str]:
        return [constant, data] if reversed_inputs else [data, constant]

    concat = OperatorIR(
        "CONCATENATION",
        [str(names["x0"]), str(names["x1"])],
        [str(names["x"])],
        options={"axis": 3, "fusedActivationFunction": "NONE"},
    )
    pre_x = OperatorIR(
        "TRANSPOSE",
        [str(names["x"]), str(names["pre_x_perm"])],
        [str(names["x_nchw"])],
    )
    pre_y = OperatorIR(
        "TRANSPOSE",
        [str(names["y"]), str(names["pre_y_perm"])],
        [str(names["y_nchw"])],
    )
    add0_inputs = [str(names["x_nchw"]), str(names["y_nchw"])]
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
    root = OperatorIR(
        "TRANSPOSE",
        [str(names["prelu1_out"]), str(names["post1_perm"])],
        [str(names["post1_out"])],
    )
    downstream = OperatorIR(
        str(downstream_type),
        [str(names["post1_out"])],
        [str(names["conv_out"])],
    )
    legacy = OperatorIR(
        "IDENTITY",
        [str(names["prelu1_out"])],
        [str(names["legacy_out"])],
    )
    tail = [root, downstream, legacy]
    if legacy_before_root:
        tail = [legacy, root, downstream]
    operators = [concat, pre_x, pre_y, add0, mul1, add1, prelu1, *tail]
    model_ir.outputs.extend(
        [str(names["conv_out"]), str(names["legacy_out"])]
    )

    constant_side = None
    if external_constant_use:
        constant_side_name = str(names["constant_side_out"])
        constant_tensor = model_ir.tensors[str(names["mul1_const"])]
        model_ir.tensors[constant_side_name] = _tensor(
            constant_side_name,
            dtype=dtype,
            shape=tuple(int(value) for value in constant_tensor.shape),
            data=None,
        )
        constant_side = OperatorIR(
            "IDENTITY",
            [str(names["mul1_const"])],
            [constant_side_name],
        )
        operators.append(constant_side)
        model_ir.outputs.append(constant_side_name)

    side_transpose = None
    if shared_post_perm:
        model_ir.inputs.append(str(names["spare_nchw"]))
        model_ir.outputs.append(str(names["spare_nhwc"]))
        model_ir.tensors[str(names["spare_nchw"])] = _tensor(
            str(names["spare_nchw"]),
            dtype=dtype,
            shape=nchw_shape,
            signature=nchw_signature,
        )
        model_ir.tensors[str(names["spare_nhwc"])] = _tensor(
            str(names["spare_nhwc"]),
            dtype=dtype,
            shape=nhwc_shape,
            signature=nhwc_signature,
        )
        side_transpose = OperatorIR(
            "TRANSPOSE",
            [str(names["spare_nchw"]), str(names["post1_perm"])],
            [str(names["spare_nhwc"])],
        )
        operators.append(side_transpose)

    model_ir.operators.extend(operators)
    names.update(
        {
            "concat_op": concat,
            "pre_x_op": pre_x,
            "pre_y_op": pre_y,
            "add0_op": add0,
            "mul1_op": mul1,
            "add1_op": add1,
            "prelu1_op": prelu1,
            "root": root,
            "downstream_op": downstream,
            "legacy_op": legacy,
            "constant_side_op": constant_side,
            "side_transpose_op": side_transpose,
        }
    )
    return names


def _late_model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_late_residual_layout")
    return model_ir, _add_late_chain(model_ir, **kwargs)


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32", "FLOAT64"))
@pytest.mark.parametrize("constant_mode", ("scalar", "raw", "nhwc"))
@pytest.mark.parametrize("reversed_inputs", (False, True))
@pytest.mark.parametrize(
    "downstream_type", ("CONV_2D", "DEPTHWISE_CONV_2D")
)
def test_sinet_late_residual_is_indexed_and_numerically_equivalent(
    dtype: str,
    constant_mode: str,
    reversed_inputs: bool,
    downstream_type: str,
) -> None:
    model_ir, names = _late_model(
        dtype=dtype,
        constant_mode=constant_mode,
        reversed_inputs=reversed_inputs,
        downstream_type=downstream_type,
        spatial_width=4 if constant_mode == "nhwc" else 3,
    )
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(1207)
    dtype_np = _NP_DTYPES[dtype]
    inputs = {
        str(names["x0"]): rng.normal(
            size=(1, 2, 4 if constant_mode == "nhwc" else 3, 2)
        ).astype(dtype_np),
        str(names["x1"]): rng.normal(
            size=(1, 2, 4 if constant_mode == "nhwc" else 3, 2)
        ).astype(dtype_np),
        str(names["y"]): rng.normal(
            size=(1, 2, 4 if constant_mode == "nhwc" else 3, 4)
        ).astype(dtype_np),
    }
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    assert second == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 0
    }
    assert names["pre_x_op"] not in model_ir.operators
    assert names["pre_y_op"] not in model_ir.operators
    assert names["add0_op"].inputs == [str(names["x"]), str(names["y"])] or (
        names["add0_op"].inputs == [str(names["y"]), str(names["x"])]
    )
    assert names["prelu1_op"].outputs == [str(names["post1_out"])]
    assert names["root"].inputs[0] == str(names["post1_out"])
    assert names["root"].outputs == [str(names["prelu1_out"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["root"].inputs[1])].data),
        np.asarray(_PRE_PERM, dtype=np.int32),
    )
    actual = _evaluate(model_ir, inputs)
    tolerance = 2e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=tolerance,
            atol=tolerance,
        )
    width = 4 if constant_mode == "nhwc" else 3
    assert model_ir.tensors[str(names["add0_out"])].shape == [1, 2, width, 4]
    assert model_ir.tensors[str(names["prelu1_out"])].shape == [1, 4, 2, width]
    assert model_ir.tensors[str(names["post1_out"])].shape == [1, 2, width, 4]
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_late_residual_groups_constants_and_clones_external_use() -> None:
    model_ir, names = _late_model(
        shared_constants=True,
        external_constant_use=True,
    )
    original_data = np.asarray(
        model_ir.tensors[str(names["mul1_const"])].data
    ).copy()

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    affine_inputs = {
        str(names["mul1_op"].inputs[0]),
        str(names["mul1_op"].inputs[1]),
        str(names["add1_op"].inputs[0]),
        str(names["add1_op"].inputs[1]),
        str(names["prelu1_op"].inputs[1]),
    }
    clone_names = {
        name for name in affine_inputs if str(name).endswith("_nhwc")
    }
    assert len(clone_names) == 1
    clone_name = next(iter(clone_names))
    assert names["constant_side_op"].inputs == [str(names["mul1_const"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["mul1_const"])].data),
        original_data,
    )
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]


def test_sinet_late_residual_clones_shared_post_permutation() -> None:
    model_ir, names = _late_model(shared_post_perm=True)
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(444)
    inputs = {
        str(names["x0"]): rng.normal(size=(1, 2, 3, 2)).astype(np.float32),
        str(names["x1"]): rng.normal(size=(1, 2, 3, 2)).astype(np.float32),
        str(names["y"]): rng.normal(size=(1, 2, 3, 4)).astype(np.float32),
        str(names["spare_nchw"]): rng.normal(size=(1, 4, 2, 3)).astype(
            np.float32
        ),
    }
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    assert names["side_transpose_op"].inputs[1] == str(names["post1_perm"])
    assert names["root"].inputs[1] != str(names["post1_perm"])
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["post1_perm"])].data),
        np.asarray(_POST_PERM, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["root"].inputs[1])].data),
        np.asarray(_PRE_PERM, dtype=np.int32),
    )
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_late_residual_preserves_public_repeated_legacy_slots() -> None:
    model_ir, names = _late_model()
    names["legacy_op"].op_type = "ADD"
    names["legacy_op"].inputs = [
        str(names["prelu1_out"]),
        str(names["prelu1_out"]),
    ]
    model_ir.outputs.append(str(names["prelu1_out"]))
    original = copy.deepcopy(model_ir)
    rng = np.random.default_rng(811)
    inputs = {
        str(names["x0"]): rng.normal(size=(1, 2, 3, 2)).astype(np.float32),
        str(names["x1"]): rng.normal(size=(1, 2, 3, 2)).astype(np.float32),
        str(names["y"]): rng.normal(size=(1, 2, 3, 4)).astype(np.float32),
    }
    expected = _evaluate(original, inputs)

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    assert names["legacy_op"].inputs == [
        str(names["prelu1_out"]),
        str(names["prelu1_out"]),
    ]
    actual = _evaluate(model_ir, inputs)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[str(output_name)],
            expected[str(output_name)],
            rtol=1e-6,
            atol=1e-6,
        )


def test_sinet_late_residual_rejects_nonbroadcasting_oriented_constant() -> None:
    model_ir, _ = _late_model(constant_mode="nhwc", spatial_width=3)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_sinet_late_residual_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("two_late_residuals")
    first = _add_late_chain(model_ir, prefix="first_")
    second = _add_late_chain(model_ir, prefix="second_")
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
        model_ir,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert candidate_stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    assert capped_stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 1
    }
    assert first["pre_x_op"] not in model_ir.operators
    assert second["pre_x_op"] not in model_ir.operators
    _assert_index_current(model_ir, graph_index)


def _append_late_fanout(
    model_ir: ModelIR,
    names: dict[str, object],
    tensor_key: str,
) -> None:
    source_name = str(names[tensor_key])
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


def _remove_late_legacy(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.remove(names["legacy_op"])
    model_ir.outputs.remove(str(names["legacy_out"]))


def _duplicate_late_add_output(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [str(names["y"])],
            [str(names["add0_out"])],
        )
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.outputs.append(str(names["post1_out"])),
        lambda model, names: setattr(
            model.tensors[str(names["post1_perm"])],
            "data",
            np.asarray([0, 1, 3, 2], dtype=np.int32),
        ),
        lambda model, names: _append_late_fanout(
            model, names, "post1_out"
        ),
        _remove_late_legacy,
        lambda model, names: names["concat_op"].options.update({"axis": 1}),
        lambda model, names: _append_late_fanout(model, names, "x_nchw"),
        lambda model, names: setattr(
            model.tensors[str(names["y"])], "shape", [1, 2, 4, 4]
        ),
        lambda model, names: names["add1_op"].options.update(
            {"fusedActivationFunction": "RELU"}
        ),
        lambda model, names: setattr(
            model.tensors[str(names["prelu1_out"])],
            "quantization",
            object(),
        ),
        lambda model, names: setattr(
            model.tensors[str(names["mul1_const"])],
            "data",
            np.full((1, 4, 1, 1), np.nan, dtype=np.float32),
        ),
        lambda model, names: setattr(
            names["downstream_op"], "op_type", "IDENTITY"
        ),
        lambda model, names: model.outputs.append(str(names["add0_out"])),
        lambda model, names: setattr(
            model.tensors[str(names["post1_out"])],
            "shape_signature",
            [-1, 2, -1, 5],
        ),
        _duplicate_late_add_output,
        lambda model, names: model.operators.insert(
            model.operators.index(names["root"]),
            model.operators.pop(model.operators.index(names["downstream_op"])),
        ),
    ),
)
def test_sinet_late_residual_rejects_unsafe_contracts_transactionally(
    mutation: _Mutation,
) -> None:
    model_ir, names = _late_model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_sinet_late_residual_rejects_legacy_consumer_before_adapter() -> None:
    model_ir, _ = _late_model(legacy_before_root=True)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(model_ir)

    assert stats == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_sinet_late_residual_apply_revalidates_stale_plan() -> None:
    model_ir, names = _late_model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_late_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    model_ir.tensors[str(names["post1_out"])].shape = [1, 2, 3, 5]
    before = _fingerprint(model_ir)

    assert not _apply_late_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_late_residual_apply_preflight_rejects_clone_collision() -> None:
    model_ir, names = _late_model(external_constant_use=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_late_candidate(model_ir, graph_index, names["root"])
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

    assert not _apply_late_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_late_residual_preflight_avoids_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("no_sinet_late_residual")
    model_ir.operators = [OperatorIR("TRANSPOSE", ["x", "p"], ["y"])]

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("index should not be allocated")

    monkeypatch.setattr(shuffle_module, "ModelIRGraphIndex", fail_index)

    assert optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
        model_ir
    ) == {
        "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains": 0
    }
