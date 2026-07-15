from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_sa_pa_mirrorpad_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains,
)


_STATS_KEY = "optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
_NP_DTYPES = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
}
_NHWC_TO_NCHW = np.asarray([0, 3, 1, 2], dtype=np.int32)
_NCHW_TO_NHWC = np.asarray([0, 2, 3, 1], dtype=np.int32)


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


def _add_constant(
    model_ir: ModelIR,
    name: str,
    data: np.ndarray,
    *,
    dtype: str = "INT32",
) -> str:
    array = np.asarray(data).copy()
    model_ir.tensors[name] = _tensor(
        name,
        dtype=dtype,
        shape=tuple(array.shape),
        data=array,
    )
    return name


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    source_mul_nchw: bool = False,
    reversed_inputs: bool = False,
    dynamic_signature: bool = False,
    external_axis_use: bool = False,
) -> dict[str, object]:
    n, h, w, c = 1, 2, 3, 1
    nhwc = (n, h, w, c)
    nhwc_sig = (-1, h, -1, c) if dynamic_signature else nhwc
    nchw = (n, c, h, w)
    nchw_sig = (-1, c, h, -1) if dynamic_signature else nchw
    sa_concat = (n, 2, h, w)
    sa_concat_sig = (-1, 2, h, -1) if dynamic_signature else sa_concat
    sa_padded = (n, 2, h + 2, w + 2)
    sa_padded_sig = (-1, 2, h + 2, -1) if dynamic_signature else sa_padded
    sa_pre = (n, h + 2, w + 2, 2)
    sa_pre_sig = (-1, h + 2, -1, 2) if dynamic_signature else sa_pre
    unsqueeze = (n, c, 1, h, w)
    unsqueeze_sig = (-1, c, 1, h, -1) if dynamic_signature else unsqueeze
    concat_pa = (n, c, 2, h, w)
    concat_pa_sig = (-1, c, 2, h, -1) if dynamic_signature else concat_pa
    reshape_pa = (n, 2 * c, h, w)
    reshape_pa_sig = (-1, 2 * c, h, -1) if dynamic_signature else reshape_pa
    pa_padded = (n, 2 * c, h + 2, w + 2)
    pa_padded_sig = (
        (-1, 2 * c, h + 2, -1) if dynamic_signature else pa_padded
    )
    pa_pre = (n, h + 2, w + 2, 2 * c)
    pa_pre_sig = (-1, h + 2, -1, 2 * c) if dynamic_signature else pa_pre

    def name(value: str) -> str:
        return f"{prefix}{value}"

    source = _add_value(
        model_ir,
        name("source"),
        dtype=dtype,
        shape=nhwc,
        signature=nhwc_sig,
        layout="NHWC",
    )
    ca = _add_value(
        model_ir,
        name("ca"),
        dtype=dtype,
        shape=nhwc,
        signature=nhwc_sig,
        layout="NHWC",
    )
    model_ir.inputs.extend((source, ca))

    constants = {
        key: _add_constant(model_ir, name(key), value)
        for key, value in (
            ("source_perm", _NHWC_TO_NCHW),
            ("ca_perm", _NHWC_TO_NCHW),
            ("sa_perm", _NCHW_TO_NHWC),
            ("pa_perm", _NCHW_TO_NHWC),
            ("gate_perm", _NHWC_TO_NCHW),
            ("axes", np.asarray([1], dtype=np.int32)),
            (
                "sa_pads",
                np.asarray(
                    [[0, 0], [0, 0], [1, 1], [1, 1]],
                    dtype=np.int32,
                ),
            ),
            (
                "pa_pads",
                np.asarray(
                    [[0, 0], [0, 0], [1, 1], [1, 1]],
                    dtype=np.int32,
                ),
            ),
            ("sa_shape", np.asarray(nchw, dtype=np.int32)),
            ("unsqueeze_shape", np.asarray(unsqueeze, dtype=np.int32)),
            ("pa_shape", np.asarray(reshape_pa, dtype=np.int32)),
        )
    }

    shapes = {
        "source_nchw": (nchw, nchw_sig, "NCHW"),
        "mean": (nchw, nchw_sig, "NCHW"),
        "maximum": (nchw, nchw_sig, "NCHW"),
        "concat_sa": (sa_concat, sa_concat_sig, "NCHW"),
        "mirror_sa": (sa_padded, sa_padded_sig, "NCHW"),
        "sa_pre": (sa_pre, sa_pre_sig, "NHWC"),
        "sa_conv": (nhwc, nhwc_sig, "NHWC"),
        "sa_nchw": (nchw, nchw_sig, "NCHW"),
        "ca_nchw": (nchw, nchw_sig, "NCHW"),
        "attention": (nchw, nchw_sig, "NCHW"),
        "unsqueeze_source": (unsqueeze, unsqueeze_sig, "UNKNOWN"),
        "unsqueeze_attention": (unsqueeze, unsqueeze_sig, "UNKNOWN"),
        "concat_pa": (concat_pa, concat_pa_sig, "UNKNOWN"),
        "reshape_pa": (reshape_pa, reshape_pa_sig, "NCHW"),
        "mirror_pa": (pa_padded, pa_padded_sig, "NCHW"),
        "pa_pre": (pa_pre, pa_pre_sig, "NHWC"),
        "pa_conv": (nhwc, nhwc_sig, "NHWC"),
        "gate_nchw": (nchw, nchw_sig, "NCHW"),
        "gate": (
            nchw if source_mul_nchw else nhwc,
            nchw_sig if source_mul_nchw else nhwc_sig,
            "NCHW" if source_mul_nchw else "NHWC",
        ),
        "output": (
            nchw if source_mul_nchw else nhwc,
            nchw_sig if source_mul_nchw else nhwc_sig,
            "NCHW" if source_mul_nchw else "NHWC",
        ),
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

    def binary(op_type: str, lhs: str, rhs: str, output: str) -> OperatorIR:
        inputs = [lhs, rhs]
        if reversed_inputs:
            inputs.reverse()
        return OperatorIR(
            op_type,
            inputs,
            [output],
            {"fusedActivationFunction": "NONE"},
        )

    root = OperatorIR(
        "TRANSPOSE",
        [source, constants["source_perm"]],
        [values["source_nchw"]],
    )
    ca_pre_op = OperatorIR(
        "TRANSPOSE",
        [ca, constants["ca_perm"]],
        [values["ca_nchw"]],
    )
    mean = OperatorIR(
        "MEAN",
        [values["source_nchw"], constants["axes"]],
        [values["mean"]],
        {"keepDims": True},
    )
    maximum = OperatorIR(
        "REDUCE_MAX",
        [values["source_nchw"], constants["axes"]],
        [values["maximum"]],
        {"keepDims": True},
    )
    unsqueeze_source = OperatorIR(
        "RESHAPE",
        [values["source_nchw"], constants["unsqueeze_shape"]],
        [values["unsqueeze_source"]],
        {"newShape": list(unsqueeze)},
    )
    concat_sa_inputs = [values["mean"], values["maximum"]]
    if reversed_inputs:
        concat_sa_inputs.reverse()
    concat_sa_op = OperatorIR(
        "CONCATENATION",
        concat_sa_inputs,
        [values["concat_sa"]],
        {"axis": 1, "fusedActivationFunction": "NONE"},
    )
    mirror_sa = OperatorIR(
        "MIRROR_PAD",
        [values["concat_sa"], constants["sa_pads"]],
        [values["mirror_sa"]],
        {"mode": "REFLECT"},
    )
    sa_pre_op = OperatorIR(
        "TRANSPOSE",
        [values["mirror_sa"], constants["sa_perm"]],
        [values["sa_pre"]],
    )
    sa_conv = OperatorIR(
        "CONV_2D",
        [values["sa_pre"]],
        [values["sa_conv"]],
        {"testKind": "sa"},
    )
    sa_reshape = OperatorIR(
        "RESHAPE",
        [values["sa_conv"], constants["sa_shape"]],
        [values["sa_nchw"]],
        {"newShape": list(nchw)},
    )
    add_attention = binary(
        "ADD",
        values["sa_nchw"],
        values["ca_nchw"],
        values["attention"],
    )
    unsqueeze_attention = OperatorIR(
        "RESHAPE",
        [values["attention"], constants["unsqueeze_shape"]],
        [values["unsqueeze_attention"]],
        {"newShape": list(unsqueeze)},
    )
    concat_pa_inputs = [
        values["unsqueeze_source"],
        values["unsqueeze_attention"],
    ]
    if reversed_inputs:
        concat_pa_inputs.reverse()
    concat_pa_op = OperatorIR(
        "CONCATENATION",
        concat_pa_inputs,
        [values["concat_pa"]],
        {"axis": 2, "fusedActivationFunction": "NONE"},
    )
    reshape_pa_op = OperatorIR(
        "RESHAPE",
        [values["concat_pa"], constants["pa_shape"]],
        [values["reshape_pa"]],
        {"newShape": list(reshape_pa)},
    )
    mirror_pa = OperatorIR(
        "MIRROR_PAD",
        [values["reshape_pa"], constants["pa_pads"]],
        [values["mirror_pa"]],
        {"mode": "REFLECT"},
    )
    pa_pre_op = OperatorIR(
        "TRANSPOSE",
        [values["mirror_pa"], constants["pa_perm"]],
        [values["pa_pre"]],
    )
    pa_conv = OperatorIR(
        "CONV_2D",
        [values["pa_pre"]],
        [values["pa_conv"]],
        {"testKind": "pa"},
    )
    gate_pre = None
    gate_input = values["pa_conv"]
    if source_mul_nchw:
        gate_pre = OperatorIR(
            "TRANSPOSE",
            [values["pa_conv"], constants["gate_perm"]],
            [values["gate_nchw"]],
        )
        gate_input = values["gate_nchw"]
    gate = OperatorIR("LOGISTIC", [gate_input], [values["gate"]])
    mul = binary(
        "MUL",
        values["gate"],
        values["source_nchw"] if source_mul_nchw else source,
        values["output"],
    )
    operators = [
        root,
        ca_pre_op,
        mean,
        maximum,
        unsqueeze_source,
        concat_sa_op,
        mirror_sa,
        sa_pre_op,
        sa_conv,
        sa_reshape,
        add_attention,
        unsqueeze_attention,
        concat_pa_op,
        reshape_pa_op,
        mirror_pa,
        pa_pre_op,
        pa_conv,
    ]
    if gate_pre is not None:
        operators.append(gate_pre)
    operators.extend((gate, mul))
    model_ir.operators.extend(operators)
    model_ir.outputs.append(values["output"])

    side = None
    if external_axis_use:
        side_name = name("axis_side")
        model_ir.tensors[side_name] = _tensor(
            side_name,
            dtype="INT32",
            shape=(1,),
        )
        side = OperatorIR("IDENTITY", [constants["axes"]], [side_name])
        model_ir.operators.append(side)
        model_ir.outputs.append(side_name)

    return {
        **constants,
        **values,
        "source": source,
        "ca": ca,
        "root": root,
        "ca_pre_op": ca_pre_op,
        "mean_op": mean,
        "maximum_op": maximum,
        "unsqueeze_source_op": unsqueeze_source,
        "concat_sa_op": concat_sa_op,
        "mirror_sa_op": mirror_sa,
        "sa_pre_op": sa_pre_op,
        "sa_reshape_op": sa_reshape,
        "add_attention_op": add_attention,
        "unsqueeze_attention_op": unsqueeze_attention,
        "concat_pa_op": concat_pa_op,
        "reshape_pa_op": reshape_pa_op,
        "mirror_pa_op": mirror_pa,
        "pa_pre_op": pa_pre_op,
        "gate_pre_op": gate_pre,
        "gate_op": gate,
        "mul_op": mul,
        "side_op": side,
    }


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_sa_pa_mirrorpad_layout")
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
        elif operator.op_type == "ADD":
            output = np.add(args[0], args[1])
        elif operator.op_type == "MUL":
            output = np.multiply(args[0], args[1])
        elif operator.op_type == "MEAN":
            output = np.mean(
                args[0],
                axis=tuple(int(value) for value in args[1].reshape(-1)),
                keepdims=bool(operator.options.get("keepDims", False)),
            )
        elif operator.op_type == "REDUCE_MAX":
            output = np.max(
                args[0],
                axis=tuple(int(value) for value in args[1].reshape(-1)),
                keepdims=bool(operator.options.get("keepDims", False)),
            )
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(args, axis=int(operator.options["axis"]))
        elif operator.op_type == "MIRROR_PAD":
            output = np.pad(
                args[0],
                tuple(tuple(int(value) for value in pair) for pair in args[1]),
                mode="reflect",
            )
        elif operator.op_type == "RESHAPE":
            output = np.reshape(
                args[0],
                tuple(int(value) for value in args[1].reshape(-1)),
            )
        elif operator.op_type == "CONV_2D":
            kind = str(operator.options["testKind"])
            cropped = args[0][:, 1:-1, 1:-1, :]
            if kind in {"sa", "pa"}:
                output = np.mean(cropped, axis=3, keepdims=True)
            else:
                raise AssertionError(kind)
        elif operator.op_type == "LOGISTIC":
            output = 1.0 / (1.0 + np.exp(-args[0]))
        elif operator.op_type == "IDENTITY":
            output = np.asarray(args[0])
        else:
            raise AssertionError(f"unsupported operator: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {str(name): np.asarray(values[str(name)]) for name in model_ir.outputs}


def _inputs(names: dict[str, object], dtype: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1177)
    return {
        str(names[key]): rng.normal(size=(1, 2, 3, 1)).astype(_NP_DTYPES[dtype])
        for key in ("source", "ca")
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
@pytest.mark.parametrize("source_mul_nchw", (False, True))
@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_sa_pa_mirrorpad_owner_is_indexed_and_numerically_equivalent(
    dtype: str,
    source_mul_nchw: bool,
    reversed_inputs: bool,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        source_mul_nchw=source_mul_nchw,
        reversed_inputs=reversed_inputs,
        dynamic_signature=True,
    )
    sample = _inputs(names, dtype)
    expected = _evaluate(copy.deepcopy(model_ir), sample)
    graph_index = ModelIRGraphIndex(model_ir)

    result = optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir,
        graph_index=graph_index,
    )

    assert result == {_STATS_KEY: 1}
    actual = _evaluate(model_ir, sample)
    assert tuple(actual) == tuple(expected)
    for output_name in expected:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=5e-3 if dtype == "FLOAT16" else 1e-6,
            atol=5e-3 if dtype == "FLOAT16" else 1e-6,
        )
    assert names["root"] not in model_ir.operators
    assert names["ca_pre_op"] not in model_ir.operators
    assert names["sa_pre_op"] not in model_ir.operators
    assert names["sa_reshape_op"] not in model_ir.operators
    assert names["pa_pre_op"] not in model_ir.operators
    assert names["concat_sa_op"].options["axis"] == 3
    assert names["concat_pa_op"].options["axis"] == 4
    assert list(np.asarray(model_ir.tensors[str(names["axes"])].data)) == [3]
    assert all(
        name in model_ir.inputs
        or name in graph_index.producers
        or model_ir.tensors[name].data is not None
        for operator in model_ir.operators
        for name in operator.inputs
    )
    _assert_index_current(model_ir, graph_index)
    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir,
        graph_index=graph_index,
    ) == {_STATS_KEY: 0}


def test_sa_pa_mirrorpad_clones_externally_used_axis_constant() -> None:
    model_ir, names = _model(external_axis_use=True)
    original = np.asarray(model_ir.tensors[str(names["axes"])].data).copy()

    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir
    ) == {_STATS_KEY: 1}

    assert np.array_equal(model_ir.tensors[str(names["axes"])].data, original)
    assert names["mean_op"].inputs[1] != names["axes"]
    assert names["maximum_op"].inputs[1] == names["mean_op"].inputs[1]
    assert names["side_op"].inputs[0] == names["axes"]


def test_sa_pa_mirrorpad_candidate_limit_and_layout_state() -> None:
    model_ir = ModelIR("two_sa_pa_chains")
    first = _add_chain(model_ir, prefix="a_")
    second = _add_chain(model_ir, prefix="b_")
    layout_state = LayoutState()
    graph_index = ModelIRGraphIndex(model_ir)

    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    assert first["root"] not in model_ir.operators
    assert second["root"] in model_ir.operators
    assert layout_state.logical_of(str(first["attention"])) == "NHWC"
    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    ) == {_STATS_KEY: 1}
    _assert_index_current(model_ir, graph_index)


Mutation = Callable[[ModelIR, dict[str, object]], None]


def _set_public_intermediate(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.outputs.append(str(names["mirror_sa"]))


def _set_public_ca_adapter_output(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.outputs.append(str(names["ca_nchw"]))


def _set_public_gate_adapter_output(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.outputs.append(str(names["gate_nchw"]))


def _unbind_source(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.inputs.remove(str(names["source"]))


def _unbind_ca(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.inputs.remove(str(names["ca"]))


def _break_source_fanout(model_ir: ModelIR, names: dict[str, object]) -> None:
    output = "extra_source_use"
    model_ir.tensors[output] = _tensor(output, shape=(1, 1, 2, 3))
    model_ir.operators.append(
        OperatorIR("IDENTITY", [str(names["source_nchw"])], [output])
    )


def _break_ca_fanout(model_ir: ModelIR, names: dict[str, object]) -> None:
    output = "extra_ca_use"
    model_ir.tensors[output] = _tensor(output, shape=(1, 1, 2, 3))
    model_ir.operators.append(
        OperatorIR("IDENTITY", [str(names["ca_nchw"])], [output])
    )


def _wrong_axis(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["concat_sa_op"].options["axis"] = 2


def _wrong_pa_axis(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["concat_pa_op"].options["axis"] = 1


def _wrong_reduce_axis(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["axes"])].data = np.asarray([2], dtype=np.int32)


def _wrong_pad_shape(model_ir: ModelIR, names: dict[str, object]) -> None:
    tensor = model_ir.tensors[str(names["sa_pads"])]
    tensor.data = np.asarray([0] * 8, dtype=np.int32)
    tensor.shape = [8]
    tensor.shape_signature = [8]


def _negative_pad(model_ir: ModelIR, names: dict[str, object]) -> None:
    data = np.asarray(model_ir.tensors[str(names["pa_pads"])].data).copy()
    data[2, 0] = -1
    model_ir.tensors[str(names["pa_pads"])].data = data


def _wrong_unsqueeze_shape(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["unsqueeze_shape"])].data = np.asarray(
        [1, 1, 2, 1, 3], dtype=np.int32
    )


def _wrong_reshape_option(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["reshape_pa_op"].options["newShape"] = [1, 1, 2, 3]


def _quantized_source(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["source"])].quantization = {"scale": [1.0]}


def _wrong_layout(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["source"])].physical_layout = "NCHW"


def _wrong_dtype(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["ca_nchw"])].dtype = "FLOAT16"


def _non_singleton_channel(model_ir: ModelIR, names: dict[str, object]) -> None:
    tensor = model_ir.tensors[str(names["source"])]
    tensor.shape[-1] = 2
    tensor.shape_signature[-1] = 2


def _fused_add(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["add_attention_op"].options["fusedActivationFunction"] = "RELU"


def _fused_mul(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["mul_op"].options["fusedActivationFunction"] = "RELU"


def _wrong_permutation(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["sa_perm"])].data = _NHWC_TO_NCHW.copy()


def _duplicate_producer(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR("IDENTITY", [str(names["source"])], [str(names["source_nchw"])]),
    )


def _duplicate_mul_output_producer(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.insert(
        -1,
        OperatorIR("IDENTITY", [str(names["source"])], [str(names["output"])]),
    )


def _wrong_gate_topology(model_ir: ModelIR, names: dict[str, object]) -> None:
    names["gate_op"].op_type = "TANH"


def _variable_constant(model_ir: ModelIR, names: dict[str, object]) -> None:
    model_ir.tensors[str(names["axes"])].is_variable = True


@pytest.mark.parametrize(
    "mutation",
    (
        _set_public_intermediate,
        _set_public_ca_adapter_output,
        _set_public_gate_adapter_output,
        _unbind_source,
        _unbind_ca,
        _break_source_fanout,
        _break_ca_fanout,
        _wrong_axis,
        _wrong_pa_axis,
        _wrong_reduce_axis,
        _wrong_pad_shape,
        _negative_pad,
        _wrong_unsqueeze_shape,
        _wrong_reshape_option,
        _quantized_source,
        _wrong_layout,
        _wrong_dtype,
        _non_singleton_channel,
        _fused_add,
        _fused_mul,
        _wrong_permutation,
        _duplicate_producer,
        _duplicate_mul_output_producer,
        _wrong_gate_topology,
        _variable_constant,
    ),
)
def test_sa_pa_mirrorpad_unsafe_contracts_are_transactional(
    mutation: Mutation,
) -> None:
    model_ir, names = _model(source_mul_nchw=True)
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir
    ) == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sa_pa_mirrorpad_stale_plan_is_revalidated() -> None:
    model_ir, names = _model(source_mul_nchw=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["concat_pa_op"].options["axis"] = 1
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sa_pa_mirrorpad_legacy_name_collision_is_transactional() -> None:
    model_ir, names = _model(source_mul_nchw=True)
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    assert plan.legacy_nhwc_output_name is not None
    model_ir.tensors[plan.legacy_nhwc_output_name] = _tensor(
        plan.legacy_nhwc_output_name,
        shape=(1,),
    )
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sa_pa_mirrorpad_rejects_unrelated_index() -> None:
    model_ir, _ = _model()
    unrelated, _ = _model()
    before = _fingerprint(model_ir)

    assert optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
        model_ir,
        graph_index=ModelIRGraphIndex(unrelated),
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before
