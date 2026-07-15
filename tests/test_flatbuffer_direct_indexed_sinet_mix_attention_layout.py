from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.sinet_mix_attention_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_sinet_mix_attention_double_logistic_nhwc_chains,
)


_STATS_KEY = "optimized_sinet_mix_attention_double_logistic_nhwc_chains"
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
    dtype: str,
    declared_shape: tuple[int, ...] | None = None,
) -> str:
    array = np.asarray(data).copy()
    shape = tuple(array.shape) if declared_shape is None else declared_shape
    model_ir.tensors[name] = _tensor(
        name,
        dtype=dtype,
        shape=shape,
        data=array,
    )
    return name


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    residual_add_mode: bool = True,
    reversed_inputs: bool = False,
    dynamic_signature: bool = False,
    external_constant_use: bool = False,
) -> dict[str, object]:
    np_dtype = _NP_DTYPES[dtype]
    n, h, w, c = 1, 2, 3, 2
    nhwc = (n, h, w, c)
    nhwc_sig = (-1, h, -1, c) if dynamic_signature else nhwc
    nchw = (n, c, h, w)
    nchw_sig = (-1, c, h, -1) if dynamic_signature else nchw
    ca_nchw = (n, c, 1, 1)
    ca_nchw_sig = (-1, c, 1, 1) if dynamic_signature else ca_nchw
    ca_nhwc = (n, 1, 1, c)
    ca_nhwc_sig = (-1, 1, 1, c) if dynamic_signature else ca_nhwc
    sa_single = (n, 1, h, w)
    sa_single_sig = (-1, 1, h, -1) if dynamic_signature else sa_single
    sa_concat = (n, 2, h, w)
    sa_concat_sig = (-1, 2, h, -1) if dynamic_signature else sa_concat
    sa_padded = (n, 2, h + 2, w + 2)
    sa_padded_sig = (
        (-1, 2, h + 2, -1) if dynamic_signature else sa_padded
    )
    sa_conv_in = (n, h + 2, w + 2, 2)
    sa_conv_in_sig = (
        (-1, h + 2, -1, 2) if dynamic_signature else sa_conv_in
    )
    sa_conv_out = (n, h, w, 1)
    sa_conv_out_sig = (-1, h, -1, 1) if dynamic_signature else sa_conv_out
    unsqueeze = (n, c, 1, h, w)
    unsqueeze_sig = (-1, c, 1, h, -1) if dynamic_signature else unsqueeze
    concat_pa = (n, c, 2, h, w)
    concat_pa_sig = (-1, c, 2, h, -1) if dynamic_signature else concat_pa
    reshape_pa = (n, 2 * c, h, w)
    reshape_pa_sig = (
        (-1, 2 * c, h, -1) if dynamic_signature else reshape_pa
    )
    pa_padded = (n, 2 * c, h + 2, w + 2)
    pa_padded_sig = (
        (-1, 2 * c, h + 2, -1) if dynamic_signature else pa_padded
    )
    pa_conv_in = (n, h + 2, w + 2, 2 * c)
    pa_conv_in_sig = (
        (-1, h + 2, -1, 2 * c)
        if dynamic_signature
        else pa_conv_in
    )

    def name(value: str) -> str:
        return f"{prefix}{value}"

    sources = {}
    source_keys = ["branch", "residual"]
    if residual_add_mode:
        source_keys = ["branch", "residual_a", "residual_b"]
    for key in source_keys:
        sources[key] = _add_value(
            model_ir,
            name(key),
            dtype=dtype,
            shape=nhwc,
            signature=nhwc_sig,
            layout="NHWC",
        )
        model_ir.inputs.append(sources[key])

    constants = {}
    for key, value in (
        ("branch_perm", _NHWC_TO_NCHW),
        ("residual_a_perm", _NHWC_TO_NCHW),
        ("residual_b_perm", _NHWC_TO_NCHW),
        ("residual_perm", _NHWC_TO_NCHW),
        ("ca_pre_perm", _NCHW_TO_NHWC),
        ("ca_post_perm", _NHWC_TO_NCHW),
        ("sa_pre_perm", _NCHW_TO_NHWC),
        ("pa_pre_perm", _NCHW_TO_NHWC),
        ("pa_post_perm", _NHWC_TO_NCHW),
        ("post_perm", _NCHW_TO_NHWC),
    ):
        constants[key] = _add_constant(
            model_ir,
            name(key),
            value,
            dtype="INT32",
        )
    constants["ca_axes"] = _add_constant(
        model_ir,
        name("ca_axes"),
        np.asarray([2, 3], dtype=np.int32),
        dtype="INT32",
    )
    constants["sa_axes"] = _add_constant(
        model_ir,
        name("sa_axes"),
        np.asarray([1], dtype=np.int32),
        dtype="INT32",
    )
    pad_data = np.asarray(
        [[0, 0], [0, 0], [1, 1], [1, 1]],
        dtype=np.int32,
    )
    constants["sa_pads"] = _add_constant(
        model_ir,
        name("sa_pads"),
        pad_data,
        dtype="INT32",
    )
    constants["pa_pads"] = _add_constant(
        model_ir,
        name("pa_pads"),
        pad_data,
        dtype="INT32",
    )
    constants["sa_shape"] = _add_constant(
        model_ir,
        name("sa_shape"),
        np.asarray(sa_single, dtype=np.int32),
        dtype="INT32",
    )
    constants["unsqueeze_shape"] = _add_constant(
        model_ir,
        name("unsqueeze_shape"),
        np.asarray(unsqueeze, dtype=np.int32),
        dtype="INT32",
    )
    constants["reshape_pa_shape"] = _add_constant(
        model_ir,
        name("reshape_pa_shape"),
        np.asarray(reshape_pa, dtype=np.int32),
        dtype="INT32",
    )
    constants["one"] = _add_constant(
        model_ir,
        name("one"),
        np.asarray(1.0, dtype=np_dtype),
        dtype=dtype,
        declared_shape=(1,),
    )

    shapes = {
        "branch_nchw": (nchw, nchw_sig, "NCHW"),
        "residual_a_nchw": (nchw, nchw_sig, "NCHW"),
        "residual_b_nchw": (nchw, nchw_sig, "NCHW"),
        "residual_nchw": (nchw, nchw_sig, "NCHW"),
        "source_nchw": (nchw, nchw_sig, "NCHW"),
        "mean_ca": (ca_nchw, ca_nchw_sig, "NCHW"),
        "ca_pre": (ca_nhwc, ca_nhwc_sig, "NHWC"),
        "ca_conv0": (ca_nhwc, ca_nhwc_sig, "NHWC"),
        "ca_conv2": (ca_nhwc, ca_nhwc_sig, "NHWC"),
        "ca_nchw": (ca_nchw, ca_nchw_sig, "NCHW"),
        "mean_sa": (sa_single, sa_single_sig, "NCHW"),
        "max_sa": (sa_single, sa_single_sig, "NCHW"),
        "concat_sa": (sa_concat, sa_concat_sig, "NCHW"),
        "mirror_sa": (sa_padded, sa_padded_sig, "NCHW"),
        "sa_pre": (sa_conv_in, sa_conv_in_sig, "NHWC"),
        "sa_conv": (sa_conv_out, sa_conv_out_sig, "NHWC"),
        "sa_nchw": (sa_single, sa_single_sig, "NCHW"),
        "attention_nchw": (nchw, nchw_sig, "NCHW"),
        "unsqueeze_source": (unsqueeze, unsqueeze_sig, "UNKNOWN"),
        "unsqueeze_attention": (unsqueeze, unsqueeze_sig, "UNKNOWN"),
        "concat_pa": (concat_pa, concat_pa_sig, "UNKNOWN"),
        "reshape_pa": (reshape_pa, reshape_pa_sig, "NCHW"),
        "mirror_pa": (pa_padded, pa_padded_sig, "NCHW"),
        "pa_pre": (pa_conv_in, pa_conv_in_sig, "NHWC"),
        "pa_conv": (nhwc, nhwc_sig, "NHWC"),
        "pa_nchw": (nchw, nchw_sig, "NCHW"),
        "gate1": (nchw, nchw_sig, "NCHW"),
        "gate2": (nchw, nchw_sig, "NCHW"),
        "mul0": (nchw, nchw_sig, "NCHW"),
        "sub": (nchw, nchw_sig, "NCHW"),
        "add2": (nchw, nchw_sig, "NCHW"),
        "mul1": (nchw, nchw_sig, "NCHW"),
        "add3": (nchw, nchw_sig, "NCHW"),
        "post": (nhwc, nhwc_sig, "NHWC"),
        "output": (nhwc, nhwc_sig, "NHWC"),
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

    def binary(
        op_type: str,
        lhs: str,
        rhs: str,
        output: str,
    ) -> OperatorIR:
        inputs = [lhs, rhs]
        if reversed_inputs:
            inputs.reverse()
        return OperatorIR(
            op_type,
            inputs,
            [output],
            {"fusedActivationFunction": "NONE"},
        )

    branch_pre = OperatorIR(
        "TRANSPOSE",
        [sources["branch"], constants["branch_perm"]],
        [values["branch_nchw"]],
    )
    operators = [branch_pre]
    residual_adapters = []
    if residual_add_mode:
        for key in ("a", "b"):
            adapter = OperatorIR(
                "TRANSPOSE",
                [
                    sources[f"residual_{key}"],
                    constants[f"residual_{key}_perm"],
                ],
                [values[f"residual_{key}_nchw"]],
            )
            residual_adapters.append(adapter)
            operators.append(adapter)
        residual = binary(
            "ADD",
            values["residual_a_nchw"],
            values["residual_b_nchw"],
            values["residual_nchw"],
        )
        operators.append(residual)
    else:
        residual = OperatorIR(
            "TRANSPOSE",
            [sources["residual"], constants["residual_perm"]],
            [values["residual_nchw"]],
        )
        residual_adapters.append(residual)
        operators.append(residual)

    source = binary(
        "ADD",
        values["branch_nchw"],
        values["residual_nchw"],
        values["source_nchw"],
    )
    mean_ca = OperatorIR(
        "MEAN",
        [values["source_nchw"], constants["ca_axes"]],
        [values["mean_ca"]],
        {"keepDims": True},
    )
    mean_sa = OperatorIR(
        "MEAN",
        [values["source_nchw"], constants["sa_axes"]],
        [values["mean_sa"]],
        {"keepDims": True},
    )
    max_sa = OperatorIR(
        "REDUCE_MAX",
        [values["source_nchw"], constants["sa_axes"]],
        [values["max_sa"]],
        {"keepDims": True},
    )
    unsqueeze_source = OperatorIR(
        "RESHAPE",
        [values["source_nchw"], constants["unsqueeze_shape"]],
        [values["unsqueeze_source"]],
        {"newShape": list(unsqueeze)},
    )
    ca_pre = OperatorIR(
        "TRANSPOSE",
        [values["mean_ca"], constants["ca_pre_perm"]],
        [values["ca_pre"]],
    )
    ca_conv0 = OperatorIR(
        "CONV_2D",
        [values["ca_pre"]],
        [values["ca_conv0"]],
        {"testKind": "identity"},
    )
    ca_conv2 = OperatorIR(
        "CONV_2D",
        [values["ca_conv0"]],
        [values["ca_conv2"]],
        {"testKind": "identity"},
    )
    ca_post = OperatorIR(
        "TRANSPOSE",
        [values["ca_conv2"], constants["ca_post_perm"]],
        [values["ca_nchw"]],
    )
    concat_sa_inputs = [values["mean_sa"], values["max_sa"]]
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
    sa_pre = OperatorIR(
        "TRANSPOSE",
        [values["mirror_sa"], constants["sa_pre_perm"]],
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
        {"newShape": list(sa_single)},
    )
    add_attention = binary(
        "ADD",
        values["ca_nchw"],
        values["sa_nchw"],
        values["attention_nchw"],
    )
    unsqueeze_attention = OperatorIR(
        "RESHAPE",
        [values["attention_nchw"], constants["unsqueeze_shape"]],
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
        [values["concat_pa"], constants["reshape_pa_shape"]],
        [values["reshape_pa"]],
        {"newShape": list(reshape_pa)},
    )
    mirror_pa = OperatorIR(
        "MIRROR_PAD",
        [values["reshape_pa"], constants["pa_pads"]],
        [values["mirror_pa"]],
        {"mode": "REFLECT"},
    )
    pa_pre = OperatorIR(
        "TRANSPOSE",
        [values["mirror_pa"], constants["pa_pre_perm"]],
        [values["pa_pre"]],
    )
    pa_conv = OperatorIR(
        "CONV_2D",
        [values["pa_pre"]],
        [values["pa_conv"]],
        {"testKind": "pa", "channels": c},
    )
    pa_post = OperatorIR(
        "TRANSPOSE",
        [values["pa_conv"], constants["pa_post_perm"]],
        [values["pa_nchw"]],
    )
    gate1 = OperatorIR("LOGISTIC", [values["pa_nchw"]], [values["gate1"]])
    gate2 = OperatorIR("LOGISTIC", [values["gate1"]], [values["gate2"]])
    mul0 = binary(
        "MUL",
        values["branch_nchw"],
        values["gate2"],
        values["mul0"],
    )
    sub = binary(
        "SUB",
        constants["one"],
        values["gate2"],
        values["sub"],
    )
    add2 = binary(
        "ADD",
        values["source_nchw"],
        values["mul0"],
        values["add2"],
    )
    mul1 = binary(
        "MUL",
        values["sub"],
        values["residual_nchw"],
        values["mul1"],
    )
    add3 = binary(
        "ADD",
        values["add2"],
        values["mul1"],
        values["add3"],
    )
    root = OperatorIR(
        "TRANSPOSE",
        [values["add3"], constants["post_perm"]],
        [values["post"]],
    )
    post_conv = OperatorIR(
        "CONV_2D",
        [values["post"]],
        [values["output"]],
        {"testKind": "identity"},
    )
    operators.extend(
        [
            source,
            mean_ca,
            mean_sa,
            max_sa,
            unsqueeze_source,
            ca_pre,
            ca_conv0,
            ca_conv2,
            ca_post,
            concat_sa_op,
            mirror_sa,
            sa_pre,
            sa_conv,
            sa_reshape,
            add_attention,
            unsqueeze_attention,
            concat_pa_op,
            reshape_pa_op,
            mirror_pa,
            pa_pre,
            pa_conv,
            pa_post,
            gate1,
            gate2,
            mul0,
            sub,
            add2,
            mul1,
            add3,
            root,
            post_conv,
        ]
    )
    model_ir.operators.extend(operators)
    model_ir.outputs.append(values["output"])

    constant_side = None
    if external_constant_use:
        output_name = name("axis_side")
        model_ir.tensors[output_name] = _tensor(
            output_name,
            dtype="INT32",
            shape=(1,),
        )
        constant_side = OperatorIR(
            "IDENTITY",
            [constants["sa_axes"]],
            [output_name],
        )
        model_ir.operators.append(constant_side)
        model_ir.outputs.append(output_name)

    return {
        **sources,
        **constants,
        **values,
        "branch_pre_op": branch_pre,
        "residual_adapters": tuple(residual_adapters),
        "residual_op": residual,
        "source_op": source,
        "mean_ca_op": mean_ca,
        "mean_sa_op": mean_sa,
        "max_sa_op": max_sa,
        "unsqueeze_source_op": unsqueeze_source,
        "ca_pre_op": ca_pre,
        "ca_post_op": ca_post,
        "concat_sa_op": concat_sa_op,
        "sa_pre_op": sa_pre,
        "sa_reshape_op": sa_reshape,
        "add_attention_op": add_attention,
        "unsqueeze_attention_op": unsqueeze_attention,
        "concat_pa_op": concat_pa_op,
        "reshape_pa_op": reshape_pa_op,
        "pa_pre_op": pa_pre,
        "pa_post_op": pa_post,
        "gate1_op": gate1,
        "mul0_op": mul0,
        "mul1_op": mul1,
        "root": root,
        "post_conv_op": post_conv,
        "constant_side_op": constant_side,
    }


def _model(**kwargs: object) -> tuple[ModelIR, dict[str, object]]:
    model_ir = ModelIR("indexed_sinet_mix_attention_layout")
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
        elif operator.op_type == "SUB":
            output = np.subtract(args[0], args[1])
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
            output = np.concatenate(
                args,
                axis=int(operator.options["axis"]),
            )
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
            kind = str(operator.options.get("testKind", "identity"))
            if kind == "identity":
                output = np.asarray(args[0])
            elif kind == "sa":
                output = np.mean(
                    args[0][:, 1:-1, 1:-1, :],
                    axis=3,
                    keepdims=True,
                )
            elif kind == "pa":
                channels = int(operator.options["channels"])
                output = args[0][:, 1:-1, 1:-1, :channels]
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


def _inputs(
    names: dict[str, object],
    dtype: str,
    *,
    residual_add_mode: bool,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(953)
    keys = (
        ("branch", "residual_a", "residual_b")
        if residual_add_mode
        else ("branch", "residual")
    )
    return {
        str(names[key]): rng.normal(size=(1, 2, 3, 2)).astype(
            _NP_DTYPES[dtype]
        )
        for key in keys
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
@pytest.mark.parametrize("residual_add_mode", (False, True))
@pytest.mark.parametrize("reversed_inputs", (False, True))
def test_sinet_mix_attention_is_indexed_and_numerically_equivalent(
    dtype: str,
    residual_add_mode: bool,
    reversed_inputs: bool,
) -> None:
    model_ir, names = _model(
        dtype=dtype,
        residual_add_mode=residual_add_mode,
        reversed_inputs=reversed_inputs,
        dynamic_signature=True,
    )
    original = copy.deepcopy(model_ir)
    inputs = _inputs(
        names,
        dtype,
        residual_add_mode=residual_add_mode,
    )
    expected = _evaluate(original, inputs)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    first = optimize_sinet_mix_attention_double_logistic_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    second = optimize_sinet_mix_attention_double_logistic_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert first == {_STATS_KEY: 1}
    assert second == {_STATS_KEY: 0}
    for key in (
        "branch_pre_op",
        "ca_pre_op",
        "ca_post_op",
        "sa_pre_op",
        "sa_reshape_op",
        "pa_pre_op",
        "pa_post_op",
        "root",
    ):
        assert names[key] not in model_ir.operators
    for adapter in names["residual_adapters"]:
        assert adapter not in model_ir.operators
    assert str(names["branch"]) in names["source_op"].inputs
    assert str(names["branch"]) in names["mul0_op"].inputs
    if residual_add_mode:
        assert set(names["residual_op"].inputs) == {
            str(names["residual_a"]),
            str(names["residual_b"]),
        }
        residual_name = str(names["residual_nchw"])
    else:
        residual_name = str(names["residual"])
    assert residual_name in names["source_op"].inputs
    assert residual_name in names["mul1_op"].inputs
    assert names["concat_sa_op"].options["axis"] == 3
    assert names["concat_pa_op"].options["axis"] == 4
    assert names["gate1_op"].inputs == [str(names["pa_conv"])]
    assert names["post_conv_op"].inputs == [str(names["add3"])]
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["ca_axes"])].data),
        np.asarray([1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["sa_axes"])].data),
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["unsqueeze_shape"])].data),
        np.asarray([1, 2, 3, 2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["reshape_pa_shape"])].data),
        np.asarray([1, 2, 3, 4], dtype=np.int32),
    )
    actual = _evaluate(model_ir, inputs)
    tolerance = 8e-3 if dtype == "FLOAT16" else 1e-6
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=tolerance,
            atol=tolerance,
        )
    assert str(names["branch_nchw"]) not in model_ir.tensors
    for key in (
        "source_nchw",
        "attention_nchw",
        "reshape_pa",
        "gate1",
        "gate2",
        "mul0",
        "sub",
        "add2",
        "mul1",
        "add3",
    ):
        tensor = model_ir.tensors[str(names[key])]
        assert tensor.shape == [1, 2, 3, 2] or key == "reshape_pa"
        assert tensor.logical_layout == "NHWC"
    assert model_ir.tensors[str(names["reshape_pa"])].shape == [1, 2, 3, 4]
    assert model_ir.tensors[str(names["unsqueeze_source"])].shape == [
        1,
        2,
        3,
        2,
        1,
    ]
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_sinet_mix_attention_clones_externally_used_constant() -> None:
    model_ir, names = _model(external_constant_use=True)
    original = np.asarray(model_ir.tensors[str(names["sa_axes"])].data).copy()

    stats = optimize_sinet_mix_attention_double_logistic_nhwc_chains(model_ir)

    assert stats == {_STATS_KEY: 1}
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[str(names["sa_axes"])].data),
        original,
    )
    assert names["constant_side_op"].inputs == [str(names["sa_axes"])]
    rewritten_names = {
        str(names["mean_sa_op"].inputs[1]),
        str(names["max_sa_op"].inputs[1]),
    }
    assert len(rewritten_names) == 1
    rewritten_name = rewritten_names.pop()
    assert rewritten_name != str(names["sa_axes"])
    np.testing.assert_array_equal(
        np.asarray(model_ir.tensors[rewritten_name].data),
        np.asarray([3], dtype=np.int32),
    )


def test_sinet_mix_attention_honors_candidate_and_total_cap() -> None:
    model_ir = ModelIR("bounded_sinet_mix_attention")
    first = _add_chain(model_ir, prefix="first_")
    second = _add_chain(model_ir, prefix="second_", residual_add_mode=False)
    graph_index = ModelIRGraphIndex(model_ir)

    candidate_stats = optimize_sinet_mix_attention_double_logistic_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        candidate=second["root"],
    )
    capped_stats = optimize_sinet_mix_attention_double_logistic_nhwc_chains(
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


def _duplicate_source_producer(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.insert(
        0,
        OperatorIR(
            "IDENTITY",
            [str(names["branch"])],
            [str(names["source_nchw"])],
        ),
    )


def _move_root_before_add3(
    model_ir: ModelIR,
    names: dict[str, object],
) -> None:
    model_ir.operators.remove(names["root"])
    add3 = next(
        operator
        for operator in model_ir.operators
        if str(names["add3"]) in operator.outputs
    )
    model_ir.operators.insert(model_ir.operators.index(add3), names["root"])


@pytest.mark.parametrize(
    "mutation",
    (
        lambda model, names: model.tensors[
            str(names["branch_perm"])
        ].data.__setitem__(slice(None), _NCHW_TO_NHWC),
        lambda model, names: model.tensors[
            str(names["ca_pre_perm"])
        ].data.__setitem__(slice(None), _NHWC_TO_NCHW),
        lambda model, names: model.tensors[
            str(names["post_perm"])
        ].data.__setitem__(slice(None), _NHWC_TO_NCHW),
        lambda model, names: names["concat_sa_op"].options.__setitem__(
            "axis", 3
        ),
        lambda model, names: names["concat_pa_op"].options.__setitem__(
            "axis", 4
        ),
        lambda model, names: names["source_op"].options.__setitem__(
            "fusedActivationFunction", "RELU"
        ),
        lambda model, names: names["mean_sa_op"].options.__setitem__(
            "keepDims", False
        ),
        lambda model, names: model.tensors[
            str(names["sa_axes"])
        ].data.__setitem__(0, 2),
        lambda model, names: model.tensors[
            str(names["sa_pads"])
        ].data.__setitem__((2, 0), -1),
        lambda model, names: model.tensors[
            str(names["unsqueeze_shape"])
        ].data.__setitem__(2, 2),
        lambda model, names: names["reshape_pa_op"].options.__setitem__(
            "newShape", [1, 2, 3, 4]
        ),
        lambda model, names: _append_fanout(model, names, "source_nchw"),
        lambda model, names: _append_fanout(model, names, "branch_nchw"),
        lambda model, names: _append_fanout(model, names, "ca_nchw"),
        lambda model, names: _append_fanout(model, names, "sa_nchw"),
        lambda model, names: _append_fanout(model, names, "pa_nchw"),
        lambda model, names: _append_fanout(model, names, "gate1"),
        lambda model, names: model.outputs.append(str(names["post"])),
        lambda model, names: model.outputs.append(str(names["add3"])),
        lambda model, names: setattr(
            model.tensors[str(names["branch"])], "dtype", "FLOAT16"
        ),
        lambda model, names: setattr(
            model.tensors[str(names["source_nchw"])],
            "shape",
            [1, 2, 2, 2],
        ),
        lambda model, names: setattr(
            model.tensors[str(names["sa_axes"])], "is_variable", True
        ),
        lambda model, names: setattr(
            model.tensors[str(names["pa_pads"])],
            "quantization",
            {"scale": [1.0]},
        ),
        lambda model, names: model.tensors[
            str(names["one"])
        ].data.__setitem__((), np.nan),
        lambda model, names: setattr(
            model.tensors[str(names["branch"])],
            "logical_layout",
            "NCHW",
        ),
        lambda model, names: setattr(
            names["post_conv_op"], "op_type", "IDENTITY"
        ),
        _duplicate_source_producer,
        _move_root_before_add3,
    ),
)
def test_sinet_mix_attention_rejects_unsafe_variants_transactionally(
    mutation: Callable[[ModelIR, dict[str, object]], None],
) -> None:
    model_ir, names = _model()
    mutation(model_ir, names)
    before = _fingerprint(model_ir)

    stats = optimize_sinet_mix_attention_double_logistic_nhwc_chains(model_ir)

    assert stats == {_STATS_KEY: 0}
    assert _fingerprint(model_ir) == before


def test_sinet_mix_attention_rejects_stale_plan_transactionally() -> None:
    model_ir, names = _model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(model_ir, graph_index, names["root"])
    assert plan is not None
    names["concat_pa_op"].options["axis"] = 4
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_mix_attention_rejects_clone_collision_transactionally() -> None:
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
    source = model_ir.tensors[str(names["sa_axes"])]
    model_ir.tensors[clone_name] = copy.deepcopy(source)
    model_ir.tensors[clone_name].name = clone_name
    before = _fingerprint(model_ir)

    assert not _apply_plan(model_ir, graph_index, plan)
    assert _fingerprint(model_ir) == before


def test_sinet_mix_attention_no_index_preflight_avoids_index_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir, _ = _model()

    def fail_index(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight should return before index construction")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.passes.sinet_mix_attention_layout.ModelIRGraphIndex",
        fail_index,
    )

    assert optimize_sinet_mix_attention_double_logistic_nhwc_chains(
        model_ir,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
