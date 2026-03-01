from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _normalize_spatial_pair(values: Any, *, default: int, label: str, node_name: str) -> list[int]:
    vals = [int(v) for v in list(values)] if values is not None else []
    if len(vals) == 0:
        return [int(default), int(default)]
    if len(vals) == 1:
        return [int(vals[0]), int(vals[0])]
    if len(vals) == 2:
        return [int(vals[0]), int(vals[1])]
    raise NotImplementedError(
        f"{label} must have length 1 or 2 for Col2Im. op={node_name} {label}={vals}"
    )


def _normalize_col2im_pads(values: Any, *, node_name: str) -> list[int]:
    pads = [int(v) for v in list(values)] if values is not None else []
    if len(pads) == 0:
        return [0, 0, 0, 0]
    if len(pads) == 2:
        return [int(pads[0]), int(pads[1]), int(pads[0]), int(pads[1])]
    if len(pads) == 4:
        return [int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])]
    raise NotImplementedError(
        f"pads must have length 2 or 4 for Col2Im. op={node_name} pads={pads}"
    )


def _add_reshape_operator(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: list[int],
) -> None:
    shape_const = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": [int(v) for v in list(new_shape)]},
        )
    )


def _tensor_signature(ctx: Any, tensor_name: str) -> list[int]:
    tensor = ctx.model_ir.tensors[tensor_name]
    if tensor.shape_signature is not None:
        return [int(v) for v in list(tensor.shape_signature)]
    return [int(v) for v in list(tensor.shape)]


def _make_pseudo_node(
    *,
    base_node: Any,
    input_names: list[str],
    output_name: str,
    attrs: dict[str, Any],
) -> Any:
    pseudo = type("Node", (), {})()
    pseudo.name = str(base_node.name)
    pseudo.op = str(base_node.op)
    pseudo.attrs = dict(attrs)
    pseudo.inputs = [type("In", (), {"name": str(name)}) for name in input_names]
    pseudo.outputs = [type("Out", (), {"name": str(output_name)})]
    return pseudo


def _decode_attr_string(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _flatten_attr_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(np.asarray(value).reshape(-1))
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, np.ndarray):
                flattened.extend(list(np.asarray(item).reshape(-1)))
            elif isinstance(item, (list, tuple)):
                flattened.extend(list(np.asarray(item).reshape(-1)))
            else:
                flattened.append(item)
        return flattened
    return [value]


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        arr = np.asarray(value).reshape(-1)
        if int(arr.size) == 0:
            return None
        value = arr[0]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    try:
        return float(value)
    except Exception:
        return None


def _clone_shape_signature(ctx: Any, src_name: str, dst_name: str) -> None:
    src_tensor = ctx.model_ir.tensors[src_name]
    dst_tensor = ctx.model_ir.tensors[dst_name]
    if src_tensor.shape_signature is not None:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape_signature)]
    else:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape)]


def _add_same_shape_tensor(ctx: Any, *, name: str, like_tensor_name: str) -> str:
    ref_tensor = ctx.model_ir.tensors[like_tensor_name]
    tensor_name = ctx.add_intermediate_tensor(
        str(name),
        dtype=str(ref_tensor.dtype),
        shape=[int(v) for v in list(ref_tensor.shape)],
    )
    _clone_shape_signature(ctx, like_tensor_name, tensor_name)
    return str(tensor_name)


def _add_fusedconv_activation_op(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
    activation: str,
    activation_params: list[Any],
) -> None:
    activation_key = str(activation).lower()
    params = _flatten_attr_values(activation_params)

    if activation_key == "relu":
        ctx.add_operator(
            OperatorIR(
                op_type="RELU",
                inputs=[input_name],
                outputs=[output_name],
            )
        )
        return

    if activation_key == "tanh":
        ctx.add_operator(
            OperatorIR(
                op_type="TANH",
                inputs=[input_name],
                outputs=[output_name],
            )
        )
        return

    if activation_key == "sigmoid":
        ctx.add_operator(
            OperatorIR(
                op_type="LOGISTIC",
                inputs=[input_name],
                outputs=[output_name],
            )
        )
        return

    if activation_key == "leakyrelu":
        alpha = 0.01
        if len(params) > 0:
            alpha_value = _to_optional_float(params[0])
            if alpha_value is None:
                raise NotImplementedError(
                    f"FusedConv LeakyRelu alpha must be scalar-convertible. op={node.name} activation_params={activation_params}"
                )
            alpha = float(alpha_value)
        ctx.add_operator(
            OperatorIR(
                op_type="LEAKY_RELU",
                inputs=[input_name],
                outputs=[output_name],
                options={"alpha": float(alpha)},
            )
        )
        return

    if activation_key == "clip":
        if len(params) == 0:
            raise NotImplementedError(
                f"FusedConv Clip requires activation_params [min,max] (or one-sided bound). op={node.name} activation_params={activation_params}"
            )
        min_value = _to_optional_float(params[0]) if len(params) >= 1 else None
        max_value = _to_optional_float(params[1]) if len(params) >= 2 else None
        if len(params) >= 1 and params[0] is not None and min_value is None:
            raise NotImplementedError(
                f"FusedConv Clip min must be scalar-convertible. op={node.name} activation_params={activation_params}"
            )
        if len(params) >= 2 and params[1] is not None and max_value is None:
            raise NotImplementedError(
                f"FusedConv Clip max must be scalar-convertible. op={node.name} activation_params={activation_params}"
            )
        if min_value is not None and max_value is not None and float(min_value) > float(max_value):
            raise NotImplementedError(
                f"FusedConv Clip requires min <= max. op={node.name} min={min_value} max={max_value}"
            )
        if (
            min_value is not None
            and abs(float(min_value) - 0.0) <= 1e-6
            and max_value is not None
            and abs(float(max_value) - 6.0) <= 1e-6
        ):
            ctx.add_operator(
                OperatorIR(
                    op_type="RELU6",
                    inputs=[input_name],
                    outputs=[output_name],
                )
            )
            return
        if (
            min_value is not None
            and abs(float(min_value) + 1.0) <= 1e-6
            and max_value is not None
            and abs(float(max_value) - 1.0) <= 1e-6
        ):
            ctx.add_operator(
                OperatorIR(
                    op_type="RELU_N1_TO_1",
                    inputs=[input_name],
                    outputs=[output_name],
                )
            )
            return
        if min_value is not None and abs(float(min_value) - 0.0) <= 1e-6 and max_value is None:
            ctx.add_operator(
                OperatorIR(
                    op_type="RELU",
                    inputs=[input_name],
                    outputs=[output_name],
                )
            )
            return
        if min_value is None and max_value is None:
            raise NotImplementedError(
                f"FusedConv Clip requires at least one concrete bound. op={node.name} activation_params={activation_params}"
            )
        if min_value is not None and max_value is None:
            min_const = ctx.add_const_tensor(
                f"{node.name}_fusedconv_clip_min",
                np.asarray(float(min_value), dtype=np.float32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MAXIMUM",
                    inputs=[input_name, min_const],
                    outputs=[output_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            return
        if min_value is None and max_value is not None:
            max_const = ctx.add_const_tensor(
                f"{node.name}_fusedconv_clip_max",
                np.asarray(float(max_value), dtype=np.float32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MINIMUM",
                    inputs=[input_name, max_const],
                    outputs=[output_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            return
        min_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_clip_min",
            np.asarray(float(min_value), dtype=np.float32),
        )
        max_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_clip_max",
            np.asarray(float(max_value), dtype=np.float32),
        )
        maxed_name = _add_same_shape_tensor(
            ctx,
            name=f"{node.name}_fusedconv_clip_maxed",
            like_tensor_name=input_name,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MAXIMUM",
                inputs=[input_name, min_const],
                outputs=[maxed_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MINIMUM",
                inputs=[maxed_name, max_const],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return

    if activation_key == "hardsigmoid":
        if len(params) < 2:
            raise NotImplementedError(
                f"FusedConv HardSigmoid requires two activation_params [alpha,beta]. op={node.name} activation_params={activation_params}"
            )
        alpha = _to_optional_float(params[0])
        beta = _to_optional_float(params[1])
        if alpha is None or beta is None:
            raise NotImplementedError(
                f"FusedConv HardSigmoid activation_params must be scalar-convertible. op={node.name} activation_params={activation_params}"
            )
        alpha_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_hardsigmoid_alpha",
            np.asarray(float(alpha), dtype=np.float32),
        )
        beta_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_hardsigmoid_beta",
            np.asarray(float(beta), dtype=np.float32),
        )
        zero_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_hardsigmoid_zero",
            np.asarray(0.0, dtype=np.float32),
        )
        one_const = ctx.add_const_tensor(
            f"{node.name}_fusedconv_hardsigmoid_one",
            np.asarray(1.0, dtype=np.float32),
        )
        mul_name = _add_same_shape_tensor(
            ctx,
            name=f"{node.name}_fusedconv_hardsigmoid_mul",
            like_tensor_name=input_name,
        )
        add_name = _add_same_shape_tensor(
            ctx,
            name=f"{node.name}_fusedconv_hardsigmoid_add",
            like_tensor_name=input_name,
        )
        max_name = _add_same_shape_tensor(
            ctx,
            name=f"{node.name}_fusedconv_hardsigmoid_max",
            like_tensor_name=input_name,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[input_name, alpha_const],
                outputs=[mul_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[mul_name, beta_const],
                outputs=[add_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MAXIMUM",
                inputs=[add_name, zero_const],
                outputs=[max_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MINIMUM",
                inputs=[max_name, one_const],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return

    raise NotImplementedError(
        f"FusedConv activation is not supported in flatbuffer_direct. op={node.name} activation={activation}"
    )


def _extract_1d_pads(raw_pads: Any, *, op_name: str, node_name: str) -> list[int]:
    pads = [int(v) for v in list(raw_pads)]
    if len(pads) == 0:
        return [0, 0]
    if len(pads) == 1:
        return [int(pads[0]), int(pads[0])]
    if len(pads) == 2:
        return [int(pads[0]), int(pads[1])]
    if len(pads) == 4:
        # Rank-1 ONNX pads may be normalized to 4D-style [h0, w0, h1, w1].
        # Pick width-axis pads, while tolerating legacy [w0, w1, w0, w1].
        if int(pads[0]) == 0 and int(pads[2]) == 0:
            return [int(pads[1]), int(pads[3])]
        if int(pads[1]) == 0 and int(pads[3]) == 0:
            return [int(pads[0]), int(pads[2])]
        if int(pads[0]) == int(pads[2]) and int(pads[1]) == int(pads[3]):
            return [int(pads[1]), int(pads[3])]
        if int(pads[0]) == int(pads[1]) and int(pads[2]) == int(pads[3]):
            return [int(pads[0]), int(pads[2])]
    raise NotImplementedError(
        f"{op_name} 1D pads must be length 2 (or normalized length 4). op={node_name} pads={pads}"
    )


def _conv1d_output_length(
    input_width: int,
    *,
    kernel: int,
    stride: int,
    dilation: int,
    pad_left: int,
    pad_right: int,
) -> int:
    effective_kernel = int((int(kernel) - 1) * int(dilation) + 1)
    numer = int(input_width) + int(pad_left) + int(pad_right) - int(effective_kernel)
    return int(numer // int(stride) + 1)


def _select_conv1d_pads_for_static_output(
    *,
    raw_pads: Any,
    default_pads: list[int],
    input_width: int,
    output_width: int,
    kernel: int,
    stride: int,
    dilation: int,
) -> list[int]:
    if int(input_width) <= 0 or int(output_width) <= 0:
        return [int(default_pads[0]), int(default_pads[1])]

    raw = [int(v) for v in list(raw_pads)]
    candidates: list[list[int]] = [
        [int(default_pads[0]), int(default_pads[1])],
    ]
    if len(raw) == 1:
        v = int(raw[0])
        candidates.extend(
            [
                [v, 0],
                [0, v],
                [v, v],
            ]
        )
    elif len(raw) == 2:
        candidates.append([int(raw[0]), int(raw[1])])
    elif len(raw) == 4:
        # Legacy 1D normalization often repeats the begin/end pair as [b, e, b, e].
        if int(raw[0]) == int(raw[2]) and int(raw[1]) == int(raw[3]):
            candidates.append([int(raw[0]), int(raw[1])])
        # Generic rank-2 style [h0, w0, h1, w1] -> pick width axis.
        if int(raw[0]) == 0 and int(raw[2]) == 0:
            candidates.append([int(raw[1]), int(raw[3])])
        if int(raw[1]) == 0 and int(raw[3]) == 0:
            candidates.append([int(raw[0]), int(raw[2])])
        if int(raw[0]) == int(raw[1]) and int(raw[2]) == int(raw[3]):
            candidates.append([int(raw[0]), int(raw[2])])
        candidates.extend(
            [
                [int(raw[0]), int(raw[1])],
                [int(raw[1]), int(raw[3])],
                [int(raw[0]), int(raw[2])],
            ]
        )

    deduped: list[list[int]] = []
    for cand in candidates:
        c0 = int(cand[0])
        c1 = int(cand[1])
        if c0 < 0 or c1 < 0:
            continue
        normalized = [c0, c1]
        if normalized not in deduped:
            deduped.append(normalized)

    for cand in deduped:
        out_len = _conv1d_output_length(
            int(input_width),
            kernel=int(kernel),
            stride=int(stride),
            dilation=int(dilation),
            pad_left=int(cand[0]),
            pad_right=int(cand[1]),
        )
        if int(out_len) == int(output_width):
            return [int(cand[0]), int(cand[1])]
    return [int(default_pads[0]), int(default_pads[1])]


def _build_conv1d_via_conv2d(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise NotImplementedError(
            f"Conv1D lowering expects rank-3 input/output. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"Conv weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 3:
        raise NotImplementedError(
            f"Conv1D weight rank must be 3. op={node.name} shape={list(weights.shape)}"
        )

    input_sig = _tensor_signature(ctx, input_name)
    output_sig = _tensor_signature(ctx, output_name)
    if len(input_sig) != 3:
        input_sig = [int(v) for v in list(input_shape)]
    if len(output_sig) != 3:
        output_sig = [int(v) for v in list(output_shape)]
    if len(input_sig) != 3:
        input_sig = [int(v) for v in list(input_shape)]
    if len(output_sig) != 3:
        output_sig = [int(v) for v in list(output_shape)]

    axis_name = ctx.add_const_tensor(
        f"{node.name}_conv1d_expand_axis",
        np.asarray([2], dtype=np.int32),
    )
    input_2d_shape = [int(input_shape[0]), int(input_shape[1]), 1, int(input_shape[2])]
    input_2d_name = ctx.add_intermediate_tensor(
        f"{node.name}_conv1d_input_nchw2d",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=input_2d_shape,
    )
    input_2d_sig = [int(input_sig[0]), int(input_sig[1]), 1, int(input_sig[2])]
    ctx.model_ir.tensors[input_2d_name].shape_signature = [int(v) for v in input_2d_sig]
    ctx.add_operator(
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=[input_name, axis_name],
            outputs=[input_2d_name],
        )
    )

    output_2d_shape = [int(output_shape[0]), int(output_shape[1]), 1, int(output_shape[2])]
    output_2d_name = ctx.add_intermediate_tensor(
        f"{node.name}_conv1d_output_nchw2d",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=output_2d_shape,
    )
    output_2d_sig = [int(output_sig[0]), int(output_sig[1]), 1, int(output_sig[2])]
    ctx.model_ir.tensors[output_2d_name].shape_signature = [int(v) for v in output_2d_sig]

    weights_2d_name = ctx.add_const_tensor(
        f"{node.name}_conv1d_filter_2d",
        np.expand_dims(weights, axis=2),
    )

    kernel_1d = int(weights.shape[2])
    raw_pads_1d = node.attrs.get("pads", [0, 0])
    pads_1d = _extract_1d_pads(
        raw_pads_1d,
        op_name="Conv",
        node_name=str(node.name),
    )
    strides_1d = [int(v) for v in list(node.attrs.get("strides", [1]))]
    dilations_1d = [int(v) for v in list(node.attrs.get("dilations", [1]))]
    if len(strides_1d) != 1 or len(dilations_1d) != 1:
        raise NotImplementedError(
            f"Conv1D strides/dilations must be length 1. op={node.name} strides={strides_1d} dilations={dilations_1d}"
        )
    pads_1d = _select_conv1d_pads_for_static_output(
        raw_pads=raw_pads_1d,
        default_pads=pads_1d,
        input_width=int(input_shape[2]),
        output_width=int(output_shape[2]),
        kernel=int(kernel_1d),
        stride=int(strides_1d[0]),
        dilation=int(dilations_1d[0]),
    )

    attrs_2d = dict(node.attrs)
    attrs_2d["kernel_shape"] = [1, kernel_1d]
    attrs_2d["pads"] = [0, int(pads_1d[0]), 0, int(pads_1d[1])]
    attrs_2d["strides"] = [1, int(strides_1d[0])]
    attrs_2d["dilations"] = [1, int(dilations_1d[0])]

    pseudo_inputs = [input_2d_name, weights_2d_name]
    if len(node.inputs) >= 3:
        pseudo_inputs.append(node.inputs[2].name)
    pseudo_node = _make_pseudo_node(
        base_node=node,
        input_names=pseudo_inputs,
        output_name=output_2d_name,
        attrs=attrs_2d,
    )
    build_conv2d_or_depthwise_op(pseudo_node, ctx)

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[output_2d_name],
            outputs=[output_name],
            options={"squeezeDims": [2]},
        )
    )


def _build_conv_transpose1d_via_conv2d(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise NotImplementedError(
            f"ConvTranspose1D lowering expects rank-3 input/output. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"ConvTranspose weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 3:
        raise NotImplementedError(
            f"ConvTranspose1D weight rank must be 3. op={node.name} shape={list(weights.shape)}"
        )

    input_sig = _tensor_signature(ctx, input_name)
    output_sig = _tensor_signature(ctx, output_name)

    axis_name = ctx.add_const_tensor(
        f"{node.name}_convtranspose1d_expand_axis",
        np.asarray([2], dtype=np.int32),
    )
    input_2d_shape = [int(input_shape[0]), int(input_shape[1]), 1, int(input_shape[2])]
    input_2d_name = ctx.add_intermediate_tensor(
        f"{node.name}_convtranspose1d_input_nchw2d",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=input_2d_shape,
    )
    input_2d_sig = [int(input_sig[0]), int(input_sig[1]), 1, int(input_sig[2])]
    ctx.model_ir.tensors[input_2d_name].shape_signature = [int(v) for v in input_2d_sig]
    ctx.add_operator(
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=[input_name, axis_name],
            outputs=[input_2d_name],
        )
    )

    output_2d_shape = [int(output_shape[0]), int(output_shape[1]), 1, int(output_shape[2])]
    output_2d_name = ctx.add_intermediate_tensor(
        f"{node.name}_convtranspose1d_output_nchw2d",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=output_2d_shape,
    )
    output_2d_sig = [int(output_sig[0]), int(output_sig[1]), 1, int(output_sig[2])]
    ctx.model_ir.tensors[output_2d_name].shape_signature = [int(v) for v in output_2d_sig]

    weights_2d_name = ctx.add_const_tensor(
        f"{node.name}_convtranspose1d_filter_2d",
        np.expand_dims(weights, axis=2),
    )

    kernel_1d = int(weights.shape[2])
    pads_1d = _extract_1d_pads(
        node.attrs.get("pads", [0, 0]),
        op_name="ConvTranspose",
        node_name=str(node.name),
    )
    strides_1d = [int(v) for v in list(node.attrs.get("strides", [1]))]
    dilations_1d = [int(v) for v in list(node.attrs.get("dilations", [1]))]
    output_padding_1d = [int(v) for v in list(node.attrs.get("output_padding", [0]))]
    if len(strides_1d) != 1 or len(dilations_1d) != 1:
        raise NotImplementedError(
            f"ConvTranspose1D strides/dilations must be length 1. op={node.name} strides={strides_1d} dilations={dilations_1d}"
        )
    if len(output_padding_1d) == 0:
        output_padding_1d = [0]
    elif len(output_padding_1d) == 2:
        output_padding_1d = [int(output_padding_1d[1])]
    elif len(output_padding_1d) != 1:
        raise NotImplementedError(
            f"ConvTranspose1D output_padding must have length 1. op={node.name} output_padding={output_padding_1d}"
        )
    if output_padding_1d[0] < 0 or output_padding_1d[0] >= int(strides_1d[0]):
        raise NotImplementedError(
            "ConvTranspose1D output_padding must satisfy "
            f"0 <= output_padding < stride. op={node.name} output_padding={output_padding_1d} strides={strides_1d}"
        )

    attrs_2d = dict(node.attrs)
    attrs_2d["kernel_shape"] = [1, kernel_1d]
    attrs_2d["pads"] = [0, int(pads_1d[0]), 0, int(pads_1d[1])]
    attrs_2d["strides"] = [1, int(strides_1d[0])]
    attrs_2d["dilations"] = [1, int(dilations_1d[0])]
    attrs_2d["output_padding"] = [0, int(output_padding_1d[0])]

    pseudo_inputs = [input_2d_name, weights_2d_name]
    if len(node.inputs) >= 3:
        pseudo_inputs.append(node.inputs[2].name)
    pseudo_node = _make_pseudo_node(
        base_node=node,
        input_names=pseudo_inputs,
        output_name=output_2d_name,
        attrs=attrs_2d,
    )
    build_conv_transpose_op(pseudo_node, ctx)

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[output_2d_name],
            outputs=[output_name],
            options={"squeezeDims": [2]},
        )
    )


def build_col2im_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    image_shape_name = node.inputs[1].name
    block_shape_name = node.inputs[2].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(image_shape_name)
    ctx.ensure_tensor(block_shape_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 3 or len(output_shape) != 4:
        raise NotImplementedError(
            f"Col2Im expects input rank=3 and output rank=4. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in input_shape + output_shape):
        raise NotImplementedError(
            f"Col2Im requires static positive input/output shapes in flatbuffer_direct. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    supported_dtypes = {"FLOAT16", "FLOAT32"}
    if input_dtype not in supported_dtypes or output_dtype not in supported_dtypes:
        raise NotImplementedError(
            "Col2Im currently supports FLOAT16/FLOAT32 input/output in flatbuffer_direct. "
            f"op={node.name} input_dtype={input_dtype} output_dtype={output_dtype}"
        )

    compute_dtype = "FLOAT32" if "FLOAT32" in {input_dtype, output_dtype} else "FLOAT16"
    compute_np_dtype = np.float32 if compute_dtype == "FLOAT32" else np.float16

    image_shape_values = ctx.get_constant_array(image_shape_name)
    if image_shape_values is None:
        raise NotImplementedError(
            f"Col2Im image_shape input must be constant for flatbuffer_direct. op={node.name}"
        )
    image_shape_values = np.asarray(image_shape_values).reshape(-1)
    if int(image_shape_values.size) != 2:
        raise NotImplementedError(
            f"Col2Im image_shape must contain 2 elements [H, W]. op={node.name} shape={list(image_shape_values.shape)}"
        )
    h_img = int(image_shape_values[0])
    w_img = int(image_shape_values[1])
    if h_img <= 0 or w_img <= 0:
        raise NotImplementedError(
            f"Col2Im image_shape values must be > 0. op={node.name} image_shape={[h_img, w_img]}"
        )

    block_shape_values = ctx.get_constant_array(block_shape_name)
    if block_shape_values is None:
        raise NotImplementedError(
            f"Col2Im block_shape input must be constant for flatbuffer_direct. op={node.name}"
        )
    block_shape_values = np.asarray(block_shape_values).reshape(-1)
    if int(block_shape_values.size) != 2:
        raise NotImplementedError(
            f"Col2Im block_shape must contain 2 elements [kH, kW]. op={node.name} shape={list(block_shape_values.shape)}"
        )
    k_h = int(block_shape_values[0])
    k_w = int(block_shape_values[1])
    if k_h <= 0 or k_w <= 0:
        raise NotImplementedError(
            f"Col2Im block_shape values must be > 0. op={node.name} block_shape={[k_h, k_w]}"
        )

    dilations = _normalize_spatial_pair(
        node.attrs.get("dilations", [1, 1]),
        default=1,
        label="dilations",
        node_name=str(node.name),
    )
    strides = _normalize_spatial_pair(
        node.attrs.get("strides", [1, 1]),
        default=1,
        label="strides",
        node_name=str(node.name),
    )
    pads = _normalize_col2im_pads(node.attrs.get("pads", [0, 0, 0, 0]), node_name=str(node.name))
    if any(int(v) < 0 for v in list(dilations) + list(strides) + list(pads)):
        raise NotImplementedError(
            f"Col2Im dilations/strides/pads must be non-negative. op={node.name} dilations={dilations} strides={strides} pads={pads}"
        )
    if any(int(v) <= 0 for v in strides + dilations):
        raise NotImplementedError(
            f"Col2Im dilations/strides must be > 0. op={node.name} dilations={dilations} strides={strides}"
        )

    dilation_h, dilation_w = [int(v) for v in dilations]
    stride_h, stride_w = [int(v) for v in strides]
    pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in pads]
    eff_k_h = (int(k_h) - 1) * int(dilation_h) + 1
    eff_k_w = (int(k_w) - 1) * int(dilation_w) + 1
    h_pad = int(h_img) + int(pad_top) + int(pad_bottom)
    w_pad = int(w_img) + int(pad_left) + int(pad_right)
    out_h = int((int(h_pad) - int(eff_k_h)) // int(stride_h) + 1)
    out_w = int((int(w_pad) - int(eff_k_w)) // int(stride_w) + 1)
    if out_h <= 0 or out_w <= 0:
        raise NotImplementedError(
            f"Col2Im produced non-positive folded spatial shape. op={node.name} out_h={out_h} out_w={out_w}"
        )

    n = int(input_shape[0])
    dim1 = int(input_shape[1])
    dim2 = int(input_shape[2])
    k_prod = int(k_h) * int(k_w)
    out_hw = int(out_h) * int(out_w)
    canonical_valid = bool(int(dim1) % int(k_prod) == 0 and int(dim2) == int(out_hw))
    swapped_valid = bool(int(dim2) % int(k_prod) == 0 and int(dim1) == int(out_hw))
    expected_c = int(output_shape[1])
    if canonical_valid and int(dim1) // int(k_prod) != int(expected_c):
        canonical_valid = False
    if swapped_valid and int(dim2) // int(k_prod) != int(expected_c):
        swapped_valid = False
    if not canonical_valid and not swapped_valid:
        raise NotImplementedError(
            "Col2Im input layout could not be resolved as [N,C*K,L] or [N,L,C*K]. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape} "
            f"k_prod={k_prod} out_hw={out_hw}"
        )

    compute_input_name = input_name
    if input_dtype != compute_dtype:
        compute_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_col2im_input_{compute_dtype.lower()}",
            dtype=compute_dtype,
            shape=[int(v) for v in list(input_shape)],
        )
        _clone_shape_signature(ctx, input_name, compute_input_name)
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[compute_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    aligned_input_name = compute_input_name
    aligned_dim1 = int(dim1)
    aligned_dim2 = int(dim2)
    if swapped_valid and not canonical_valid:
        aligned_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_col2im_input_aligned",
            dtype=compute_dtype,
            shape=[int(n), int(dim2), int(dim1)],
        )
        make_transpose(
            ctx=ctx,
            input_name=compute_input_name,
            output_name=aligned_input_name,
            perm_values=[0, 2, 1],
        )
        aligned_dim1 = int(dim2)
        aligned_dim2 = int(dim1)

    if int(aligned_dim2) != int(out_hw) or int(aligned_dim1) % int(k_prod) != 0:
        raise NotImplementedError(
            "Col2Im aligned input is inconsistent with folded shape. "
            f"op={node.name} aligned_dim1={aligned_dim1} aligned_dim2={aligned_dim2} "
            f"k_prod={k_prod} out_hw={out_hw}"
        )
    c = int(aligned_dim1) // int(k_prod)
    if [int(output_shape[0]), int(output_shape[1]), int(output_shape[2]), int(output_shape[3])] != [
        int(n),
        int(c),
        int(h_img),
        int(w_img),
    ]:
        raise NotImplementedError(
            f"Col2Im output shape mismatch. op={node.name} "
            f"expected={[n, c, h_img, w_img]} actual={output_shape}"
        )

    cols_nckhw_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_cols_nckhw",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(k_prod), int(out_h), int(out_w)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=aligned_input_name,
        output_name=cols_nckhw_name,
        new_shape=[int(n), int(c), int(k_prod), int(out_h), int(out_w)],
    )

    # ONNX Col2Im canonical lowering:
    # [N, C, K, OH, OW] -> [N, C, OH, OW, K] -> [N*C, OH, OW, K] -> [N*C, K, OH, OW]
    cols_nchwk_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_cols_nchwk",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w), int(k_prod)],
    )
    make_transpose(
        ctx=ctx,
        input_name=cols_nckhw_name,
        output_name=cols_nchwk_name,
        perm_values=[0, 1, 3, 4, 2],
    )

    cols_folded_nhwc_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_cols_folded_nhwc",
        dtype=compute_dtype,
        shape=[int(n) * int(c), int(out_h), int(out_w), int(k_prod)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=cols_nchwk_name,
        output_name=cols_folded_nhwc_name,
        new_shape=[int(n) * int(c), int(out_h), int(out_w), int(k_prod)],
    )

    cols_folded_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_cols_folded_nchw",
        dtype=compute_dtype,
        shape=[int(n) * int(c), int(k_prod), int(out_h), int(out_w)],
    )
    make_transpose(
        ctx=ctx,
        input_name=cols_folded_nhwc_name,
        output_name=cols_folded_name,
        perm_values=[0, 3, 1, 2],
    )

    ky = np.repeat(np.arange(int(k_h), dtype=np.int32), int(k_w))
    kx = np.tile(np.arange(int(k_w), dtype=np.int32), int(k_h))
    positions = ky * int(dilation_h) * int(eff_k_w) + kx * int(dilation_w)
    one_hot = np.eye(int(eff_k_h) * int(eff_k_w), dtype=np.float32)[positions]
    kernel_cin_cout_hw = np.reshape(one_hot, [int(k_prod), int(eff_k_h), int(eff_k_w)])
    kernel_cin_cout_hw = np.expand_dims(kernel_cin_cout_hw, axis=1)
    kernel_cin_cout_hw = np.asarray(kernel_cin_cout_hw, dtype=compute_np_dtype)
    kernel_name = ctx.add_const_tensor(
        f"{node.name}_col2im_transpose_conv_kernel",
        kernel_cin_cout_hw,
    )

    deconv_output_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_transpose_conv_out",
        dtype=compute_dtype,
        shape=[int(n) * int(c), 1, int(h_pad), int(w_pad)],
    )
    pseudo_node = _make_pseudo_node(
        base_node=node,
        input_names=[cols_folded_name, kernel_name],
        output_name=deconv_output_name,
        attrs={
            "group": 1,
            "kernel_shape": [int(eff_k_h), int(eff_k_w)],
            "strides": [int(stride_h), int(stride_w)],
            "dilations": [1, 1],
            "pads": [0, 0, 0, 0],
            "output_padding": [0, 0],
            "auto_pad": "VALID",
        },
    )
    build_conv_transpose_op(pseudo_node, ctx)

    folded_padded_name = ctx.add_intermediate_tensor(
        f"{node.name}_col2im_folded_padded",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(h_pad), int(w_pad)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=deconv_output_name,
        output_name=folded_padded_name,
        new_shape=[int(n), int(c), int(h_pad), int(w_pad)],
    )

    output_compute_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
        f"{node.name}_col2im_output_{compute_dtype.lower()}",
        dtype=compute_dtype,
        shape=[int(v) for v in list(output_shape)],
    )
    if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
        crop_begin = ctx.add_const_tensor(
            f"{node.name}_col2im_crop_begin",
            np.asarray([0, 0, int(pad_top), int(pad_left)], dtype=np.int32),
        )
        crop_size = ctx.add_const_tensor(
            f"{node.name}_col2im_crop_size",
            np.asarray([int(n), int(c), int(h_img), int(w_img)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[folded_padded_name, crop_begin, crop_size],
                outputs=[output_compute_name],
            )
        )
    else:
        _add_reshape_operator(
            ctx=ctx,
            input_name=folded_padded_name,
            output_name=output_compute_name,
            new_shape=[int(n), int(c), int(h_img), int(w_img)],
        )

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[output_compute_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def _dummy_input_from_ort_meta(meta: Any) -> np.ndarray:
    shape = []
    for dim in list(getattr(meta, "shape", [])):
        if isinstance(dim, int) and dim > 0:
            shape.append(int(dim))
        else:
            shape.append(1)
    type_str = str(getattr(meta, "type", "tensor(float)")).lower()
    if "float16" in type_str:
        dtype = np.float16
    elif "float" in type_str:
        dtype = np.float32
    elif "int64" in type_str:
        dtype = np.int64
    elif "int32" in type_str:
        dtype = np.int32
    elif "int16" in type_str:
        dtype = np.int16
    elif "int8" in type_str:
        dtype = np.int8
    elif "uint8" in type_str:
        dtype = np.uint8
    elif "bool" in type_str:
        dtype = np.bool_
    else:
        dtype = np.float32
    return np.zeros(shape, dtype=dtype)


def _infer_convtranspose_io_shapes_with_onnxruntime(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> tuple[list[int] | None, list[int] | None]:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return None, None
    try:
        import onnx
        import onnxruntime as ort
    except Exception:
        return None, None
    try:
        work_model = onnx.ModelProto()
        work_model.CopyFrom(onnx_model)
        existing_outputs = {str(v.name) for v in list(work_model.graph.output)}
        for tensor_name in [input_name, output_name]:
            if str(tensor_name) in existing_outputs:
                continue
            work_model.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    str(tensor_name),
                    onnx.TensorProto.FLOAT,
                    None,
                )
            )
        sess = ort.InferenceSession(
            work_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        feed = {
            inp.name: _dummy_input_from_ort_meta(inp)
            for inp in sess.get_inputs()
        }
        outputs = sess.run([str(input_name), str(output_name)], feed)
        input_shape = [int(v) for v in list(np.asarray(outputs[0]).shape)]
        output_shape = [int(v) for v in list(np.asarray(outputs[1]).shape)]
        return input_shape, output_shape
    except Exception:
        return None, None


def _infer_conv_transpose_output_shape_nchw(
    *,
    node: Any,
    input_shape_nchw: list[int],
    weights: np.ndarray,
) -> list[int]:
    group = int(node.attrs.get("group", 1))
    kernel_shape_attr = node.attrs.get("kernel_shape", None)
    if kernel_shape_attr is None:
        kernel_h = int(weights.shape[2])
        kernel_w = int(weights.shape[3])
    else:
        kernel_h, kernel_w = [int(v) for v in list(kernel_shape_attr)]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(strides) == 0:
        strides = [1, 1]
    elif len(strides) == 1:
        strides = [int(strides[0]), int(strides[0])]
    elif len(strides) != 2:
        raise NotImplementedError(
            f"ConvTranspose strides must have length 2 for shape inference. op={node.name} strides={strides}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
    if len(output_padding) == 0:
        output_padding = [0, 0]
    elif len(output_padding) == 1:
        output_padding = [int(output_padding[0]), int(output_padding[0])]
    elif len(output_padding) != 2:
        raise NotImplementedError(
            "ConvTranspose output_padding must have length 2 for shape inference. "
            f"op={node.name} output_padding={output_padding}"
        )

    in_n = int(input_shape_nchw[0])
    in_h = int(input_shape_nchw[2])
    in_w = int(input_shape_nchw[3])
    out_c = int(weights.shape[1]) * int(group)

    eff_kh = (kernel_h - 1) * int(dilations[0]) + 1
    eff_kw = (kernel_w - 1) * int(dilations[1]) + 1
    out_h = int(strides[0]) * (in_h - 1) + eff_kh + int(output_padding[0]) - int(pads[0]) - int(pads[2])
    out_w = int(strides[1]) * (in_w - 1) + eff_kw + int(output_padding[1]) - int(pads[1]) - int(pads[3])
    return [in_n, out_c, out_h, out_w]


def _infer_rank4_conv_output_signature(
    *,
    input_signature_nchw: list[int],
    output_shape_nchw: list[int],
    existing_output_signature_nchw: list[int] | None = None,
) -> list[int]:
    signature = [int(v) for v in list(output_shape_nchw)]
    if len(signature) != 4:
        return signature
    if existing_output_signature_nchw is not None and len(existing_output_signature_nchw) == 4:
        for axis in range(4):
            if int(existing_output_signature_nchw[axis]) < 0:
                signature[axis] = -1
    if len(input_signature_nchw) == 4:
        if int(input_signature_nchw[0]) < 0:
            signature[0] = -1
        if int(input_signature_nchw[2]) < 0:
            signature[2] = -1
        if int(input_signature_nchw[3]) < 0:
            signature[3] = -1
    return [int(v) for v in signature]


def _infer_conv2d_output_shape_nchw(
    *,
    node: Any,
    input_shape_nchw: list[int],
    weights: np.ndarray,
) -> list[int]:
    kernel_shape_attr = node.attrs.get("kernel_shape", None)
    if kernel_shape_attr is None:
        kernel_h = int(weights.shape[2])
        kernel_w = int(weights.shape[3])
    else:
        kernel_h, kernel_w = [int(v) for v in list(kernel_shape_attr)]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(raw_pads) < 4:
        raw_pads = [0, 0, 0, 0]
    pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in raw_pads[:4]]
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()

    n = int(input_shape_nchw[0]) if len(input_shape_nchw) > 0 else 1
    in_h = int(input_shape_nchw[2]) if len(input_shape_nchw) > 2 else -1
    in_w = int(input_shape_nchw[3]) if len(input_shape_nchw) > 3 else -1
    out_c = int(weights.shape[0])

    eff_kh = int((kernel_h - 1) * int(dilations[0]) + 1)
    eff_kw = int((kernel_w - 1) * int(dilations[1]) + 1)

    def _infer_dim(in_dim: int, eff_k: int, stride: int, pad_before: int, pad_after: int) -> int:
        if int(in_dim) <= 0:
            return -1
        if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            return int(np.ceil(float(in_dim) / float(stride)))
        numer = int(in_dim) + int(pad_before) + int(pad_after) - int(eff_k)
        return int(np.floor(float(numer) / float(stride)) + 1)

    out_h = _infer_dim(in_h, eff_kh, int(strides[0]), pad_top, pad_bottom)
    out_w = _infer_dim(in_w, eff_kw, int(strides[1]), pad_left, pad_right)
    return [int(n), int(out_c), int(out_h), int(out_w)]


def _materialize_fusedconv_placeholder_output_shape(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    weight_name: str,
    output_name: str,
) -> None:
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is None:
        return
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if output_shape != [1]:
        return
    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    if len(input_shape) != 4:
        return
    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        return
    weights = np.asarray(weights)
    if weights.ndim != 4:
        return

    inferred = _infer_conv2d_output_shape_nchw(
        node=node,
        input_shape_nchw=input_shape,
        weights=weights,
    )
    output_tensor.shape = [int(v) if int(v) > 0 else 1 for v in inferred]
    output_tensor.shape_signature = [int(v) if int(v) > 0 else -1 for v in inferred]


def _resolve_conv_transpose_padding(node: Any) -> str:
    auto_pad_raw = node.attrs.get("auto_pad", "NOTSET")
    auto_pad = (
        auto_pad_raw.decode("utf-8")
        if isinstance(auto_pad_raw, (bytes, bytearray))
        else str(auto_pad_raw)
    )
    auto_pad = str(auto_pad).upper()
    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        return "SAME"
    if auto_pad in ["VALID", "NOTSET"]:
        # flatbuffer_direct feeds explicit output_shape to TRANSPOSE_CONV,
        # so VALID is usable even when ONNX pads are non-zero.
        return "VALID"
    raise NotImplementedError(
        "ConvTranspose currently supports auto_pad=SAME_* or zero pads with auto_pad in {NOTSET,VALID}. "
        f"op={node.name} auto_pad={auto_pad_raw} pads={node.attrs.get('pads', [0,0,0,0])}"
    )


def _resolve_conv_padding_and_explicit_pads(
    *,
    node: Any,
    input_shape_nchw: list[int],
    output_shape_nchw: list[int],
) -> tuple[str, list[int] | None]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(raw_pads) < 4:
        raw_pads = [0, 0, 0, 0]
    pads = [int(raw_pads[0]), int(raw_pads[1]), int(raw_pads[2]), int(raw_pads[3])]
    pad_top, pad_left, pad_bottom, pad_right = pads
    pads_axes_opposite_same = bool((pad_top == pad_bottom) and (pad_left == pad_right))

    if auto_pad == "NOTSET":
        # SAME is safe only when it preserves tensor extent and padding orientation
        # does not affect sampled coordinates.
        if (
            pads_axes_opposite_same
            and len(input_shape_nchw) == 4
            and len(output_shape_nchw) == 4
            and list(input_shape_nchw[2:]) == list(output_shape_nchw[2:])
        ):
            return "SAME", None
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None

    if auto_pad == "SAME_UPPER":
        return "SAME", None
    if auto_pad == "VALID":
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None
    if auto_pad == "SAME_LOWER":
        raise NotImplementedError(
            f"Conv auto_pad=SAME_LOWER is not supported in flatbuffer_direct. op={node.name}"
        )
    raise NotImplementedError(
        f"Conv auto_pad attribute is invalid for flatbuffer_direct. op={node.name} auto_pad={auto_pad}"
    )


def _resolve_conv3d_padding_and_explicit_pads(
    *,
    node: Any,
    input_shape_ncdhw: list[int],
    output_shape_ncdhw: list[int],
) -> tuple[str, list[int] | None]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0, 0, 0]))]
    if len(raw_pads) < 6:
        raw_pads = [0, 0, 0, 0, 0, 0]
    pads = [int(raw_pads[0]), int(raw_pads[1]), int(raw_pads[2]), int(raw_pads[3]), int(raw_pads[4]), int(raw_pads[5])]
    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = pads
    pads_axes_opposite_same = bool(
        (pad_front == pad_back)
        and (pad_top == pad_bottom)
        and (pad_left == pad_right)
    )

    if auto_pad == "NOTSET":
        if (
            pads_axes_opposite_same
            and len(input_shape_ncdhw) == 5
            and len(output_shape_ncdhw) == 5
            and list(input_shape_ncdhw[2:]) == list(output_shape_ncdhw[2:])
        ):
            return "SAME", None
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None

    if auto_pad == "SAME_UPPER":
        return "SAME", None
    if auto_pad == "VALID":
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None
    if auto_pad == "SAME_LOWER":
        raise NotImplementedError(
            f"Conv auto_pad=SAME_LOWER is not supported in flatbuffer_direct. op={node.name}"
        )
    raise NotImplementedError(
        f"Conv auto_pad attribute is invalid for flatbuffer_direct. op={node.name} auto_pad={auto_pad}"
    )


def _build_conv_transpose3d_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 5 or len(output_shape) != 5:
        raise NotImplementedError(
            f"ConvTranspose3D supports rank-5 input/output only in flatbuffer_direct. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in output_shape):
        raise NotImplementedError(
            f"ConvTranspose3D requires static positive output shape in flatbuffer_direct. op={node.name} output_shape={output_shape}"
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"ConvTranspose weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 5:
        raise NotImplementedError(
            f"ConvTranspose3D weight rank must be 5. op={node.name} weight_shape={list(weights.shape)}"
        )

    group = int(node.attrs.get("group", 1))
    if group != 1:
        raise NotImplementedError(
            f"ConvTranspose3D currently supports group=1 only. op={node.name} group={group}"
        )

    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1, 1]))]
    if len(strides) == 0:
        strides = [1, 1, 1]
    elif len(strides) == 1:
        strides = [int(strides[0]), int(strides[0]), int(strides[0])]
    elif len(strides) != 3:
        raise NotImplementedError(
            f"ConvTranspose3D strides must have length 3 in flatbuffer_direct. op={node.name} strides={strides}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1, 1]))]
    if len(dilations) == 0:
        dilations = [1, 1, 1]
    elif len(dilations) == 1:
        dilations = [int(dilations[0]), int(dilations[0]), int(dilations[0])]
    elif len(dilations) != 3:
        raise NotImplementedError(
            f"ConvTranspose3D dilations must have length 3 in flatbuffer_direct. op={node.name} dilations={dilations}"
        )
    if dilations != [1, 1, 1]:
        raise NotImplementedError(
            f"ConvTranspose3D dilations must be [1,1,1] in flatbuffer_direct. op={node.name} dilations={dilations}"
        )
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
    if len(output_padding) == 0:
        output_padding = [0, 0, 0]
    elif len(output_padding) == 1:
        output_padding = [int(output_padding[0]), int(output_padding[0]), int(output_padding[0])]
    elif len(output_padding) != 3:
        raise NotImplementedError(
            "ConvTranspose3D output_padding must have length 3 in flatbuffer_direct. "
            f"op={node.name} output_padding={output_padding}"
        )
    if any(v < 0 for v in output_padding):
        raise NotImplementedError(
            f"ConvTranspose3D output_padding must be non-negative in flatbuffer_direct. op={node.name} output_padding={output_padding}"
        )
    if any(int(v) >= int(s) for v, s in zip(output_padding, strides)):
        raise NotImplementedError(
            "ConvTranspose3D output_padding must satisfy "
            f"0 <= output_padding < stride in flatbuffer_direct. op={node.name} output_padding={output_padding} strides={strides}"
        )

    # ONNX ConvTranspose3D weights are [C_in, C_out/group, kD, kH, kW].
    # TFLite CONV_3D_TRANSPOSE expects [kD, kH, kW, C_out, C_in].
    w_deconv = np.transpose(weights, (2, 3, 4, 1, 0)).astype(np.float32)
    w_name = ctx.add_const_tensor(
        f"{node.name}_conv3d_transpose_filter",
        w_deconv,
    )

    ndhwc_input_shape = [
        int(input_shape[0]),
        int(input_shape[2]),
        int(input_shape[3]),
        int(input_shape[4]),
        int(input_shape[1]),
    ]
    ndhwc_output_shape = [
        int(output_shape[0]),
        int(output_shape[2]),
        int(output_shape[3]),
        int(output_shape[4]),
        int(output_shape[1]),
    ]
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0, 0, 0]))]
    if len(raw_pads) < 6:
        raw_pads = [0, 0, 0, 0, 0, 0]
    pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = [int(v) for v in raw_pads[:6]]
    needs_spatial_crop = any(
        int(v) != 0 for v in [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]
    )
    ndhwc_transpose_conv_output_shape = [int(v) for v in list(ndhwc_output_shape)]
    if needs_spatial_crop:
        raw_out_d = int(ndhwc_output_shape[1]) + int(pad_front) + int(pad_back) - int(output_padding[0])
        raw_out_h = int(ndhwc_output_shape[2]) + int(pad_top) + int(pad_bottom) - int(output_padding[1])
        raw_out_w = int(ndhwc_output_shape[3]) + int(pad_left) + int(pad_right) - int(output_padding[2])
        if raw_out_d <= 0 or raw_out_h <= 0 or raw_out_w <= 0:
            raise NotImplementedError(
                "ConvTranspose3D explicit pads produce invalid pre-crop output shape. "
                f"op={node.name} output_shape={ndhwc_output_shape} pads={raw_pads} output_padding={output_padding}"
            )
        ndhwc_transpose_conv_output_shape = [
            int(ndhwc_output_shape[0]),
            int(raw_out_d),
            int(raw_out_h),
            int(raw_out_w),
            int(ndhwc_output_shape[4]),
        ]

    x_ndhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_ndhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=ndhwc_input_shape,
    )
    x_ndhwc = make_transpose(
        ctx,
        input_name,
        x_ndhwc,
        [0, 2, 3, 4, 1],
        allow_elide_inverse_chain=True,
    )

    out_shape_name = ctx.add_const_tensor(
        f"{node.name}_conv3d_transpose_output_shape",
        np.asarray(ndhwc_transpose_conv_output_shape, dtype=np.int32),
    )

    out_channels = int(ndhwc_output_shape[4])
    bias_values = None
    if len(node.inputs) >= 3:
        bias_values = ctx.get_constant_array(node.inputs[2].name)
    if bias_values is None:
        bias_values = np.zeros((out_channels,), dtype=np.float32)
    bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
    if int(bias_values.size) != out_channels:
        raise NotImplementedError(
            f"ConvTranspose3D bias size must match output channels. op={node.name} bias_size={int(bias_values.size)} out_channels={out_channels}"
        )
    b_name = ctx.add_const_tensor(
        f"{node.name}_conv3d_transpose_bias",
        bias_values,
    )

    y_ndhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_ndhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=ndhwc_transpose_conv_output_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONV_3D_TRANSPOSE",
            inputs=[out_shape_name, w_name, x_ndhwc, b_name],
            outputs=[y_ndhwc],
            options={
                "padding": _resolve_conv_transpose_padding(node),
                "strideD": int(strides[0]),
                "strideH": int(strides[1]),
                "strideW": int(strides[2]),
                "dilationDFactor": int(dilations[0]),
                "dilationHFactor": int(dilations[1]),
                "dilationWFactor": int(dilations[2]),
                "fusedActivationFunction": "NONE",
            },
        )
    )

    y_after_crop_ndhwc = y_ndhwc
    if needs_spatial_crop:
        crop_begin_name = ctx.add_const_tensor(
            f"{node.name}_conv3d_transpose_crop_begin",
            np.asarray([0, int(pad_front), int(pad_top), int(pad_left), 0], dtype=np.int32),
        )
        crop_end_name = ctx.add_const_tensor(
            f"{node.name}_conv3d_transpose_crop_end",
            np.asarray(
                [
                    int(ndhwc_transpose_conv_output_shape[0]),
                    int(ndhwc_transpose_conv_output_shape[1]) - int(pad_back) + int(output_padding[0]),
                    int(ndhwc_transpose_conv_output_shape[2]) - int(pad_bottom) + int(output_padding[1]),
                    int(ndhwc_transpose_conv_output_shape[3]) - int(pad_right) + int(output_padding[2]),
                    int(ndhwc_transpose_conv_output_shape[4]),
                ],
                dtype=np.int32,
            ),
        )
        crop_stride_name = ctx.add_const_tensor(
            f"{node.name}_conv3d_transpose_crop_stride",
            np.asarray([1, 1, 1, 1, 1], dtype=np.int32),
        )
        y_cropped_ndhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_ndhwc_cropped",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=ndhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[y_ndhwc, crop_begin_name, crop_end_name, crop_stride_name],
                outputs=[y_cropped_ndhwc],
                options={
                    "beginMask": 0,
                    "endMask": 0,
                    "ellipsisMask": 0,
                    "newAxisMask": 0,
                    "shrinkAxisMask": 0,
                    "offset": False,
                },
            )
        )
        y_after_crop_ndhwc = y_cropped_ndhwc

    output_tensor = ctx.model_ir.tensors[output_name]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    ndhwc_output_signature = list(ndhwc_output_shape)
    if len(output_signature) == 5:
        ndhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[4]),
            int(output_signature[1]),
        ]
    ctx.model_ir.tensors[y_after_crop_ndhwc].shape_signature = [int(v) for v in ndhwc_output_signature]

    make_transpose(
        ctx,
        y_after_crop_ndhwc,
        output_name,
        [0, 4, 1, 2, 3],
    )


def build_conv_transpose_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    original_input_signature = [int(v) for v in list(input_signature)]
    original_output_signature = [int(v) for v in list(output_signature)]

    def _shape_from_rank3_signature(signature: list[int]) -> list[int] | None:
        if len(signature) != 3:
            return None
        return [int(v) if int(v) > 0 else 1 for v in list(signature)]

    weights_1d = ctx.get_constant_array(weight_name)
    rank3_input_from_signature = _shape_from_rank3_signature(input_signature)
    if len(input_shape) != 3 and rank3_input_from_signature is not None:
        input_shape = [int(v) for v in list(rank3_input_from_signature)]
        input_tensor.shape = [int(v) for v in list(input_shape)]
    rank3_output_from_signature = _shape_from_rank3_signature(output_signature)
    if len(output_shape) != 3 and rank3_output_from_signature is not None:
        output_shape = [int(v) for v in list(rank3_output_from_signature)]
        output_tensor.shape = [int(v) for v in list(output_shape)]
    if (
        weights_1d is not None
        and np.asarray(weights_1d).ndim == 3
        and (len(input_shape) != 3 or len(output_shape) != 3)
    ):
        inferred_input_shape, inferred_output_shape = _infer_convtranspose_io_shapes_with_onnxruntime(
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
        )
        if inferred_input_shape is not None and len(inferred_input_shape) == 3:
            input_shape = [int(v) for v in list(inferred_input_shape)]
            input_tensor.shape = [int(v) for v in list(input_shape)]
            if len(input_signature) == 3:
                input_tensor.shape_signature = [int(v) for v in list(input_signature)]
            else:
                input_tensor.shape_signature = [int(v) for v in list(inferred_input_shape)]
        if inferred_output_shape is not None and len(inferred_output_shape) == 3:
            output_shape = [int(v) for v in list(inferred_output_shape)]
            output_tensor.shape = [int(v) for v in list(output_shape)]
            if len(output_signature) == 3:
                output_tensor.shape_signature = [int(v) for v in list(output_signature)]
            else:
                output_tensor.shape_signature = [int(v) for v in list(inferred_output_shape)]
    if (
        weights_1d is not None
        and np.asarray(weights_1d).ndim == 3
        and (len(input_shape) != 3 or len(output_shape) != 3)
    ):
        weights_1d_arr = np.asarray(weights_1d)
        group = int(node.attrs.get("group", 1))
        in_channels = int(weights_1d_arr.shape[0])
        out_channels = int(weights_1d_arr.shape[1]) * int(group)
        if len(input_shape) != 3:
            input_shape = [1, int(in_channels), 1]
            input_tensor.shape = [int(v) for v in list(input_shape)]
            if len(input_signature) != 3:
                input_tensor.shape_signature = [-1, int(in_channels), -1]
        if len(output_shape) != 3:
            output_shape = [1, int(out_channels), 1]
            output_tensor.shape = [int(v) for v in list(output_shape)]
            if len(output_signature) != 3:
                output_tensor.shape_signature = [-1, int(out_channels), -1]
    if (
        len(input_shape) == 3
        and len(output_shape) == 3
        and weights_1d is not None
        and np.asarray(weights_1d).ndim == 3
    ):
        _build_conv_transpose1d_via_conv2d(node, ctx)
        return
    if (
        len(input_shape) == 5
        and len(output_shape) == 5
        and weights_1d is not None
        and np.asarray(weights_1d).ndim == 5
    ):
        _build_conv_transpose3d_op(node, ctx)
        return

    def _shape_from_rank4_signature(signature: list[int]) -> list[int] | None:
        if len(signature) != 4:
            return None
        return [int(v) if int(v) > 0 else 1 for v in list(signature)]

    rank4_input_from_signature = _shape_from_rank4_signature(input_signature)
    if len(input_shape) != 4 and rank4_input_from_signature is not None:
        input_shape = [int(v) for v in list(rank4_input_from_signature)]
        input_tensor.shape = [int(v) for v in list(input_shape)]

    rank4_output_from_signature = _shape_from_rank4_signature(output_signature)
    if len(output_shape) != 4 and rank4_output_from_signature is not None:
        output_shape = [int(v) for v in list(rank4_output_from_signature)]
        output_tensor.shape = [int(v) for v in list(output_shape)]

    if len(input_shape) != 4 or len(output_shape) != 4:
        inferred_input_shape, inferred_output_shape = _infer_convtranspose_io_shapes_with_onnxruntime(
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
        )
        if inferred_input_shape is not None and len(inferred_input_shape) == 4:
            input_shape = [int(v) for v in list(inferred_input_shape)]
            input_tensor.shape = [int(v) for v in list(input_shape)]
            if len(input_signature) == 4:
                input_tensor.shape_signature = [int(v) for v in list(input_signature)]
            else:
                input_tensor.shape_signature = [int(v) for v in list(inferred_input_shape)]
        if inferred_output_shape is not None and len(inferred_output_shape) == 4:
            output_shape = [int(v) for v in list(inferred_output_shape)]
            output_tensor.shape = [int(v) for v in list(output_shape)]
            if len(output_signature) == 4:
                output_tensor.shape_signature = [int(v) for v in list(output_signature)]
            else:
                output_tensor.shape_signature = [int(v) for v in list(inferred_output_shape)]
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"ConvTranspose supports rank-4 input only in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )

    if len(original_input_signature) != 4:
        input_signature = [int(input_shape[0]), int(input_shape[1]), -1, -1]
        input_tensor.shape_signature = [int(v) for v in list(input_signature)]
    else:
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_shape)]
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"ConvTranspose weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 4:
        raise NotImplementedError(
            f"ConvTranspose weight rank must be 4. op={node.name} weight_shape={list(weights.shape)}"
        )

    group = int(node.attrs.get("group", 1))
    if group != 1:
        raise NotImplementedError(
            f"ConvTranspose currently supports group=1 only. op={node.name} group={group}"
        )
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(strides) == 0:
        strides = [1, 1]
    elif len(strides) == 1:
        strides = [int(strides[0]), int(strides[0])]
    elif len(strides) != 2:
        raise NotImplementedError(
            f"ConvTranspose strides must have length 2 in flatbuffer_direct. op={node.name} strides={strides}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NotImplementedError(
            f"ConvTranspose dilations must be [1,1] in flatbuffer_direct. op={node.name} dilations={dilations}"
        )
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
    if len(output_padding) == 0:
        output_padding = [0, 0]
    elif len(output_padding) == 1:
        output_padding = [int(output_padding[0]), int(output_padding[0])]
    elif len(output_padding) != 2:
        raise NotImplementedError(
            "ConvTranspose output_padding must have length 2 in flatbuffer_direct. "
            f"op={node.name} output_padding={output_padding}"
        )
    if any(v < 0 for v in output_padding):
        raise NotImplementedError(
            f"ConvTranspose output_padding must be non-negative in flatbuffer_direct. op={node.name} output_padding={output_padding}"
        )
    if any(int(v) >= int(s) for v, s in zip(output_padding, strides)):
        raise NotImplementedError(
            "ConvTranspose output_padding must satisfy "
            f"0 <= output_padding < stride in flatbuffer_direct. op={node.name} output_padding={output_padding} strides={strides}"
        )

    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(output_shape) != 4:
        output_shape = _infer_conv_transpose_output_shape_nchw(
            node=node,
            input_shape_nchw=input_shape,
            weights=weights,
        )
        output_tensor.shape = [int(v) for v in list(output_shape)]
        if len(output_signature) == 4:
            output_tensor.shape_signature = [int(v) for v in list(output_signature)]
        else:
            output_tensor.shape_signature = [int(v) for v in list(output_shape)]
    output_shape = [int(v) for v in list(output_shape)]

    if any(int(v) <= 0 for v in output_shape):
        raise NotImplementedError(
            f"ConvTranspose requires static positive output shape in flatbuffer_direct. op={node.name} output_shape={output_shape}"
        )

    if len(original_output_signature) != 4:
        output_signature = [int(output_shape[0]), int(output_shape[1]), -1, -1]
        output_tensor.shape_signature = [int(v) for v in list(output_signature)]
    else:
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_shape)]
        )

    # ONNX ConvTranspose weights are [C_in, C_out/group, kH, kW].
    # TFLite TRANSPOSE_CONV expects [C_out, kH, kW, C_in].
    w_deconv = np.transpose(weights, (1, 2, 3, 0)).astype(np.float32)
    w_name = ctx.add_const_tensor(
        f"{node.name}_transpose_conv_filter",
        w_deconv,
    )

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(raw_pads) < 4:
        raw_pads = [0, 0, 0, 0]
    pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in raw_pads[:4]]
    needs_spatial_crop = any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right])
    nhwc_transpose_conv_output_shape = [int(v) for v in list(nhwc_output_shape)]
    if needs_spatial_crop:
        raw_out_h = int(nhwc_output_shape[1]) + int(pad_top) + int(pad_bottom) - int(output_padding[0])
        raw_out_w = int(nhwc_output_shape[2]) + int(pad_left) + int(pad_right) - int(output_padding[1])
        if raw_out_h <= 0 or raw_out_w <= 0:
            raise NotImplementedError(
                f"ConvTranspose explicit pads produce invalid pre-crop output shape. "
                f"op={node.name} output_shape={nhwc_output_shape} pads={raw_pads} output_padding={output_padding}"
            )
        nhwc_transpose_conv_output_shape = [
            int(nhwc_output_shape[0]),
            int(raw_out_h),
            int(raw_out_w),
            int(nhwc_output_shape[3]),
        ]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    use_dynamic_output_shape = (
        len(original_output_signature) != 4
        or any(int(v) <= 0 for v in list(original_output_signature))
    )
    if use_dynamic_output_shape and needs_spatial_crop:
        pads_are_symmetric = (
            int(pad_top) == int(pad_bottom)
            and int(pad_left) == int(pad_right)
        )
        if (not pads_are_symmetric) or any(int(v) != 0 for v in output_padding):
            raise NotImplementedError(
                "ConvTranspose with explicit pads requires static output shape in flatbuffer_direct "
                "unless pads are symmetric and output_padding is zero. "
                f"op={node.name} output_signature={original_output_signature} "
                f"pads={raw_pads} output_padding={output_padding}"
            )
        # For symmetric explicit pads, dynamic output-shape math below already
        # accounts for pads. Skip explicit crop to keep the shape dynamic.
        needs_spatial_crop = False
        nhwc_transpose_conv_output_shape = [int(v) for v in list(nhwc_output_shape)]
    if use_dynamic_output_shape:
        kernel_shape_attr = node.attrs.get("kernel_shape", None)
        if kernel_shape_attr is None:
            kernel_h = int(weights.shape[2])
            kernel_w = int(weights.shape[3])
        else:
            kernel_h, kernel_w = [int(v) for v in list(kernel_shape_attr)]
        pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
        eff_kh = (int(kernel_h) - 1) * int(dilations[0]) + 1
        eff_kw = (int(kernel_w) - 1) * int(dilations[1]) + 1
        adjust_h = int(eff_kh + int(output_padding[0]) - int(pads[0]) - int(pads[2]))
        adjust_w = int(eff_kw + int(output_padding[1]) - int(pads[1]) - int(pads[3]))

        shape_vec = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_input_shape_vec",
            dtype="INT32",
            shape=[4],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[x_nhwc],
                outputs=[shape_vec],
                options={"outType": "INT32"},
            )
        )

        def _slice_dim(idx: int, suffix: str) -> str:
            begin_name = ctx.add_const_tensor(
                f"{node.name}_transpose_conv_shape_begin_{suffix}",
                np.asarray([int(idx)], dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{node.name}_transpose_conv_shape_size_{suffix}",
                np.asarray([1], dtype=np.int32),
            )
            out_name = ctx.add_intermediate_tensor(
                f"{node.name}_transpose_conv_dim_{suffix}",
                dtype="INT32",
                shape=[1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[shape_vec, begin_name, size_name],
                    outputs=[out_name],
                )
            )
            return out_name

        n_vec = _slice_dim(0, "n")
        h_vec = _slice_dim(1, "h")
        w_vec = _slice_dim(2, "w")
        one_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_one",
            np.asarray([1], dtype=np.int32),
        )
        stride_h_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_stride_h",
            np.asarray([int(strides[0])], dtype=np.int32),
        )
        stride_w_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_stride_w",
            np.asarray([int(strides[1])], dtype=np.int32),
        )
        adjust_h_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_adjust_h",
            np.asarray([int(adjust_h)], dtype=np.int32),
        )
        adjust_w_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_adjust_w",
            np.asarray([int(adjust_w)], dtype=np.int32),
        )
        out_c_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_out_c",
            np.asarray([int(nhwc_output_shape[3])], dtype=np.int32),
        )

        h_minus_one = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_h_minus_one",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[h_vec, one_vec],
                outputs=[h_minus_one],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        h_scaled = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_h_scaled",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[h_minus_one, stride_h_vec],
                outputs=[h_scaled],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        out_h_vec = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_out_h",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[h_scaled, adjust_h_vec],
                outputs=[out_h_vec],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        w_minus_one = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_w_minus_one",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[w_vec, one_vec],
                outputs=[w_minus_one],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        w_scaled = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_w_scaled",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[w_minus_one, stride_w_vec],
                outputs=[w_scaled],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        out_w_vec = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_out_w",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[w_scaled, adjust_w_vec],
                outputs=[out_w_vec],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        out_shape_name = ctx.add_intermediate_tensor(
            f"{node.name}_transpose_conv_output_shape",
            dtype="INT32",
            shape=[4],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[n_vec, out_h_vec, out_w_vec, out_c_vec],
                outputs=[out_shape_name],
                options={
                    "axis": 0,
                    "fusedActivationFunction": "NONE",
                },
            )
        )
    else:
        out_shape_name = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_output_shape",
            np.asarray(nhwc_transpose_conv_output_shape, dtype=np.int32),
        )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_transpose_conv_output_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE_CONV",
            inputs=[out_shape_name, w_name, x_nhwc],
            outputs=[y_nhwc],
                options={
                    "padding": _resolve_conv_transpose_padding(node),
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                },
            )
        )

    y_after_crop_nhwc = y_nhwc
    if needs_spatial_crop:
        crop_begin_name = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_crop_begin",
            np.asarray([0, int(pad_top), int(pad_left), 0], dtype=np.int32),
        )
        crop_end_name = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_crop_end",
            np.asarray(
                [
                    int(nhwc_transpose_conv_output_shape[0]),
                    int(nhwc_transpose_conv_output_shape[1]) - int(pad_bottom) + int(output_padding[0]),
                    int(nhwc_transpose_conv_output_shape[2]) - int(pad_right) + int(output_padding[1]),
                    int(nhwc_transpose_conv_output_shape[3]),
                ],
                dtype=np.int32,
            ),
        )
        crop_stride_name = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_crop_stride",
            np.asarray([1, 1, 1, 1], dtype=np.int32),
        )
        y_cropped_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc_cropped",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=nhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[y_nhwc, crop_begin_name, crop_end_name, crop_stride_name],
                outputs=[y_cropped_nhwc],
                options={
                    "beginMask": 0,
                    "endMask": 0,
                    "ellipsisMask": 0,
                    "newAxisMask": 0,
                    "shrinkAxisMask": 0,
                    "offset": False,
                },
            )
        )
        y_after_crop_nhwc = y_cropped_nhwc

    y_final_nhwc = y_after_crop_nhwc
    if len(node.inputs) >= 3:
        bias_name = node.inputs[2].name
        bias_values = ctx.get_constant_array(bias_name)
        if bias_values is None:
            raise NotImplementedError(
                f"ConvTranspose bias must be constant when provided. op={node.name}"
            )
        bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
        out_channels = int(nhwc_output_shape[3])
        if int(bias_values.size) != out_channels:
            raise NotImplementedError(
                f"ConvTranspose bias size must match output channels. "
                f"op={node.name} bias_size={int(bias_values.size)} out_channels={out_channels}"
            )
        bias_const = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_bias",
            bias_values.reshape(1, 1, 1, out_channels).astype(np.float32),
        )
        y_bias_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc_bias",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=nhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[y_after_crop_nhwc, bias_const],
                outputs=[y_bias_nhwc],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        y_final_nhwc = y_bias_nhwc

    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    nhwc_output_signature = list(nhwc_output_shape)
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]
    elif len(input_signature) == 4:
        nhwc_output_signature = [
            int(input_signature[0]),
            -1,
            -1,
            int(nhwc_output_shape[3]),
        ]
    ctx.model_ir.tensors[y_final_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]

    make_transpose(
        ctx,
        y_final_nhwc,
        output_name,
        [0, 3, 1, 2],
    )


def _build_conv3d_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 5 or len(output_shape) != 5:
        raise NotImplementedError(
            f"Only 3D Conv (rank=5) is supported by CONV_3D lowering. op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"Conv weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 5:
        raise NotImplementedError(
            f"Conv3D weight rank must be 5. op={node.name} shape={weights.shape}"
        )

    group = int(node.attrs.get("group", 1))
    if group != 1:
        raise NotImplementedError(
            f"Conv3D currently supports group=1 only. op={node.name} group={group}"
        )

    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1, 1]))]
    if len(strides) == 0:
        strides = [1, 1, 1]
    elif len(strides) == 1:
        strides = [int(strides[0]), int(strides[0]), int(strides[0])]
    elif len(strides) != 3:
        raise NotImplementedError(
            f"Conv3D strides must have length 3 in flatbuffer_direct. op={node.name} strides={strides}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1, 1]))]
    if len(dilations) == 0:
        dilations = [1, 1, 1]
    elif len(dilations) == 1:
        dilations = [int(dilations[0]), int(dilations[0]), int(dilations[0])]
    elif len(dilations) != 3:
        raise NotImplementedError(
            f"Conv3D dilations must have length 3 in flatbuffer_direct. op={node.name} dilations={dilations}"
        )

    padding, explicit_pads = _resolve_conv3d_padding_and_explicit_pads(
        node=node,
        input_shape_ncdhw=input_shape,
        output_shape_ncdhw=output_shape,
    )
    ndhwc_input_shape = [
        int(input_shape[0]),
        int(input_shape[2]),
        int(input_shape[3]),
        int(input_shape[4]),
        int(input_shape[1]),
    ]
    ndhwc_output_shape = [
        int(output_shape[0]),
        int(output_shape[2]),
        int(output_shape[3]),
        int(output_shape[4]),
        int(output_shape[1]),
    ]

    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    ndhwc_output_signature = list(ndhwc_output_shape)
    if len(output_signature) == 5:
        ndhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[4]),
            int(output_signature[1]),
        ]
        if len(input_signature) == 5:
            if int(input_signature[0]) < 0:
                ndhwc_output_signature[0] = -1
            if int(input_signature[2]) < 0:
                ndhwc_output_signature[1] = -1
            if int(input_signature[3]) < 0:
                ndhwc_output_signature[2] = -1
            if int(input_signature[4]) < 0:
                ndhwc_output_signature[3] = -1

    x_ndhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_ndhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=ndhwc_input_shape,
    )
    x_ndhwc = make_transpose(
        ctx,
        input_name,
        x_ndhwc,
        [0, 2, 3, 4, 1],
        allow_elide_inverse_chain=True,
    )
    x_ndhwc_conv = x_ndhwc
    if explicit_pads is not None:
        pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right = [int(v) for v in explicit_pads]
        if any(int(v) != 0 for v in [pad_front, pad_top, pad_left, pad_back, pad_bottom, pad_right]):
            x_tensor = ctx.model_ir.tensors[x_ndhwc_conv]
            padded_shape = list(x_tensor.shape)
            padded_shape[1] = int(padded_shape[1]) + int(pad_front) + int(pad_back)
            padded_shape[2] = int(padded_shape[2]) + int(pad_top) + int(pad_bottom)
            padded_shape[3] = int(padded_shape[3]) + int(pad_left) + int(pad_right)
            x_ndhwc_padded = ctx.add_intermediate_tensor(
                f"{node.name}_input_ndhwc_padded",
                dtype=ctx.get_tensor_dtype(x_ndhwc_conv),
                shape=padded_shape,
            )
            x_ndhwc_padded_tensor = ctx.model_ir.tensors[x_ndhwc_padded]
            x_sig = (
                list(x_tensor.shape_signature)
                if x_tensor.shape_signature is not None
                else list(x_tensor.shape)
            )
            if len(x_sig) == 5:
                padded_sig = list(x_sig)
                if int(padded_sig[1]) >= 0:
                    padded_sig[1] = int(padded_sig[1]) + int(pad_front) + int(pad_back)
                if int(padded_sig[2]) >= 0:
                    padded_sig[2] = int(padded_sig[2]) + int(pad_top) + int(pad_bottom)
                if int(padded_sig[3]) >= 0:
                    padded_sig[3] = int(padded_sig[3]) + int(pad_left) + int(pad_right)
                x_ndhwc_padded_tensor.shape_signature = [int(v) for v in padded_sig]

            pads_name = ctx.add_const_tensor(
                f"{node.name}_pads_ndhwc",
                np.asarray(
                    [
                        [0, 0],
                        [pad_front, pad_back],
                        [pad_top, pad_bottom],
                        [pad_left, pad_right],
                        [0, 0],
                    ],
                    dtype=np.int32,
                ),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="PAD",
                    inputs=[x_ndhwc_conv, pads_name],
                    outputs=[x_ndhwc_padded],
                )
            )
            x_ndhwc_conv = x_ndhwc_padded

    # ONNX Conv3D weights are OI(DHW); TFLite CONV_3D expects DHWIO.
    w_conv = np.transpose(weights, (2, 3, 4, 1, 0))
    w_name = ctx.add_const_tensor(
        f"{node.name}_conv3d_filter",
        w_conv.astype(np.float32),
    )

    out_channels = int(weights.shape[0])
    bias_values = None
    if len(node.inputs) >= 3:
        bias_values = ctx.get_constant_array(node.inputs[2].name)
    if bias_values is None:
        bias_values = np.zeros((out_channels,), dtype=np.float32)
    bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
    if int(bias_values.size) != out_channels:
        raise NotImplementedError(
            "Conv3D bias size must match output channels. "
            f"op={node.name} bias_size={int(bias_values.size)} out_channels={out_channels}"
        )
    b_name = ctx.add_const_tensor(
        f"{node.name}_conv3d_bias",
        bias_values,
    )

    y_ndhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_ndhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=ndhwc_output_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONV_3D",
            inputs=[x_ndhwc_conv, w_name, b_name],
            outputs=[y_ndhwc],
            options={
                "padding": padding,
                "strideD": int(strides[0]),
                "strideH": int(strides[1]),
                "strideW": int(strides[2]),
                "dilationDFactor": int(dilations[0]),
                "dilationHFactor": int(dilations[1]),
                "dilationWFactor": int(dilations[2]),
                "fusedActivationFunction": "NONE",
            },
        )
    )
    ctx.model_ir.tensors[y_ndhwc].shape_signature = [int(v) for v in ndhwc_output_signature]

    make_transpose(
        ctx,
        y_ndhwc,
        output_name,
        [0, 4, 1, 2, 3],
    )


def build_conv2d_or_depthwise_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT32"
    compute_input_name = str(input_name)

    if input_dtype != compute_dtype:
        compute_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_conv_input_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in list(input_shape)],
        )
        _clone_shape_signature(ctx, input_name, compute_input_name)
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[compute_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    if len(input_shape) == 3 and len(output_shape) == 3:
        conv1d_output_name = str(output_name)
        if output_dtype != compute_dtype:
            conv1d_output_name = ctx.add_intermediate_tensor(
                f"{node.name}_conv1d_f32_output",
                dtype=compute_dtype,
                shape=[int(v) for v in list(output_shape)],
            )
            _clone_shape_signature(ctx, output_name, conv1d_output_name)
        pseudo_node = _make_pseudo_node(
            base_node=node,
            input_names=[compute_input_name, str(weight_name)] + (
                [str(node.inputs[2].name)] if len(node.inputs) >= 3 else []
            ),
            output_name=conv1d_output_name,
            attrs=dict(node.attrs),
        )
        _build_conv1d_via_conv2d(pseudo_node, ctx)
        if conv1d_output_name != str(output_name):
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[conv1d_output_name],
                    outputs=[str(output_name)],
                    options={
                        "inDataType": compute_dtype,
                        "outDataType": output_dtype,
                    },
                )
            )
        return
    if len(input_shape) == 5 and len(output_shape) == 5:
        weights_3d = ctx.get_constant_array(weight_name)
        if weights_3d is not None and np.asarray(weights_3d).ndim == 5:
            _build_conv3d_op(node, ctx)
            return
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(f"Only 2D Conv (rank=4) is supported. op={node.name}")

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"Conv weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 4:
        raise NotImplementedError(
            f"Conv weight rank must be 4. op={node.name} shape={weights.shape}"
        )

    strides = list(node.attrs.get("strides", [1, 1]))
    dilations = list(node.attrs.get("dilations", [1, 1]))
    group = int(node.attrs.get("group", 1))

    nchw_input = input_shape
    nchw_output = output_shape
    padding, explicit_pads = _resolve_conv_padding_and_explicit_pads(
        node=node,
        input_shape_nchw=nchw_input,
        output_shape_nchw=nchw_output,
    )
    nhwc_input_shape = [nchw_input[0], nchw_input[2], nchw_input[3], nchw_input[1]]
    nhwc_output_shape = [nchw_output[0], nchw_output[2], nchw_output[3], nchw_output[1]]
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(nchw_input)
    )
    existing_output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == 4
        else None
    )
    output_signature = _infer_rank4_conv_output_signature(
        input_signature_nchw=input_signature,
        output_shape_nchw=nchw_output,
        existing_output_signature_nchw=existing_output_signature,
    )
    nhwc_output_signature = list(nhwc_output_shape)
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=compute_dtype,
        shape=nhwc_input_shape,
    )
    x_nhwc = make_transpose(
        ctx,
        compute_input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )
    x_nhwc_conv = x_nhwc
    if explicit_pads is not None:
        pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in explicit_pads]
        if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
            x_tensor = ctx.model_ir.tensors[x_nhwc_conv]
            padded_shape = list(x_tensor.shape)
            padded_shape[1] = int(padded_shape[1]) + int(pad_top) + int(pad_bottom)
            padded_shape[2] = int(padded_shape[2]) + int(pad_left) + int(pad_right)
            x_nhwc_padded = ctx.add_intermediate_tensor(
                f"{node.name}_input_nhwc_padded",
                dtype=ctx.get_tensor_dtype(x_nhwc_conv),
                shape=padded_shape,
            )
            x_nhwc_padded_tensor = ctx.model_ir.tensors[x_nhwc_padded]
            x_sig = (
                list(x_tensor.shape_signature)
                if x_tensor.shape_signature is not None
                else list(x_tensor.shape)
            )
            if len(x_sig) == 4:
                padded_sig = list(x_sig)
                if int(padded_sig[1]) >= 0:
                    padded_sig[1] = int(padded_sig[1]) + int(pad_top) + int(pad_bottom)
                if int(padded_sig[2]) >= 0:
                    padded_sig[2] = int(padded_sig[2]) + int(pad_left) + int(pad_right)
                x_nhwc_padded_tensor.shape_signature = [int(v) for v in padded_sig]

            pads_name = ctx.add_const_tensor(
                f"{node.name}_pads_nhwc",
                np.asarray(
                    [
                        [0, 0],
                        [pad_top, pad_bottom],
                        [pad_left, pad_right],
                        [0, 0],
                    ],
                    dtype=np.int32,
                ),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="PAD",
                    inputs=[x_nhwc_conv, pads_name],
                    outputs=[x_nhwc_padded],
                )
            )
            x_nhwc_conv = x_nhwc_padded

    in_channels = int(nchw_input[1])
    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )

    if is_depthwise:
        depth_multiplier = out_channels // group
        w_dw = weights.reshape(out_channels, weights.shape[2], weights.shape[3])
        w_dw = np.transpose(w_dw, (1, 2, 0))
        w_dw = np.expand_dims(w_dw, axis=0)
        w_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_filter",
            w_dw.astype(np.float32),
        )

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((out_channels,), dtype=np.float32)
        b_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_bias",
            np.asarray(bias_values, dtype=np.float32).reshape(-1),
        )

        y_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc",
            dtype=compute_dtype,
            shape=nhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTHWISE_CONV_2D",
                inputs=[x_nhwc_conv, w_name, b_name],
                outputs=[y_nhwc],
                options={
                    "padding": padding,
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                    "dilationHFactor": int(dilations[0]),
                    "dilationWFactor": int(dilations[1]),
                    "depthMultiplier": int(depth_multiplier),
                    "fusedActivationFunction": "NONE",
                },
            )
        )
    else:
        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((out_channels,), dtype=np.float32)
        bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
        if int(bias_values.size) != out_channels:
            raise NotImplementedError(
                "Conv bias size must match output channels. "
                f"op={node.name} bias_size={int(bias_values.size)} out_channels={out_channels}"
            )

        if group != 1:
            if (
                group <= 0
                or in_channels <= 0
                or (in_channels % group) != 0
                or (out_channels % group) != 0
            ):
                raise NotImplementedError(
                    "Grouped Conv requires channels divisible by group. "
                    f"op={node.name} group={group} in_channels={in_channels} out_channels={out_channels}"
                )
            if weight_in_channels_per_group != (in_channels // group):
                raise NotImplementedError(
                    "Grouped Conv weights are inconsistent with input channels/group. "
                    f"op={node.name} group={group} "
                    f"weight_in_channels_per_group={weight_in_channels_per_group} "
                    f"in_channels={in_channels}"
                )

            # TFLite CONV_2D supports grouped convolution when filter input
            # channels are per-group channels (OHWI with I=C/group).
            if not bool(getattr(ctx, "disable_group_convolution", False)):
                w_conv = np.transpose(weights, (0, 2, 3, 1))
                w_name = ctx.add_const_tensor(
                    f"{node.name}_conv_filter",
                    w_conv.astype(np.float32),
                )
                b_name = ctx.add_const_tensor(
                    f"{node.name}_conv_bias",
                    np.asarray(bias_values, dtype=np.float32).reshape(-1),
                )
                y_nhwc = ctx.add_intermediate_tensor(
                    f"{node.name}_output_nhwc",
                    dtype=compute_dtype,
                    shape=nhwc_output_shape,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CONV_2D",
                        inputs=[x_nhwc_conv, w_name, b_name],
                        outputs=[y_nhwc],
                        options={
                            "padding": padding,
                            "strideH": int(strides[0]),
                            "strideW": int(strides[1]),
                            "dilationHFactor": int(dilations[0]),
                            "dilationWFactor": int(dilations[1]),
                            "fusedActivationFunction": "NONE",
                        },
                    )
                )
            else:
                x_group_shape = list(ctx.model_ir.tensors[x_nhwc_conv].shape)
                x_group_signature = (
                    list(ctx.model_ir.tensors[x_nhwc_conv].shape_signature)
                    if ctx.model_ir.tensors[x_nhwc_conv].shape_signature is not None
                    else list(x_group_shape)
                )

                group_inputs: list[str] = []
                x_group_split_axis = ctx.add_const_tensor(
                    f"{node.name}_group_split_axis",
                    np.asarray(3, dtype=np.int32),
                )
                split_outputs: list[str] = []
                for group_idx in range(int(group)):
                    x_group_name = ctx.add_intermediate_tensor(
                        f"{node.name}_group{group_idx}_input_nhwc",
                        dtype=ctx.get_tensor_dtype(x_nhwc_conv),
                        shape=[
                            int(x_group_shape[0]),
                            int(x_group_shape[1]),
                            int(x_group_shape[2]),
                            int(weight_in_channels_per_group),
                        ],
                    )
                    x_group_sig = list(x_group_signature)
                    if len(x_group_sig) == 4:
                        if int(x_group_sig[3]) >= 0 and int(x_group_sig[3]) % int(group) == 0:
                            x_group_sig[3] = int(x_group_sig[3]) // int(group)
                        else:
                            x_group_sig[3] = -1
                        ctx.model_ir.tensors[x_group_name].shape_signature = [int(v) for v in x_group_sig]
                    split_outputs.append(x_group_name)
                ctx.add_operator(
                    OperatorIR(
                        op_type="SPLIT",
                        inputs=[x_group_split_axis, x_nhwc_conv],
                        outputs=split_outputs,
                        options={"numSplits": int(group)},
                    )
                )
                group_inputs = split_outputs
                out_channels_per_group = int(out_channels // group)
                group_conv_outputs: list[str] = []
                for group_idx in range(int(group)):
                    out_begin = int(group_idx * out_channels_per_group)
                    out_end = int(out_begin + out_channels_per_group)
                    group_weights = weights[out_begin:out_end, :, :, :]
                    w_group_conv = np.transpose(group_weights, (0, 2, 3, 1))
                    w_group_name = ctx.add_const_tensor(
                        f"{node.name}_group{group_idx}_conv_filter",
                        w_group_conv.astype(np.float32),
                    )
                    b_group_name = ctx.add_const_tensor(
                        f"{node.name}_group{group_idx}_conv_bias",
                        bias_values[out_begin:out_end].astype(np.float32),
                    )
                    y_group_name = ctx.add_intermediate_tensor(
                        f"{node.name}_group{group_idx}_output_nhwc",
                        dtype=compute_dtype,
                        shape=[
                            int(nhwc_output_shape[0]),
                            int(nhwc_output_shape[1]),
                            int(nhwc_output_shape[2]),
                            int(out_channels_per_group),
                        ],
                    )
                    y_group_signature = list(nhwc_output_signature)
                    if len(y_group_signature) == 4:
                        if int(y_group_signature[3]) >= 0 and int(y_group_signature[3]) % int(group) == 0:
                            y_group_signature[3] = int(y_group_signature[3]) // int(group)
                        else:
                            y_group_signature[3] = -1
                        ctx.model_ir.tensors[y_group_name].shape_signature = [int(v) for v in y_group_signature]
                    ctx.add_operator(
                        OperatorIR(
                            op_type="CONV_2D",
                            inputs=[group_inputs[group_idx], w_group_name, b_group_name],
                            outputs=[y_group_name],
                            options={
                                "padding": padding,
                                "strideH": int(strides[0]),
                                "strideW": int(strides[1]),
                                "dilationHFactor": int(dilations[0]),
                                "dilationWFactor": int(dilations[1]),
                                "fusedActivationFunction": "NONE",
                            },
                        )
                    )
                    group_conv_outputs.append(y_group_name)

                y_nhwc = ctx.add_intermediate_tensor(
                    f"{node.name}_output_nhwc",
                    dtype=compute_dtype,
                    shape=nhwc_output_shape,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CONCATENATION",
                        inputs=group_conv_outputs,
                        outputs=[y_nhwc],
                        options={
                            "axis": 3,
                            "fusedActivationFunction": "NONE",
                        },
                    )
                )
        else:
            # ONNX Conv weights are OIHW; TFLite CONV_2D expects OHWI.
            w_conv = np.transpose(weights, (0, 2, 3, 1))
            w_name = ctx.add_const_tensor(
                f"{node.name}_conv_filter",
                w_conv.astype(np.float32),
            )
            b_name = ctx.add_const_tensor(
                f"{node.name}_conv_bias",
                np.asarray(bias_values, dtype=np.float32).reshape(-1),
            )

            y_nhwc = ctx.add_intermediate_tensor(
                f"{node.name}_output_nhwc",
                dtype=compute_dtype,
                shape=nhwc_output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONV_2D",
                    inputs=[x_nhwc_conv, w_name, b_name],
                    outputs=[y_nhwc],
                    options={
                        "padding": padding,
                        "strideH": int(strides[0]),
                        "strideW": int(strides[1]),
                        "dilationHFactor": int(dilations[0]),
                        "dilationWFactor": int(dilations[1]),
                        "fusedActivationFunction": "NONE",
                    },
                )
            )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]

    nchw_compute_output_name = str(output_name)
    if output_dtype != compute_dtype:
        nchw_compute_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_output_nchw_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in list(output_shape)],
        )
        _clone_shape_signature(ctx, output_name, nchw_compute_output_name)

    make_transpose(
        ctx,
        y_nhwc,
        nchw_compute_output_name,
        [0, 3, 1, 2],
    )
    if nchw_compute_output_name != str(output_name):
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[nchw_compute_output_name],
                outputs=[str(output_name)],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_fused_conv_op(node: Any, ctx: Any) -> None:
    output_name = str(node.outputs[0].name)
    input_names = [str(inp.name) for inp in list(node.inputs)]
    for tensor_name in input_names:
        ctx.ensure_tensor(tensor_name)
    ctx.ensure_tensor(output_name)
    if len(input_names) >= 2:
        _materialize_fusedconv_placeholder_output_shape(
            node=node,
            ctx=ctx,
            input_name=input_names[0],
            weight_name=input_names[1],
            output_name=output_name,
        )
    pre_activation_output_name = _add_same_shape_tensor(
        ctx,
        name=f"{node.name}_fusedconv_pre_activation",
        like_tensor_name=output_name,
    )

    conv_attrs = dict(node.attrs)
    conv_attrs.pop("activation", None)
    conv_attrs.pop("activation_params", None)
    pseudo_node = _make_pseudo_node(
        base_node=node,
        input_names=input_names,
        output_name=pre_activation_output_name,
        attrs=conv_attrs,
    )
    build_conv2d_or_depthwise_op(pseudo_node, ctx)

    activation = _decode_attr_string(node.attrs.get("activation", "Relu"), "Relu")
    activation_params = _flatten_attr_values(node.attrs.get("activation_params", []))
    _add_fusedconv_activation_op(
        node=node,
        ctx=ctx,
        input_name=pre_activation_output_name,
        output_name=output_name,
        activation=activation,
        activation_params=activation_params,
    )
