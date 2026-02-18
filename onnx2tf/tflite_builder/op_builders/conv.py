from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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
    pads_1d = _extract_1d_pads(
        node.attrs.get("pads", [0, 0]),
        op_name="Conv",
        node_name=str(node.name),
    )
    strides_1d = [int(v) for v in list(node.attrs.get("strides", [1]))]
    dilations_1d = [int(v) for v in list(node.attrs.get("dilations", [1]))]
    if len(strides_1d) != 1 or len(dilations_1d) != 1:
        raise NotImplementedError(
            f"Conv1D strides/dilations must be length 1. op={node.name} strides={strides_1d} dilations={dilations_1d}"
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
    elif len(output_padding_1d) != 1:
        raise NotImplementedError(
            f"ConvTranspose1D output_padding must have length 1. op={node.name} output_padding={output_padding_1d}"
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
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", [0, 0]))]

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

    weights_1d = ctx.get_constant_array(weight_name)
    if (
        len(input_shape) == 3
        and len(output_shape) == 3
        and weights_1d is not None
        and np.asarray(weights_1d).ndim == 3
    ):
        _build_conv_transpose1d_via_conv2d(node, ctx)
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
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NotImplementedError(
            f"ConvTranspose dilations must be [1,1] in flatbuffer_direct. op={node.name} dilations={dilations}"
        )
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", [0, 0]))]
    if any(v != 0 for v in output_padding):
        raise NotImplementedError(
            f"ConvTranspose output_padding must be [0,0] in flatbuffer_direct. op={node.name} output_padding={output_padding}"
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
        raise NotImplementedError(
            "ConvTranspose with explicit pads requires static output shape in flatbuffer_direct. "
            f"op={node.name} output_signature={original_output_signature} pads={raw_pads}"
        )
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
            np.asarray([int(dilations[0] * 0 + node.attrs.get('strides', [1, 1])[0])], dtype=np.int32),
        )
        stride_w_vec = ctx.add_const_tensor(
            f"{node.name}_transpose_conv_stride_w",
            np.asarray([int(dilations[1] * 0 + node.attrs.get('strides', [1, 1])[1])], dtype=np.int32),
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
                "strideH": int(node.attrs.get("strides", [1, 1])[0]),
                "strideW": int(node.attrs.get("strides", [1, 1])[1]),
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


def build_conv2d_or_depthwise_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) == 3 and len(output_shape) == 3:
        _build_conv1d_via_conv2d(node, ctx)
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
            dtype=ctx.get_tensor_dtype(output_name),
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
                    dtype=ctx.get_tensor_dtype(output_name),
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
                        dtype=ctx.get_tensor_dtype(output_name),
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
                    dtype=ctx.get_tensor_dtype(output_name),
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
                dtype=ctx.get_tensor_dtype(output_name),
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

    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
