from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose, resolve_padding


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
        pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
        if len(pads) == 4 and sum(abs(int(v)) for v in pads) == 0:
            return "VALID"
    raise NotImplementedError(
        "ConvTranspose currently supports auto_pad=SAME_* or zero pads with auto_pad in {NOTSET,VALID}. "
        f"op={node.name} auto_pad={auto_pad_raw} pads={node.attrs.get('pads', [0,0,0,0])}"
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
            np.asarray(nhwc_output_shape, dtype=np.int32),
        )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
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

    y_final_nhwc = y_nhwc
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
                inputs=[y_nhwc, bias_const],
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
    padding = resolve_padding(node)

    nchw_input = input_shape
    nchw_output = output_shape
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

    in_channels = int(nchw_input[1])
    out_channels = int(weights.shape[0])
    is_depthwise = group == in_channels and weights.shape[1] == 1 and group > 1

    if is_depthwise:
        depth_multiplier = out_channels // in_channels
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
                inputs=[x_nhwc, w_name, b_name],
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
        if group != 1:
            raise NotImplementedError(
                "Grouped Conv is not supported except depthwise. "
                f"op={node.name} group={group}"
            )
        # ONNX Conv weights are OIHW; TFLite CONV_2D expects OHWI.
        w_conv = np.transpose(weights, (0, 2, 3, 1))
        w_name = ctx.add_const_tensor(
            f"{node.name}_conv_filter",
            w_conv.astype(np.float32),
        )

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((out_channels,), dtype=np.float32)
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
                inputs=[x_nhwc, w_name, b_name],
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
