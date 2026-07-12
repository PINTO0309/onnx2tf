from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized import (
    _infer_pool_output_hw_for_qlinear,
    _normalize_axis,
    _promote_internal_uint8_tensor_to_int8,
    _require_const,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
)
from onnx2tf.tflite_builder.op_builders.shared import (
    make_transpose,
    resolve_padding,
)


def build_qlinear_average_pool_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearAveragePool input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearAveragePool input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearAveragePool output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearAveragePool output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"QLinearAveragePool supports only rank-4 input for flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", [1, 1]))]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(kernel) != 2 or len(strides) != 2:
        raise NotImplementedError(
            f"QLinearAveragePool supports only 2D pooling in flatbuffer_direct. "
            f"op={node.name} kernel_shape={kernel} strides={strides}"
        )
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in [0, 1]:
        raise NotImplementedError(
            f"QLinearAveragePool ceil_mode must be 0 or 1 for flatbuffer_direct. op={node.name} ceil_mode={ceil_mode}"
        )
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(pads) < 4:
        pads = [0, 0, 0, 0]
    if ceil_mode == 1:
        if auto_pad not in ["NOTSET", "SAME", "SAME_UPPER", "SAME_LOWER"]:
            raise NotImplementedError(
                f"QLinearAveragePool ceil_mode=1 supports auto_pad NOTSET/SAME only. op={node.name} auto_pad={auto_pad}"
            )
        if auto_pad == "NOTSET" and any(int(v) != 0 for v in pads):
            raise NotImplementedError(
                f"QLinearAveragePool ceil_mode=1 with auto_pad=NOTSET requires pads=[0,0,0,0]. op={node.name} pads={pads}"
            )
    if int(node.attrs.get("count_include_pad", 0)) != 0:
        raise NotImplementedError(
            f"QLinearAveragePool count_include_pad must be 0 for flatbuffer_direct. op={node.name}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NotImplementedError(
            f"QLinearAveragePool dilations must be [1,1] for flatbuffer_direct. "
            f"op={node.name} dilations={dilations}"
        )

    if ceil_mode == 1:
        # TFLite has no explicit ceil_mode flag; use SAME-style output sizing for this supported subset.
        padding = "SAME"
    else:
        padding = resolve_padding(node)

    input_tensor = ctx.model_ir.tensors[x_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    if len(output_shape) != 4:
        out_h, out_w = _infer_pool_output_hw_for_qlinear(
            node=node,
            input_h=int(input_shape[2]),
            input_w=int(input_shape[3]),
            kernel_h=int(kernel[0]),
            kernel_w=int(kernel[1]),
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
            ceil_mode=ceil_mode,
        )
        output_shape = [
            int(input_shape[0]),
            int(input_shape[1]),
            int(out_h),
            int(out_w),
        ]
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.shape = list(output_shape)
        output_signature = list(output_shape)
        if len(input_signature) == 4:
            output_signature[0] = int(input_signature[0])
            output_signature[1] = int(input_signature[1])
        output_tensor.shape_signature = list(output_signature)

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0
        if np.asarray(x_scale).size <= 1
        else _normalize_axis(1, 4),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0
        if np.asarray(y_scale).size <= 1
        else _normalize_axis(1, 4),
    )

    nhwc_input_shape = [
        int(input_shape[0]),
        int(input_shape[2]),
        int(input_shape[3]),
        int(input_shape[1]),
    ]
    nhwc_output_shape = [
        int(output_shape[0]),
        int(output_shape[2]),
        int(output_shape[3]),
        int(output_shape[1]),
    ]
    output_tensor = ctx.model_ir.tensors[output_name]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    nhwc_output_signature = [int(v) for v in nhwc_output_shape]
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]
    nhwc_input_signature = [
        int(input_signature[0]),
        int(input_signature[2]),
        int(input_signature[3]),
        int(input_signature[1]),
    ]

    def _lower_via_dequantize_avgpool_quantize() -> None:
        dq_out = ctx.add_intermediate_tensor(
            f"{node.name}_dq_out",
            dtype="FLOAT32",
            shape=list(input_shape),
        )
        ctx.model_ir.tensors[dq_out].shape_signature = list(input_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[x_name],
                outputs=[dq_out],
            )
        )

        x_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_input_nhwc",
            dtype="FLOAT32",
            shape=nhwc_input_shape,
        )
        ctx.model_ir.tensors[x_nhwc].shape_signature = list(nhwc_input_signature)
        x_nhwc = make_transpose(
            ctx,
            dq_out,
            x_nhwc,
            [0, 2, 3, 1],
            allow_elide_inverse_chain=True,
        )

        y_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc",
            dtype="FLOAT32",
            shape=nhwc_output_shape,
        )
        ctx.model_ir.tensors[y_nhwc].shape_signature = [
            int(v) for v in nhwc_output_signature
        ]
        ctx.add_operator(
            OperatorIR(
                op_type="AVERAGE_POOL_2D",
                inputs=[x_nhwc],
                outputs=[y_nhwc],
                options={
                    "padding": padding,
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                    "filterHeight": int(kernel[0]),
                    "filterWidth": int(kernel[1]),
                    "fusedActivationFunction": "NONE",
                },
            )
        )

        pool_out_nchw = ctx.add_intermediate_tensor(
            f"{node.name}_pool_out_nchw",
            dtype="FLOAT32",
            shape=list(output_shape),
        )
        ctx.model_ir.tensors[pool_out_nchw].shape_signature = list(output_signature)
        make_transpose(
            ctx,
            y_nhwc,
            pool_out_nchw,
            [0, 3, 1, 2],
        )

        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[pool_out_nchw],
                outputs=[output_name],
            )
        )

    # Prefer quantized AVERAGE_POOL_2D and avoid DEQUANTIZE/QUANTIZE bridges.
    # Keep this conservative: only per-tensor activation quantization.
    can_use_quantized_pool = (
        len(input_shape) == 4
        and len(output_shape) == 4
        and np.asarray(x_scale).size <= 1
        and np.asarray(y_scale).size <= 1
        and np.array_equal(
            np.asarray(x_scale).reshape(-1),
            np.asarray(y_scale).reshape(-1),
        )
        and np.array_equal(
            np.asarray(x_zero).reshape(-1),
            np.asarray(y_zero).reshape(-1),
        )
    )
    if not can_use_quantized_pool:
        _lower_via_dequantize_avgpool_quantize()
        return

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(x_name),
        shape=nhwc_input_shape,
    )
    ctx.model_ir.tensors[x_nhwc].shape_signature = list(nhwc_input_signature)
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_nhwc,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0,
    )
    x_nhwc = make_transpose(
        ctx,
        x_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [
        int(v) for v in nhwc_output_signature
    ]
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=y_nhwc,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="AVERAGE_POOL_2D",
            inputs=[x_nhwc],
            outputs=[y_nhwc],
            options={
                "padding": padding,
                "strideH": int(strides[0]),
                "strideW": int(strides[1]),
                "filterHeight": int(kernel[0]),
                "filterWidth": int(kernel[1]),
                "fusedActivationFunction": "NONE",
            },
        )
    )

    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )


def build_qlinear_global_average_pool_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearGlobalAveragePool input scale")
    x_zero = _require_const(
        ctx, x_zero_name, "QLinearGlobalAveragePool input zero_point"
    )
    y_scale = _require_const(ctx, y_scale_name, "QLinearGlobalAveragePool output scale")
    y_zero = _require_const(
        ctx, y_zero_name, "QLinearGlobalAveragePool output zero_point"
    )

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    input_tensor = ctx.model_ir.tensors[x_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    channels_last = bool(int(node.attrs.get("channels_last", 0)))
    if len(input_shape) < 3 and input_shape != [1]:
        raise NotImplementedError(
            f"QLinearGlobalAveragePool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    spatial_axes = (
        [int(v) for v in range(1, len(input_shape) - 1)]
        if channels_last
        else [int(v) for v in range(2, len(input_shape))]
    )
    if len(spatial_axes) == 0:
        raise NotImplementedError(
            f"QLinearGlobalAveragePool requires at least one spatial axis. op={node.name} input_shape={input_shape}"
        )

    output_shape = [int(v) for v in input_shape]
    output_signature = list(input_signature)
    for axis in spatial_axes:
        output_shape[axis] = 1
        if axis < len(output_signature):
            output_signature[axis] = 1
    ctx.model_ir.tensors[output_name].shape = list(output_shape)
    ctx.model_ir.tensors[output_name].shape_signature = list(output_signature)

    input_rank = len(input_shape)
    output_rank = len(output_shape)
    input_channel_axis = (input_rank - 1) if channels_last else 1
    output_channel_axis = (output_rank - 1) if channels_last else 1
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0
        if np.asarray(x_scale).size <= 1
        else _normalize_axis(input_channel_axis, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0
        if np.asarray(y_scale).size <= 1
        else _normalize_axis(output_channel_axis, output_rank),
    )

    def _lower_via_dequantize_mean_quantize() -> None:
        dq_out = ctx.add_intermediate_tensor(
            f"{node.name}_dq_out",
            dtype="FLOAT32",
            shape=input_shape,
        )
        ctx.model_ir.tensors[dq_out].shape_signature = list(input_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[x_name],
                outputs=[dq_out],
            )
        )

        axes_const = ctx.add_const_tensor(
            f"{node.name}_mean_axes",
            np.asarray(spatial_axes, dtype=np.int32),
        )
        mean_out = ctx.add_intermediate_tensor(
            f"{node.name}_mean_out",
            dtype="FLOAT32",
            shape=output_shape,
        )
        ctx.model_ir.tensors[mean_out].shape_signature = list(output_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="MEAN",
                inputs=[dq_out, axes_const],
                outputs=[mean_out],
                options={"keepDims": True},
            )
        )

        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[mean_out],
                outputs=[output_name],
            )
        )

    def _lower_via_quantized_global_average_pool() -> bool:
        if len(input_shape) != 4 or len(output_shape) != 4:
            return False
        if np.asarray(x_scale).size > 1 or np.asarray(y_scale).size > 1:
            return False
        # TFLite's quantized AVERAGE_POOL_2D preserves the input integer
        # encoding; it does not requantize between distinct input/output
        # scales or zero points. QLinearGlobalAveragePool explicitly permits
        # different encodings, so use the float MEAN bridge in that case.
        if not np.array_equal(
            np.asarray(x_scale).reshape(-1),
            np.asarray(y_scale).reshape(-1),
        ) or not np.array_equal(
            np.asarray(x_zero).reshape(-1),
            np.asarray(y_zero).reshape(-1),
        ):
            return False

        if channels_last:
            x_nhwc = x_name
            nhwc_input_shape = [int(v) for v in list(input_shape)]
            nhwc_input_signature = [int(v) for v in list(input_signature)]
            y_nhwc = output_name
        else:
            nhwc_input_shape = [
                int(input_shape[0]),
                int(input_shape[2]),
                int(input_shape[3]),
                int(input_shape[1]),
            ]
            nhwc_input_signature = [
                int(input_signature[0]),
                int(input_signature[2]),
                int(input_signature[3]),
                int(input_signature[1]),
            ]
            x_nhwc = ctx.add_intermediate_tensor(
                f"{node.name}_input_nhwc",
                dtype=ctx.get_tensor_dtype(x_name),
                shape=nhwc_input_shape,
            )
            x_nhwc_tensor = ctx.model_ir.tensors[x_nhwc]
            x_nhwc_tensor.shape_signature = [int(v) for v in list(nhwc_input_signature)]
            _set_tensor_quantization(
                ctx=ctx,
                tensor_name=x_nhwc,
                scale=x_scale,
                zero_point=x_zero,
                quantized_dimension=0,
            )
            x_nhwc = make_transpose(
                ctx,
                x_name,
                x_nhwc,
                [0, 2, 3, 1],
                allow_elide_inverse_chain=True,
            )

            nhwc_output_shape = [
                int(output_shape[0]),
                int(output_shape[2]),
                int(output_shape[3]),
                int(output_shape[1]),
            ]
            nhwc_output_signature = [
                int(output_signature[0]),
                int(output_signature[2]),
                int(output_signature[3]),
                int(output_signature[1]),
            ]
            y_nhwc = ctx.add_intermediate_tensor(
                f"{node.name}_output_nhwc",
                dtype=ctx.get_tensor_dtype(output_name),
                shape=nhwc_output_shape,
            )
            y_nhwc_tensor = ctx.model_ir.tensors[y_nhwc]
            y_nhwc_tensor.shape_signature = [
                int(v) for v in list(nhwc_output_signature)
            ]
            _set_tensor_quantization(
                ctx=ctx,
                tensor_name=y_nhwc,
                scale=y_scale,
                zero_point=y_zero,
                quantized_dimension=0,
            )

        filter_h = int(nhwc_input_shape[1]) if len(nhwc_input_shape) > 1 else 0
        filter_w = int(nhwc_input_shape[2]) if len(nhwc_input_shape) > 2 else 0
        if filter_h <= 0 or filter_w <= 0:
            return False

        ctx.add_operator(
            OperatorIR(
                op_type="AVERAGE_POOL_2D",
                inputs=[x_nhwc],
                outputs=[y_nhwc],
                options={
                    "padding": "VALID",
                    "strideH": 1,
                    "strideW": 1,
                    "filterHeight": int(filter_h),
                    "filterWidth": int(filter_w),
                    "fusedActivationFunction": "NONE",
                },
            )
        )

        if not channels_last:
            make_transpose(
                ctx,
                y_nhwc,
                output_name,
                [0, 3, 1, 2],
            )
            if len(ctx.model_ir.operators) > 0:
                last_op = ctx.model_ir.operators[-1]
                if (
                    str(last_op.op_type) == "TRANSPOSE"
                    and len(last_op.outputs) == 1
                    and str(last_op.outputs[0]) == str(output_name)
                ):
                    last_opts = (
                        dict(last_op.options)
                        if isinstance(last_op.options, dict)
                        else {}
                    )
                    last_opts["__preserve_layout_boundary__"] = True
                    last_op.options = last_opts
        return True

    # Prefer quantized builtin lowering when possible so layout optimizations can
    # propagate NHWC contracts through concat/conv chains.
    if not _lower_via_quantized_global_average_pool():
        _lower_via_dequantize_mean_quantize()
