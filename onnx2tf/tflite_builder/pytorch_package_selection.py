from __future__ import annotations

import re

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_first_logical_layout,
    normalize_logical_layout,
)


def _should_prefer_tflite_backed_package(model_ir: ModelIR) -> bool:
    op_types = [str(op.op_type) for op in model_ir.operators]
    recurrent_or_control_ops = {
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(op_type in recurrent_or_control_ops for op_type in op_types):
        return False
    has_length_like_input = False
    for input_name in model_ir.inputs:
        canonical = re.sub(r"[^0-9a-z]+", "_", str(input_name).lower()).strip("_")
        if canonical.endswith(
            ("length", "lengths", "len", "lens", "seq_len", "seq_lens")
        ):
            has_length_like_input = True
            break
    if has_length_like_input:
        return False
    if any(op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} for op_type in op_types):
        return True
    for op in model_ir.operators:
        if str(op.op_type) != "SOFTMAX" or len(op.inputs) == 0:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        if input_tensor is None:
            continue
        if is_channel_first_logical_layout(
            normalize_logical_layout(input_tensor.logical_layout)
        ):
            return True
    conv_like_count = sum(
        1 for op_type in op_types if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    )
    strided_slice_count = sum(1 for op_type in op_types if op_type == "STRIDED_SLICE")
    concat_count = sum(1 for op_type in op_types if op_type == "CONCATENATION")
    resize_count = sum(
        1
        for op_type in op_types
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
    )
    split_count = sum(1 for op_type in op_types if op_type == "SPLIT")
    softmax_count = sum(1 for op_type in op_types if op_type == "SOFTMAX")
    nhwc_named_tensor_count = sum(
        1
        for tensor_name in model_ir.tensors.keys()
        if str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))
    )
    has_rank3_channel_first_output = any(
        len(list(model_ir.tensors[str(output_name)].shape)) == 3
        and normalize_logical_layout(model_ir.tensors[str(output_name)].logical_layout)
        == "NCW"
        for output_name in model_ir.outputs
        if str(output_name) in model_ir.tensors
    )
    if (
        has_rank3_channel_first_output
        and conv_like_count >= 20
        and strided_slice_count >= 4
        and concat_count >= 4
    ):
        return True
    if (
        conv_like_count >= 40
        and nhwc_named_tensor_count >= 40
        and (resize_count >= 4 or softmax_count >= 1 or split_count >= 1)
    ):
        return True
    if conv_like_count >= 60 and nhwc_named_tensor_count >= 80 and resize_count >= 2:
        return True
    if conv_like_count >= 15 and nhwc_named_tensor_count >= 30 and resize_count >= 3:
        return True
    return False


def _should_prefer_saved_model_backed_package(model_ir: ModelIR) -> bool:
    return _should_prefer_tflite_backed_package(model_ir)
