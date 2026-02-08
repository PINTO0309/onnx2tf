from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.op_builders import (
    build_binary_op,
    build_concat_op,
    build_conv2d_or_depthwise_op,
    build_fully_connected_from_gemm_or_matmul,
    build_identity_op,
    build_logistic_op,
    build_pool2d_op,
    build_reshape_op,
    build_softmax_op,
    build_transpose_op,
)


def dispatch_node(node: Any, ctx: Any) -> None:
    op = node.op
    if op == "Add":
        build_binary_op(node, ctx, "ADD")
        return
    if op == "Sub":
        build_binary_op(node, ctx, "SUB")
        return
    if op == "Mul":
        build_binary_op(node, ctx, "MUL")
        return
    if op == "Div":
        build_binary_op(node, ctx, "DIV")
        return
    if op == "Sigmoid":
        build_logistic_op(node, ctx)
        return
    if op == "Softmax":
        build_softmax_op(node, ctx)
        return
    if op == "Reshape":
        build_reshape_op(node, ctx)
        return
    if op == "Transpose":
        build_transpose_op(node, ctx)
        return
    if op == "Concat":
        build_concat_op(node, ctx)
        return
    if op == "Identity":
        build_identity_op(node, ctx)
        return
    if op == "Conv":
        build_conv2d_or_depthwise_op(node, ctx)
        return
    if op == "AveragePool":
        build_pool2d_op(node, ctx, "AVERAGE_POOL_2D")
        return
    if op == "MaxPool":
        build_pool2d_op(node, ctx, "MAX_POOL_2D")
        return
    if op in ["Gemm", "MatMul"]:
        build_fully_connected_from_gemm_or_matmul(node, ctx)
        return
    raise NotImplementedError(f"ONNX op is not supported by flatbuffer_direct: {op} ({node.name})")
