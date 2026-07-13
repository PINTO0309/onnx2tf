from __future__ import annotations

from typing import Set

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.pytorch_emitters import (
    _DIRECT_CODEGEN_BINARY_FUNCTIONS,
    _DIRECT_CODEGEN_MODULE_OP_TYPES,
    _DIRECT_CODEGEN_UNARY_EXPRESSIONS,
)
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    SUPPORTED_TORCH_KERNEL_OP_TYPES,
)
from onnx2tf.tflite_builder.passes.pytorch_normalization import (
    _collect_model_op_types,
)


_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] = (
    set(_DIRECT_CODEGEN_MODULE_OP_TYPES)
    | set(_DIRECT_CODEGEN_UNARY_EXPRESSIONS.keys())
    | set(_DIRECT_CODEGEN_BINARY_FUNCTIONS.keys())
    | {
        "ARG_MAX",
        "ARG_MIN",
        "AVERAGE_POOL_2D",
        "BATCH_MATMUL",
        "CAST",
        "CONCATENATION",
        "CUMSUM",
        "DEPTH_TO_SPACE",
        "EXPAND_DIMS",
        "FILL",
        "GATHER",
        "GATHER_ND",
        "MAX_POOL_2D",
        "LOCAL_RESPONSE_NORMALIZATION",
        "MEAN",
        "MIRROR_PAD",
        "NON_MAX_SUPPRESSION_V4",
        "PACK",
        "PAD",
        "PADV2",
        "RANDOM_STANDARD_NORMAL",
        "RANGE",
        "REDUCE_ANY",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "REVERSE_V2",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SCATTER_ND",
        "SHAPE",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUM",
        "TOPK_V2",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
        "SELECT",
        "SELECT_V2",
        "WHERE",
    }
)


def get_supported_pytorch_kernel_op_types() -> Set[str]:
    return set(SUPPORTED_TORCH_KERNEL_OP_TYPES)


def _ensure_supported_ops(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {
            op_type
            for op_type in _collect_model_op_types(model_ir)
            if op_type not in SUPPORTED_TORCH_KERNEL_OP_TYPES
            and op_type not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES
            and op_type not in {"MODEL"}
        }
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _ensure_no_custom_ops(model_ir: ModelIR) -> None:
    custom_ops = sorted(
        {str(op.op_type) for op in model_ir.operators if str(op.op_type) == "CUSTOM"}
    )
    if len(custom_ops) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support CUSTOM ops."
        )
