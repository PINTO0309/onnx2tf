from __future__ import annotations

from typing import Any, Callable, Set

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
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

_RUNTIME_SUPPORTED_CUSTOM_CODES: Set[str] = {
    "ONNX_SLICE",
}


def _is_channel_last_layout_for_codegen(logical_layout: Any) -> bool:
    return str(logical_layout).upper() in {"NWC", "NHWC", "NDHWC"}


def _can_emit_direct_module_call_for_codegen(
    *,
    model_ir: ModelIR,
    is_channel_last_layout_fn: Callable[[Any], bool],
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"}:
        return False
    if len(op.outputs) != 1:
        return False
    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if input_tensor is None or output_tensor is None:
        return False
    if is_channel_last_layout_fn(
        input_tensor.logical_layout
    ) or is_channel_last_layout_fn(output_tensor.logical_layout):
        return False
    expected_rank = 5 if op_type == "CONV_3D" else 4
    if (
        len(input_tensor.shape) != expected_rank
        or len(output_tensor.shape) != expected_rank
    ):
        return False
    if int(input_tensor.shape[1]) <= 0 or int(output_tensor.shape[1]) <= 0:
        return False
    return True


def get_supported_pytorch_kernel_op_types() -> Set[str]:
    return set(SUPPORTED_TORCH_KERNEL_OP_TYPES)


def _raise_for_unsupported_ops(op_types: Set[str]) -> None:
    unsupported = sorted(
        {
            op_type
            for op_type in op_types
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


def _ensure_supported_ops(model_ir: ModelIR) -> None:
    _raise_for_unsupported_ops(_collect_model_op_types(model_ir))


def _ensure_native_export_supported_ops(model_ir: ModelIR) -> None:
    root_op_types = {str(op.op_type) for op in model_ir.operators}
    if "CUSTOM" in root_op_types:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support CUSTOM ops."
        )

    all_op_types = set(root_op_types)
    for subgraph in model_ir.subgraphs:
        all_op_types.update(_collect_model_op_types(subgraph))
    _raise_for_unsupported_ops(all_op_types)


def _ensure_direct_codegen_supported(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {
            str(op.op_type)
            for op in model_ir.operators
            if str(op.op_type) not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES
        }
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _is_direct_codegen_unsupported_error(ex: BaseException) -> bool:
    return (
        "Native PyTorch-like model.py codegen does not support some op types in this model."
        in str(ex)
    )


def _ensure_no_custom_ops(model_ir: ModelIR) -> None:
    custom_ops = sorted(
        {str(op.op_type) for op in model_ir.operators if str(op.op_type) == "CUSTOM"}
    )
    if len(custom_ops) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support CUSTOM ops."
        )


def _supports_runtime_wrapper_model_ir(model_ir: ModelIR) -> bool:
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type in SUPPORTED_TORCH_KERNEL_OP_TYPES:
            continue
        if op_type == "CUSTOM":
            custom_code = str(op.options.get("customCode", "")).upper()
            if custom_code in _RUNTIME_SUPPORTED_CUSTOM_CODES:
                continue
        return False
    return True
