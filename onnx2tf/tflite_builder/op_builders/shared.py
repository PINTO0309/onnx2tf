from __future__ import annotations

import copy
from typing import Any, List, Optional

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR


def _read_transpose_perm(ctx: Any, op: OperatorIR) -> Optional[List[int]]:
    if str(op.op_type) != "TRANSPOSE":
        return None
    if len(op.inputs) < 2:
        return None
    perm_tensor = ctx.model_ir.tensors.get(op.inputs[1], None)
    if perm_tensor is None or perm_tensor.data is None:
        return None
    perm = np.asarray(perm_tensor.data).reshape(-1)
    if perm.size == 0:
        return None
    return [int(v) for v in perm.tolist()]


def _is_inverse_perm(perm_a: List[int], perm_b: List[int]) -> bool:
    if len(perm_a) != len(perm_b):
        return False
    rank = len(perm_a)
    if sorted(perm_a) != [int(i) for i in range(rank)]:
        return False
    if sorted(perm_b) != [int(i) for i in range(rank)]:
        return False
    for idx, value in enumerate(perm_a):
        if perm_b[value] != idx:
            return False
    return True


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return QuantParamIR(
            scale=list(quantization.scale),
            zero_point=list(quantization.zero_point),
            quantized_dimension=int(quantization.quantized_dimension),
            min=list(quantization.min) if quantization.min is not None else None,
            max=list(quantization.max) if quantization.max is not None else None,
        )
    return copy.deepcopy(quantization)


def _quant_param_scale_count(quantization: Any) -> int:
    if quantization is None:
        return 0
    if isinstance(quantization, QuantParamIR):
        return int(len(list(quantization.scale)))
    if isinstance(quantization, dict):
        scale = quantization.get("scale", [])
        if isinstance(scale, np.ndarray):
            return int(scale.size)
        if isinstance(scale, (list, tuple)):
            return int(len(scale))
    return 0


def _get_quantized_dimension(quantization: Any) -> Optional[int]:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return int(quantization.quantized_dimension)
    if isinstance(quantization, dict) and "quantized_dimension" in quantization:
        try:
            return int(quantization.get("quantized_dimension", 0))
        except Exception:
            return None
    return None


def _set_quantized_dimension(quantization: Any, qdim: int) -> None:
    if quantization is None:
        return
    if isinstance(quantization, QuantParamIR):
        quantization.quantized_dimension = int(qdim)
        return
    if isinstance(quantization, dict):
        quantization["quantized_dimension"] = int(qdim)


def _invert_perm(perm: List[int]) -> Optional[List[int]]:
    rank = len(list(perm))
    if sorted(int(v) for v in perm) != [int(i) for i in range(rank)]:
        return None
    inv = [0] * rank
    for out_axis, in_axis in enumerate(perm):
        inv[int(in_axis)] = int(out_axis)
    return inv


def _remap_quantized_dimension_for_transpose(quantization: Any, perm_values: List[int]) -> Any:
    if quantization is None:
        return None
    scale_count = _quant_param_scale_count(quantization)
    if scale_count <= 1:
        return quantization
    qdim = _get_quantized_dimension(quantization)
    if qdim is None:
        return quantization
    inv = _invert_perm([int(v) for v in list(perm_values)])
    if inv is None:
        return quantization
    if 0 <= int(qdim) < len(inv):
        _set_quantized_dimension(quantization, int(inv[int(qdim)]))
    return quantization


def resolve_padding(node: Any) -> str:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET"))
    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        return "SAME"
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 4 and sum([abs(int(v)) for v in pads]) == 0:
        return "VALID"
    if len(pads) == 4:
        top, left, bottom, right = [int(v) for v in pads]
        if top == bottom and left == right and top >= 0 and left >= 0:
            return "SAME"
        # ONNX exports from TF frequently encode SAME padding as asymmetric pads
        # (e.g. [0, 0, 1, 1]). TFLite SAME supports this distribution internally.
        if (
            top >= 0
            and left >= 0
            and bottom >= 0
            and right >= 0
            and abs(top - bottom) <= 1
            and abs(left - right) <= 1
        ):
            return "SAME"
    raise NotImplementedError(
        "Only zero pads, symmetric pads, or SAME auto_pad are supported in flatbuffer_direct. "
        f"op={node.name} pads={pads} auto_pad={auto_pad}"
    )


def make_transpose(
    ctx: Any,
    input_name: str,
    output_name: str,
    perm_values: List[int],
    allow_elide_inverse_chain: bool = False,
) -> str:
    if allow_elide_inverse_chain:
        producer_idx = None
        producer_op = None
        for op_idx in range(len(ctx.model_ir.operators) - 1, -1, -1):
            op = ctx.model_ir.operators[op_idx]
            if output_name in op.outputs:
                # output_name is already produced by some op, so do not alias.
                producer_op = None
                producer_idx = None
                break
            if input_name in op.outputs:
                producer_op = op
                producer_idx = int(op_idx)
                break
        if producer_op is not None and str(producer_op.op_type) == "TRANSPOSE":
            prev_perm = _read_transpose_perm(ctx, producer_op)
            if prev_perm is not None and _is_inverse_perm(prev_perm, [int(v) for v in perm_values]):
                prev_input_name = producer_op.inputs[0]
                prev_input_shape = list(ctx.get_tensor_shape(prev_input_name))
                output_shape = list(ctx.get_tensor_shape(output_name))
                if prev_input_shape == output_shape:
                    consumer_count = int(ctx.tensor_consumer_count.get(str(input_name), 0))
                    if (
                        consumer_count <= 1
                        and str(input_name) not in set(ctx.graph_output_names)
                        and producer_idx is not None
                    ):
                        # input_name is an ONNX edge consumed only by this transpose.
                        # Remove the previous transpose as well to avoid leaving dead ops.
                        del ctx.model_ir.operators[int(producer_idx)]
                    return prev_input_name

    perm_name = ctx.add_const_tensor(
        f"{output_name}_perm",
        np.asarray(perm_values, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_name],
            outputs=[output_name],
        )
    )
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if input_tensor is not None and output_tensor is not None:
        perm = [int(v) for v in perm_values]
        if len(input_tensor.shape) == len(perm):
            output_tensor.shape = [int(input_tensor.shape[int(axis)]) for axis in perm]
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        if len(input_signature) == len(perm):
            output_tensor.shape_signature = [int(input_signature[int(axis)]) for axis in perm]
        output_tensor.quantization = _clone_quantization(input_tensor.quantization)
        output_tensor.quantization = _remap_quantized_dimension_for_transpose(
            output_tensor.quantization,
            perm,
        )
    return output_name
