from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


_DYNAMIC_RANGE_KERNEL_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
}

_DYNAMIC_RANGE_CONST_DEQUANT_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "CONCATENATION",
}


def _clone_model_ir(model_ir: ModelIR) -> ModelIR:
    clone = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
    )
    clone.inputs = list(model_ir.inputs)
    clone.outputs = list(model_ir.outputs)
    clone.operators = [
        OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            version=op.version,
        )
        for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        clone.tensors[name] = TensorIR(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else None,
            data=tensor.data.copy() if isinstance(tensor.data, np.ndarray) else tensor.data,
            is_variable=tensor.is_variable,
            quantization=(
                dict(tensor.quantization)
                if isinstance(tensor.quantization, dict)
                else QuantParamIR(
                    scale=list(tensor.quantization.scale),
                    zero_point=list(tensor.quantization.zero_point),
                    quantized_dimension=int(tensor.quantization.quantized_dimension),
                    min=list(tensor.quantization.min)
                    if tensor.quantization.min is not None
                    else None,
                    max=list(tensor.quantization.max)
                    if tensor.quantization.max is not None
                    else None,
                )
                if isinstance(tensor.quantization, QuantParamIR)
                else tensor.quantization
            ),
        )
    return clone


def _normalize_quant_type(quant_type: str) -> str:
    quant_type = str(quant_type).strip().lower()
    if quant_type not in ["per-channel", "per-tensor"]:
        raise ValueError(
            "flatbuffer_direct quant_type must be one of [\"per-channel\", \"per-tensor\"]. "
            f"got: {quant_type}"
        )
    return quant_type


def _normalize_calibration_method(calibration_method: str) -> str:
    m = str(calibration_method).strip().lower()
    if m not in ["max", "percentile"]:
        raise ValueError(
            "flatbuffer_direct calibration_method must be one of [\"max\", \"percentile\"]. "
            f"got: {calibration_method}"
        )
    return m


def _normalize_calibration_percentile(calibration_percentile: float) -> float:
    p = float(calibration_percentile)
    if p <= 0.0 or p > 100.0:
        raise ValueError(
            "flatbuffer_direct calibration_percentile must be in (0.0, 100.0]. "
            f"got: {calibration_percentile}"
        )
    return p


def _normalize_positive_int(v: int, *, name: str) -> int:
    iv = int(v)
    if iv <= 0:
        raise ValueError(f"flatbuffer_direct {name} must be > 0. got: {v}")
    return iv


def _normalize_non_negative_float(v: float, *, name: str) -> float:
    fv = float(v)
    if fv < 0.0:
        raise ValueError(f"flatbuffer_direct {name} must be >= 0.0. got: {v}")
    return fv


def _compute_abs_limit(
    data: np.ndarray,
    *,
    calibration_method: str,
    calibration_percentile: float,
    axis: Optional[Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    abs_data = np.abs(np.asarray(data, dtype=np.float32))
    if calibration_method == "max":
        return np.max(abs_data, axis=axis, keepdims=keepdims)
    return np.percentile(
        abs_data,
        q=calibration_percentile,
        axis=axis,
        keepdims=keepdims,
    ).astype(np.float32)


def _symmetric_int8_quantize(
    data: np.ndarray,
    *,
    calibration_method: str,
    calibration_percentile: float,
    scale_floor: float,
) -> Tuple[np.ndarray, float]:
    data = np.asarray(data, dtype=np.float32)
    abs_limit = float(
        _compute_abs_limit(
            data,
            calibration_method=calibration_method,
            calibration_percentile=calibration_percentile,
        )
    )
    if abs_limit == 0.0:
        scale = max(scale_floor, 1.0)
        q = np.zeros_like(data, dtype=np.int8)
        return q, scale
    scale = max(abs_limit / 127.0, scale_floor)
    q = np.round(data / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, float(scale)


def _symmetric_int8_quantize_per_channel(
    data: np.ndarray,
    *,
    axis: int,
    calibration_method: str,
    calibration_percentile: float,
    scale_floor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float32)
    if axis < 0:
        axis += data.ndim
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"Invalid quantization axis {axis} for data rank {data.ndim}")

    reduce_axes = tuple(i for i in range(data.ndim) if i != axis)
    abs_limit = _compute_abs_limit(
        data,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        axis=reduce_axes,
        keepdims=True,
    )
    scales = np.maximum(abs_limit / 127.0, scale_floor).astype(np.float32)
    q = np.round(data / scales)
    q = np.clip(q, -127, 127).astype(np.int8)
    scales_1d = np.squeeze(scales, axis=reduce_axes).astype(np.float32)
    return q, scales_1d


def _is_float_constant_tensor(
    *,
    tensor: TensorIR,
    graph_input_names: Set[str],
) -> bool:
    if tensor.name in graph_input_names:
        return False
    if tensor.is_variable:
        return False
    if tensor.dtype != "FLOAT32":
        return False
    if not isinstance(tensor.data, np.ndarray):
        return False
    return True


def _quantize_tensor_inplace(
    *,
    tensor: TensorIR,
    quant_mode: str = "per-tensor",
    quantized_dimension: int = 0,
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> bool:
    if tensor.dtype == "INT8" and isinstance(tensor.quantization, QuantParamIR):
        return True
    if tensor.dtype != "FLOAT32":
        return False
    if not isinstance(tensor.data, np.ndarray):
        return False

    data = np.asarray(tensor.data, dtype=np.float32)
    if int(data.size) < min_numel:
        return False

    raw_abs_max = float(np.max(np.abs(data))) if data.size > 0 else 0.0
    if raw_abs_max < min_abs_max:
        return False

    if quant_mode == "per-tensor":
        q_data, scale = _symmetric_int8_quantize(
            data,
            calibration_method=calibration_method,
            calibration_percentile=calibration_percentile,
            scale_floor=scale_floor,
        )
        tensor.dtype = "INT8"
        tensor.data = q_data
        tensor.quantization = QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            min=[float(np.min(q_data.astype(np.float32) * scale))],
            max=[float(np.max(q_data.astype(np.float32) * scale))],
            quantized_dimension=0,
        )
    elif quant_mode == "per-channel":
        q_data, scales = _symmetric_int8_quantize_per_channel(
            data,
            axis=quantized_dimension,
            calibration_method=calibration_method,
            calibration_percentile=calibration_percentile,
            scale_floor=scale_floor,
        )
        tensor.dtype = "INT8"
        tensor.data = q_data
        tensor.quantization = QuantParamIR(
            scale=[float(v) for v in scales.reshape(-1).tolist()],
            zero_point=[0 for _ in range(int(scales.size))],
            min=None,
            max=None,
            quantized_dimension=int(quantized_dimension),
        )
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")
    return True


def _make_unique_tensor_name(base: str, tensors: Dict[str, TensorIR]) -> str:
    if base not in tensors:
        return base
    serial = 1
    while f"{base}_{serial}" in tensors:
        serial += 1
    return f"{base}_{serial}"


def _kernel_weight_quant_axis(op_type: str, tensor: TensorIR) -> int:
    if op_type in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
        return len(tensor.shape) - 1
    if op_type == "FULLY_CONNECTED":
        return 0
    return 0


def _const_quant_mode_and_axis(
    *,
    quant_type: str,
    tensor: TensorIR,
) -> Tuple[str, int]:
    if quant_type == "per-channel" and len(tensor.shape) >= 2 and int(tensor.shape[-1]) > 1:
        return "per-channel", len(tensor.shape) - 1
    return "per-tensor", 0


def _normalize_quantization_controls(
    *,
    calibration_method: str,
    calibration_percentile: float,
    min_numel: int,
    min_abs_max: float,
    scale_floor: float,
) -> Dict[str, object]:
    return {
        "calibration_method": _normalize_calibration_method(calibration_method),
        "calibration_percentile": _normalize_calibration_percentile(calibration_percentile),
        "min_numel": _normalize_positive_int(min_numel, name="quant_min_numel"),
        "min_abs_max": _normalize_non_negative_float(min_abs_max, name="quant_min_abs_max"),
        "scale_floor": _normalize_non_negative_float(scale_floor, name="quant_scale_floor"),
    }


def build_dynamic_range_quantized_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> ModelIR:
    clone = _clone_model_ir(model_ir)
    graph_input_names = set(clone.inputs)
    quant_type = _normalize_quant_type(quant_type)
    controls = _normalize_quantization_controls(
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )

    quantized_tensor_names: Set[str] = set()
    dequantized_tensor_map: Dict[str, str] = {}

    rewritten_ops: List[OperatorIR] = []

    def ensure_dequantized_tensor(quant_tensor_name: str) -> Optional[str]:
        if quant_tensor_name in dequantized_tensor_map:
            return dequantized_tensor_map[quant_tensor_name]

        if quant_tensor_name not in clone.tensors:
            return None
        q_tensor = clone.tensors[quant_tensor_name]
        if q_tensor.dtype != "INT8":
            return None

        deq_name = _make_unique_tensor_name(f"{quant_tensor_name}_dequantized", clone.tensors)
        clone.tensors[deq_name] = TensorIR(
            name=deq_name,
            dtype="FLOAT32",
            shape=list(q_tensor.shape),
            shape_signature=list(q_tensor.shape_signature)
            if q_tensor.shape_signature is not None
            else list(q_tensor.shape),
            data=None,
            is_variable=False,
            quantization=None,
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[quant_tensor_name],
                outputs=[deq_name],
                options={},
            )
        )
        dequantized_tensor_map[quant_tensor_name] = deq_name
        return deq_name

    for op in clone.operators:
        new_op = OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            version=op.version,
        )

        if new_op.op_type in _DYNAMIC_RANGE_KERNEL_OPS and len(new_op.inputs) >= 2:
            weight_name = new_op.inputs[1]
            tensor = clone.tensors.get(weight_name)
            if tensor is not None and _is_float_constant_tensor(
                tensor=tensor,
                graph_input_names=graph_input_names,
            ):
                quant_mode = "per-channel" if quant_type == "per-channel" else "per-tensor"
                quant_axis = _kernel_weight_quant_axis(new_op.op_type, tensor)
                if _quantize_tensor_inplace(
                    tensor=tensor,
                    quant_mode=quant_mode,
                    quantized_dimension=quant_axis,
                    calibration_method=str(controls["calibration_method"]),
                    calibration_percentile=float(controls["calibration_percentile"]),
                    min_numel=int(controls["min_numel"]),
                    min_abs_max=float(controls["min_abs_max"]),
                    scale_floor=float(controls["scale_floor"]),
                ):
                    quantized_tensor_names.add(weight_name)

        if new_op.op_type in _DYNAMIC_RANGE_CONST_DEQUANT_OPS:
            for idx, input_name in enumerate(list(new_op.inputs)):
                tensor = clone.tensors.get(input_name)
                if tensor is None:
                    continue
                if _is_float_constant_tensor(tensor=tensor, graph_input_names=graph_input_names):
                    const_mode, const_axis = _const_quant_mode_and_axis(
                        quant_type=quant_type,
                        tensor=tensor,
                    )
                    if _quantize_tensor_inplace(
                        tensor=tensor,
                        quant_mode=const_mode,
                        quantized_dimension=const_axis,
                        calibration_method=str(controls["calibration_method"]),
                        calibration_percentile=float(controls["calibration_percentile"]),
                        min_numel=int(controls["min_numel"]),
                        min_abs_max=float(controls["min_abs_max"]),
                        scale_floor=float(controls["scale_floor"]),
                    ):
                        quantized_tensor_names.add(input_name)
                tensor = clone.tensors.get(input_name)
                if tensor is None or tensor.dtype != "INT8":
                    continue
                deq_name = ensure_dequantized_tensor(input_name)
                if deq_name is not None:
                    new_op.inputs[idx] = deq_name

        rewritten_ops.append(new_op)

    clone.operators = rewritten_ops

    if len(quantized_tensor_names) == 0:
        raise NotImplementedError(
            "flatbuffer_direct dynamic-range quantization requires at least one quantizable float32 constant tensor "
            "(kernel weights for CONV_2D/DEPTHWISE_CONV_2D/FULLY_CONNECTED or constants used by "
            "ADD/SUB/MUL/DIV/CONCATENATION)."
        )

    clone.description = f"{clone.description} (dynamic_range_quantized)"
    return clone


def build_integer_quantized_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> ModelIR:
    clone = build_dynamic_range_quantized_model_ir(
        model_ir,
        quant_type=quant_type,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )
    clone.description = f"{clone.description} (integer_quantized_limited)"
    return clone


def _normalize_quant_io_dtype(dtype: str) -> Tuple[str, QuantParamIR]:
    d = str(dtype).strip().lower()
    if d == "int8":
        return "INT8", QuantParamIR(scale=[1.0 / 128.0], zero_point=[0], quantized_dimension=0)
    if d == "uint8":
        return "UINT8", QuantParamIR(scale=[1.0 / 255.0], zero_point=[128], quantized_dimension=0)
    if d == "int16":
        return "INT16", QuantParamIR(scale=[1.0 / 32768.0], zero_point=[0], quantized_dimension=0)
    if d == "float32":
        return "FLOAT32", QuantParamIR(scale=[1.0], zero_point=[0], quantized_dimension=0)
    raise ValueError(
        "flatbuffer_direct full integer quantization supports input/output dtype "
        f"int8/uint8/int16/float32. got: {dtype}"
    )


def build_full_integer_quantized_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    input_quant_dtype: str = "int8",
    output_quant_dtype: str = "int8",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> ModelIR:
    clone = build_integer_quantized_model_ir(
        model_ir,
        quant_type=quant_type,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )

    input_dtype_name, input_qparams = _normalize_quant_io_dtype(input_quant_dtype)
    output_dtype_name, output_qparams = _normalize_quant_io_dtype(output_quant_dtype)

    pre_ops: List[OperatorIR] = []
    post_ops: List[OperatorIR] = []

    new_inputs: List[str] = []
    for input_name in list(clone.inputs):
        old_tensor = clone.tensors[input_name]
        if input_dtype_name == "FLOAT32":
            new_inputs.append(input_name)
            continue

        q_input_name = _make_unique_tensor_name(f"{input_name}_quantized_input", clone.tensors)
        clone.tensors[q_input_name] = TensorIR(
            name=q_input_name,
            dtype=input_dtype_name,
            shape=list(old_tensor.shape),
            shape_signature=list(old_tensor.shape_signature)
            if old_tensor.shape_signature is not None
            else list(old_tensor.shape),
            data=None,
            is_variable=False,
            quantization=QuantParamIR(
                scale=list(input_qparams.scale),
                zero_point=list(input_qparams.zero_point),
                quantized_dimension=int(input_qparams.quantized_dimension),
            ),
        )
        pre_ops.append(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[q_input_name],
                outputs=[input_name],
                options={},
            )
        )
        new_inputs.append(q_input_name)

    new_outputs: List[str] = []
    for output_name in list(clone.outputs):
        old_tensor = clone.tensors[output_name]
        if output_dtype_name == "FLOAT32":
            new_outputs.append(output_name)
            continue

        q_output_name = _make_unique_tensor_name(f"{output_name}_quantized_output", clone.tensors)
        clone.tensors[q_output_name] = TensorIR(
            name=q_output_name,
            dtype=output_dtype_name,
            shape=list(old_tensor.shape),
            shape_signature=list(old_tensor.shape_signature)
            if old_tensor.shape_signature is not None
            else list(old_tensor.shape),
            data=None,
            is_variable=False,
            quantization=QuantParamIR(
                scale=list(output_qparams.scale),
                zero_point=list(output_qparams.zero_point),
                quantized_dimension=int(output_qparams.quantized_dimension),
            ),
        )
        post_ops.append(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[output_name],
                outputs=[q_output_name],
                options={},
            )
        )
        new_outputs.append(q_output_name)

    clone.inputs = new_inputs
    clone.outputs = new_outputs
    clone.operators = pre_ops + clone.operators + post_ops
    clone.description = f"{clone.description} (full_integer_quantized_limited)"
    return clone


def build_integer_quantized_with_int16_act_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> ModelIR:
    clone = build_integer_quantized_model_ir(
        model_ir,
        quant_type=quant_type,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )
    clone.description = f"{clone.description} (integer_quant_with_int16_act_limited)"
    return clone


def build_full_integer_quantized_with_int16_act_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
) -> ModelIR:
    clone = build_full_integer_quantized_model_ir(
        model_ir,
        quant_type=quant_type,
        input_quant_dtype="int16",
        output_quant_dtype="int16",
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )
    clone.description = f"{clone.description} (full_integer_quant_with_int16_act_limited)"
    return clone
