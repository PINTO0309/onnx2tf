from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, TypedDict

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
    clone_operator_ir,
    clone_tensor_ir,
)


class StrictFullIntegerQuantizationError(RuntimeError):
    pass


@dataclass
class TensorCalibrationRange:
    min_value: float
    max_value: float
    num_samples: int = 0


@dataclass
class QuantizedModelResult:
    model_ir: ModelIR
    report: Dict[str, Any]


@dataclass
class _StrictQuantizationReporter:
    enabled: bool
    payload: Dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        full_integer_io: bool,
        calibration_ranges: Dict[str, TensorCalibrationRange],
    ) -> "_StrictQuantizationReporter":
        if not enabled:
            return cls(enabled=False, payload={})
        return cls(
            enabled=True,
            payload={
                "mode": "full_integer" if full_integer_io else "integer_float_io",
                "strict": True,
                "supported_ops": sorted(_STRICT_FULL_INTEGER_SUPPORTED_OPS),
                "tensor_ranges": {
                    str(name): _tensor_range_to_report(value)
                    for name, value in calibration_ranges.items()
                },
                "quantized_tensors": {},
                "quantized_ops": [],
                "failures": [],
            },
        )

    def record_tensor(
        self,
        *,
        tensor_name: str,
        dtype: str,
        kind: str,
        qparams: Optional[QuantParamIR],
    ) -> None:
        if not self.enabled:
            return
        self.payload["quantized_tensors"][str(tensor_name)] = {
            "dtype": dtype,
            "kind": kind,
            "qparams": _quant_param_to_report(qparams),
        }

    def record_operator(self, *, index: int, operator: OperatorIR) -> None:
        if not self.enabled:
            return
        self.payload["quantized_ops"].append(
            {
                "index": int(index),
                "op_type": str(operator.op_type),
                "inputs": [str(value) for value in operator.inputs],
                "outputs": [str(value) for value in operator.outputs],
            }
        )


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

_STRICT_FULL_INTEGER_SUPPORTED_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MAXIMUM",
    "MINIMUM",
    "LESS",
    "LOGICAL_NOT",
    "WHERE",
    "SHAPE",
    "SELECT",
    "SELECT_V2",
    "CAST",
    "CONCATENATION",
    "RESHAPE",
    "TRANSPOSE",
    "SQUEEZE",
    "EXPAND_DIMS",
    "SLICE",
    "STRIDED_SLICE",
    "SPLIT",
    "GATHER",
    "GATHER_ND",
    "TILE",
    "SPACE_TO_DEPTH",
    "DEPTH_TO_SPACE",
    "PAD",
    "MEAN",
    "SUM",
    "REDUCE_MAX",
    "REDUCE_MIN",
    "REDUCE_PROD",
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "ABS",
    "NEG",
    "EXP",
    "RELU",
    "RELU6",
    "RELU_N1_TO_1",
    "LEAKY_RELU",
    "HARD_SWISH",
    "TANH",
    "PRELU",
    "SOFTMAX",
    "LOGISTIC",
    "RESIZE_NEAREST_NEIGHBOR",
    "RESIZE_BILINEAR",
    "SCATTER_ND",
    "TOPK_V2",
    "ARG_MAX",
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "TRANSPOSE_CONV",
    "FULLY_CONNECTED",
    "BATCH_MATMUL",
}

_STRICT_FULL_INTEGER_PASSTHROUGH_OPS = {
    "RESHAPE",
    "TRANSPOSE",
    "SQUEEZE",
    "EXPAND_DIMS",
}

_STRICT_FULL_INTEGER_SAME_QPARAM_INDICES: Dict[
    str,
    Tuple[Optional[Set[int]], Optional[Set[int]]],
] = {
    "CONCATENATION": (None, None),
    "MEAN": ({0}, None),
    "AVERAGE_POOL_2D": ({0}, None),
    "MAX_POOL_2D": ({0}, None),
    "SLICE": ({0}, None),
    "STRIDED_SLICE": ({0}, None),
    "SPLIT": ({1}, None),
    "GATHER": ({0}, None),
    "GATHER_ND": ({0}, {0}),
    "TILE": ({0}, None),
    "SPACE_TO_DEPTH": ({0}, None),
    "DEPTH_TO_SPACE": ({0}, None),
    "PAD": ({0}, None),
    "SCATTER_ND": ({1}, {0}),
    "SELECT": ({1, 2}, None),
    "SELECT_V2": ({1, 2}, None),
}

_STRICT_FULL_INTEGER_INT16_ACT_UNSUPPORTED_OPS = {
    "SCATTER_ND": "LiteRT SCATTER_ND does not support INT16 updates.",
}


class _QuantizationControls(TypedDict):
    calibration_method: str
    calibration_percentile: float
    min_numel: int
    min_abs_max: float
    scale_floor: float


def strict_int16_activation_skip_reasons(model_ir: ModelIR) -> List[str]:
    reasons: List[str] = []
    seen: Set[str] = set()
    for op_idx, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        reason = _STRICT_FULL_INTEGER_INT16_ACT_UNSUPPORTED_OPS.get(op_type, None)
        if reason is None:
            continue
        key = f"{op_type}:{reason}"
        if key in seen:
            continue
        seen.add(key)
        reasons.append(f"index={op_idx} op_type={op_type}: {reason}")
    return reasons


def _clone_model_ir(model_ir: ModelIR) -> ModelIR:
    clone = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
    )
    clone.inputs = list(model_ir.inputs)
    clone.outputs = list(model_ir.outputs)
    clone.operators = [
        clone_operator_ir(op, options=dict(op.options))
        for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        clone.tensors[name] = clone_tensor_ir(
            tensor,
            dtype=tensor.dtype,
            data=tensor.data,
            normalize_layouts=False,
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


def _normalize_tensor_detail_name(name: Any) -> str:
    value = name.decode("utf-8") if isinstance(name, bytes) else str(name)
    if value.endswith(":0"):
        value = value[:-2]
    return value


def load_calibration_samples(
    *,
    custom_input_op_name_np_data_path: Optional[List[Any]],
    input_names: List[str],
) -> List[Dict[str, np.ndarray]]:
    if not custom_input_op_name_np_data_path:
        raise StrictFullIntegerQuantizationError(
            "flatbuffer_direct strict integer quantization requires "
            "custom_input_op_name_np_data_path entries with input name, npy path, mean, and std."
        )

    input_set = {str(v) for v in input_names}
    loaded: Dict[str, np.ndarray] = {}
    sample_count: Optional[int] = None
    for param in custom_input_op_name_np_data_path:
        if not isinstance(param, (list, tuple)) or len(param) != 4:
            raise StrictFullIntegerQuantizationError(
                "flatbuffer_direct strict integer quantization requires each calibration entry "
                "to be [input_name, numpy_file_path, mean, std]."
            )
        input_name = str(param[0])
        if input_name not in input_set:
            raise StrictFullIntegerQuantizationError(
                f"Calibration data references unknown input: {input_name}"
            )
        raw = np.load(str(param[1]))
        mean = np.asarray(param[2], dtype=np.float32)
        std = np.asarray(param[3], dtype=np.float32)
        if np.any(std == 0):
            raise StrictFullIntegerQuantizationError(
                f"Calibration std contains zero for input: {input_name}"
            )
        normalized = ((np.asarray(raw, dtype=np.float32) - mean) / std).astype(np.float32)
        if int(normalized.ndim) == 0:
            raise StrictFullIntegerQuantizationError(
                f"Calibration data must include a sample dimension: {input_name}"
            )
        if sample_count is None:
            sample_count = int(normalized.shape[0])
        elif int(normalized.shape[0]) != int(sample_count):
            raise StrictFullIntegerQuantizationError(
                "Calibration sample count mismatch: "
                f"input={input_name} count={int(normalized.shape[0])} expected={sample_count}"
            )
        loaded[input_name] = normalized

    missing = sorted(input_set.difference(loaded.keys()))
    if missing:
        raise StrictFullIntegerQuantizationError(
            f"Missing calibration data for input(s): {', '.join(missing)}"
        )
    if sample_count is None or int(sample_count) <= 0:
        raise StrictFullIntegerQuantizationError("Calibration data is empty.")

    samples: List[Dict[str, np.ndarray]] = []
    for idx in range(int(sample_count)):
        samples.append(
            {
                input_name: np.expand_dims(data[idx], axis=0)
                for input_name, data in loaded.items()
            }
        )
    return samples


def collect_calibration_ranges_from_tflite(
    *,
    tflite_path: str,
    calibration_samples: List[Dict[str, np.ndarray]],
) -> Dict[str, TensorCalibrationRange]:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    try:
        interpreter = Interpreter(
            model_path=tflite_path,
            experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    except TypeError:
        interpreter = Interpreter(model_path=tflite_path)

    ranges: Dict[str, TensorCalibrationRange] = {}

    def update(name: str, value: np.ndarray) -> None:
        arr = np.asarray(value)
        if arr.size == 0 or not np.issubdtype(arr.dtype, np.floating):
            return
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        if not np.isfinite(mn) or not np.isfinite(mx):
            return
        current = ranges.get(str(name), None)
        if current is None:
            ranges[str(name)] = TensorCalibrationRange(mn, mx, 1)
        else:
            current.min_value = min(float(current.min_value), mn)
            current.max_value = max(float(current.max_value), mx)
            current.num_samples = int(current.num_samples) + 1

    for sample in calibration_samples:
        interpreter.allocate_tensors()
        input_details = list(interpreter.get_input_details())
        by_name = {
            _normalize_tensor_detail_name(detail.get("name", "")): detail
            for detail in input_details
        }
        for idx, (sample_name, sample_value) in enumerate(sample.items()):
            detail = by_name.get(str(sample_name), None)
            if detail is None:
                if idx >= len(input_details):
                    raise StrictFullIntegerQuantizationError(
                        f"TFLite input not found for calibration sample: {sample_name}"
                    )
                detail = input_details[idx]
            value = np.asarray(sample_value, dtype=np.float32)
            expected_shape = [int(v) for v in list(detail.get("shape", []))]
            if expected_shape and list(value.shape) != expected_shape:
                try:
                    interpreter.resize_tensor_input(int(detail["index"]), value.shape, strict=False)
                    interpreter.allocate_tensors()
                    input_details = list(interpreter.get_input_details())
                    by_name = {
                        _normalize_tensor_detail_name(item.get("name", "")): item
                        for item in input_details
                    }
                    detail = by_name.get(str(sample_name), detail)
                except Exception as ex:
                    raise StrictFullIntegerQuantizationError(
                        f"Failed to resize calibration input {sample_name} to {list(value.shape)}: {ex}"
                    ) from ex
            interpreter.set_tensor(int(detail["index"]), value)
            update(str(sample_name), value)
            update(_normalize_tensor_detail_name(detail.get("name", sample_name)), value)
        interpreter.invoke()
        for detail in list(interpreter.get_tensor_details()):
            name = _normalize_tensor_detail_name(detail.get("name", ""))
            if name == "":
                continue
            try:
                value = interpreter.get_tensor(int(detail["index"]))
            except Exception:
                continue
            update(name, value)

    return ranges


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


def activation_qparams_from_range(
    *,
    min_value: float,
    max_value: float,
    dtype: str,
    scale_floor: float = 1e-8,
) -> QuantParamIR:
    dtype_u = str(dtype).upper()
    if dtype_u == "INT8":
        qmin, qmax = -128, 127
    elif dtype_u == "UINT8":
        qmin, qmax = 0, 255
    elif dtype_u == "INT16":
        abs_limit = max(abs(float(min_value)), abs(float(max_value)), 0.0)
        scale = max(abs_limit / 32767.0, float(scale_floor), 1.0 if abs_limit == 0.0 else float(scale_floor))
        return QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
            min=[float(min_value)],
            max=[float(max_value)],
        )
    else:
        raise ValueError(f"Unsupported activation quantization dtype: {dtype}")

    mn = float(min_value)
    mx = float(max_value)
    if not np.isfinite(mn) or not np.isfinite(mx):
        raise ValueError(f"Non-finite calibration range: min={min_value} max={max_value}")
    if mn > mx:
        mn, mx = mx, mn
    mn = min(mn, 0.0)
    mx = max(mx, 0.0)
    if mx == mn:
        return QuantParamIR(
            scale=[max(float(scale_floor), 1.0)],
            zero_point=[0],
            quantized_dimension=0,
            min=[mn],
            max=[mx],
        )

    scale = max((mx - mn) / float(qmax - qmin), float(scale_floor))
    zero_point = int(np.round(float(qmin) - (mn / scale)))
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return QuantParamIR(
        scale=[float(scale)],
        zero_point=[zero_point],
        quantized_dimension=0,
        min=[mn],
        max=[mx],
    )


def fixed_activation_qparams_for_op(
    *,
    op_type: str,
    dtype: str,
) -> Optional[QuantParamIR]:
    dtype_u = str(dtype).upper()
    op_u = str(op_type).upper()
    if op_u == "SOFTMAX":
        if dtype_u == "UINT8":
            return QuantParamIR(scale=[1.0 / 256.0], zero_point=[0], quantized_dimension=0, min=[0.0], max=[1.0])
        if dtype_u == "INT8":
            return QuantParamIR(scale=[1.0 / 256.0], zero_point=[-128], quantized_dimension=0, min=[0.0], max=[1.0])
        if dtype_u == "INT16":
            return QuantParamIR(scale=[1.0 / 32768.0], zero_point=[0], quantized_dimension=0, min=[0.0], max=[1.0])
    if op_u == "LOGISTIC":
        if dtype_u == "UINT8":
            return QuantParamIR(scale=[1.0 / 256.0], zero_point=[0], quantized_dimension=0, min=[0.0], max=[1.0])
        if dtype_u == "INT8":
            return QuantParamIR(scale=[1.0 / 256.0], zero_point=[-128], quantized_dimension=0, min=[0.0], max=[1.0])
        if dtype_u == "INT16":
            return QuantParamIR(scale=[1.0 / 32768.0], zero_point=[0], quantized_dimension=0, min=[0.0], max=[1.0])
    if op_u == "TANH":
        if dtype_u == "UINT8":
            return QuantParamIR(scale=[1.0 / 128.0], zero_point=[128], quantized_dimension=0, min=[-1.0], max=[1.0])
        if dtype_u == "INT8":
            return QuantParamIR(scale=[1.0 / 128.0], zero_point=[0], quantized_dimension=0, min=[-1.0], max=[1.0])
        if dtype_u == "INT16":
            return QuantParamIR(scale=[1.0 / 32768.0], zero_point=[0], quantized_dimension=0, min=[-1.0], max=[1.0])
    return None


def _clone_quant_param(quantization: QuantParamIR) -> QuantParamIR:
    return QuantParamIR(
        scale=list(quantization.scale),
        zero_point=list(quantization.zero_point),
        quantized_dimension=int(quantization.quantized_dimension),
        min=list(quantization.min) if quantization.min is not None else None,
        max=list(quantization.max) if quantization.max is not None else None,
    )


def _quant_param_to_report(quantization: Optional[QuantParamIR]) -> Optional[Dict[str, Any]]:
    if quantization is None:
        return None
    return {
        "scale": [float(v) for v in list(quantization.scale)],
        "zero_point": [int(v) for v in list(quantization.zero_point)],
        "quantized_dimension": int(quantization.quantized_dimension),
        "min": [float(v) for v in list(quantization.min)] if quantization.min is not None else None,
        "max": [float(v) for v in list(quantization.max)] if quantization.max is not None else None,
    }


def _tensor_range_to_report(value: TensorCalibrationRange) -> Dict[str, Any]:
    return {
        "min": float(value.min_value),
        "max": float(value.max_value),
        "num_samples": int(value.num_samples),
    }


def _used_tensor_names(model_ir: ModelIR) -> Set[str]:
    used = set(str(v) for v in list(model_ir.inputs) + list(model_ir.outputs))
    for op in model_ir.operators:
        used.update(str(v) for v in op.inputs if str(v) != "")
        used.update(str(v) for v in op.outputs if str(v) != "")
    return used


def _elide_identity_operators(model_ir: ModelIR) -> None:
    candidate_ops = list(model_ir.operators)
    if not any(str(op.op_type) == "IDENTITY" for op in candidate_ops):
        return

    graph_index = ModelIRGraphIndex(model_ir)
    retained_ops: List[OperatorIR] = []
    removed_ops: List[OperatorIR] = []
    replacements: Dict[str, str] = {}
    graph_outputs = {str(v) for v in model_ir.outputs}

    def resolve(name: str) -> str:
        current = str(name)
        seen: Set[str] = set()
        while current in replacements and current not in seen:
            seen.add(current)
            current = replacements[current]
        return current

    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None:
            continue
        new_inputs = [
            resolve(str(value)) if str(value) != "" else value
            for value in op.inputs
        ]
        if new_inputs != list(op.inputs):
            graph_index.replace_operator_inputs(int(op_index), new_inputs)
        if str(op.op_type) != "IDENTITY" or len(op.inputs) != 1 or len(op.outputs) != 1:
            retained_ops.append(op)
            continue
        input_name = resolve(str(op.inputs[0]))
        output_name = str(op.outputs[0])
        if output_name in graph_outputs:
            producer = next(
                (
                    candidate
                    for candidate in reversed(retained_ops)
                    if input_name in [str(v) for v in candidate.outputs]
                ),
                None,
            )
            if producer is not None and input_name not in {str(v) for v in model_ir.inputs}:
                producer_index = graph_index.operator_index(producer)
                if producer_index is None:
                    retained_ops.append(op)
                    continue
                graph_index.replace_operator_outputs(
                    int(producer_index),
                    [
                        output_name if str(value) == input_name else value
                        for value in producer.outputs
                    ],
                )
                replacements[input_name] = output_name
                removed_ops.append(op)
                continue
        replacements[output_name] = input_name
        removed_ops.append(op)

    for op in retained_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None:
            continue
        new_inputs = [
            resolve(str(value)) if str(value) != "" else value
            for value in op.inputs
        ]
        new_outputs = [
            resolve(str(value)) if str(value) != "" else value
            for value in op.outputs
        ]
        if new_inputs != list(op.inputs):
            graph_index.replace_operator_inputs(int(op_index), new_inputs)
        if new_outputs != list(op.outputs):
            graph_index.replace_operator_outputs(int(op_index), new_outputs)
    model_ir.outputs = [resolve(str(v)) for v in model_ir.outputs]
    remove_indices = [
        index
        for op in removed_ops
        if (index := graph_index.operator_index(op)) is not None
    ]
    graph_index.remove_operators(remove_indices)


def _same_qparam_tensor_names_for_op(op: OperatorIR) -> List[str]:
    policy = _STRICT_FULL_INTEGER_SAME_QPARAM_INDICES.get(str(op.op_type), None)
    if policy is None:
        return []
    input_indices, output_indices = policy
    names: List[str] = []
    if input_indices is None:
        names.extend(str(v) for v in op.inputs if str(v) != "")
    else:
        names.extend(
            str(input_name)
            for idx, input_name in enumerate(op.inputs)
            if int(idx) in input_indices and str(input_name) != ""
        )
    if output_indices is None:
        names.extend(str(v) for v in op.outputs if str(v) != "")
    else:
        names.extend(
            str(output_name)
            for idx, output_name in enumerate(op.outputs)
            if int(idx) in output_indices and str(output_name) != ""
        )
    return names


def _derive_same_qparams_from_available_tensor(
    *,
    model_ir: ModelIR,
    tensor_names: Iterable[str],
    calibration_ranges: Dict[str, TensorCalibrationRange],
    dtype: str,
    scale_floor: float,
) -> Optional[QuantParamIR]:
    names = [str(name) for name in tensor_names if str(name) != ""]
    dtype_u = str(dtype).upper()
    for tensor_name in names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        if tensor.dtype == dtype_u and isinstance(tensor.quantization, QuantParamIR):
            return _clone_quant_param(tensor.quantization)
    for tensor_name in names:
        if tensor_name not in calibration_ranges:
            continue
        rng = calibration_ranges[tensor_name]
        return activation_qparams_from_range(
            min_value=float(rng.min_value),
            max_value=float(rng.max_value),
            dtype=dtype_u,
            scale_floor=float(scale_floor),
        )
    for tensor_name in names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or tensor.dtype != "FLOAT32" or not isinstance(tensor.data, np.ndarray):
            continue
        data = np.asarray(tensor.data, dtype=np.float32)
        if data.size == 0:
            continue
        return activation_qparams_from_range(
            min_value=float(np.min(data)),
            max_value=float(np.max(data)),
            dtype=dtype_u,
            scale_floor=float(scale_floor),
        )
    return None


def _is_float_activation_tensor(
    *,
    tensor: TensorIR,
    graph_input_names: Set[str],
) -> bool:
    if tensor.dtype != "FLOAT32":
        return False
    if tensor.is_variable:
        return False
    if tensor.name in graph_input_names:
        return True
    return tensor.data is None


def _require_tensor_range(
    *,
    tensor_name: str,
    calibration_ranges: Dict[str, TensorCalibrationRange],
) -> TensorCalibrationRange:
    if tensor_name not in calibration_ranges:
        raise StrictFullIntegerQuantizationError(
            f"Missing calibration range for tensor: {tensor_name}"
        )
    return calibration_ranges[tensor_name]


def _activation_dtype_for_tensor(
    *,
    tensor_name: str,
    model_ir: ModelIR,
    full_integer_io: bool,
    input_quant_dtype: str,
    output_quant_dtype: str,
    internal_activation_dtype: str,
) -> str:
    if full_integer_io and tensor_name in set(model_ir.inputs):
        return str(input_quant_dtype).upper()
    if full_integer_io and tensor_name in set(model_ir.outputs):
        return str(output_quant_dtype).upper()
    return str(internal_activation_dtype).upper()


def _ensure_activation_quantized(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    calibration_ranges: Dict[str, TensorCalibrationRange],
    dtype: str,
    scale_floor: float,
    reporter: _StrictQuantizationReporter,
    fixed_qparams: Optional[QuantParamIR] = None,
) -> QuantParamIR:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        raise StrictFullIntegerQuantizationError(f"Missing tensor: {tensor_name}")
    if tensor.dtype not in {"FLOAT32", "INT8", "UINT8", "INT16"}:
        return tensor.quantization if isinstance(tensor.quantization, QuantParamIR) else QuantParamIR(scale=[1.0], zero_point=[0])
    if fixed_qparams is not None:
        qparams = _clone_quant_param(fixed_qparams)
    elif isinstance(tensor.quantization, QuantParamIR) and tensor.dtype == str(dtype).upper():
        qparams = _clone_quant_param(tensor.quantization)
    else:
        rng = _require_tensor_range(
            tensor_name=str(tensor_name),
            calibration_ranges=calibration_ranges,
        )
        qparams = activation_qparams_from_range(
            min_value=float(rng.min_value),
            max_value=float(rng.max_value),
            dtype=str(dtype),
            scale_floor=float(scale_floor),
        )
    tensor.dtype = str(dtype).upper()
    tensor.quantization = qparams
    tensor.data = None
    reporter.record_tensor(
        tensor_name=str(tensor_name),
        dtype=tensor.dtype,
        kind="activation",
        qparams=qparams,
    )
    return qparams


def _quantize_float_constant_with_qparams(
    *,
    tensor: TensorIR,
    dtype: str,
    qparams: QuantParamIR,
) -> None:
    if not isinstance(tensor.data, np.ndarray):
        raise StrictFullIntegerQuantizationError(
            f"Constant tensor requires ndarray data for quantization: {tensor.name}"
        )
    scale = float(qparams.scale[0])
    zero_point = int(qparams.zero_point[0])
    dtype_u = str(dtype).upper()
    if dtype_u == "INT8":
        qmin, qmax, np_dtype = -128, 127, np.int8
    elif dtype_u == "UINT8":
        qmin, qmax, np_dtype = 0, 255, np.uint8
    elif dtype_u == "INT16":
        qmin, qmax, np_dtype = -32768, 32767, np.int16
    else:
        raise StrictFullIntegerQuantizationError(f"Unsupported constant quant dtype: {dtype}")
    q = np.round(np.asarray(tensor.data, dtype=np.float32) / scale) + zero_point
    tensor.data = np.clip(q, qmin, qmax).astype(np_dtype)
    tensor.dtype = dtype_u
    tensor.quantization = _clone_quant_param(qparams)


def _quantize_weight_tensor(
    *,
    tensor: TensorIR,
    quant_type: str,
    quantized_dimension: int,
    calibration_method: str,
    calibration_percentile: float,
    scale_floor: float,
    reporter: _StrictQuantizationReporter,
) -> QuantParamIR:
    if not isinstance(tensor.data, np.ndarray):
        raise StrictFullIntegerQuantizationError(
            f"Weight tensor must be constant for strict full integer quantization: {tensor.name}"
        )
    if quant_type == "per-channel" and int(tensor.data.ndim) >= 2:
        q_data, scales = _symmetric_int8_quantize_per_channel(
            np.asarray(tensor.data, dtype=np.float32),
            axis=int(quantized_dimension),
            calibration_method=str(calibration_method),
            calibration_percentile=float(calibration_percentile),
            scale_floor=float(scale_floor),
        )
        qparams = QuantParamIR(
            scale=[float(v) for v in scales.reshape(-1).tolist()],
            zero_point=[0 for _ in range(int(scales.size))],
            quantized_dimension=int(quantized_dimension),
        )
    else:
        q_data, scale = _symmetric_int8_quantize(
            np.asarray(tensor.data, dtype=np.float32),
            calibration_method=str(calibration_method),
            calibration_percentile=float(calibration_percentile),
            scale_floor=float(scale_floor),
        )
        qparams = QuantParamIR(scale=[float(scale)], zero_point=[0], quantized_dimension=0)
    tensor.data = q_data
    tensor.dtype = "INT8"
    tensor.quantization = qparams
    reporter.record_tensor(
        tensor_name=tensor.name,
        dtype="INT8",
        kind="weight",
        qparams=qparams,
    )
    return qparams


def _quantize_bias_tensor(
    *,
    tensor: TensorIR,
    input_qparams: QuantParamIR,
    weight_qparams: QuantParamIR,
    reporter: _StrictQuantizationReporter,
) -> None:
    if not isinstance(tensor.data, np.ndarray):
        raise StrictFullIntegerQuantizationError(
            f"Bias tensor must be constant for strict full integer quantization: {tensor.name}"
        )
    input_scale = float(input_qparams.scale[0])
    scales = np.asarray(weight_qparams.scale, dtype=np.float64).reshape(-1)
    if scales.size == 0:
        raise StrictFullIntegerQuantizationError(f"Weight qparams have no scales for bias: {tensor.name}")
    if scales.size == 1:
        bias_scales = np.asarray([input_scale * float(scales[0])], dtype=np.float64)
    else:
        bias_scales = input_scale * scales
    bias_values = np.asarray(tensor.data, dtype=np.float64).reshape(-1)
    if bias_scales.size not in {1, bias_values.size}:
        raise StrictFullIntegerQuantizationError(
            f"Bias scale count mismatch: tensor={tensor.name} bias={bias_values.size} scales={bias_scales.size}"
        )
    denom = bias_scales if bias_scales.size > 1 else np.full_like(bias_values, float(bias_scales[0]))
    q_bias = np.round(bias_values / denom)
    tensor.data = np.clip(q_bias, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
    tensor.dtype = "INT32"
    tensor.quantization = QuantParamIR(
        scale=[float(v) for v in bias_scales.reshape(-1).tolist()],
        zero_point=[0 for _ in range(int(bias_scales.size))],
        quantized_dimension=int(weight_qparams.quantized_dimension),
    )
    reporter.record_tensor(
        tensor_name=tensor.name,
        dtype="INT32",
        kind="bias",
        qparams=tensor.quantization,
    )


def _kernel_weight_quant_axis_for_strict(op_type: str, tensor: TensorIR) -> int:
    if op_type == "DEPTHWISE_CONV_2D":
        return max(0, len(tensor.shape) - 1)
    if op_type in {"CONV_2D", "FULLY_CONNECTED", "TRANSPOSE_CONV"}:
        return 0
    if op_type == "BATCH_MATMUL":
        return 0
    return 0


def _force_same_qparams(
    *,
    model_ir: ModelIR,
    tensor_names: Iterable[str],
    qparams: QuantParamIR,
    dtype: str,
    reporter: _StrictQuantizationReporter,
) -> None:
    for tensor_name in tensor_names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            raise StrictFullIntegerQuantizationError(f"Missing tensor: {tensor_name}")
        if tensor.data is not None and tensor.dtype == "FLOAT32":
            _quantize_float_constant_with_qparams(
                tensor=tensor,
                dtype=dtype,
                qparams=qparams,
            )
        else:
            tensor.dtype = str(dtype).upper()
            tensor.quantization = _clone_quant_param(qparams)
        reporter.record_tensor(
            tensor_name=str(tensor_name),
            dtype=tensor.dtype,
            kind="activation",
            qparams=(
                tensor.quantization
                if isinstance(tensor.quantization, QuantParamIR)
                else None
            ),
        )


def _normalize_quantization_controls(
    *,
    calibration_method: str,
    calibration_percentile: float,
    min_numel: int,
    min_abs_max: float,
    scale_floor: float,
) -> _QuantizationControls:
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

    graph_index = ModelIRGraphIndex(clone)
    candidate_ops = list(clone.operators)

    def ensure_dequantized_tensor(
        quant_tensor_name: str,
        *,
        before_op: OperatorIR,
    ) -> Optional[str]:
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
        before_index = graph_index.operator_index(before_op)
        if before_index is None:
            return None
        graph_index.insert_operator(
            int(before_index),
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[quant_tensor_name],
                outputs=[deq_name],
                options={},
            ),
        )
        dequantized_tensor_map[quant_tensor_name] = deq_name
        return deq_name

    for op in candidate_ops:
        if op.op_type in _DYNAMIC_RANGE_KERNEL_OPS and len(op.inputs) >= 2:
            weight_name = op.inputs[1]
            tensor = clone.tensors.get(weight_name)
            if tensor is not None and _is_float_constant_tensor(
                tensor=tensor,
                graph_input_names=graph_input_names,
            ):
                quant_mode = "per-channel" if quant_type == "per-channel" else "per-tensor"
                quant_axis = _kernel_weight_quant_axis(op.op_type, tensor)
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

        if op.op_type in _DYNAMIC_RANGE_CONST_DEQUANT_OPS:
            new_inputs = list(op.inputs)
            for idx, input_name in enumerate(list(op.inputs)):
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
                deq_name = ensure_dequantized_tensor(
                    input_name,
                    before_op=op,
                )
                if deq_name is not None:
                    new_inputs[idx] = deq_name
            op_index = graph_index.operator_index(op)
            if op_index is not None and new_inputs != list(op.inputs):
                graph_index.replace_operator_inputs(int(op_index), new_inputs)

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
    calibration_ranges: Optional[Dict[str, TensorCalibrationRange]] = None,
    return_report: bool = False,
) -> Any:
    result = _build_strict_full_integer_model_ir(
        model_ir,
        quant_type=quant_type,
        input_quant_dtype="int8",
        output_quant_dtype="int8",
        internal_activation_dtype="int8",
        full_integer_io=False,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
        calibration_ranges=calibration_ranges,
        collect_report=bool(return_report),
    )
    result.model_ir.description = f"{result.model_ir.description} (strict_integer_quantized)"
    return result if return_report else result.model_ir


def _build_strict_full_integer_model_ir(
    model_ir: ModelIR,
    *,
    quant_type: str,
    input_quant_dtype: str,
    output_quant_dtype: str,
    internal_activation_dtype: str,
    full_integer_io: bool,
    calibration_method: str,
    calibration_percentile: float,
    min_numel: int,
    min_abs_max: float,
    scale_floor: float,
    calibration_ranges: Optional[Dict[str, TensorCalibrationRange]],
    collect_report: bool = True,
) -> QuantizedModelResult:
    if calibration_ranges is None or len(calibration_ranges) == 0:
        raise StrictFullIntegerQuantizationError(
            "flatbuffer_direct strict integer quantization requires calibration data."
        )

    quant_type = _normalize_quant_type(quant_type)
    controls = _normalize_quantization_controls(
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
    )
    clone = _clone_model_ir(model_ir)
    _elide_identity_operators(clone)
    graph_index = ModelIRGraphIndex(clone)
    internal_dtype = str(internal_activation_dtype).upper()
    input_dtype = str(input_quant_dtype).upper()
    output_dtype = str(output_quant_dtype).upper()
    if input_dtype == "FLOAT32" or output_dtype == "FLOAT32":
        raise StrictFullIntegerQuantizationError(
            "strict full integer quantization does not allow FLOAT32 input/output dtype."
        )

    reporter = _StrictQuantizationReporter.create(
        enabled=bool(collect_report),
        full_integer_io=bool(full_integer_io),
        calibration_ranges=calibration_ranges,
    )

    def activation_dtype(tensor_name: str) -> str:
        return _activation_dtype_for_tensor(
            tensor_name=str(tensor_name),
            model_ir=clone,
            full_integer_io=bool(full_integer_io),
            input_quant_dtype=input_dtype,
            output_quant_dtype=output_dtype,
            internal_activation_dtype=internal_dtype,
        )

    pre_ops: List[OperatorIR] = []
    post_ops: List[OperatorIR] = []

    def replace_all_operator_inputs(old_name: str, new_name: str) -> None:
        for op_index in sorted(set(graph_index.consumer_indices(str(old_name)))):
            op = clone.operators[int(op_index)]
            new_inputs = [
                str(new_name) if str(value) == str(old_name) else value
                for value in op.inputs
            ]
            if new_inputs != list(op.inputs):
                graph_index.replace_operator_inputs(int(op_index), new_inputs)

    def replace_producer_outputs(old_name: str, new_name: str) -> None:
        producer = graph_index.producer(str(old_name))
        if producer is None:
            return
        producer_index = graph_index.operator_index(producer)
        if producer_index is None:
            return
        graph_index.replace_operator_outputs(
            int(producer_index),
            [
                str(new_name) if str(value) == str(old_name) else value
                for value in producer.outputs
            ],
        )

    if not full_integer_io:
        for input_name in list(clone.inputs):
            old_tensor = clone.tensors[str(input_name)]
            if old_tensor.dtype != "FLOAT32":
                continue
            q_name = _make_unique_tensor_name(f"{input_name}_quantized_internal", clone.tensors)
            qparams = activation_qparams_from_range(
                min_value=_require_tensor_range(
                    tensor_name=str(input_name),
                    calibration_ranges=calibration_ranges,
                ).min_value,
                max_value=_require_tensor_range(
                    tensor_name=str(input_name),
                    calibration_ranges=calibration_ranges,
                ).max_value,
                dtype=internal_dtype,
                scale_floor=float(controls["scale_floor"]),
            )
            clone.tensors[q_name] = TensorIR(
                name=q_name,
                dtype=internal_dtype,
                shape=list(old_tensor.shape),
                shape_signature=list(old_tensor.shape_signature) if old_tensor.shape_signature is not None else list(old_tensor.shape),
                data=None,
                is_variable=False,
                quantization=qparams,
                logical_layout=old_tensor.logical_layout,
            )
            pre_ops.append(OperatorIR(op_type="QUANTIZE", inputs=[str(input_name)], outputs=[q_name]))
            replace_all_operator_inputs(str(input_name), q_name)
            reporter.record_tensor(
                tensor_name=q_name,
                dtype=internal_dtype,
                kind="activation",
                qparams=qparams,
            )
        for output_name in list(clone.outputs):
            old_tensor = clone.tensors[str(output_name)]
            if old_tensor.dtype != "FLOAT32":
                continue
            q_name = str(output_name)
            float_name = _make_unique_tensor_name(f"{output_name}_dequantized_output", clone.tensors)
            clone.tensors[float_name] = TensorIR(
                name=float_name,
                dtype="FLOAT32",
                shape=list(old_tensor.shape),
                shape_signature=list(old_tensor.shape_signature) if old_tensor.shape_signature is not None else list(old_tensor.shape),
                data=None,
                is_variable=False,
                quantization=None,
                logical_layout=old_tensor.logical_layout,
            )
            post_ops.append(OperatorIR(op_type="DEQUANTIZE", inputs=[q_name], outputs=[float_name]))
            clone.outputs = [float_name if str(v) == q_name else v for v in clone.outputs]
    else:
        for input_name in list(clone.inputs):
            old_tensor = clone.tensors[str(input_name)]
            if old_tensor.dtype != "FLOAT32":
                continue
            input_range = _require_tensor_range(
                tensor_name=str(input_name),
                calibration_ranges=calibration_ranges,
            )
            external_qparams = activation_qparams_from_range(
                min_value=float(input_range.min_value),
                max_value=float(input_range.max_value),
                dtype=input_dtype,
                scale_floor=float(controls["scale_floor"]),
            )
            old_tensor.dtype = input_dtype
            old_tensor.quantization = external_qparams
            old_tensor.data = None
            reporter.record_tensor(
                tensor_name=str(input_name),
                dtype=input_dtype,
                kind="graph_input",
                qparams=external_qparams,
            )
            if input_dtype != internal_dtype:
                internal_name = _make_unique_tensor_name(f"{input_name}_quantized_internal", clone.tensors)
                internal_qparams = activation_qparams_from_range(
                    min_value=float(input_range.min_value),
                    max_value=float(input_range.max_value),
                    dtype=internal_dtype,
                    scale_floor=float(controls["scale_floor"]),
                )
                clone.tensors[internal_name] = TensorIR(
                    name=internal_name,
                    dtype=internal_dtype,
                    shape=list(old_tensor.shape),
                    shape_signature=list(old_tensor.shape_signature) if old_tensor.shape_signature is not None else list(old_tensor.shape),
                    data=None,
                    is_variable=False,
                    quantization=internal_qparams,
                    logical_layout=old_tensor.logical_layout,
                )
                pre_ops.append(OperatorIR(op_type="QUANTIZE", inputs=[str(input_name)], outputs=[internal_name]))
                replace_all_operator_inputs(str(input_name), internal_name)
                reporter.record_tensor(
                    tensor_name=internal_name,
                    dtype=internal_dtype,
                    kind="activation",
                    qparams=internal_qparams,
                )

        for output_name in list(clone.outputs):
            old_tensor = clone.tensors[str(output_name)]
            if old_tensor.dtype != "FLOAT32":
                continue
            if output_dtype == internal_dtype:
                continue
            producer_op = graph_index.producer(str(output_name))
            fixed_external = fixed_activation_qparams_for_op(
                op_type=str(producer_op.op_type) if producer_op is not None else "",
                dtype=output_dtype,
            )
            if fixed_external is None:
                output_range = _require_tensor_range(
                    tensor_name=str(output_name),
                    calibration_ranges=calibration_ranges,
                )
                external_qparams = activation_qparams_from_range(
                    min_value=float(output_range.min_value),
                    max_value=float(output_range.max_value),
                    dtype=output_dtype,
                    scale_floor=float(controls["scale_floor"]),
                )
            else:
                external_qparams = fixed_external
            internal_name = _make_unique_tensor_name(f"{output_name}_quantized_internal", clone.tensors)
            clone.tensors[internal_name] = TensorIR(
                name=internal_name,
                dtype="FLOAT32",
                shape=list(old_tensor.shape),
                shape_signature=list(old_tensor.shape_signature) if old_tensor.shape_signature is not None else list(old_tensor.shape),
                data=None,
                is_variable=False,
                quantization=None,
                logical_layout=old_tensor.logical_layout,
            )
            replace_producer_outputs(str(output_name), internal_name)
            old_tensor.dtype = output_dtype
            old_tensor.quantization = external_qparams
            old_tensor.data = None
            post_ops.append(OperatorIR(op_type="QUANTIZE", inputs=[internal_name], outputs=[str(output_name)]))
            reporter.record_tensor(
                tensor_name=str(output_name),
                dtype=output_dtype,
                kind="graph_output",
                qparams=external_qparams,
            )

    for op_idx, op in enumerate(clone.operators):
        op_type = str(op.op_type)
        if op_type not in _STRICT_FULL_INTEGER_SUPPORTED_OPS:
            raise StrictFullIntegerQuantizationError(
                f"Unsupported op for strict full integer quantization: index={op_idx} op_type={op_type}"
            )

        for input_name in list(op.inputs):
            if str(input_name) == "":
                continue
            tensor = clone.tensors.get(str(input_name), None)
            if tensor is None:
                raise StrictFullIntegerQuantizationError(
                    f"Missing input tensor for strict full integer quantization: op={op_type} tensor={input_name}"
                )
            if tensor.dtype == "FLOAT32" and tensor.data is None:
                _ensure_activation_quantized(
                    model_ir=clone,
                    tensor_name=str(input_name),
                    calibration_ranges=calibration_ranges,
                    dtype=activation_dtype(str(input_name)),
                    scale_floor=float(controls["scale_floor"]),
                    reporter=reporter,
                )

        output_fixed = fixed_activation_qparams_for_op(
            op_type=op_type,
            dtype=activation_dtype(str(op.outputs[0])) if len(op.outputs) > 0 else internal_dtype,
        )
        for output_name in list(op.outputs):
            if str(output_name) == "":
                continue
            out_tensor = clone.tensors.get(str(output_name), None)
            if out_tensor is None:
                raise StrictFullIntegerQuantizationError(
                    f"Missing output tensor for strict full integer quantization: op={op_type} tensor={output_name}"
                )
            if out_tensor.dtype == "FLOAT32":
                if op_type in _STRICT_FULL_INTEGER_PASSTHROUGH_OPS and len(op.inputs) > 0:
                    ref = clone.tensors[str(op.inputs[0])]
                    if not isinstance(ref.quantization, QuantParamIR):
                        raise StrictFullIntegerQuantizationError(
                            f"Passthrough op input is not quantized: op={op_type} tensor={op.inputs[0]}"
                        )
                    out_tensor.dtype = ref.dtype
                    out_tensor.quantization = _clone_quant_param(ref.quantization)
                    out_tensor.data = None
                    reporter.record_tensor(
                        tensor_name=str(output_name),
                        dtype=out_tensor.dtype,
                        kind="activation",
                        qparams=out_tensor.quantization,
                    )
                else:
                    same_qparam_names = (
                        _same_qparam_tensor_names_for_op(op)
                        if op_type in _STRICT_FULL_INTEGER_SAME_QPARAM_INDICES
                        else []
                    )
                    fallback_qparams = None
                    if output_fixed is None and str(output_name) not in calibration_ranges:
                        fallback_qparams = _derive_same_qparams_from_available_tensor(
                            model_ir=clone,
                            tensor_names=same_qparam_names or [str(v) for v in op.inputs],
                            calibration_ranges=calibration_ranges,
                            dtype=activation_dtype(str(output_name)),
                            scale_floor=float(controls["scale_floor"]),
                        )
                    if fallback_qparams is not None and same_qparam_names:
                        _force_same_qparams(
                            model_ir=clone,
                            tensor_names=same_qparam_names,
                            qparams=fallback_qparams,
                            dtype=activation_dtype(str(output_name)),
                            reporter=reporter,
                        )
                    else:
                        _ensure_activation_quantized(
                            model_ir=clone,
                            tensor_name=str(output_name),
                            calibration_ranges=calibration_ranges,
                            dtype=activation_dtype(str(output_name)),
                            scale_floor=float(controls["scale_floor"]),
                            reporter=reporter,
                            fixed_qparams=output_fixed or fallback_qparams,
                        )

        if op_type in {"FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL", "TRANSPOSE_CONV"}:
            activation_input_idx = 2 if op_type == "TRANSPOSE_CONV" else 0
            weight_input_idx = 1
            if len(op.inputs) <= max(activation_input_idx, weight_input_idx):
                raise StrictFullIntegerQuantizationError(f"{op_type} requires a weight input.")
            input_tensor = clone.tensors[str(op.inputs[activation_input_idx])]
            if not isinstance(input_tensor.quantization, QuantParamIR):
                raise StrictFullIntegerQuantizationError(
                    f"{op_type} input tensor is missing qparams: {op.inputs[activation_input_idx]}"
                )
            weight_tensor = clone.tensors[str(op.inputs[weight_input_idx])]
            weight_qparams = _quantize_weight_tensor(
                tensor=weight_tensor,
                quant_type=quant_type,
                quantized_dimension=_kernel_weight_quant_axis_for_strict(op_type, weight_tensor),
                calibration_method=str(controls["calibration_method"]),
                calibration_percentile=float(controls["calibration_percentile"]),
                scale_floor=float(controls["scale_floor"]),
                reporter=reporter,
            )
            bias_input_idx = 3 if op_type == "TRANSPOSE_CONV" else 2
            if len(op.inputs) > bias_input_idx and str(op.inputs[bias_input_idx]) != "":
                bias_tensor = clone.tensors.get(str(op.inputs[bias_input_idx]), None)
                if bias_tensor is not None and bias_tensor.data is not None:
                    _quantize_bias_tensor(
                        tensor=bias_tensor,
                        input_qparams=input_tensor.quantization,
                        weight_qparams=weight_qparams,
                        reporter=reporter,
                    )
        elif op_type in _STRICT_FULL_INTEGER_SAME_QPARAM_INDICES and len(op.outputs) > 0:
            out_tensor = clone.tensors[str(op.outputs[0])]
            if not isinstance(out_tensor.quantization, QuantParamIR):
                if out_tensor.dtype == "FLOAT32":
                    raise StrictFullIntegerQuantizationError(
                        f"{op_type} output tensor is missing qparams: {op.outputs[0]}"
                    )
            else:
                _force_same_qparams(
                    model_ir=clone,
                    tensor_names=_same_qparam_tensor_names_for_op(op),
                    qparams=out_tensor.quantization,
                    dtype=out_tensor.dtype,
                    reporter=reporter,
                )
        else:
            for input_name in list(op.inputs):
                tensor = clone.tensors.get(str(input_name), None)
                if tensor is None or tensor.data is None or tensor.dtype != "FLOAT32":
                    continue
                rng = TensorCalibrationRange(
                    min_value=float(np.min(np.asarray(tensor.data, dtype=np.float32))),
                    max_value=float(np.max(np.asarray(tensor.data, dtype=np.float32))),
                    num_samples=1,
                )
                qparams = activation_qparams_from_range(
                    min_value=rng.min_value,
                    max_value=rng.max_value,
                    dtype=activation_dtype(str(input_name)),
                    scale_floor=float(controls["scale_floor"]),
                )
                _quantize_float_constant_with_qparams(
                    tensor=tensor,
                    dtype=activation_dtype(str(input_name)),
                    qparams=qparams,
                )
                reporter.record_tensor(
                    tensor_name=str(input_name),
                    dtype=tensor.dtype,
                    kind="constant",
                    qparams=qparams,
                )

        reporter.record_operator(index=int(op_idx), operator=op)

    for op_index, pre_op in enumerate(pre_ops):
        graph_index.insert_operator(int(op_index), pre_op)
    for post_op in post_ops:
        graph_index.append_operator(post_op)
    _validate_strict_full_integer_model_ir(
        model_ir=clone,
        allow_float_boundary=not full_integer_io,
    )
    return QuantizedModelResult(model_ir=clone, report=reporter.payload)


def _validate_strict_full_integer_model_ir(
    *,
    model_ir: ModelIR,
    allow_float_boundary: bool,
) -> None:
    allowed_boundary_float_tensors: Set[str] = set()
    if allow_float_boundary:
        allowed_boundary_float_tensors.update(str(v) for v in model_ir.inputs)
        allowed_boundary_float_tensors.update(str(v) for v in model_ir.outputs)

    used_tensors = _used_tensor_names(model_ir)
    for op_idx, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type == "QUANTIZE":
            continue
        if op_type == "DEQUANTIZE" and allow_float_boundary:
            continue
        if op_type not in _STRICT_FULL_INTEGER_SUPPORTED_OPS:
            raise StrictFullIntegerQuantizationError(
                f"Unsupported op remains in strict full integer model: index={op_idx} op_type={op_type}"
            )

    for tensor_name in sorted(used_tensors):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            raise StrictFullIntegerQuantizationError(f"Used tensor is missing: {tensor_name}")
        if tensor.dtype == "FLOAT32":
            if str(tensor_name) in allowed_boundary_float_tensors:
                continue
            raise StrictFullIntegerQuantizationError(
                f"Float tensor remains in strict full integer model: {tensor_name}"
            )
        if tensor.dtype in {"INT8", "UINT8", "INT16"} and tensor.quantization is None:
            raise StrictFullIntegerQuantizationError(
                f"Quantized tensor is missing qparams: {tensor_name}"
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
    calibration_ranges: Optional[Dict[str, TensorCalibrationRange]] = None,
    return_report: bool = False,
) -> Any:
    result = _build_strict_full_integer_model_ir(
        model_ir,
        quant_type=quant_type,
        input_quant_dtype=input_quant_dtype,
        output_quant_dtype=output_quant_dtype,
        internal_activation_dtype="int8",
        full_integer_io=True,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
        calibration_ranges=calibration_ranges,
        collect_report=bool(return_report),
    )
    result.model_ir.description = f"{result.model_ir.description} (strict_full_integer_quantized)"
    return result if return_report else result.model_ir


def build_integer_quantized_with_int16_act_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
    calibration_ranges: Optional[Dict[str, TensorCalibrationRange]] = None,
) -> Any:
    result = _build_strict_full_integer_model_ir(
        model_ir,
        quant_type=quant_type,
        input_quant_dtype="int16",
        output_quant_dtype="int16",
        internal_activation_dtype="int16",
        full_integer_io=False,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
        calibration_ranges=calibration_ranges,
        collect_report=False,
    )
    result.model_ir.description = f"{result.model_ir.description} (strict_integer_quant_with_int16_act)"
    return result.model_ir


def build_full_integer_quantized_with_int16_act_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
    calibration_method: str = "max",
    calibration_percentile: float = 99.99,
    min_numel: int = 1,
    min_abs_max: float = 0.0,
    scale_floor: float = 1e-8,
    calibration_ranges: Optional[Dict[str, TensorCalibrationRange]] = None,
) -> Any:
    result = _build_strict_full_integer_model_ir(
        model_ir,
        quant_type=quant_type,
        input_quant_dtype="int16",
        output_quant_dtype="int16",
        internal_activation_dtype="int16",
        full_integer_io=True,
        calibration_method=calibration_method,
        calibration_percentile=calibration_percentile,
        min_numel=min_numel,
        min_abs_max=min_abs_max,
        scale_floor=scale_floor,
        calibration_ranges=calibration_ranges,
        collect_report=False,
    )
    result.model_ir.description = f"{result.model_ir.description} (strict_full_integer_quant_with_int16_act)"
    return result.model_ir
