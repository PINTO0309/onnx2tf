from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, QuantParamIR, TensorIR


def _append_tensor_lineage_event(
    *,
    model_ir: ModelIR,
    event: Dict[str, Any],
) -> None:
    if "tensor_lineage_events" not in model_ir.metadata:
        model_ir.metadata["tensor_lineage_events"] = []
    events = model_ir.metadata["tensor_lineage_events"]
    if not isinstance(events, list):
        events = []
        model_ir.metadata["tensor_lineage_events"] = events
    normalized_event = dict(event)
    normalized_event["event_index"] = int(len(events))
    events.append(normalized_event)


def _build_tensor_producer_map(model_ir: ModelIR) -> Dict[str, int]:
    producers: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[output_name] = op_idx
    return producers


def _topologically_sort_operators(model_ir: ModelIR) -> Dict[str, int]:
    """
    Reorder operators so every producer appears before its consumers.

    This is a safety net for late graph-rewrite passes that may rewrite tensor
    names without physically moving operators.
    """
    op_count = len(model_ir.operators)
    if op_count <= 1:
        return {"reordered_operators": 0, "cycle_detected": 0}

    producers = _build_tensor_producer_map(model_ir)
    dependencies: List[set[int]] = [set() for _ in range(op_count)]
    dependents: List[set[int]] = [set() for _ in range(op_count)]

    for consumer_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            producer_idx = producers.get(str(input_name), None)
            if producer_idx is None or int(producer_idx) == int(consumer_idx):
                continue
            if int(producer_idx) in dependencies[int(consumer_idx)]:
                continue
            dependencies[int(consumer_idx)].add(int(producer_idx))
            dependents[int(producer_idx)].add(int(consumer_idx))

    indegree = [int(len(v)) for v in dependencies]
    ready = [int(idx) for idx, deg in enumerate(indegree) if int(deg) == 0]
    ready.sort()
    sorted_indices: List[int] = []

    while len(ready) > 0:
        current = int(ready.pop(0))
        sorted_indices.append(current)
        for dependent_idx in sorted(list(dependents[current])):
            indegree[int(dependent_idx)] -= 1
            if int(indegree[int(dependent_idx)]) == 0:
                ready.append(int(dependent_idx))
        ready.sort()

    if len(sorted_indices) != op_count:
        return {"reordered_operators": 0, "cycle_detected": 1}

    if sorted_indices == [int(v) for v in range(op_count)]:
        return {"reordered_operators": 0, "cycle_detected": 0}

    original_ops = list(model_ir.operators)
    model_ir.operators = [original_ops[int(idx)] for idx in sorted_indices]
    return {
        "reordered_operators": int(
            sum(1 for new_idx, old_idx in enumerate(sorted_indices) if int(new_idx) != int(old_idx))
        ),
        "cycle_detected": 0,
    }


def _rename_tensor_globally(
    model_ir: ModelIR,
    old_name: str,
    new_name: str,
) -> None:
    if old_name == new_name:
        return

    for op in model_ir.operators:
        if len(op.inputs) > 0:
            op.inputs = [new_name if input_name == old_name else input_name for input_name in op.inputs]
        if len(op.outputs) > 0:
            op.outputs = [new_name if output_name == old_name else output_name for output_name in op.outputs]

    if len(model_ir.inputs) > 0:
        model_ir.inputs = [new_name if input_name == old_name else input_name for input_name in model_ir.inputs]
    if len(model_ir.outputs) > 0:
        model_ir.outputs = [new_name if output_name == old_name else output_name for output_name in model_ir.outputs]

    old_tensor = model_ir.tensors.get(old_name, None)
    if old_tensor is None:
        if new_name in model_ir.tensors:
            del model_ir.tensors[new_name]
        return
    old_tensor.name = new_name
    if new_name in model_ir.tensors and new_name != old_name:
        del model_ir.tensors[new_name]
    del model_ir.tensors[old_name]
    model_ir.tensors[new_name] = old_tensor
    _append_tensor_lineage_event(
        model_ir=model_ir,
        event={
            "kind": "rename_tensor",
            "old_name": str(old_name),
            "new_name": str(new_name),
        },
    )


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


def _quantize_prelu_alpha(
    alpha: np.ndarray,
    target_dtype: str,
) -> Tuple[np.ndarray, QuantParamIR]:
    alpha = np.asarray(alpha, dtype=np.float32)
    if str(target_dtype) == "INT8":
        max_abs = float(np.max(np.abs(alpha))) if alpha.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.clip(np.round(alpha / scale), -128, 127).astype(np.int8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if str(target_dtype) == "UINT8":
        mn = float(np.min(alpha)) if alpha.size > 0 else 0.0
        mx = float(np.max(alpha)) if alpha.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        quantized = np.clip(np.round(alpha / scale) + zp, 0, 255).astype(np.uint8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(f"Quantized PRELU alpha requires INT8/UINT8. got={target_dtype}")


def _quantize_tensor_per_tensor(
    values: np.ndarray,
    target_dtype: str,
) -> Tuple[np.ndarray, QuantParamIR]:
    values = np.asarray(values, dtype=np.float32)
    if str(target_dtype) == "INT8":
        max_abs = float(np.max(np.abs(values))) if values.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        quantized = np.clip(np.round(values / scale), -128, 127).astype(np.int8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if str(target_dtype) == "UINT8":
        mn = float(np.min(values)) if values.size > 0 else 0.0
        mx = float(np.max(values)) if values.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        quantized = np.clip(np.round(values / scale) + zp, 0, 255).astype(np.uint8)
        return quantized, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(f"Per-tensor quantization supports INT8/UINT8 only. got={target_dtype}")


def _get_per_tensor_scale_zero_point(quantization: Any) -> Optional[Tuple[float, int]]:
    if quantization is None:
        return None
    if not _is_per_tensor_quantization(quantization):
        return None
    if isinstance(quantization, QuantParamIR):
        if len(list(quantization.scale)) == 0 or len(list(quantization.zero_point)) == 0:
            return None
        return float(quantization.scale[0]), int(quantization.zero_point[0])
    if isinstance(quantization, dict):
        try:
            scale = np.asarray(quantization.get("scale", []), dtype=np.float32).reshape(-1)
            zero_point = np.asarray(quantization.get("zero_point", []), dtype=np.int64).reshape(-1)
            if scale.size == 0 or zero_point.size == 0:
                return None
            return float(scale[0]), int(zero_point[0])
        except Exception:
            return None
    return None


def _is_same_per_tensor_quantization(
    quantization_a: Any,
    quantization_b: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    qparams_a = _get_per_tensor_scale_zero_point(quantization_a)
    qparams_b = _get_per_tensor_scale_zero_point(quantization_b)
    if qparams_a is None or qparams_b is None:
        return False
    scale_a, zp_a = qparams_a
    scale_b, zp_b = qparams_b
    if int(zp_a) != int(zp_b):
        return False
    return bool(np.isclose(float(scale_a), float(scale_b), rtol=rtol, atol=atol))


def _quant_scale_count(quantization: Any) -> int:
    if quantization is None:
        return 0
    if isinstance(quantization, QuantParamIR):
        return len(list(quantization.scale))
    if isinstance(quantization, dict):
        scale = quantization.get("scale", None)
        if scale is None:
            return 0
        if isinstance(scale, (list, tuple)):
            return len(list(scale))
        try:
            return int(np.asarray(scale).size)
        except Exception:
            return 0
    return 0


def _is_per_tensor_quantization(quantization: Any) -> bool:
    count = _quant_scale_count(quantization)
    return count <= 1


def _permute_shape(shape: Optional[List[int]], perm: List[int]) -> Optional[List[int]]:
    if shape is None:
        return None
    rank = len(shape)
    if len(perm) != rank:
        return None
    if sorted(perm) != [int(i) for i in range(rank)]:
        return None
    return [int(shape[int(axis)]) for axis in perm]


def _shapes_equal(shape_a: Optional[List[int]], shape_b: Optional[List[int]]) -> bool:
    if shape_a is None or shape_b is None:
        return False
    if len(shape_a) != len(shape_b):
        return False
    return all(int(a) == int(b) for a, b in zip(shape_a, shape_b))


def _is_unknown_shape(shape: Optional[List[int]]) -> bool:
    if shape is None:
        return True
    if len(shape) == 0:
        return True
    if any(int(dim) < 0 for dim in shape):
        return True
    # Internal placeholder shape used before static shape propagation.
    if len(shape) == 1 and int(shape[0]) == 1:
        return True
    return False


def _shapes_match_if_known(shape_a: Optional[List[int]], shape_b: Optional[List[int]]) -> bool:
    if _is_unknown_shape(shape_a) or _is_unknown_shape(shape_b):
        return True
    return _shapes_equal(shape_a, shape_b)


def _all_per_tensor_quantized(tensors: List[Optional[TensorIR]]) -> bool:
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.quantization is None:
            continue
        if not _is_per_tensor_quantization(tensor.quantization):
            return False
    return True
