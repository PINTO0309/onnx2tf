from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
)
from onnx2tf.tflite_builder.ir import ModelIR

_TFLITE_DTYPE_TO_NUMPY_DTYPE = {
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
    "FLOAT64": np.dtype(np.float64),
    "INT8": np.dtype(np.int8),
    "INT16": np.dtype(np.int16),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "UINT8": np.dtype(np.uint8),
    "UINT16": np.dtype(np.uint16),
    "UINT32": np.dtype(np.uint32),
    "UINT64": np.dtype(np.uint64),
    "BOOL": np.dtype(np.bool_),
    "STRING": np.dtype(np.object_),
}


def _append_tensor_lineage_event(*, model_ir: ModelIR, event: Dict[str, Any]) -> None:
    events = model_ir.metadata.setdefault("tensor_lineage_events", [])
    if not isinstance(events, list):
        events = []
        model_ir.metadata["tensor_lineage_events"] = events
    normalized = dict(event)
    normalized["event_index"] = len(events)
    events.append(normalized)


def _cast_const_array_to_tflite_dtype(
    values: Any,
    out_dtype: str,
) -> Optional[np.ndarray]:
    dtype_key = str(out_dtype).upper()
    target_dtype = _TFLITE_DTYPE_TO_NUMPY_DTYPE.get(dtype_key, None)
    if target_dtype is None:
        return None
    try:
        src = np.asarray(values)
        if dtype_key == "BOOL":
            return np.asarray(src != 0, dtype=np.bool_)
        return np.asarray(src, dtype=target_dtype)
    except Exception:
        return None


def _evaluate_constant_pool2d(
    *,
    input_data: Any,
    op_type: str,
    padding: str,
    stride_h: int,
    stride_w: int,
    filter_h: int,
    filter_w: int,
) -> Optional[np.ndarray]:
    try:
        x = np.asarray(input_data)
    except Exception:
        return None
    if x.ndim != 4:
        return None
    if int(stride_h) <= 0 or int(stride_w) <= 0 or int(filter_h) <= 0 or int(filter_w) <= 0:
        return None

    x_work = np.asarray(x)
    n_dim, in_h, in_w, channels = [int(v) for v in list(x_work.shape)]
    if min(n_dim, in_h, in_w, channels) <= 0:
        return None

    padding_mode = str(padding).upper()
    if padding_mode not in {"SAME", "VALID"}:
        return None

    if padding_mode == "SAME":
        out_h = int(np.ceil(float(in_h) / float(stride_h)))
        out_w = int(np.ceil(float(in_w) / float(stride_w)))
        pad_along_h = max((int(out_h) - 1) * int(stride_h) + int(filter_h) - int(in_h), 0)
        pad_along_w = max((int(out_w) - 1) * int(stride_w) + int(filter_w) - int(in_w), 0)
        pad_top = int(pad_along_h // 2)
        pad_bottom = int(pad_along_h - pad_top)
        pad_left = int(pad_along_w // 2)
        pad_right = int(pad_along_w - pad_left)
    else:
        out_h = ((int(in_h) - int(filter_h)) // int(stride_h)) + 1
        out_w = ((int(in_w) - int(filter_w)) // int(stride_w)) + 1
        pad_top = pad_bottom = pad_left = pad_right = 0

    if int(out_h) <= 0 or int(out_w) <= 0:
        return None

    pool_type = str(op_type).upper()
    if pool_type == "AVERAGE_POOL_2D":
        x_eval = np.asarray(x_work, dtype=np.float64)
        pad_value = 0.0
        out_dtype = np.float64
        reduce_fn = np.mean
    elif pool_type == "MAX_POOL_2D":
        if x_work.dtype == np.bool_:
            x_eval = np.asarray(x_work, dtype=np.bool_)
            pad_value = False
            out_dtype = np.bool_
        elif np.issubdtype(x_work.dtype, np.integer):
            x_eval = np.asarray(x_work)
            pad_value = np.iinfo(x_eval.dtype).min
            out_dtype = x_eval.dtype
        else:
            x_eval = np.asarray(x_work, dtype=np.float64)
            pad_value = -np.inf
            out_dtype = np.float64
        reduce_fn = np.max
    else:
        return None

    x_eval = np.pad(
        x_eval,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )
    out = np.empty((int(n_dim), int(out_h), int(out_w), int(channels)), dtype=out_dtype)
    for out_y in range(int(out_h)):
        start_y = int(out_y) * int(stride_h)
        end_y = start_y + int(filter_h)
        for out_x in range(int(out_w)):
            start_x = int(out_x) * int(stride_w)
            end_x = start_x + int(filter_w)
            window = x_eval[:, start_y:end_y, start_x:end_x, :]
            out[:, out_y, out_x, :] = reduce_fn(window, axis=(1, 2))
    return np.asarray(out)


def _evaluate_constant_pad(
    *,
    input_data: Any,
    pads_data: Any,
    pad_value_data: Optional[Any] = None,
) -> Optional[np.ndarray]:
    try:
        x = np.asarray(input_data)
        pads = np.asarray(pads_data, dtype=np.int64)
    except Exception:
        return None
    if pads.ndim != 2 or pads.shape[1] != 2 or pads.shape[0] != x.ndim:
        return None
    if np.any(pads < 0):
        return None

    if pad_value_data is None:
        if x.dtype == np.bool_:
            pad_value: Any = False
        else:
            pad_value = 0
    else:
        try:
            pad_value_arr = np.asarray(pad_value_data)
        except Exception:
            return None
        if pad_value_arr.size != 1:
            return None
        pad_value = pad_value_arr.reshape(-1)[0].item()

    try:
        return np.pad(
            x,
            [(int(v[0]), int(v[1])) for v in pads.tolist()],
            mode="constant",
            constant_values=pad_value,
        )
    except Exception:
        return None


def _evaluate_constant_scatter_nd(
    *,
    indices_data: Any,
    updates_data: Any,
    shape_data: Any,
) -> Optional[np.ndarray]:
    try:
        indices = np.asarray(indices_data, dtype=np.int64)
        updates = np.asarray(updates_data)
        dense_shape = np.asarray(shape_data, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if indices.ndim < 1 or dense_shape.ndim != 1:
        return None
    if indices.shape[-1] < 0 or np.any(dense_shape < 0):
        return None

    index_depth = int(indices.shape[-1])
    output_rank = int(dense_shape.size)
    if int(index_depth) > int(output_rank):
        return None

    prefix_shape = tuple(int(v) for v in list(indices.shape[:-1]))
    suffix_shape = tuple(int(v) for v in list(dense_shape[index_depth:].tolist()))
    if tuple(int(v) for v in list(updates.shape)) != prefix_shape + suffix_shape:
        return None

    try:
        out = np.zeros(tuple(int(v) for v in list(dense_shape.tolist())), dtype=updates.dtype)
    except Exception:
        return None

    flat_indices = indices.reshape((-1, int(index_depth)))
    flat_updates = updates.reshape((int(flat_indices.shape[0]),) + suffix_shape)

    if int(index_depth) == 0:
        try:
            np.add(out, np.sum(flat_updates, axis=0), out=out)
        except Exception:
            return None
        return out

    upper_bounds = np.asarray(dense_shape[:index_depth], dtype=np.int64)
    if upper_bounds.size != int(index_depth):
        return None
    if np.any(flat_indices < 0) or np.any(flat_indices >= upper_bounds.reshape((1, -1))):
        return None

    try:
        for update_idx, target_index in enumerate(flat_indices):
            out[tuple(int(v) for v in list(target_index.tolist()))] += flat_updates[int(update_idx)]
    except Exception:
        return None
    return out


def _evaluate_constant_binary_elementwise(
    *,
    lhs_data: Any,
    rhs_data: Any,
    op_type: str,
) -> Optional[np.ndarray]:
    try:
        lhs = np.asarray(lhs_data)
        rhs = np.asarray(rhs_data)
    except Exception:
        return None
    if not (
        np.issubdtype(lhs.dtype, np.floating)
        and np.issubdtype(rhs.dtype, np.floating)
    ):
        return None
    try:
        if str(op_type) == "ADD":
            out = np.add(lhs, rhs)
        elif str(op_type) == "SUB":
            out = np.subtract(lhs, rhs)
        elif str(op_type) == "MUL":
            out = np.multiply(lhs, rhs)
        elif str(op_type) == "DIV":
            out = np.divide(lhs, rhs)
        else:
            return None
    except Exception:
        return None
    if not np.all(np.isfinite(out)):
        return None
    return np.asarray(out)


def _optimize_constant_input_cast_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold CAST when its input tensor already has embedded constant data.

    Rewrite:
      CONST(in) -> CAST -> out
    Into:
      CONST(out_casted_data) and remove CAST op.
    """
    rewritten = 0

    while True:
        changed = False
        for cast_idx, cast_op in enumerate(model_ir.operators):
            if str(cast_op.op_type) != "CAST" or len(cast_op.inputs) != 1 or len(cast_op.outputs) != 1:
                continue
            if bool(
                cast_op.options.get(
                    "preserveRuntimeCastForQuantizedAccumulator",
                    False,
                )
            ):
                continue

            in_name = str(cast_op.inputs[0])
            out_name = str(cast_op.outputs[0])
            if in_name == "" or out_name == "":
                continue
            if in_name == out_name:
                continue

            in_tensor = model_ir.tensors.get(in_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if in_tensor is None or out_tensor is None:
                continue
            if bool(in_tensor.is_variable) or in_tensor.data is None:
                continue

            out_dtype = str(cast_op.options.get("outDataType", "")).upper()
            if out_dtype == "":
                out_dtype = str(out_tensor.dtype).upper()
            if out_dtype == "":
                continue

            casted = _cast_const_array_to_tflite_dtype(in_tensor.data, out_dtype)
            if casted is None:
                continue

            out_tensor.data = casted
            out_tensor.dtype = str(out_dtype)
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "fold_constant_cast",
                    "input_name": str(in_name),
                    "output_name": str(out_name),
                    "out_dtype": str(out_dtype),
                },
            )

            del model_ir.operators[int(cast_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_constant_input_cast_chains": int(rewritten)}


def _optimize_constant_input_pool_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold constant-input pool operators by materializing their outputs as constant tensors.

    Rewrite:
      CONST(in) -> (AVERAGE_POOL_2D|MAX_POOL_2D) -> out
    Into:
      CONST(out_pooled_data) and remove the pool op.
    """
    rewritten = 0

    while True:
        changed = False
        for pool_idx, pool_op in enumerate(model_ir.operators):
            op_type = str(pool_op.op_type)
            if op_type not in {"AVERAGE_POOL_2D", "MAX_POOL_2D"}:
                continue
            if len(pool_op.inputs) != 1 or len(pool_op.outputs) != 1:
                continue

            in_name = str(pool_op.inputs[0])
            out_name = str(pool_op.outputs[0])
            if in_name == "" or out_name == "" or in_name == out_name:
                continue

            in_tensor = model_ir.tensors.get(in_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if in_tensor is None or out_tensor is None:
                continue
            if bool(in_tensor.is_variable) or in_tensor.data is None:
                continue
            if not _is_fully_known_positive_shape(in_tensor.shape):
                continue

            pooled = _evaluate_constant_pool2d(
                input_data=in_tensor.data,
                op_type=op_type,
                padding=str(pool_op.options.get("padding", "SAME")),
                stride_h=int(pool_op.options.get("strideH", 1)),
                stride_w=int(pool_op.options.get("strideW", 1)),
                filter_h=int(pool_op.options.get("filterHeight", 1)),
                filter_w=int(pool_op.options.get("filterWidth", 1)),
            )
            if pooled is None:
                continue

            out_dtype = str(out_tensor.dtype).upper()
            casted = _cast_const_array_to_tflite_dtype(pooled, out_dtype)
            if casted is None:
                continue

            out_tensor.data = casted
            out_tensor.dtype = str(out_dtype)
            out_tensor.shape = [int(v) for v in list(casted.shape)]
            out_tensor.shape_signature = [int(v) for v in list(casted.shape)]
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "fold_constant_pool",
                    "op_type": str(op_type),
                    "input_name": str(in_name),
                    "output_name": str(out_name),
                },
            )

            del model_ir.operators[int(pool_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_constant_input_pool_chains": int(rewritten)}


def _optimize_constant_input_pad_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold constant-input PAD/PADV2 operators by materializing their outputs as constant tensors.

    Rewrite:
      CONST(in) + CONST(pads) [+ CONST(pad_value)] -> (PAD|PADV2) -> out
    Into:
      CONST(out_padded_data) and remove the pad op.
    """
    rewritten = 0

    while True:
        changed = False
        for pad_idx, pad_op in enumerate(model_ir.operators):
            op_type = str(pad_op.op_type)
            if op_type not in {"PAD", "PADV2"}:
                continue
            if len(pad_op.outputs) != 1:
                continue
            if len(pad_op.inputs) not in {2, 3}:
                continue

            in_name = str(pad_op.inputs[0])
            pads_name = str(pad_op.inputs[1])
            pad_value_name = str(pad_op.inputs[2]) if len(pad_op.inputs) == 3 else None
            out_name = str(pad_op.outputs[0])
            if in_name == "" or pads_name == "" or out_name == "" or in_name == out_name:
                continue

            in_tensor = model_ir.tensors.get(in_name, None)
            pads_tensor = model_ir.tensors.get(pads_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            pad_value_tensor = model_ir.tensors.get(pad_value_name, None) if pad_value_name is not None else None
            if in_tensor is None or pads_tensor is None or out_tensor is None:
                continue
            if bool(in_tensor.is_variable) or in_tensor.data is None:
                continue
            if bool(pads_tensor.is_variable) or pads_tensor.data is None:
                continue
            if pad_value_name is not None and (pad_value_tensor is None or bool(pad_value_tensor.is_variable) or pad_value_tensor.data is None):
                continue

            padded = _evaluate_constant_pad(
                input_data=in_tensor.data,
                pads_data=pads_tensor.data,
                pad_value_data=None if pad_value_tensor is None else pad_value_tensor.data,
            )
            if padded is None:
                continue

            out_dtype = str(out_tensor.dtype).upper()
            casted = _cast_const_array_to_tflite_dtype(padded, out_dtype)
            if casted is None:
                continue

            out_tensor.data = casted
            out_tensor.dtype = str(out_dtype)
            out_tensor.shape = [int(v) for v in list(casted.shape)]
            out_tensor.shape_signature = [int(v) for v in list(casted.shape)]
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "fold_constant_pad",
                    "op_type": str(op_type),
                    "input_name": str(in_name),
                    "pads_name": str(pads_name),
                    "output_name": str(out_name),
                },
            )

            del model_ir.operators[int(pad_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_constant_input_pad_chains": int(rewritten)}


def _optimize_constant_input_scatter_nd_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold constant-input SCATTER_ND operators by materializing their dense outputs.

    Rewrite:
      CONST(indices) + CONST(updates) + CONST(shape) -> SCATTER_ND -> out
    Into:
      CONST(out_dense_data) and remove the SCATTER_ND op.
    """
    rewritten = 0

    while True:
        changed = False
        for scatter_idx, scatter_op in enumerate(model_ir.operators):
            if str(scatter_op.op_type) != "SCATTER_ND":
                continue
            if len(scatter_op.inputs) != 3 or len(scatter_op.outputs) != 1:
                continue

            indices_name = str(scatter_op.inputs[0])
            updates_name = str(scatter_op.inputs[1])
            shape_name = str(scatter_op.inputs[2])
            out_name = str(scatter_op.outputs[0])
            if "" in {indices_name, updates_name, shape_name, out_name}:
                continue

            indices_tensor = model_ir.tensors.get(indices_name, None)
            updates_tensor = model_ir.tensors.get(updates_name, None)
            shape_tensor = model_ir.tensors.get(shape_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if (
                indices_tensor is None
                or updates_tensor is None
                or shape_tensor is None
                or out_tensor is None
            ):
                continue
            if bool(indices_tensor.is_variable) or indices_tensor.data is None:
                continue
            if bool(updates_tensor.is_variable) or updates_tensor.data is None:
                continue
            if bool(shape_tensor.is_variable) or shape_tensor.data is None:
                continue

            scattered = _evaluate_constant_scatter_nd(
                indices_data=indices_tensor.data,
                updates_data=updates_tensor.data,
                shape_data=shape_tensor.data,
            )
            if scattered is None:
                continue

            out_dtype = str(out_tensor.dtype).upper()
            casted = _cast_const_array_to_tflite_dtype(scattered, out_dtype)
            if casted is None:
                continue

            out_tensor.data = casted
            out_tensor.dtype = str(out_dtype)
            out_tensor.shape = [int(v) for v in list(casted.shape)]
            out_tensor.shape_signature = [int(v) for v in list(casted.shape)]
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "fold_constant_scatter_nd",
                    "indices_name": str(indices_name),
                    "updates_name": str(updates_name),
                    "shape_name": str(shape_name),
                    "output_name": str(out_name),
                },
            )

            del model_ir.operators[int(scatter_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_constant_input_scatter_nd_chains": int(rewritten)}


def _optimize_constant_binary_elementwise_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold strict binary floating-point elementwise ops when both operands are constant.
    """
    rewritten = 0

    while True:
        changed = False
        for op_idx, op in enumerate(model_ir.operators):
            op_type = str(op.op_type)
            if op_type not in {"ADD", "SUB", "MUL", "DIV"}:
                continue
            if len(op.inputs) != 2 or len(op.outputs) != 1:
                continue
            if str(op.options.get("fusedActivationFunction", "NONE")).upper() != "NONE":
                continue

            lhs_name = str(op.inputs[0])
            rhs_name = str(op.inputs[1])
            out_name = str(op.outputs[0])
            lhs_tensor = model_ir.tensors.get(lhs_name, None)
            rhs_tensor = model_ir.tensors.get(rhs_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if lhs_tensor is None or rhs_tensor is None or out_tensor is None:
                continue
            if bool(lhs_tensor.is_variable) or lhs_tensor.data is None:
                continue
            if bool(rhs_tensor.is_variable) or rhs_tensor.data is None:
                continue

            folded = _evaluate_constant_binary_elementwise(
                lhs_data=lhs_tensor.data,
                rhs_data=rhs_tensor.data,
                op_type=op_type,
            )
            if folded is None:
                continue

            out_dtype = str(out_tensor.dtype).upper()
            casted = _cast_const_array_to_tflite_dtype(folded, out_dtype)
            if casted is None:
                continue

            out_tensor.data = casted
            out_tensor.dtype = str(out_dtype)
            out_tensor.shape = [int(v) for v in list(casted.shape)]
            out_tensor.shape_signature = [int(v) for v in list(casted.shape)]
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "fold_constant_binary_elementwise",
                    "op_type": str(op_type),
                    "lhs_name": str(lhs_name),
                    "rhs_name": str(rhs_name),
                    "output_name": str(out_name),
                },
            )

            del model_ir.operators[int(op_idx)]
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_constant_binary_elementwise_chains": int(rewritten)}
