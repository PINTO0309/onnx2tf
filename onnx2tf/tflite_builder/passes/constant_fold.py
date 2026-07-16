from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    preflight_any_operator,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
    _is_singleton_constant_tensor,
    _prune_unused_tensors,
    _read_singleton_constant_float,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.tensor_buffer_builder import (
    tflite_dtype_from_numpy,
)

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


def _optimize_mul_square_anchor_constant_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold a guarded MUL(const)->square->MUL(const)->MUL(const) chain."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(str(operator.op_type) == "MUL" for operator in model_ir.operators):
            return {"optimized_yolo_decode_mul_square_anchor_chains": 0}
        active_index = ModelIRGraphIndex(model_ir)
    elif len(active_index.operator_indices("MUL")) == 0:
        return {"optimized_yolo_decode_mul_square_anchor_chains": 0}

    model_outputs = {str(name) for name in model_ir.outputs}
    rewritten = 0

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _consumed_only_by(tensor_name: str, operator_index: int) -> bool:
        consumers = active_index.consumer_indices(tensor_name)
        return bool(consumers) and {
            int(value) for value in consumers
        } == {int(operator_index)}

    while True:
        changed = False
        for scale_index in active_index.operator_indices("MUL"):
            scale_multiply = model_ir.operators[int(scale_index)]
            if len(scale_multiply.inputs) != 2 or len(scale_multiply.outputs) != 1:
                continue

            scale_input0 = str(scale_multiply.inputs[0])
            scale_input1 = str(scale_multiply.inputs[1])
            scale_tensor0 = model_ir.tensors.get(scale_input0)
            scale_tensor1 = model_ir.tensors.get(scale_input1)
            if scale_tensor0 is not None and scale_tensor0.data is not None:
                scale_constant_name = scale_input0
                anchor_output = scale_input1
            elif scale_tensor1 is not None and scale_tensor1.data is not None:
                scale_constant_name = scale_input1
                anchor_output = scale_input0
            else:
                continue
            if anchor_output in model_outputs:
                continue
            if not _consumed_only_by(anchor_output, int(scale_index)):
                continue

            anchor_multiply = active_index.producer(anchor_output)
            anchor_index = (
                active_index.operator_index(anchor_multiply)
                if anchor_multiply is not None
                else None
            )
            if (
                anchor_multiply is None
                or anchor_index is None
                or str(anchor_multiply.op_type) != "MUL"
                or len(anchor_multiply.inputs) != 2
                or len(anchor_multiply.outputs) != 1
                or str(anchor_multiply.outputs[0]) != anchor_output
            ):
                continue

            anchor_input0 = str(anchor_multiply.inputs[0])
            anchor_input1 = str(anchor_multiply.inputs[1])
            anchor_tensor0 = model_ir.tensors.get(anchor_input0)
            anchor_tensor1 = model_ir.tensors.get(anchor_input1)
            if anchor_tensor0 is not None and anchor_tensor0.data is not None:
                anchor_constant_name = anchor_input0
                square_output = anchor_input1
                anchor_constant_position = 0
            elif anchor_tensor1 is not None and anchor_tensor1.data is not None:
                anchor_constant_name = anchor_input1
                square_output = anchor_input0
                anchor_constant_position = 1
            else:
                continue
            if square_output in model_outputs:
                continue
            if not _consumed_only_by(square_output, int(anchor_index)):
                continue

            square_multiply = active_index.producer(square_output)
            square_index = (
                active_index.operator_index(square_multiply)
                if square_multiply is not None
                else None
            )
            if (
                square_multiply is None
                or square_index is None
                or str(square_multiply.op_type) != "MUL"
                or len(square_multiply.inputs) != 2
                or len(square_multiply.outputs) != 1
                or str(square_multiply.outputs[0]) != square_output
                or str(square_multiply.inputs[0])
                != str(square_multiply.inputs[1])
            ):
                continue

            scaled_source = str(square_multiply.inputs[0])
            if scaled_source in model_outputs:
                continue
            if not _consumed_only_by(scaled_source, int(square_index)):
                continue
            pre_multiply = active_index.producer(scaled_source)
            pre_index = (
                active_index.operator_index(pre_multiply)
                if pre_multiply is not None
                else None
            )
            if (
                pre_multiply is None
                or pre_index is None
                or str(pre_multiply.op_type) != "MUL"
                or len(pre_multiply.inputs) != 2
                or len(pre_multiply.outputs) != 1
                or str(pre_multiply.outputs[0]) != scaled_source
            ):
                continue

            pre_input0 = str(pre_multiply.inputs[0])
            pre_input1 = str(pre_multiply.inputs[1])
            if _is_singleton_constant_tensor(model_ir, pre_input0):
                pre_constant_name = pre_input0
                source_name = pre_input1
            elif _is_singleton_constant_tensor(model_ir, pre_input1):
                pre_constant_name = pre_input1
                source_name = pre_input0
            else:
                continue
            pre_scale = _read_singleton_constant_float(
                model_ir,
                pre_constant_name,
            )
            if pre_scale is None or not np.isfinite(pre_scale):
                continue

            anchor_tensor = model_ir.tensors.get(anchor_constant_name)
            scale_tensor = model_ir.tensors.get(scale_constant_name)
            if (
                anchor_tensor is None
                or anchor_tensor.data is None
                or scale_tensor is None
                or scale_tensor.data is None
            ):
                continue
            anchor_data = np.asarray(anchor_tensor.data)
            scale_data = np.asarray(scale_tensor.data)
            if not np.issubdtype(anchor_data.dtype, np.floating):
                continue
            if not np.issubdtype(scale_data.dtype, np.floating):
                continue
            try:
                fused_data = (
                    anchor_data.astype(np.float32, copy=False)
                    * float(pre_scale)
                    * float(pre_scale)
                    * scale_data.astype(np.float32, copy=False)
                )
            except Exception:
                continue
            if not np.all(np.isfinite(fused_data)):
                continue
            fused_data = fused_data.astype(anchor_data.dtype, copy=False)

            # All topology and numeric guards are complete before adding the
            # fused constant or changing an edge.
            fused_name = _unique_tensor_name(
                f"{anchor_constant_name}_mulsq_fused"
            )
            fused_shape, fused_signature = normalize_onnx_shape(
                list(fused_data.shape)
            )
            model_ir.tensors[fused_name] = TensorIR(
                name=fused_name,
                dtype=tflite_dtype_from_numpy(fused_data.dtype),
                shape=[int(value) for value in fused_shape],
                shape_signature=[int(value) for value in fused_signature],
                data=fused_data,
                is_variable=False,
                quantization=_clone_quantization(anchor_tensor.quantization),
            )
            if layout_state is not None:
                layout_state.set(
                    fused_name,
                    logical=model_ir.tensors[fused_name].logical_layout,
                    physical=model_ir.tensors[fused_name].physical_layout,
                )
            _set_operator_inputs(
                model_ir=model_ir,
                op=square_multiply,
                new_inputs=[source_name, source_name],
                graph_index=active_index,
            )
            anchor_inputs = [str(name) for name in anchor_multiply.inputs]
            anchor_inputs[int(anchor_constant_position)] = fused_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=anchor_multiply,
                new_inputs=anchor_inputs,
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=anchor_multiply,
                new_outputs=[str(scale_multiply.outputs[0])],
                graph_index=active_index,
            )
            current_scale_index = active_index.operator_index(scale_multiply)
            current_pre_index = active_index.operator_index(pre_multiply)
            if current_scale_index is None or current_pre_index is None:
                raise RuntimeError("MUL-square fold operator disappeared")
            active_index.remove_operators(
                [int(current_scale_index), int(current_pre_index)]
            )
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "optimized_yolo_decode_mul_square_anchor_chains": int(rewritten),
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


def _optimize_constant_input_cast_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold CAST when its input tensor already has embedded constant data.

    Rewrite:
      CONST(in) -> CAST -> out
    Into:
      CONST(out_casted_data) and remove CAST op.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        for cast_idx in graph_index.operator_indices("CAST"):
            cast_op = model_ir.operators[int(cast_idx)]
            if len(cast_op.inputs) != 1 or len(cast_op.outputs) != 1:
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

            graph_index.remove_operator(int(cast_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_constant_input_cast_chains": int(rewritten)}


def _optimize_constant_input_pool_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold constant-input pool operators by materializing their outputs as constant tensors.

    Rewrite:
      CONST(in) -> (AVERAGE_POOL_2D|MAX_POOL_2D) -> out
    Into:
      CONST(out_pooled_data) and remove the pool op.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        for pool_idx in graph_index.operator_indices_for_types(
            {"AVERAGE_POOL_2D", "MAX_POOL_2D"}
        ):
            pool_op = model_ir.operators[int(pool_idx)]
            op_type = str(pool_op.op_type)
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

            graph_index.remove_operator(int(pool_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_constant_input_pool_chains": int(rewritten)}


def _optimize_constant_input_pad_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold constant-input PAD/PADV2 operators by materializing their outputs as constant tensors.

    Rewrite:
      CONST(in) + CONST(pads) [+ CONST(pad_value)] -> (PAD|PADV2) -> out
    Into:
      CONST(out_padded_data) and remove the pad op.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        for pad_idx in graph_index.operator_indices_for_types(
            {"PAD", "PADV2"}
        ):
            pad_op = model_ir.operators[int(pad_idx)]
            op_type = str(pad_op.op_type)
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

            graph_index.remove_operator(int(pad_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_constant_input_pad_chains": int(rewritten)}


def run_constant_input_fold_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Materialize constant Pad, Pool, then Cast chains in fixed order."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_any_operator(
            candidate_model,
            lambda op: str(op.op_type)
            in {"PAD", "PADV2", "AVERAGE_POOL_2D", "MAX_POOL_2D", "CAST"},
        )

    def _has_pad(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) in {"PAD", "PADV2"}
            for op in pass_state.model_ir.operators
        )

    def _has_pool(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) in {"AVERAGE_POOL_2D", "MAX_POOL_2D"}
            for op in pass_state.model_ir.operators
        )

    def _has_cast(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) == "CAST" for op in pass_state.model_ir.operators
        )

    def _run_pad(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_constant_input_pad_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("optimized_constant_input_pad_chains", 0)),
        }

    def _run_pool(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_constant_input_pool_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("optimized_constant_input_pool_chains", 0)),
        }

    def _run_cast(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_constant_input_cast_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("optimized_constant_input_cast_chains", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.constant_input_pad",
                phase=PassPhase.CANONICALIZE,
                priority=10,
                callback=_run_pad,
                precondition=_has_pad,
                transactional=True,
            ),
            PassSpec(
                pass_id="canonicalize.constant_input_pool",
                phase=PassPhase.CANONICALIZE,
                priority=20,
                callback=_run_pool,
                precondition=_has_pool,
                transactional=True,
            ),
            PassSpec(
                pass_id="canonicalize.constant_input_cast",
                phase=PassPhase.CANONICALIZE,
                priority=30,
                callback=_run_cast,
                precondition=_has_cast,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            "optimized_constant_input_pad_chains": 0,
            "optimized_constant_input_pool_chains": 0,
            "optimized_constant_input_cast_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_constant_input_scatter_nd_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold constant-input SCATTER_ND operators by materializing their dense outputs.

    Rewrite:
      CONST(indices) + CONST(updates) + CONST(shape) -> SCATTER_ND -> out
    Into:
      CONST(out_dense_data) and remove the SCATTER_ND op.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        for scatter_idx in graph_index.operator_indices("SCATTER_ND"):
            scatter_op = model_ir.operators[int(scatter_idx)]
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

            graph_index.remove_operator(int(scatter_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_constant_input_scatter_nd_chains": int(rewritten)}


def _optimize_constant_binary_elementwise_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Fold strict binary floating-point elementwise ops when both operands are constant.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        for op_idx in graph_index.operator_indices_for_types(
            {"ADD", "SUB", "MUL", "DIV"}
        ):
            op = model_ir.operators[int(op_idx)]
            op_type = str(op.op_type)
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

            graph_index.remove_operator(int(op_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_constant_binary_elementwise_chains": int(rewritten)}
