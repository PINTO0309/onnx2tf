from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _invert_perm,
    _is_fully_known_positive_shape,
    _normalize_squeeze_axes_for_rank,
    _permute_shape,
    _read_const_ints_from_tensor,
    _topologically_sort_operators,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.dynamic_reshape_resolution import (
    _resolve_reshape_new_shape_from_static_input,
)


def _infer_slice_output_shape_and_resolved_params(
    input_shape: Optional[List[int]],
    begin_vals: Optional[List[int]],
    size_vals: Optional[List[int]],
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    if not _is_fully_known_positive_shape(input_shape):
        return None, None, None
    if begin_vals is None or size_vals is None:
        return None, None, None

    in_shape = [int(v) for v in list(input_shape)]
    begin = [int(v) for v in list(begin_vals)]
    size = [int(v) for v in list(size_vals)]
    rank = len(in_shape)
    if len(begin) != rank or len(size) != rank:
        return None, None, None

    resolved_begin: List[int] = []
    resolved_size: List[int] = []
    out_shape: List[int] = []
    for axis, dim in enumerate(in_shape):
        b = int(begin[axis])
        if b < 0:
            b += int(dim)
        b = max(0, min(int(b), int(dim)))
        remain = max(int(dim) - int(b), 0)

        raw_size = int(size[axis])
        if raw_size == -1:
            s = int(remain)
        elif raw_size >= 0:
            s = int(raw_size)
        else:
            return None, None, None

        out_dim = max(min(int(s), int(remain)), 0)
        resolved_begin.append(int(b))
        resolved_size.append(int(s))
        out_shape.append(int(out_dim))

    return out_shape, resolved_begin, resolved_size


def _infer_slice_output_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    begin_vals: Optional[List[int]],
    size_vals: Optional[List[int]],
) -> Optional[List[int]]:
    if not _is_fully_known_positive_shape(input_shape):
        return None
    if begin_vals is None or size_vals is None:
        return None

    in_shape = [int(v) for v in list(input_shape)]
    rank = len(in_shape)
    begin = [int(v) for v in list(begin_vals)]
    size = [int(v) for v in list(size_vals)]
    if len(begin) != rank or len(size) != rank:
        return None

    if input_signature is not None and len(list(input_signature)) == rank:
        in_signature = [int(v) for v in list(input_signature)]
    else:
        in_signature = [int(v) for v in list(in_shape)]

    out_signature: List[int] = []
    for axis, dim in enumerate(in_shape):
        b = int(begin[axis])
        if b < 0:
            b += int(dim)
        b = max(0, min(int(b), int(dim)))
        remain = max(int(dim) - int(b), 0)

        raw_size = int(size[axis])
        if raw_size == -1:
            if int(in_signature[axis]) < 0:
                out_signature.append(-1)
            else:
                out_signature.append(int(remain))
            continue
        if raw_size < 0:
            return None
        if int(in_signature[axis]) < 0:
            # Preserve explicit static slice sizes on dynamic axes. Clamping to
            # placeholder `input_shape` (often 1) can corrupt downstream shape
            # metadata and break kernels that validate state widths statically.
            out_signature.append(int(max(int(raw_size), 0)))
        else:
            out_signature.append(int(max(min(int(raw_size), int(remain)), 0)))

    return out_signature


def _infer_batch_matmul_output_shape_and_signature(
    shape_a: Optional[List[int]],
    shape_b: Optional[List[int]],
    signature_a: Optional[List[int]],
    signature_b: Optional[List[int]],
    adj_x: bool,
    adj_y: bool,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    if not _is_fully_known_positive_shape(shape_a) or not _is_fully_known_positive_shape(shape_b):
        return None, None
    a_shape = [int(v) for v in list(shape_a)]
    b_shape = [int(v) for v in list(shape_b)]
    if len(a_shape) < 2 or len(b_shape) < 2:
        return None, None

    a_sig = (
        [int(v) for v in list(signature_a)]
        if signature_a is not None and len(list(signature_a)) == len(a_shape)
        else list(a_shape)
    )
    b_sig = (
        [int(v) for v in list(signature_b)]
        if signature_b is not None and len(list(signature_b)) == len(b_shape)
        else list(b_shape)
    )

    a_batch_shape = [int(v) for v in a_shape[:-2]]
    b_batch_shape = [int(v) for v in b_shape[:-2]]
    if len(a_batch_shape) == 0:
        batch_shape = list(b_batch_shape)
    elif len(b_batch_shape) == 0:
        batch_shape = list(a_batch_shape)
    else:
        batch_shape = _broadcast_static_shapes(a_batch_shape, b_batch_shape)
    if batch_shape is None:
        return None, None

    a_m_idx = -1 if bool(adj_x) else -2
    a_k_idx = -2 if bool(adj_x) else -1
    b_k_idx = -1 if bool(adj_y) else -2
    b_n_idx = -2 if bool(adj_y) else -1

    a_m = int(a_shape[a_m_idx])
    a_k = int(a_shape[a_k_idx])
    b_k = int(b_shape[b_k_idx])
    b_n = int(b_shape[b_n_idx])

    a_k_sig = int(a_sig[a_k_idx])
    b_k_sig = int(b_sig[b_k_idx])
    if a_k != b_k and a_k_sig >= 0 and b_k_sig >= 0:
        return None, None

    out_shape = [int(v) for v in list(batch_shape)] + [int(a_m), int(b_n)]

    batch_sig = _broadcast_shape_signatures(a_sig[:-2], b_sig[:-2])
    if batch_sig is None:
        batch_sig = [int(v) for v in list(batch_shape)]
    out_sig_m = -1 if int(a_sig[a_m_idx]) < 0 else int(a_m)
    out_sig_n = -1 if int(b_sig[b_n_idx]) < 0 else int(b_n)
    out_signature = [int(v) for v in list(batch_sig)] + [int(out_sig_m), int(out_sig_n)]

    return out_shape, out_signature


def _infer_rank4_signature_from_input(
    *,
    input_signature: Optional[List[int]],
    output_shape: Optional[List[int]],
    existing_output_signature: Optional[List[int]] = None,
    propagate_channel: bool = False,
) -> Optional[List[int]]:
    if output_shape is None or len(list(output_shape)) != 4:
        return None
    signature = [int(v) for v in list(output_shape)]

    normalized_input_signature = (
        [int(v) for v in list(input_signature)]
        if input_signature is not None and len(list(input_signature)) == 4
        else None
    )
    dynamic_from_input = [False, False, False, False]
    if normalized_input_signature is not None:
        dynamic_from_input[0] = int(normalized_input_signature[0]) < 0
        dynamic_from_input[1] = int(normalized_input_signature[1]) < 0
        dynamic_from_input[2] = int(normalized_input_signature[2]) < 0
        dynamic_from_input[3] = bool(propagate_channel) and int(normalized_input_signature[3]) < 0

    # Preserve existing dynamic marks only when input-derived dynamics can explain
    # that axis. This avoids stale all-unknown placeholders dominating new
    # operator-local inferences (e.g. static channel axis from filters).
    if (
        existing_output_signature is not None
        and len(list(existing_output_signature)) == 4
    ):
        if normalized_input_signature is None:
            for axis in range(4):
                if int(existing_output_signature[axis]) < 0:
                    signature[axis] = -1
        else:
            for axis in range(4):
                if int(existing_output_signature[axis]) < 0 and bool(dynamic_from_input[axis]):
                    signature[axis] = -1

    if normalized_input_signature is not None:
        if int(normalized_input_signature[0]) < 0:
            signature[0] = -1
        if int(normalized_input_signature[1]) < 0:
            signature[1] = -1
        if int(normalized_input_signature[2]) < 0:
            signature[2] = -1
        if bool(propagate_channel) and int(normalized_input_signature[3]) < 0:
            signature[3] = -1
    return signature


def _normalize_reduce_axes_for_rank(
    axes: Optional[List[int]],
    rank: int,
) -> Optional[List[int]]:
    if axes is None:
        return None
    if rank < 0:
        return None
    normalized: List[int] = []
    for axis in list(axes):
        a = int(axis)
        if a < 0:
            a += int(rank)
        if a < 0 or a >= int(rank):
            return None
        if int(a) not in normalized:
            normalized.append(int(a))
    return normalized


def _infer_reduce_output_shape_and_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    axes: Optional[List[int]],
    keep_dims: bool,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    if input_shape is None:
        return None, None
    in_shape = [int(v) for v in list(input_shape)]
    rank = len(in_shape)
    if rank == 0:
        return list(in_shape), list(in_shape)
    normalized_axes = _normalize_reduce_axes_for_rank(axes, rank)
    if normalized_axes is None:
        return None, None

    if input_signature is not None and len(list(input_signature)) == rank:
        in_signature = [int(v) for v in list(input_signature)]
    else:
        in_signature = [int(v) for v in list(in_shape)]

    if bool(keep_dims):
        out_shape = [int(v) for v in list(in_shape)]
        out_signature = [int(v) for v in list(in_signature)]
        for axis in normalized_axes:
            out_shape[int(axis)] = 1
            out_signature[int(axis)] = 1
        return out_shape, out_signature

    reduced_axes = set(int(v) for v in normalized_axes)
    out_shape = [int(in_shape[idx]) for idx in range(rank) if idx not in reduced_axes]
    out_signature = [int(in_signature[idx]) for idx in range(rank) if idx not in reduced_axes]
    return out_shape, out_signature


def _parse_axes_option(raw_axes: Any) -> List[int]:
    if raw_axes is None:
        return []
    if isinstance(raw_axes, (list, tuple, np.ndarray)):
        return [int(v) for v in list(raw_axes)]
    return [int(raw_axes)]


def _infer_squeeze_output_shape_and_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    squeeze_axes: Optional[List[int]],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    if input_shape is None:
        return None, None
    in_shape = [int(v) for v in list(input_shape)]
    rank = len(in_shape)
    if input_signature is not None and len(list(input_signature)) == rank:
        in_signature = [int(v) for v in list(input_signature)]
    else:
        in_signature = [int(v) for v in list(in_shape)]

    axes_list = [int(v) for v in list(squeeze_axes)] if squeeze_axes is not None else []
    normalized_axes = _normalize_squeeze_axes_for_rank(axes_list, rank)
    if normalized_axes is None:
        return None, None
    if len(normalized_axes) == 0:
        normalized_axes = [
            int(idx)
            for idx in range(rank)
            if idx < len(in_signature) and int(in_signature[idx]) == 1
        ]
        if len(normalized_axes) == 0:
            normalized_axes = [
                int(idx)
                for idx, dim in enumerate(in_shape)
                if int(dim) == 1
            ]

    remove_axes = set(int(v) for v in normalized_axes)
    out_shape = [int(in_shape[idx]) for idx in range(rank) if idx not in remove_axes]
    out_signature = [int(in_signature[idx]) for idx in range(rank) if idx not in remove_axes]
    return out_shape, out_signature


def _infer_conv_out_dim(
    in_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: str,
) -> Optional[int]:
    if any(int(v) <= 0 for v in [in_size, kernel_size, stride, dilation]):
        return None
    effective_kernel = int((int(kernel_size) - 1) * int(dilation) + 1)
    mode = str(padding).upper()
    if mode == "SAME":
        return int((int(in_size) + int(stride) - 1) // int(stride))
    if mode == "VALID":
        return int((int(in_size) - int(effective_kernel)) // int(stride) + 1)
    return None


def reconcile_static_tensor_shapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    include_mutation_count: bool = False,
) -> Dict[str, int]:
    """
    Recompute static tensor shapes after aggressive graph rewrites.

    Some transpose-bridge optimizations intentionally relax local shape guards and can
    leave stale static metadata. This pass performs a conservative forward fixed-point
    shape propagation and syncs `shape` / `shape_signature` for common TFLite ops.
    """
    updated_tensors = 0
    mutation_count = 0
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    producer_by_output = (
        {
            str(output_name): model_ir.operators[int(operator_index)]
            for output_name, operator_index in active_index.producers.items()
            if str(output_name) != ""
        }
        if active_index is not None
        else {
            str(output_name): op
            for op in model_ir.operators
            for output_name in op.outputs
            if str(output_name) != ""
        }
    )

    def _update_tensor_shape(
        tensor_name: str,
        new_shape: Optional[List[int]],
        new_shape_signature: Optional[List[int]] = None,
    ) -> bool:
        nonlocal mutation_count, updated_tensors
        if new_shape is None:
            return False
        if not _is_fully_known_positive_shape(new_shape):
            return False
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            return False
        normalized = [int(v) for v in list(new_shape)]
        if new_shape_signature is not None and len(list(new_shape_signature)) == len(normalized):
            signature = [int(v) for v in list(new_shape_signature)]
        else:
            existing_signature = (
                list(tensor.shape_signature)
                if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == len(normalized)
                else None
            )
            if existing_signature is None:
                signature = [int(v) for v in list(normalized)]
            else:
                # Preserve unknown dims inferred upstream to avoid collapsing dynamic axes
                # into placeholder static 1s during conservative shape reconciliation.
                signature = [
                    int(existing_signature[idx]) if int(existing_signature[idx]) < 0 else int(normalized[idx])
                    for idx in range(len(normalized))
                ]
        if tensor.shape == normalized and tensor.shape_signature == signature:
            return False
        tensor.shape = normalized
        tensor.shape_signature = signature
        updated_tensors += 1
        mutation_count += 1
        return True

    def _set_operator_option(operator: Any, key: str, value: Any) -> bool:
        nonlocal mutation_count
        if key in operator.options and operator.options[key] == value:
            return False
        operator.options[key] = value
        mutation_count += 1
        return True

    def _write_const_ints_tracked(tensor: Any, values: List[int]) -> bool:
        nonlocal mutation_count
        changed_tensor = _write_const_ints_to_tensor(tensor, values)
        if changed_tensor:
            mutation_count += 1
        return bool(changed_tensor)

    def _set_tensor_vector_metadata(tensor: Any, length: int) -> bool:
        nonlocal mutation_count
        normalized = [int(length)]
        if tensor.shape == normalized and tensor.shape_signature == normalized:
            return False
        tensor.shape = list(normalized)
        tensor.shape_signature = list(normalized)
        mutation_count += 1
        return True

    def _set_tensor_shape_signature(tensor: Any, signature: List[int]) -> bool:
        nonlocal mutation_count
        normalized = [int(value) for value in signature]
        if tensor.shape_signature == normalized:
            return False
        tensor.shape_signature = normalized
        mutation_count += 1
        return True

    def _set_int32_const_tensor(tensor: Any, values: List[int]) -> bool:
        nonlocal mutation_count
        normalized = [int(value) for value in values]
        current = _read_const_ints_from_tensor(tensor)
        target_shape = [int(len(normalized))]
        if (
            current == normalized
            and str(tensor.dtype).upper() == "INT32"
            and tensor.shape == target_shape
            and tensor.shape_signature == target_shape
        ):
            return False
        tensor.data = np.asarray(normalized, dtype=np.int32)
        tensor.dtype = "INT32"
        tensor.shape = list(target_shape)
        tensor.shape_signature = list(target_shape)
        mutation_count += 1
        return True

    max_passes = 32
    for _ in range(max_passes):
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            inputs = [str(v) for v in list(op.inputs)]
            outputs = [str(v) for v in list(op.outputs)]
            if len(outputs) == 0:
                continue

            # Pass-through ops: output shape == first input shape.
            if op_type in {
                "QUANTIZE",
                "DEQUANTIZE",
                "SOFTMAX",
                "LOGISTIC",
                "TANH",
                "RELU",
                "RELU6",
                "RELU_0_TO_1",
                "PRELU",
                "HARD_SWISH",
                "LEAKY_RELU",
                "GELU",
                "CAST",
                "NEG",
                "ABS",
                "EXP",
                "SQRT",
                "FLOOR",
                "ROUND",
            }:
                if len(inputs) >= 1:
                    in_tensor = model_ir.tensors.get(inputs[0], None)
                    if in_tensor is not None and _is_fully_known_positive_shape(in_tensor.shape):
                        in_signature = (
                            list(in_tensor.shape_signature)
                            if in_tensor.shape_signature is not None
                            else list(in_tensor.shape)
                        )
                        changed |= _update_tensor_shape(
                            outputs[0],
                            list(in_tensor.shape),
                            in_signature,
                        )
                continue

            if op_type == "TRANSPOSE" and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                out_tensor = model_ir.tensors.get(outputs[0], None)
                perm_tensor = model_ir.tensors.get(inputs[1], None)
                perm = _read_const_ints_from_tensor(perm_tensor)
                if perm is None:
                    continue
                perm = [int(v) for v in perm]
                if in_tensor is not None and _is_fully_known_positive_shape(in_tensor.shape):
                    out_shape = _permute_shape(list(in_tensor.shape), perm)
                    in_signature = (
                        list(in_tensor.shape_signature)
                        if in_tensor.shape_signature is not None
                        else list(in_tensor.shape)
                    )
                    out_signature = _permute_shape(in_signature, perm)
                    changed |= _update_tensor_shape(
                        outputs[0],
                        out_shape,
                        out_signature,
                    )
                if out_tensor is not None and _is_fully_known_positive_shape(out_tensor.shape):
                    inv_perm = _invert_perm(perm)
                    if inv_perm is not None:
                        in_shape = _permute_shape(list(out_tensor.shape), inv_perm)
                        out_signature = (
                            list(out_tensor.shape_signature)
                            if out_tensor.shape_signature is not None
                            else list(out_tensor.shape)
                        )
                        in_signature = _permute_shape(out_signature, inv_perm)
                        changed |= _update_tensor_shape(
                            inputs[0],
                            in_shape,
                            in_signature,
                        )
                continue

            if op_type == "GATHER" and len(inputs) >= 2 and len(outputs) == 1:
                params_tensor = model_ir.tensors.get(inputs[0], None)
                indices_tensor = model_ir.tensors.get(inputs[1], None)
                if params_tensor is None or indices_tensor is None:
                    continue
                if not _is_fully_known_positive_shape(list(params_tensor.shape)):
                    continue
                params_shape = [int(v) for v in list(params_tensor.shape)]
                params_signature = (
                    [int(v) for v in list(params_tensor.shape_signature)]
                    if params_tensor.shape_signature is not None
                    else list(params_shape)
                )
                indices_shape = [int(v) for v in list(indices_tensor.shape)]
                indices_signature = (
                    [int(v) for v in list(indices_tensor.shape_signature)]
                    if indices_tensor.shape_signature is not None
                    else list(indices_shape)
                )
                axis = int(op.options.get("axis", 0))
                batch_dims = int(op.options.get("batchDims", 0))
                if batch_dims != 0:
                    continue
                rank = len(params_shape)
                if rank <= 0:
                    continue
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    continue
                out_shape = (
                    [int(v) for v in params_shape[:axis]]
                    + [int(v) for v in indices_shape]
                    + [int(v) for v in params_shape[axis + 1:]]
                )
                out_signature = (
                    [int(v) for v in params_signature[:axis]]
                    + [int(v) for v in indices_signature]
                    + [int(v) for v in params_signature[axis + 1:]]
                )
                if len(out_shape) == 0:
                    out_shape = [1]
                if len(out_signature) == 0:
                    out_signature = [1]
                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type == "GATHER_ND" and len(inputs) >= 2 and len(outputs) == 1:
                params_tensor = model_ir.tensors.get(inputs[0], None)
                indices_tensor = model_ir.tensors.get(inputs[1], None)
                if params_tensor is None or indices_tensor is None:
                    continue
                if not _is_fully_known_positive_shape(list(params_tensor.shape)):
                    continue
                params_shape = [int(v) for v in list(params_tensor.shape)]
                params_signature = (
                    [int(v) for v in list(params_tensor.shape_signature)]
                    if params_tensor.shape_signature is not None
                    else list(params_shape)
                )
                indices_shape = [int(v) for v in list(indices_tensor.shape)]
                indices_signature = (
                    [int(v) for v in list(indices_tensor.shape_signature)]
                    if indices_tensor.shape_signature is not None
                    else list(indices_shape)
                )
                if len(indices_shape) < 1:
                    continue
                gather_dims = int(indices_shape[-1])
                if len(indices_signature) > 0 and int(indices_signature[-1]) > 0:
                    gather_dims = int(indices_signature[-1])
                if gather_dims <= 0 or gather_dims > len(params_shape):
                    continue
                out_shape = (
                    [int(v) for v in indices_shape[:-1]]
                    + [int(v) for v in params_shape[gather_dims:]]
                )
                out_signature = (
                    [int(v) for v in indices_signature[:-1]]
                    + [int(v) for v in params_signature[gather_dims:]]
                )
                if len(out_shape) == 0:
                    out_shape = [1]
                if len(out_signature) == 0:
                    out_signature = [1]
                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type in {"BIDIRECTIONAL_SEQUENCE_LSTM", "UNIDIRECTIONAL_SEQUENCE_LSTM"} \
                and len(inputs) >= 1 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                output_tensor = model_ir.tensors.get(outputs[0], None)
                input_producer = producer_by_output.get(inputs[0], None)
                input_weight_tensor = (
                    model_ir.tensors.get(inputs[1], None)
                    if len(inputs) >= 2
                    else None
                )
                if (
                    input_tensor is not None
                    and input_producer is not None
                    and str(input_producer.op_type) == "RESHAPE"
                    and len(input_producer.inputs) >= 1
                    and list(
                        input_producer.options.get("onnxRawNewShape", [])
                    ) == [0, 0, -1]
                    and not bool(input_producer.options.get("allowZero", False))
                    and input_weight_tensor is not None
                    and len(list(input_weight_tensor.shape)) == 2
                ):
                    reshape_source_tensor = model_ir.tensors.get(
                        str(input_producer.inputs[0]),
                        None,
                    )
                    if (
                        reshape_source_tensor is not None
                        and len(list(reshape_source_tensor.shape)) >= 2
                        and _is_fully_known_positive_shape(
                            list(reshape_source_tensor.shape)
                        )
                    ):
                        source_shape = [
                            int(v) for v in list(reshape_source_tensor.shape)
                        ]
                        source_signature = (
                            [
                                int(v)
                                for v in list(
                                    reshape_source_tensor.shape_signature
                                )
                            ]
                            if reshape_source_tensor.shape_signature is not None
                            and len(list(reshape_source_tensor.shape_signature))
                            == len(source_shape)
                            else list(source_shape)
                        )
                        inferred_input_shape = [
                            int(source_shape[0]),
                            int(source_shape[1]),
                            int(input_weight_tensor.shape[1]),
                        ]
                        inferred_input_signature = [
                            int(source_signature[0]),
                            int(source_signature[1]),
                            int(input_weight_tensor.shape[1]),
                        ]
                        changed |= _update_tensor_shape(
                            inputs[0],
                            inferred_input_shape,
                            inferred_input_signature,
                        )
                        input_tensor = model_ir.tensors.get(inputs[0], None)
                if (
                    input_tensor is None
                    or output_tensor is None
                    or len(list(input_tensor.shape)) != 3
                    or len(list(output_tensor.shape)) != 3
                    or not _is_fully_known_positive_shape(list(input_tensor.shape))
                ):
                    continue
                input_shape = [int(v) for v in list(input_tensor.shape)]
                input_signature = (
                    [int(v) for v in list(input_tensor.shape_signature)]
                    if input_tensor.shape_signature is not None
                    else list(input_shape)
                )
                output_shape = [int(v) for v in list(output_tensor.shape)]
                output_signature_existing = (
                    [int(v) for v in list(output_tensor.shape_signature)]
                    if output_tensor.shape_signature is not None
                    else list(output_shape)
                )
                time_major = bool(op.options.get("timeMajor", True))
                if time_major:
                    inferred_output_shape = [int(input_shape[0]), int(input_shape[1]), int(output_shape[2])]
                    inferred_output_signature = [
                        int(input_signature[0]),
                        int(input_signature[1]),
                        int(output_signature_existing[2]),
                    ]
                else:
                    inferred_output_shape = [int(input_shape[1]), int(input_shape[0]), int(output_shape[2])]
                    inferred_output_signature = [
                        int(input_signature[1]),
                        int(input_signature[0]),
                        int(output_signature_existing[2]),
                    ]
                changed |= _update_tensor_shape(
                    outputs[0],
                    inferred_output_shape,
                    inferred_output_signature,
                )
                continue

            if op_type == "SPLIT" and len(inputs) >= 2 and len(outputs) >= 1:
                axis_tensor = model_ir.tensors.get(inputs[0], None)
                input_tensor = model_ir.tensors.get(inputs[1], None)
                axis_values = _read_const_ints_from_tensor(axis_tensor)
                if (
                    axis_values is None
                    or len(axis_values) == 0
                    or input_tensor is None
                    or not _is_fully_known_positive_shape(list(input_tensor.shape))
                ):
                    continue
                input_shape = [int(v) for v in list(input_tensor.shape)]
                rank = len(input_shape)
                axis = int(axis_values[0])
                if axis < 0:
                    axis += rank
                if axis < 0 or axis >= rank:
                    continue
                num_splits = int(op.options.get("numSplits", len(outputs)))
                if num_splits <= 0 or len(outputs) != num_splits:
                    continue
                if int(input_shape[axis]) % num_splits != 0:
                    continue
                split_dim = int(input_shape[axis] // num_splits)
                input_signature = (
                    [int(v) for v in list(input_tensor.shape_signature)]
                    if input_tensor.shape_signature is not None
                    else list(input_shape)
                )
                split_signature_dim = -1
                if axis < len(input_signature):
                    sig_dim = int(input_signature[axis])
                    if sig_dim > 0 and sig_dim % num_splits == 0:
                        split_signature_dim = int(sig_dim // num_splits)
                for output_name in outputs:
                    out_shape = [int(v) for v in list(input_shape)]
                    out_shape[axis] = int(split_dim)
                    out_signature = [int(v) for v in list(input_signature)]
                    out_signature[axis] = int(split_signature_dim if split_signature_dim > 0 else -1)
                    changed |= _update_tensor_shape(
                        output_name,
                        out_shape,
                        out_signature,
                    )
                continue

            if op_type == "EXPAND_DIMS" and len(inputs) >= 2 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                axis_tensor = model_ir.tensors.get(inputs[1], None)
                axis_values = _read_const_ints_from_tensor(axis_tensor)
                if (
                    input_tensor is None
                    or axis_values is None
                    or len(axis_values) == 0
                    or not _is_fully_known_positive_shape(list(input_tensor.shape))
                ):
                    continue
                input_shape = [int(v) for v in list(input_tensor.shape)]
                input_signature = (
                    [int(v) for v in list(input_tensor.shape_signature)]
                    if input_tensor.shape_signature is not None
                    else list(input_shape)
                )
                out_rank = len(input_shape) + 1
                axis = int(axis_values[0])
                if axis < 0:
                    axis += out_rank
                if axis < 0 or axis >= out_rank:
                    continue
                out_shape = [int(v) for v in list(input_shape)]
                out_signature = [int(v) for v in list(input_signature)]
                out_shape.insert(axis, 1)
                out_signature.insert(axis, 1)
                changed |= _update_tensor_shape(
                    outputs[0],
                    out_shape,
                    out_signature,
                )
                continue

            if op_type in {"ADD", "SUB", "MUL", "DIV", "FLOOR_MOD", "MAXIMUM", "MINIMUM"} and len(inputs) >= 2 and len(outputs) == 1:
                in0 = model_ir.tensors.get(inputs[0], None)
                in1 = model_ir.tensors.get(inputs[1], None)
                shape0 = list(in0.shape) if in0 is not None else None
                shape1 = list(in1.shape) if in1 is not None else None
                sig0 = (
                    list(in0.shape_signature)
                    if in0 is not None and in0.shape_signature is not None
                    else (list(shape0) if shape0 is not None else None)
                )
                sig1 = (
                    list(in1.shape_signature)
                    if in1 is not None and in1.shape_signature is not None
                    else (list(shape1) if shape1 is not None else None)
                )
                out_shape = _broadcast_static_shapes(shape0, shape1)
                out_signature = _broadcast_shape_signatures(sig0, sig1)
                if out_shape is not None:
                    changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type == "BATCH_MATMUL" and len(inputs) >= 2 and len(outputs) == 1:
                in0 = model_ir.tensors.get(inputs[0], None)
                in1 = model_ir.tensors.get(inputs[1], None)
                shape0 = list(in0.shape) if in0 is not None else None
                shape1 = list(in1.shape) if in1 is not None else None
                sig0 = (
                    list(in0.shape_signature)
                    if in0 is not None and in0.shape_signature is not None
                    else (list(shape0) if shape0 is not None else None)
                )
                sig1 = (
                    list(in1.shape_signature)
                    if in1 is not None and in1.shape_signature is not None
                    else (list(shape1) if shape1 is not None else None)
                )
                out_shape, out_signature = _infer_batch_matmul_output_shape_and_signature(
                    shape_a=shape0,
                    shape_b=shape1,
                    signature_a=sig0,
                    signature_b=sig1,
                    adj_x=bool(op.options.get("adjX", False)),
                    adj_y=bool(op.options.get("adjY", False)),
                )
                if out_shape is not None:
                    changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type == "CONCATENATION" and len(inputs) >= 1 and len(outputs) == 1:
                axis = op.options.get("axis", None)
                if axis is None:
                    continue
                in_shapes: List[List[int]] = []
                in_signatures: List[List[int]] = []
                ranks: List[int] = []
                valid = True
                for input_name in inputs:
                    t = model_ir.tensors.get(input_name, None)
                    if t is None or not _is_fully_known_positive_shape(t.shape):
                        valid = False
                        break
                    shape = [int(v) for v in list(t.shape)]
                    in_shapes.append(shape)
                    in_signatures.append(
                        [int(v) for v in list(t.shape_signature)]
                        if t.shape_signature is not None
                        else list(shape)
                    )
                    ranks.append(len(shape))
                if not valid or len(in_shapes) == 0 or len(set(ranks)) != 1:
                    continue
                rank = int(ranks[0])
                axis_new = int(axis)
                if axis_new < 0:
                    axis_new += rank
                if axis_new < 0 or axis_new >= rank:
                    continue
                out_shape = list(in_shapes[0])
                compatible = True
                for shape in in_shapes[1:]:
                    for dim_idx in range(rank):
                        if dim_idx == axis_new:
                            continue
                        if int(shape[dim_idx]) != int(out_shape[dim_idx]):
                            compatible = False
                            break
                    if not compatible:
                        break
                    out_shape[axis_new] += int(shape[axis_new])
                if compatible:
                    out_signature = [int(v) for v in list(out_shape)]
                    if len(in_signatures) == len(in_shapes):
                        for dim_idx in range(rank):
                            dim_values = [int(sig[dim_idx]) for sig in in_signatures]
                            if dim_idx == axis_new:
                                if any(int(v) < 0 for v in dim_values):
                                    out_signature[dim_idx] = -1
                                else:
                                    out_signature[dim_idx] = int(sum(int(v) for v in dim_values))
                            else:
                                if any(int(v) < 0 for v in dim_values):
                                    out_signature[dim_idx] = -1
                    changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type == "RESHAPE" and len(inputs) >= 1 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                output_tensor = model_ir.tensors.get(outputs[0], None)
                if input_tensor is None:
                    continue
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                if "onnxExpandDimsAxis" in op.options:
                    input_shape = [int(v) for v in list(input_tensor.shape)]
                    out_rank = len(input_shape) + 1
                    axis = int(op.options.get("onnxExpandDimsAxis", 0))
                    if axis < 0:
                        axis += out_rank
                    if 0 <= axis < out_rank:
                        out_shape = [int(v) for v in input_shape]
                        out_signature = [int(v) for v in input_signature]
                        out_shape.insert(axis, 1)
                        out_signature.insert(axis, 1)
                        _set_operator_option(
                            op,
                            "newShape",
                            [int(v) for v in out_shape],
                        )
                        if len(inputs) >= 2:
                            shape_tensor = model_ir.tensors.get(inputs[1], None)
                            if shape_tensor is not None and shape_tensor.data is not None:
                                changed |= _write_const_ints_tracked(
                                    shape_tensor,
                                    [int(v) for v in out_shape],
                                )
                                _set_tensor_vector_metadata(
                                    shape_tensor,
                                    len(out_shape),
                                )
                        changed |= _update_tensor_shape(
                            outputs[0],
                            out_shape,
                            out_signature,
                        )
                    continue
                if "onnxSqueezeDims" in op.options:
                    squeeze_axes = _parse_axes_option(
                        op.options.get("onnxSqueezeDims", [])
                    )
                    out_shape, out_signature = (
                        _infer_squeeze_output_shape_and_signature(
                            input_shape=list(input_tensor.shape),
                            input_signature=input_signature,
                            squeeze_axes=squeeze_axes,
                        )
                    )
                    if len(squeeze_axes) > 0:
                        input_shape = [int(v) for v in list(input_tensor.shape)]
                        normalized_squeeze_axes: List[int] = []
                        for squeeze_axis in squeeze_axes:
                            normalized_axis = int(squeeze_axis)
                            if normalized_axis < 0:
                                normalized_axis += len(input_shape)
                            if 0 <= normalized_axis < len(input_shape):
                                normalized_squeeze_axes.append(int(normalized_axis))
                        has_non_singleton_axis = any(
                            int(input_shape[axis]) > 1
                            for axis in normalized_squeeze_axes
                        )
                        existing_output_shape = (
                            [int(v) for v in list(output_tensor.shape)]
                            if output_tensor is not None
                            else []
                        )
                        if has_non_singleton_axis and len(existing_output_shape) > 0:
                            dynamic_axis = next(
                                (
                                    int(axis)
                                    for axis, dim in enumerate(existing_output_shape)
                                    if int(dim) == 1
                                ),
                                0,
                            )
                            safe_shape = [int(v) for v in existing_output_shape]
                            safe_shape[int(dynamic_axis)] = -1
                            _set_operator_option(
                                op,
                                "newShape",
                                [int(v) for v in safe_shape],
                            )
                            _set_operator_option(
                                op,
                                "preserveDynamicShape",
                                True,
                            )
                            _set_operator_option(
                                op,
                                "speculativeBranchSafe",
                                True,
                            )
                            if len(inputs) >= 2:
                                shape_tensor = model_ir.tensors.get(inputs[1], None)
                                if shape_tensor is not None and shape_tensor.data is not None:
                                    changed |= _write_const_ints_tracked(
                                        shape_tensor,
                                        [int(v) for v in safe_shape],
                                    )
                            if output_tensor is not None:
                                _set_tensor_shape_signature(
                                    output_tensor,
                                    [int(v) for v in safe_shape],
                                )
                            changed = True
                            continue
                    if out_shape is None:
                        continue
                    if (
                        out_shape is not None
                        and _is_fully_known_positive_shape(out_shape)
                    ):
                        _set_operator_option(
                            op,
                            "newShape",
                            [int(v) for v in list(out_shape)],
                        )
                        if len(inputs) >= 2:
                            shape_tensor = model_ir.tensors.get(inputs[1], None)
                            if (
                                shape_tensor is not None
                                and shape_tensor.data is not None
                            ):
                                changed |= _write_const_ints_tracked(
                                    shape_tensor,
                                    [int(v) for v in list(out_shape)],
                                )
                                _set_tensor_vector_metadata(
                                    shape_tensor,
                                    len(out_shape),
                                )
                        changed |= _update_tensor_shape(
                            outputs[0],
                            out_shape,
                            out_signature,
                        )
                    continue
                if "onnxFlattenAxis" in op.options:
                    existing_flatten_new_shape = op.options.get("newShape", [])
                    try:
                        existing_flatten_new_shape = [
                            int(v)
                            for v in np.asarray(
                                existing_flatten_new_shape
                            ).reshape(-1).tolist()
                        ]
                    except Exception:
                        existing_flatten_new_shape = []
                    input_shape = [int(v) for v in list(input_tensor.shape)]
                    input_rank = len(input_shape)
                    semantic_input_shape = op.options.get(
                        "onnxFlattenInputShape",
                        [],
                    )
                    try:
                        semantic_input_shape = [
                            int(v)
                            for v in np.asarray(
                                semantic_input_shape
                            ).reshape(-1).tolist()
                        ]
                    except Exception:
                        semantic_input_shape = []
                    if not (
                        len(semantic_input_shape) == input_rank
                        and _is_fully_known_positive_shape(
                            semantic_input_shape
                        )
                        and _is_fully_known_positive_shape(input_shape)
                        and int(np.prod(semantic_input_shape, dtype=np.int64))
                        == int(np.prod(input_shape, dtype=np.int64))
                    ):
                        semantic_input_shape = list(input_shape)
                    flatten_signature_basis = (
                        list(semantic_input_shape)
                        if semantic_input_shape != input_shape
                        else list(input_signature)
                    )
                    flatten_axis = int(op.options.get("onnxFlattenAxis", 1))
                    if flatten_axis < 0:
                        flatten_axis += input_rank
                    if (
                        0 <= flatten_axis <= input_rank
                        and _is_fully_known_positive_shape(input_shape)
                    ):
                        def _flatten_product(values: List[int]) -> int:
                            product = 1
                            for value in values:
                                product *= int(value)
                            return int(product)

                        flatten_shape = [
                            _flatten_product(
                                semantic_input_shape[:flatten_axis]
                            ),
                            _flatten_product(
                                semantic_input_shape[flatten_axis:]
                            ),
                        ]
                        flatten_signature = [
                            (
                                -1
                                if any(
                                    int(v) < 0
                                    for v in flatten_signature_basis[:flatten_axis]
                                )
                                else _flatten_product(
                                    flatten_signature_basis[:flatten_axis]
                                )
                            ),
                            (
                                -1
                                if any(
                                    int(v) < 0
                                    for v in flatten_signature_basis[flatten_axis:]
                                )
                                else _flatten_product(
                                    flatten_signature_basis[flatten_axis:]
                                )
                            ),
                        ]
                        flatten_consumer_feature_dim = op.options.get(
                            "onnxFlattenConsumerFeatureDim",
                            None,
                        )
                        if (
                            flatten_consumer_feature_dim is not None
                            and int(flatten_consumer_feature_dim) > 0
                            and int(flatten_signature[1])
                            != int(flatten_consumer_feature_dim)
                        ):
                            flatten_signature[1] = int(
                                flatten_consumer_feature_dim
                            )
                            flatten_shape[1] = int(
                                flatten_consumer_feature_dim
                            )
                        _set_operator_option(
                            op,
                            "newShape",
                            (
                                []
                                if len(existing_flatten_new_shape) == 0
                                else [int(v) for v in flatten_signature]
                            ),
                        )
                        if len(inputs) >= 2:
                            shape_tensor = model_ir.tensors.get(inputs[1], None)
                            if shape_tensor is not None and shape_tensor.data is not None:
                                _set_int32_const_tensor(
                                    shape_tensor,
                                    flatten_signature,
                                )
                        changed |= _update_tensor_shape(
                            outputs[0],
                            flatten_shape,
                            flatten_signature,
                        )
                        continue
                raw_new_shape = op.options.get("newShape", [])
                try:
                    new_shape = [int(v) for v in np.asarray(raw_new_shape).reshape(-1).tolist()]
                except Exception:
                    new_shape = []
                has_onnx_raw_new_shape = "onnxRawNewShape" in op.options
                onnx_raw_shape = op.options.get("onnxRawNewShape", [])
                try:
                    onnx_raw_shape_list = [
                        int(v) for v in np.asarray(onnx_raw_shape).reshape(-1).tolist()
                    ]
                except Exception:
                    onnx_raw_shape_list = []
                if (
                    bool(op.options.get("onnxBoundaryShapeHint", False))
                    and len(new_shape) > 0
                    and all(int(dim) > 0 for dim in new_shape)
                ):
                    resolved = [int(v) for v in new_shape]
                elif (
                    has_onnx_raw_new_shape
                    and len(onnx_raw_shape_list) > 0
                    and all(int(dim) > 0 for dim in onnx_raw_shape_list)
                ):
                    resolved = [int(v) for v in onnx_raw_shape_list]
                else:
                    resolved = _resolve_reshape_new_shape_from_static_input(
                        new_shape=new_shape,
                        input_signature=input_signature,
                        allow_zero=(
                            bool(op.options.get("allowZero"))
                            if "allowZero" in op.options
                            else None
                        ),
                    )
                if resolved is not None and _is_fully_known_positive_shape(resolved):
                    out_signature = (
                        [int(v) for v in list(output_tensor.shape_signature)]
                        if output_tensor is not None
                        and output_tensor.shape_signature is not None
                        and len(list(output_tensor.shape_signature)) == len(resolved)
                        else [int(v) for v in list(resolved)]
                    )
                    if len(input_signature) == len(resolved) and input_tensor.shape is not None:
                        input_shape = [int(v) for v in list(input_tensor.shape)]
                        for axis, dim in enumerate(input_signature):
                            if (
                                int(dim) < 0
                                and axis < len(input_shape)
                                and int(input_shape[axis]) == int(resolved[axis])
                            ):
                                out_signature[axis] = -1
                    changed |= _update_tensor_shape(outputs[0], resolved, out_signature)
                continue

            if op_type == "SQUEEZE" and len(inputs) >= 1 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                if input_tensor is None or not _is_fully_known_positive_shape(input_tensor.shape):
                    continue
                raw_axes = op.options.get("squeezeDims", [])
                squeeze_axes = _parse_axes_option(raw_axes)
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                out_shape, out_signature = _infer_squeeze_output_shape_and_signature(
                    input_shape=list(input_tensor.shape),
                    input_signature=input_signature,
                    squeeze_axes=squeeze_axes,
                )
                if out_shape is None or not _is_fully_known_positive_shape(out_shape):
                    continue
                changed |= _update_tensor_shape(
                    outputs[0],
                    out_shape,
                    out_signature,
                )
                continue

            if op_type == "MEAN" and len(inputs) >= 2 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                axes_tensor = model_ir.tensors.get(inputs[1], None)
                if input_tensor is None or not _is_fully_known_positive_shape(input_tensor.shape):
                    continue
                axes_vals = _read_const_ints_from_tensor(axes_tensor)
                keep_dims = bool(op.options.get("keepDims", False))
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                out_shape, out_signature = _infer_reduce_output_shape_and_signature(
                    input_shape=list(input_tensor.shape),
                    input_signature=input_signature,
                    axes=axes_vals,
                    keep_dims=keep_dims,
                )
                if out_shape is None or not _is_fully_known_positive_shape(out_shape):
                    continue
                changed |= _update_tensor_shape(
                    outputs[0],
                    out_shape,
                    out_signature,
                )
                continue

            if (
                op_type == "STRIDED_SLICE"
                and len(inputs) >= 4
                and len(outputs) == 1
            ):
                input_tensor = model_ir.tensors.get(inputs[0], None)
                begin_values = _read_const_ints_from_tensor(
                    model_ir.tensors.get(inputs[1], None)
                )
                end_values = _read_const_ints_from_tensor(
                    model_ir.tensors.get(inputs[2], None)
                )
                stride_values = _read_const_ints_from_tensor(
                    model_ir.tensors.get(inputs[3], None)
                )
                if (
                    input_tensor is None
                    or not _is_fully_known_positive_shape(input_tensor.shape)
                    or begin_values is None
                    or end_values is None
                    or stride_values is None
                ):
                    continue
                input_shape = [int(v) for v in list(input_tensor.shape)]
                rank = len(input_shape)
                if not (
                    len(begin_values) == rank
                    and len(end_values) == rank
                    and len(stride_values) == rank
                    and all(int(v) > 0 for v in stride_values)
                    and int(op.options.get("ellipsisMask", 0)) == 0
                    and int(op.options.get("newAxisMask", 0)) == 0
                    and int(op.options.get("shrinkAxisMask", 0)) == 0
                ):
                    continue
                begin_mask = int(op.options.get("beginMask", 0))
                end_mask = int(op.options.get("endMask", 0))
                out_shape: List[int] = []
                for axis, dim in enumerate(input_shape):
                    stride = int(stride_values[axis])
                    if ((begin_mask >> axis) & 1) != 0:
                        start = 0
                    else:
                        start = int(begin_values[axis])
                        start = (
                            max(int(dim) + start, 0)
                            if start < 0
                            else min(start, int(dim))
                        )
                    if ((end_mask >> axis) & 1) != 0:
                        stop = int(dim)
                    else:
                        stop = int(end_values[axis])
                        stop = (
                            max(int(dim) + stop, 0)
                            if stop < 0
                            else min(stop, int(dim))
                        )
                    out_shape.append(
                        int(max((int(stop) - int(start) + stride - 1) // stride, 0))
                    )
                if _is_fully_known_positive_shape(out_shape):
                    input_signature = (
                        [int(v) for v in list(input_tensor.shape_signature)]
                        if input_tensor.shape_signature is not None
                        else list(input_shape)
                    )
                    out_signature = [
                        -1 if int(input_signature[axis]) < 0 else int(out_shape[axis])
                        for axis in range(rank)
                    ]
                    changed |= _update_tensor_shape(
                        outputs[0], out_shape, out_signature
                    )
                continue

            if op_type == "SLICE" and len(inputs) >= 3 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                begin_tensor = model_ir.tensors.get(inputs[1], None)
                size_tensor = model_ir.tensors.get(inputs[2], None)
                if input_tensor is None:
                    continue
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                has_dynamic_input_dim = any(int(v) < 0 for v in input_signature)
                begin_vals = _read_const_ints_from_tensor(begin_tensor)
                size_vals = _read_const_ints_from_tensor(size_tensor)
                out_shape, resolved_begin, resolved_size = _infer_slice_output_shape_and_resolved_params(
                    input_shape=list(input_tensor.shape),
                    begin_vals=begin_vals,
                    size_vals=size_vals,
                )
                if out_shape is None or resolved_begin is None or resolved_size is None:
                    continue
                out_signature = _infer_slice_output_signature(
                    input_shape=list(input_tensor.shape),
                    input_signature=input_signature,
                    begin_vals=begin_vals,
                    size_vals=size_vals,
                )
                if (
                    bool(op.options.get("preserveDynamicShape", False))
                    and out_signature is not None
                    and len(out_signature) == len(size_vals)
                ):
                    out_signature = [
                        -1 if int(size_vals[axis]) == -1 else int(size_vals[axis])
                        for axis in range(len(size_vals))
                    ]
                    output_tensor = model_ir.tensors.get(outputs[0], None)
                    if output_tensor is not None:
                        _set_tensor_shape_signature(
                            output_tensor,
                            [int(v) for v in out_signature],
                        )
                shape_for_update = [int(v) for v in list(out_shape)]
                if (
                    has_dynamic_input_dim
                    and out_signature is not None
                    and len(out_signature) == len(shape_for_update)
                ):
                    existing_output_shape = []
                    output_tensor = model_ir.tensors.get(outputs[0], None)
                    if output_tensor is not None:
                        existing_output_shape = [int(v) for v in list(output_tensor.shape)]
                    merged_shape: List[int] = []
                    for idx, dim in enumerate(shape_for_update):
                        sig_dim = int(out_signature[idx])
                        if sig_dim > 0:
                            merged_shape.append(int(sig_dim))
                            continue
                        if idx < len(existing_output_shape) and int(existing_output_shape[idx]) > 0:
                            merged_shape.append(int(existing_output_shape[idx]))
                            continue
                        merged_shape.append(int(dim) if int(dim) > 0 else 1)
                    shape_for_update = [int(v) for v in list(merged_shape)]
                out_shape_is_fully_known_positive = _is_fully_known_positive_shape(shape_for_update)
                if out_shape_is_fully_known_positive:
                    changed |= _update_tensor_shape(outputs[0], shape_for_update, out_signature)
                # Keep runtime-driven SLICE semantics when input dimensions are dynamic.
                # Also avoid mutating begin/size when inferred output dims include 0 because
                # stale static metadata can collapse valid slicing ranges into empty outputs.
                if (
                    out_shape_is_fully_known_positive
                    and not has_dynamic_input_dim
                    and not bool(op.options.get("preserveDynamicShape", False))
                ):
                    changed |= _write_const_ints_tracked(begin_tensor, resolved_begin)
                    changed |= _write_const_ints_tracked(size_tensor, resolved_size)
                continue

            if op_type in {"PAD", "MIRROR_PAD"} and len(inputs) >= 2 and len(outputs) == 1:
                input_tensor = model_ir.tensors.get(inputs[0], None)
                pads_tensor = model_ir.tensors.get(inputs[1], None)
                if (
                    input_tensor is None
                    or pads_tensor is None
                    or not _is_fully_known_positive_shape(input_tensor.shape)
                ):
                    continue
                in_shape = [int(v) for v in list(input_tensor.shape)]
                rank = len(in_shape)
                pads_vals = _read_const_ints_from_tensor(pads_tensor)
                if pads_vals is None or len(pads_vals) != int(rank * 2):
                    continue
                pads_shape = (
                    [int(v) for v in list(pads_tensor.shape)]
                    if pads_tensor.shape is not None
                    else []
                )

                pad_pairs: List[Tuple[int, int]] = []
                if len(pads_shape) == 2 and int(pads_shape[0]) == int(rank) and int(pads_shape[1]) == 2:
                    for axis in range(rank):
                        before = int(pads_vals[int(axis * 2)])
                        after = int(pads_vals[int(axis * 2 + 1)])
                        pad_pairs.append((before, after))
                elif len(pads_shape) == 2 and int(pads_shape[0]) == 2 and int(pads_shape[1]) == int(rank):
                    for axis in range(rank):
                        before = int(pads_vals[int(axis)])
                        after = int(pads_vals[int(rank + axis)])
                        pad_pairs.append((before, after))
                else:
                    for axis in range(rank):
                        before = int(pads_vals[int(axis * 2)])
                        after = int(pads_vals[int(axis * 2 + 1)])
                        pad_pairs.append((before, after))

                if any(int(before) < 0 or int(after) < 0 for before, after in pad_pairs):
                    continue

                out_shape: List[int] = []
                valid_out = True
                for axis in range(rank):
                    before, after = pad_pairs[axis]
                    dim = int(in_shape[axis]) + int(before) + int(after)
                    if int(dim) <= 0:
                        valid_out = False
                        break
                    out_shape.append(int(dim))
                if not valid_out:
                    continue

                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(in_shape)
                )
                out_signature: List[int] = []
                for axis in range(rank):
                    sig_dim = int(input_signature[axis]) if axis < len(input_signature) else int(in_shape[axis])
                    if int(sig_dim) < 0:
                        out_signature.append(-1)
                    else:
                        out_signature.append(int(out_shape[axis]))

                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                filter_tensor = model_ir.tensors.get(inputs[1], None)
                out_tensor = model_ir.tensors.get(outputs[0], None)
                if (
                    in_tensor is None
                    or filter_tensor is None
                    or not _is_fully_known_positive_shape(in_tensor.shape)
                    or not _is_fully_known_positive_shape(filter_tensor.shape)
                ):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                filter_shape = [int(v) for v in list(filter_tensor.shape)]
                if len(in_shape) != 4 or len(filter_shape) != 4:
                    continue
                padding = str(op.options.get("padding", "SAME"))
                stride_h = int(op.options.get("strideH", 1))
                stride_w = int(op.options.get("strideW", 1))
                dilation_h = int(op.options.get("dilationHFactor", 1))
                dilation_w = int(op.options.get("dilationWFactor", 1))
                kernel_h = int(filter_shape[1])
                kernel_w = int(filter_shape[2])
                out_h = _infer_conv_out_dim(in_shape[1], kernel_h, stride_h, dilation_h, padding)
                out_w = _infer_conv_out_dim(in_shape[2], kernel_w, stride_w, dilation_w, padding)
                if out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0:
                    continue
                if op_type == "CONV_2D":
                    out_c = int(filter_shape[0])
                else:
                    # TFLite depthwise filter layout: [1, KH, KW, OC]
                    out_c = int(filter_shape[3])
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(out_c)]
                input_signature = (
                    list(in_tensor.shape_signature)
                    if in_tensor.shape_signature is not None
                    else list(in_shape)
                )
                existing_out_signature = (
                    list(out_tensor.shape_signature)
                    if out_tensor is not None and out_tensor.shape_signature is not None
                    else None
                )
                out_signature = _infer_rank4_signature_from_input(
                    input_signature=input_signature,
                    output_shape=out_shape,
                    existing_output_signature=existing_out_signature,
                    propagate_channel=False,
                )
                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type in {"AVERAGE_POOL_2D", "MAX_POOL_2D"} and len(inputs) >= 1 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                out_tensor = model_ir.tensors.get(outputs[0], None)
                if in_tensor is None or not _is_fully_known_positive_shape(in_tensor.shape):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                if len(in_shape) != 4:
                    continue

                padding = str(op.options.get("padding", "SAME")).upper()
                stride_h = int(op.options.get("strideH", 1))
                stride_w = int(op.options.get("strideW", 1))
                filter_h = int(op.options.get("filterHeight", 1))
                filter_w = int(op.options.get("filterWidth", 1))

                out_h = _infer_conv_out_dim(in_shape[1], filter_h, stride_h, 1, padding)
                out_w = _infer_conv_out_dim(in_shape[2], filter_w, stride_w, 1, padding)

                # GlobalAveragePool lowering can become stale after aggressive transpose/layout
                # rewrites. Recover invalid VALID+stride=1 average-pool metadata by snapping
                # pool kernel to the current input spatial size.
                if (
                    op_type == "AVERAGE_POOL_2D"
                    and (out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0)
                    and padding == "VALID"
                    and int(stride_h) == 1
                    and int(stride_w) == 1
                ):
                    if int(filter_h) > int(in_shape[1]) or int(filter_w) > int(in_shape[2]):
                        filter_h = int(in_shape[1])
                        filter_w = int(in_shape[2])
                        _set_operator_option(op, "filterHeight", int(filter_h))
                        _set_operator_option(op, "filterWidth", int(filter_w))
                        out_h = _infer_conv_out_dim(in_shape[1], filter_h, stride_h, 1, padding)
                        out_w = _infer_conv_out_dim(in_shape[2], filter_w, stride_w, 1, padding)

                if out_h is None or out_w is None or int(out_h) <= 0 or int(out_w) <= 0:
                    continue
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(in_shape[3])]
                input_signature = (
                    list(in_tensor.shape_signature)
                    if in_tensor.shape_signature is not None
                    else list(in_shape)
                )
                existing_out_signature = (
                    list(out_tensor.shape_signature)
                    if out_tensor is not None and out_tensor.shape_signature is not None
                    else None
                )
                out_signature = _infer_rank4_signature_from_input(
                    input_signature=input_signature,
                    output_shape=out_shape,
                    existing_output_signature=existing_out_signature,
                    propagate_channel=True,
                )
                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

            if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"} and len(inputs) >= 2 and len(outputs) == 1:
                in_tensor = model_ir.tensors.get(inputs[0], None)
                size_tensor = model_ir.tensors.get(inputs[1], None)
                out_tensor = model_ir.tensors.get(outputs[0], None)
                if in_tensor is None or not _is_fully_known_positive_shape(in_tensor.shape):
                    continue
                in_shape = [int(v) for v in list(in_tensor.shape)]
                if len(in_shape) != 4:
                    continue

                out_h = None
                out_w = None
                size_source = "unknown"
                size_vals = _read_const_ints_from_tensor(size_tensor)
                size_tensor_is_const = bool(size_tensor is not None and size_tensor.data is not None)
                onnx_sizes_hw = op.options.get("onnxSizesHW", None)
                if isinstance(onnx_sizes_hw, (list, tuple)) and len(onnx_sizes_hw) >= 2:
                    try:
                        out_h = int(onnx_sizes_hw[0])
                        out_w = int(onnx_sizes_hw[1])
                        size_source = "onnx_sizes"
                    except Exception:
                        out_h = None
                        out_w = None
                        size_source = "unknown"
                if out_h is None or out_w is None:
                    onnx_scales_hw = op.options.get("onnxScalesHW", None)
                    if isinstance(onnx_scales_hw, (list, tuple)) and len(onnx_scales_hw) >= 2:
                        try:
                            out_h = int(round(float(in_shape[1]) * float(onnx_scales_hw[0])))
                            out_w = int(round(float(in_shape[2]) * float(onnx_scales_hw[1])))
                            size_source = "onnx_scales"
                        except Exception:
                            out_h = None
                            out_w = None
                            size_source = "unknown"
                if out_h is None or out_w is None:
                    if size_vals is None or len(size_vals) < 2:
                        continue
                    out_h = int(size_vals[0])
                    out_w = int(size_vals[1])
                    size_source = "const_size"
                if out_h <= 0 or out_w <= 0:
                    continue
                # Synchronize resize-size constants with ONNX-derived hints so downstream
                # prepare-time shape inference follows the reconciled NHWC metadata.
                if size_tensor_is_const and size_source in {"onnx_sizes", "onnx_scales"}:
                    changed |= _write_const_ints_tracked(
                        size_tensor,
                        [int(out_h), int(out_w)],
                    )
                out_shape = [int(in_shape[0]), int(out_h), int(out_w), int(in_shape[3])]
                input_signature = (
                    list(in_tensor.shape_signature)
                    if in_tensor.shape_signature is not None
                    else list(in_shape)
                )
                existing_out_signature = (
                    list(out_tensor.shape_signature)
                    if out_tensor is not None and out_tensor.shape_signature is not None
                    else None
                )
                out_signature = _infer_rank4_signature_from_input(
                    input_signature=input_signature,
                    output_shape=out_shape,
                    existing_output_signature=existing_out_signature,
                    propagate_channel=True,
                )
                if len(out_signature) == 4 and len(input_signature) == 4 and size_source == "onnx_scales":
                    if int(input_signature[1]) < 0:
                        out_signature[1] = -1
                    if int(input_signature[2]) < 0:
                        out_signature[2] = -1
                changed |= _update_tensor_shape(outputs[0], out_shape, out_signature)
                continue

        if not changed:
            break

    details = {"reconciled_static_tensor_shapes": int(updated_tensors)}
    if include_mutation_count:
        details["reconciled_static_shape_mutations"] = int(mutation_count)
    return details


def run_static_shape_topology_reconciliation(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Reconcile static shapes, then restore operator topology."""

    shape_stats = reconcile_static_tensor_shapes(
        model_ir,
        include_mutation_count=True,
    )
    sort_stats = _topologically_sort_operators(model_ir)
    return {
        "reconciled_static_tensor_shapes": int(
            shape_stats.get("reconciled_static_tensor_shapes", 0)
        ),
        "reconciled_static_shape_mutations": int(
            shape_stats.get("reconciled_static_shape_mutations", 0)
        ),
        "reordered_operators": int(sort_stats.get("reordered_operators", 0)),
        "cycle_detected": int(sort_stats.get("cycle_detected", 0)),
    }
