from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    ModelIRPreflightResult,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_PERM_NCDHW_TO_NDHWC = [0, 2, 3, 4, 1]
_REDUCE_OPS = {"SUM", "MEAN", "REDUCE_MAX"}
_SCATTER_LAYOUT_OPS = {"SCATTER_ND", "ADD", "SUB", "MUL"}
_SCATTER_CONST_INPUT_LAYOUT_OPS = {"ADD", "SUB", "MUL"}
_ALLOWED_OPS = {
    "SLICE",
    "MUL",
    "SUM",
    "SQRT",
    "DIV",
    "MEAN",
    "RESHAPE",
    "CAST",
    "SCATTER_ND",
    "SUB",
    "ADD",
    "CONCATENATION",
    "REDUCE_MAX",
}


@dataclass(frozen=True)
class _CostVolumeScatterCandidate:
    post_op: OperatorIR
    conv_op: OperatorIR
    candidate_ops: Tuple[OperatorIR, ...]
    boundary_plans: Tuple[Tuple[OperatorIR, str, str], ...]


def _normalize_axis(axis: int, rank: int) -> Optional[int]:
    value = int(axis)
    if value < 0:
        value += int(rank)
    if value < 0 or value >= int(rank):
        return None
    return int(value)


def _indices_in_bounds(coords: np.ndarray, shape: List[int]) -> bool:
    if int(coords.ndim) != 2 or int(coords.shape[1]) != int(len(shape)):
        return False
    for axis, dimension in enumerate(shape):
        dim = int(dimension)
        if dim <= 0:
            return False
        axis_values = coords[:, int(axis)]
        if np.any(axis_values < 0) or np.any(axis_values >= dim):
            return False
    return True


def _const_ints_for_input(
    model_ir: ModelIR,
    op: OperatorIR,
    input_index: int,
) -> Optional[List[int]]:
    if int(input_index) < 0 or int(input_index) >= len(op.inputs):
        return None
    return _read_const_ints_from_tensor(
        model_ir.tensors.get(str(op.inputs[int(input_index)]), None)
    )


def _validate_candidate_constants(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    candidate_ops: Tuple[OperatorIR, ...],
) -> bool:
    for op in candidate_ops:
        op_type = str(op.op_type)
        if op_type == "SLICE":
            begin_values = _const_ints_for_input(model_ir, op, 1)
            size_values = _const_ints_for_input(model_ir, op, 2)
            if (
                begin_values is None
                or size_values is None
                or len(begin_values) != 4
                or len(size_values) != 4
            ):
                return False
        elif op_type in _REDUCE_OPS:
            axes_values = _const_ints_for_input(model_ir, op, 1)
            if axes_values is None or len(axes_values) == 0:
                return False
            if any(_normalize_axis(axis, 4) is None for axis in axes_values):
                return False
        elif op_type == "CONCATENATION":
            rank_hint = 4
            if len(op.inputs) > 0:
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                if input_tensor is not None and input_tensor.shape is not None:
                    rank_hint = int(len(list(input_tensor.shape)))
            if int(rank_hint) == 4:
                if not isinstance(op.options, dict):
                    return False
                raw_axis = int(op.options.get("axis", 1))
                if _normalize_axis(raw_axis, 4) is None:
                    return False
        elif op_type == "SCATTER_ND":
            shape_values = _const_ints_for_input(model_ir, op, 2)
            if shape_values is None or len(shape_values) != 5:
                return False
            target_shape = _permute_shape(
                [int(value) for value in shape_values],
                _PERM_NCDHW_TO_NDHWC,
            )
            if target_shape is None or len(op.inputs) < 1:
                return False
            indices_name = str(op.inputs[0])
            indices_producer = graph_index.producer(indices_name)
            const_name = indices_name
            if (
                indices_producer is not None
                and str(indices_producer.op_type) == "CAST"
                and len(indices_producer.inputs) == 1
            ):
                const_name = str(indices_producer.inputs[0])
            const_tensor = model_ir.tensors.get(const_name, None)
            if const_tensor is None or const_tensor.data is None:
                return False
            coordinate_array = np.asarray(const_tensor.data)
            if (
                int(coordinate_array.size) == 0
                or int(coordinate_array.ndim) == 0
                or int(coordinate_array.shape[-1]) != 5
            ):
                return False
            flat = coordinate_array.reshape(-1, 5)
            remapped = flat[:, [int(value) for value in _PERM_NCDHW_TO_NDHWC]]
            if not (
                _indices_in_bounds(flat, target_shape)
                or _indices_in_bounds(remapped, target_shape)
            ):
                return False
    return True


def _resolve_cost_volume_scatter_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_CostVolumeScatterCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for post_op in model_ir.operators:
        post_index = graph_index.operator_index(post_op)
        if (
            post_index is None
            or str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or _read_transpose_perm(model_ir, post_op) != _PERM_NCDHW_TO_NDHWC
        ):
            continue
        post_input = str(post_op.inputs[0])
        post_output = str(post_op.outputs[0])
        if post_input in model_outputs or post_output in model_outputs:
            continue
        if set(graph_index.consumer_indices(post_input)) != {int(post_index)}:
            continue
        post_users = graph_index.consumer_indices(post_output)
        if len(post_users) != 1:
            continue
        conv_op = model_ir.operators[int(post_users[0])]
        if (
            str(conv_op.op_type) != "CONV_3D"
            or len(conv_op.inputs) < 1
            or str(conv_op.inputs[0]) != post_output
        ):
            continue
        root_index = graph_index.producers.get(post_input, None)
        if root_index is None:
            continue

        candidate_indices: set[int] = set()
        boundary_indices: set[int] = set()
        traversal_ok = True
        stack = [int(root_index)]
        visited: set[int] = set()
        while stack:
            current_index = int(stack.pop())
            if current_index in visited:
                continue
            visited.add(current_index)
            current_op = model_ir.operators[current_index]
            if str(current_op.op_type) == "TRANSPOSE":
                if _read_transpose_perm(model_ir, current_op) == _PERM_NHWC_TO_NCHW:
                    boundary_indices.add(current_index)
                    continue
                traversal_ok = False
                break
            if str(current_op.op_type) not in _ALLOWED_OPS:
                traversal_ok = False
                break
            candidate_indices.add(current_index)
            for input_name in current_op.inputs:
                parent_index = graph_index.producers.get(str(input_name), None)
                if parent_index is not None:
                    stack.append(int(parent_index))
        if (
            not traversal_ok
            or len(candidate_indices) == 0
            or len(boundary_indices) != 2
        ):
            continue

        boundary_plans: List[Tuple[OperatorIR, str, str]] = []
        for boundary_index in sorted(boundary_indices):
            boundary_op = model_ir.operators[int(boundary_index)]
            boundary_input = str(boundary_op.inputs[0])
            boundary_output = str(boundary_op.outputs[0])
            boundary_users = set(graph_index.consumer_indices(boundary_output))
            if (
                boundary_input in model_outputs
                or boundary_output in model_outputs
                or len(boundary_users) == 0
                or any(user not in candidate_indices for user in boundary_users)
            ):
                traversal_ok = False
                break
            boundary_plans.append(
                (boundary_op, boundary_input, boundary_output)
            )
        if not traversal_ok:
            continue
        candidate_ops = tuple(
            model_ir.operators[index] for index in sorted(candidate_indices)
        )
        if not _validate_candidate_constants(
            model_ir,
            graph_index,
            candidate_ops,
        ):
            continue
        return _CostVolumeScatterCandidate(
            post_op=post_op,
            conv_op=conv_op,
            candidate_ops=candidate_ops,
            boundary_plans=tuple(boundary_plans),
        )
    return None


def _has_cost_volume_scatter_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_cost_volume_scatter_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_cost_volume_scatter_ndhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove NCHW/NCDHW adapter transposes in cost-volume ScatterND accumulation motifs.

    Target:
      desc0_nhwc --T(0,3,1,2)--> desc0_nchw
      desc1_nhwc --T(0,3,1,2)--> desc1_nchw
      (SLICE/SUM/SQRT/DIV/MEAN/RESHAPE/SCATTER_ND chain in NCHW+NCDHW space)
      vol_ncdhw --T(0,2,3,4,1)--> vol_ndhwc --CONV_3D-->

    Rewrite:
      - Keep rank-4 path in NHWC: rewrite SLICE and reduce axes accordingly.
      - Keep ScatterND result tensors in NDHWC: rewrite ScatterND shape and indices.
      - Remove the two leading NHWC->NCHW transposes and trailing NCDHW->NDHWC transpose.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = _PERM_NHWC_TO_NCHW
    perm_nchw_to_nhwc = _PERM_NCHW_TO_NHWC
    perm_ncdhw_to_ndhwc = _PERM_NCDHW_TO_NDHWC
    reduce_ops = _REDUCE_OPS
    scatter_layout_ops = _SCATTER_LAYOUT_OPS
    scatter_const_input_layout_ops = _SCATTER_CONST_INPUT_LAYOUT_OPS

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _ensure_exclusive_const_input(
        *,
        op: OperatorIR,
        op_idx: int,
        input_index: int,
        consumers: Dict[str, List[int]],
    ) -> Optional[TensorIR]:
        if int(input_index) < 0 or int(input_index) >= len(op.inputs):
            return None
        tensor_name = str(op.inputs[int(input_index)])
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None or tensor.data is None:
            return None
        tensor_consumers = [int(v) for v in consumers.get(tensor_name, [])]
        if len(tensor_consumers) <= 1:
            return tensor
        if set(tensor_consumers) == {int(op_idx)}:
            return tensor
        new_name = _unique_tensor_name(f"{tensor_name}_layout")
        new_data = np.asarray(tensor.data).copy()
        new_shape_signature = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else [int(v) for v in list(tensor.shape)]
        )
        model_ir.tensors[new_name] = TensorIR(
            name=new_name,
            dtype=str(tensor.dtype),
            shape=[int(v) for v in list(tensor.shape)],
            shape_signature=[int(v) for v in list(new_shape_signature)],
            data=new_data,
            is_variable=bool(tensor.is_variable),
            quantization=_clone_quantization(tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=str(new_name),
            graph_index=graph_index,
        )
        return model_ir.tensors.get(str(new_name), None)

    def _remap_axes_const(
        *,
        op: OperatorIR,
        op_idx: int,
        consumers: Dict[str, List[int]],
        input_index: int,
        rank: int,
        axis_map: List[int],
    ) -> bool:
        axes_tensor = _ensure_exclusive_const_input(
            op=op,
            op_idx=int(op_idx),
            input_index=int(input_index),
            consumers=consumers,
        )
        axes_vals = _read_const_ints_from_tensor(axes_tensor)
        if axes_vals is None or len(axes_vals) == 0:
            return False
        mapped_axes: List[int] = []
        for axis in axes_vals:
            normalized = _normalize_axis(int(axis), int(rank))
            if normalized is None:
                return False
            mapped_axes.append(int(axis_map[int(normalized)]))
        _write_const_ints_to_tensor(axes_tensor, [int(v) for v in mapped_axes])
        return True

    def _remap_slice_consts(
        *,
        op: OperatorIR,
        op_idx: int,
        consumers: Dict[str, List[int]],
    ) -> bool:
        if len(op.inputs) < 3:
            return False
        begin_tensor = _ensure_exclusive_const_input(
            op=op,
            op_idx=int(op_idx),
            input_index=1,
            consumers=consumers,
        )
        size_tensor = _ensure_exclusive_const_input(
            op=op,
            op_idx=int(op_idx),
            input_index=2,
            consumers=consumers,
        )
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None:
            return False
        if len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        new_begin = [int(begin_vals[int(idx)]) for idx in perm_nchw_to_nhwc]
        new_size = [int(size_vals[int(idx)]) for idx in perm_nchw_to_nhwc]
        _write_const_ints_to_tensor(begin_tensor, new_begin)
        _write_const_ints_to_tensor(size_tensor, new_size)
        return True

    def _remap_concat_axis(
        *,
        op: OperatorIR,
        rank: int,
        axis_map: List[int],
    ) -> bool:
        if not isinstance(op.options, dict):
            return False
        raw_axis = int(op.options.get("axis", 1))
        if raw_axis < 0:
            raw_axis += int(rank)
        if raw_axis < 0 or raw_axis >= int(rank):
            return False
        op.options["axis"] = int(axis_map[int(raw_axis)])
        return True

    def _remap_scatter_shape_const(
        *,
        op: OperatorIR,
        op_idx: int,
        consumers: Dict[str, List[int]],
    ) -> Optional[List[int]]:
        if len(op.inputs) < 3:
            return None
        shape_tensor = _ensure_exclusive_const_input(
            op=op,
            op_idx=int(op_idx),
            input_index=2,
            consumers=consumers,
        )
        shape_vals = _read_const_ints_from_tensor(shape_tensor)
        if shape_vals is None or len(shape_vals) != 5:
            return None
        mapped_shape = _permute_shape(
            [int(v) for v in list(shape_vals)],
            perm_ncdhw_to_ndhwc,
        )
        if mapped_shape is None:
            return None
        _write_const_ints_to_tensor(shape_tensor, [int(v) for v in list(mapped_shape)])
        return [int(v) for v in list(mapped_shape)]

    def _remap_scatter_indices_const(
        *,
        op: OperatorIR,
        op_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        target_shape: List[int],
    ) -> bool:
        if len(op.inputs) < 1:
            return False
        indices_name = str(op.inputs[0])
        const_owner_op = op
        const_owner_idx = int(op_idx)
        const_input_index = 0

        indices_prod_idx = producers.get(indices_name, None)
        if indices_prod_idx is not None:
            indices_prod_op = model_ir.operators[int(indices_prod_idx)]
            if str(indices_prod_op.op_type) == "CAST" and len(indices_prod_op.inputs) == 1:
                const_owner_op = indices_prod_op
                const_owner_idx = int(indices_prod_idx)
                const_input_index = 0

        const_tensor = _ensure_exclusive_const_input(
            op=const_owner_op,
            op_idx=int(const_owner_idx),
            input_index=int(const_input_index),
            consumers=consumers,
        )
        if const_tensor is None or const_tensor.data is None:
            return False

        coord_array = np.asarray(const_tensor.data)
        if int(coord_array.size) == 0:
            return False
        if int(coord_array.shape[-1]) != 5:
            return False
        flat = coord_array.reshape(-1, 5).copy()

        def _indices_in_bounds(coords: np.ndarray, shape: List[int]) -> bool:
            if int(coords.shape[1]) != int(len(shape)):
                return False
            for axis in range(int(len(shape))):
                dim = int(shape[axis])
                if dim <= 0:
                    return False
                axis_vals = coords[:, int(axis)]
                if np.any(axis_vals < 0):
                    return False
                if np.any(axis_vals >= dim):
                    return False
            return True

        remapped_flat = flat[:, [int(v) for v in perm_ncdhw_to_ndhwc]]
        selected_flat = flat
        if _indices_in_bounds(flat, target_shape):
            selected_flat = flat
        elif _indices_in_bounds(remapped_flat, target_shape):
            selected_flat = remapped_flat
        else:
            return False

        remapped = selected_flat.reshape(coord_array.shape).astype(coord_array.dtype, copy=False)
        if not np.array_equal(remapped, coord_array):
            const_tensor.data = np.asarray(remapped)
            const_tensor.shape = [int(v) for v in list(remapped.shape)]
            const_tensor.shape_signature = [int(v) for v in list(remapped.shape)]
        return True

    def _permute_tensor_data_and_metadata_if_rank_matches(
        tensor: Optional[TensorIR],
        perm: List[int],
    ) -> None:
        if tensor is None:
            return
        if tensor.data is not None:
            try:
                data_array = np.asarray(tensor.data)
                if int(data_array.ndim) == int(len(perm)):
                    transposed = np.transpose(data_array, axes=perm)
                    tensor.data = np.asarray(transposed)
                    tensor.shape = [int(v) for v in list(transposed.shape)]
                    tensor.shape_signature = [int(v) for v in list(transposed.shape)]
                    return
            except Exception:
                pass
        _permute_tensor_metadata_if_rank_matches(tensor, perm)

    while True:
        changed = False
        candidate = _resolve_cost_volume_scatter_candidate(model_ir, graph_index)
        if candidate is None:
            break
        consumers = graph_index.consumers
        producers = graph_index.producers

        for candidate in [candidate]:
            post_op = candidate.post_op
            conv_op = candidate.conv_op
            post_idx = graph_index.operator_index(post_op)
            if post_idx is None:
                break
            post_in_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            candidate_indices = [
                graph_index.operator_index(op) for op in candidate.candidate_ops
            ]
            boundary_plans = [
                (graph_index.operator_index(op), boundary_in, boundary_out)
                for op, boundary_in, boundary_out in candidate.boundary_plans
            ]
            if any(index is None for index in candidate_indices) or any(
                index is None for index, _, _ in boundary_plans
            ):
                break
            candidate_indices = [int(index) for index in candidate_indices]
            boundary_plans = [
                (int(index), boundary_in, boundary_out)
                for index, boundary_in, boundary_out in boundary_plans
            ]

            apply_ok = True
            for op_idx in sorted(list(candidate_indices)):
                op = model_ir.operators[int(op_idx)]
                op_type = str(op.op_type)
                if op_type == "SLICE":
                    if not _remap_slice_consts(
                        op=op,
                        op_idx=int(op_idx),
                        consumers=consumers,
                    ):
                        apply_ok = False
                        break
                elif op_type in reduce_ops:
                    if not _remap_axes_const(
                        op=op,
                        op_idx=int(op_idx),
                        consumers=consumers,
                        input_index=1,
                        rank=4,
                        axis_map=perm_nhwc_to_nchw,
                    ):
                        apply_ok = False
                        break
                elif op_type == "CONCATENATION":
                    rank_hint = 4
                    if len(op.inputs) > 0:
                        in_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                        if in_tensor is not None and in_tensor.shape is not None:
                            rank_hint = int(len(list(in_tensor.shape)))
                    if int(rank_hint) == 4:
                        if not _remap_concat_axis(
                            op=op,
                            rank=4,
                            axis_map=perm_nhwc_to_nchw,
                        ):
                            apply_ok = False
                            break
                elif op_type == "SCATTER_ND":
                    mapped_shape = _remap_scatter_shape_const(
                        op=op,
                        op_idx=int(op_idx),
                        consumers=consumers,
                    )
                    if mapped_shape is None:
                        apply_ok = False
                        break
                    if not _remap_scatter_indices_const(
                        op=op,
                        op_idx=int(op_idx),
                        producers=producers,
                        consumers=consumers,
                        target_shape=[int(v) for v in list(mapped_shape)],
                    ):
                        apply_ok = False
                        break

            if not apply_ok:
                continue

            for _, boundary_in, boundary_out in boundary_plans:
                _replace_tensor_inputs(
                    model_ir=model_ir,
                    src_name=str(boundary_out),
                    dst_name=str(boundary_in),
                    graph_index=graph_index,
                )

            conv_inputs = [
                str(post_in_name) if str(v) == str(post_out_name) else str(v)
                for v in list(conv_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv_op,
                new_inputs=conv_inputs,
                graph_index=graph_index,
            )

            for op_idx in sorted(list(candidate_indices)):
                op = model_ir.operators[int(op_idx)]
                if str(op.op_type) not in scatter_const_input_layout_ops:
                    continue
                for input_name in list(op.inputs):
                    input_tensor = model_ir.tensors.get(str(input_name), None)
                    if input_tensor is None:
                        continue
                    rank = (
                        len(list(input_tensor.shape))
                        if input_tensor.shape is not None
                        else (
                            len(list(input_tensor.shape_signature))
                            if input_tensor.shape_signature is not None
                            else 0
                        )
                    )
                    if int(rank) != 5:
                        continue
                    producer_idx = producers.get(str(input_name), None)
                    if producer_idx is not None:
                        continue
                    _permute_tensor_data_and_metadata_if_rank_matches(
                        input_tensor,
                        perm_ncdhw_to_ndhwc,
                    )

            for op_idx in sorted(list(candidate_indices)):
                op = model_ir.operators[int(op_idx)]
                for output_name in list(op.outputs):
                    output_tensor = model_ir.tensors.get(str(output_name), None)
                    if output_tensor is None:
                        continue
                    rank = (
                        len(list(output_tensor.shape))
                        if output_tensor.shape is not None
                        else (
                            len(list(output_tensor.shape_signature))
                            if output_tensor.shape_signature is not None
                            else 0
                        )
                    )
                    if int(rank) == 4:
                        _permute_tensor_metadata_if_rank_matches(
                            output_tensor,
                            perm_nchw_to_nhwc,
                        )
                    elif int(rank) == 5 and str(op.op_type) in scatter_layout_ops:
                        _permute_tensor_metadata_if_rank_matches(
                            output_tensor,
                            perm_ncdhw_to_ndhwc,
                        )

            remove_indices = sorted(
                [int(post_idx)] + [int(v[0]) for v in list(boundary_plans)],
                reverse=True,
            )
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_cost_volume_scatter_ndhwc_chains": int(rewritten)}


def run_cost_volume_scatter_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate a validated cost-volume ScatterND island to NHWC/NDHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "SCATTER_ND", "CONV_3D"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = "optimized_transpose_cost_volume_scatter_ndhwc_chains"

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_cost_volume_scatter_ndhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.cost_volume_scatter_ndhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_cost_volume_scatter_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={stats_key: 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
