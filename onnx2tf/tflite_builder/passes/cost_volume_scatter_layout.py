from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
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
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_transpose_cost_volume_scatter_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
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
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_ncdhw_to_ndhwc = [0, 2, 3, 4, 1]
    perm_ndhwc_to_ncdhw = [0, 4, 1, 2, 3]
    reduce_ops = {"SUM", "MEAN", "REDUCE_MAX"}
    scatter_layout_ops = {"SCATTER_ND", "ADD", "SUB", "MUL"}
    scatter_const_input_layout_ops = {"ADD", "SUB", "MUL"}
    allowed_ops = {
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
        )
        return model_ir.tensors.get(str(new_name), None)

    def _normalize_axis(axis: int, rank: int) -> Optional[int]:
        value = int(axis)
        if value < 0:
            value += int(rank)
        if value < 0 or value >= int(rank):
            return None
        return int(value)

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
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != perm_ncdhw_to_ndhwc
            ):
                continue

            post_in_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if (
                post_in_name in model_outputs
                or post_out_name in model_outputs
            ):
                continue
            if set(int(v) for v in consumers.get(post_in_name, [])) != {int(post_idx)}:
                continue

            post_users = [int(v) for v in consumers.get(post_out_name, [])]
            if len(post_users) != 1:
                continue
            conv_idx = int(post_users[0])
            conv_op = model_ir.operators[int(conv_idx)]
            if (
                str(conv_op.op_type) != "CONV_3D"
                or len(conv_op.inputs) < 1
                or str(conv_op.inputs[0]) != post_out_name
            ):
                continue

            root_idx = producers.get(post_in_name, None)
            if root_idx is None:
                continue

            candidate_indices: set[int] = set()
            boundary_indices: set[int] = set()
            traversal_ok = True
            stack: List[int] = [int(root_idx)]
            visited: set[int] = set()

            while len(stack) > 0:
                current_idx = int(stack.pop())
                if int(current_idx) in visited:
                    continue
                visited.add(int(current_idx))
                current_op = model_ir.operators[int(current_idx)]

                if str(current_op.op_type) == "TRANSPOSE":
                    current_perm = _read_transpose_perm(model_ir, current_op)
                    if current_perm == perm_nhwc_to_nchw:
                        boundary_indices.add(int(current_idx))
                        continue
                    traversal_ok = False
                    break

                if str(current_op.op_type) not in allowed_ops:
                    traversal_ok = False
                    break

                candidate_indices.add(int(current_idx))
                for input_name in list(current_op.inputs):
                    parent_idx = producers.get(str(input_name), None)
                    if parent_idx is not None:
                        stack.append(int(parent_idx))

            if not traversal_ok:
                continue
            if len(candidate_indices) == 0:
                continue
            if len(boundary_indices) != 2:
                continue

            boundary_plans: List[Tuple[int, str, str]] = []
            for boundary_idx in sorted(list(boundary_indices)):
                boundary_op = model_ir.operators[int(boundary_idx)]
                if (
                    str(boundary_op.op_type) != "TRANSPOSE"
                    or len(boundary_op.inputs) < 2
                    or len(boundary_op.outputs) != 1
                    or _read_transpose_perm(model_ir, boundary_op) != perm_nhwc_to_nchw
                ):
                    traversal_ok = False
                    break
                boundary_in = str(boundary_op.inputs[0])
                boundary_out = str(boundary_op.outputs[0])
                if boundary_in in model_outputs or boundary_out in model_outputs:
                    traversal_ok = False
                    break
                boundary_users = set(int(v) for v in consumers.get(boundary_out, []))
                if len(boundary_users) == 0:
                    traversal_ok = False
                    break
                if any(int(user_idx) not in candidate_indices for user_idx in boundary_users):
                    traversal_ok = False
                    break
                boundary_plans.append((int(boundary_idx), str(boundary_in), str(boundary_out)))
            if not traversal_ok:
                continue

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
                )

            conv_inputs = [
                str(post_in_name) if str(v) == str(post_out_name) else str(v)
                for v in list(conv_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv_op,
                new_inputs=conv_inputs,
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
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_cost_volume_scatter_ndhwc_chains": int(rewritten)}

