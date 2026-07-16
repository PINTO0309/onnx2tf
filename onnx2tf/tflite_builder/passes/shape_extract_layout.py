from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _clone_quantization,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove NHWC->NCHW transposes used only for SHAPE extraction.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw --SHAPE--> s_nchw
      s_nchw --GATHER/SLICE(...)--> ...

    Rewrite:
      x_nhwc --SHAPE--> s_nchw'
      - Remap GATHER indices by transpose permutation.
      - For SLICE on rank-1 shape vectors, remap selected indices.
        If remapped indices are non-contiguous, convert SLICE to GATHER.
      - Remove the now-redundant TRANSPOSE.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _unique_tensor_name(base: str) -> str:
        if base not in model_ir.tensors:
            return base
        suffix = 1
        while f"{base}_{suffix}" in model_ir.tensors:
            suffix += 1
        return f"{base}_{suffix}"

    def _materialize_const_tensor(
        *,
        base_name: str,
        template_tensor: Optional[TensorIR],
        values: List[int],
    ) -> str:
        new_name = _unique_tensor_name(base_name)
        np_dtype = np.int32
        dtype_name = "INT32"
        quant = None
        if template_tensor is not None:
            dtype_name = str(template_tensor.dtype)
            quant = _clone_quantization(template_tensor.quantization)
            if template_tensor.data is not None:
                try:
                    np_dtype = np.asarray(template_tensor.data).dtype
                except Exception:
                    np_dtype = np.int32
        const_values = [int(v) for v in list(values)]
        model_ir.tensors[new_name] = TensorIR(
            name=new_name,
            dtype=dtype_name,
            shape=[int(len(const_values))],
            shape_signature=[int(len(const_values))],
            data=np.asarray(const_values, dtype=np_dtype),
            is_variable=False,
            quantization=quant,
        )
        return new_name

    def _assign_const_input(
        *,
        op_idx: int,
        input_index: int,
        values: List[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        op = model_ir.operators[int(op_idx)]
        if int(input_index) < 0 or int(input_index) >= len(op.inputs):
            return False
        input_name = str(op.inputs[int(input_index)])
        input_tensor = model_ir.tensors.get(input_name, None)
        if input_tensor is None:
            return False
        user_set = set(int(v) for v in consumers.get(input_name, []))
        if user_set == {int(op_idx)}:
            return _write_const_ints_to_tensor(input_tensor, [int(v) for v in list(values)])
        cloned_name = _materialize_const_tensor(
            base_name=f"{input_name}_shape_remap",
            template_tensor=input_tensor,
            values=[int(v) for v in list(values)],
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=cloned_name,
        )
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for transpose_idx, transpose_op in enumerate(model_ir.operators):
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
            ):
                continue
            perm = _read_transpose_perm(model_ir, transpose_op)
            if perm != perm_nhwc_to_nchw:
                continue

            transpose_input_name = str(transpose_op.inputs[0])
            transpose_output_name = str(transpose_op.outputs[0])
            if transpose_output_name in model_outputs:
                continue

            transpose_users = [int(v) for v in consumers.get(transpose_output_name, [])]
            if len(transpose_users) != 1:
                continue
            shape_idx = int(transpose_users[0])
            shape_op = model_ir.operators[int(shape_idx)]
            if (
                str(shape_op.op_type) != "SHAPE"
                or len(shape_op.inputs) != 1
                or len(shape_op.outputs) != 1
                or str(shape_op.inputs[0]) != transpose_output_name
            ):
                continue

            shape_output_name = str(shape_op.outputs[0])
            if shape_output_name in model_outputs:
                continue
            shape_users = [int(v) for v in consumers.get(shape_output_name, [])]
            if len(shape_users) == 0:
                continue

            gather_rewrites: List[Tuple[int, List[int]]] = []
            slice_contiguous_rewrites: List[Tuple[int, List[int], List[int]]] = []
            slice_to_gather_rewrites: List[Tuple[int, List[int]]] = []
            rewrite_supported = True
            rank = int(len(perm))

            for user_idx in shape_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "GATHER":
                    if (
                        len(user_op.inputs) < 2
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != shape_output_name
                    ):
                        rewrite_supported = False
                        break
                    axis = int(user_op.options.get("axis", 0))
                    if axis < 0:
                        axis += 1
                    if axis != 0:
                        rewrite_supported = False
                        break
                    indices_tensor = model_ir.tensors.get(str(user_op.inputs[1]), None)
                    indices_vals = _read_const_ints_from_tensor(indices_tensor)
                    if indices_vals is None or len(indices_vals) == 0:
                        rewrite_supported = False
                        break
                    remapped_indices: List[int] = []
                    for index_value in indices_vals:
                        index = int(index_value)
                        if index < 0:
                            index += int(rank)
                        if index < 0 or index >= int(rank):
                            rewrite_supported = False
                            break
                        remapped_indices.append(int(perm[int(index)]))
                    if not rewrite_supported:
                        break
                    gather_rewrites.append((int(user_idx), remapped_indices))
                    continue

                if user_type == "SLICE":
                    if (
                        len(user_op.inputs) < 3
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != shape_output_name
                    ):
                        rewrite_supported = False
                        break
                    begin_tensor = model_ir.tensors.get(str(user_op.inputs[1]), None)
                    size_tensor = model_ir.tensors.get(str(user_op.inputs[2]), None)
                    begin_vals = _read_const_ints_from_tensor(begin_tensor)
                    size_vals = _read_const_ints_from_tensor(size_tensor)
                    if begin_vals is None or size_vals is None:
                        rewrite_supported = False
                        break
                    if len(begin_vals) != 1 or len(size_vals) != 1:
                        rewrite_supported = False
                        break
                    begin = int(begin_vals[0])
                    size = int(size_vals[0])
                    if begin < 0:
                        begin += int(rank)
                    begin = max(0, min(int(begin), int(rank)))
                    if size == -1:
                        end = int(rank)
                    elif size < -1:
                        rewrite_supported = False
                        break
                    else:
                        end = min(int(rank), int(begin + size))
                    if int(end) <= int(begin):
                        rewrite_supported = False
                        break
                    selected = [int(v) for v in range(int(begin), int(end))]
                    remapped_selected = [int(perm[int(v)]) for v in selected]
                    is_contiguous = all(
                        int(remapped_selected[int(i)]) == int(remapped_selected[0] + i)
                        for i in range(len(remapped_selected))
                    )
                    if is_contiguous:
                        new_begin = [int(remapped_selected[0])]
                        new_size = [int(len(remapped_selected))]
                        slice_contiguous_rewrites.append((int(user_idx), new_begin, new_size))
                    else:
                        slice_to_gather_rewrites.append((int(user_idx), remapped_selected))
                    continue

                rewrite_supported = False
                break

            if not rewrite_supported:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=shape_op,
                input_index=0,
                new_input_name=transpose_input_name,
            )

            for gather_idx, remapped_indices in gather_rewrites:
                _assign_const_input(
                    op_idx=int(gather_idx),
                    input_index=1,
                    values=[int(v) for v in remapped_indices],
                    consumers=consumers,
                )

            for slice_idx, new_begin, new_size in slice_contiguous_rewrites:
                _assign_const_input(
                    op_idx=int(slice_idx),
                    input_index=1,
                    values=[int(v) for v in list(new_begin)],
                    consumers=consumers,
                )
                _assign_const_input(
                    op_idx=int(slice_idx),
                    input_index=2,
                    values=[int(v) for v in list(new_size)],
                    consumers=consumers,
                )

            for slice_idx, remapped_indices in slice_to_gather_rewrites:
                slice_op = model_ir.operators[int(slice_idx)]
                begin_input_name = str(slice_op.inputs[1])
                begin_tensor = model_ir.tensors.get(begin_input_name, None)
                gather_indices_name = _materialize_const_tensor(
                    base_name=f"{begin_input_name}_gather",
                    template_tensor=begin_tensor,
                    values=[int(v) for v in remapped_indices],
                )
                slice_op.op_type = "GATHER"
                slice_op.version = 1
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=slice_op,
                    new_inputs=[shape_output_name, gather_indices_name],
                )
                slice_op.options = {"axis": 0, "batchDims": 0}
                slice_out_name = str(slice_op.outputs[0])
                slice_out_tensor = model_ir.tensors.get(slice_out_name, None)
                if slice_out_tensor is not None:
                    slice_out_tensor.shape = [int(len(remapped_indices))]
                    slice_out_tensor.shape_signature = [int(len(remapped_indices))]

            del model_ir.operators[int(transpose_idx)]
            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_shape_extract_nhwc_to_nchw_chains": int(optimized)}
