from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Repair partial NHWC propagation around mixed spatial-attention reductions.

    Target broken sketch:
      x_nhwc --T(0,3,1,2)--> x_nchw
      mean_nhwc = MEAN(x_nhwc, axes=[3], keepDims=1)
      max_nchw  = REDUCE_MAX(x_nchw, axes=[1], keepDims=1)
      CONCAT(mean_nhwc, max_nchw, axis=1) -> MIRROR_PAD(nchw pads) -> T(0,2,3,1) -> CONV

    Rewrite:
      - move REDUCE_MAX to x_nhwc and remap axes to NHWC
      - rewrite CONCAT axis to NHWC channel axis
      - rewrite MIRROR_PAD pairs to NHWC ordering
      - remove the redundant MIRROR_PAD output transpose
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _normalize_axis(axis: int, rank: int) -> Optional[int]:
        value = int(axis)
        if value < 0:
            value += int(rank)
        if value < 0 or value >= int(rank):
            return None
        return int(value)

    def _rewrite_reduce_axes_nchw_to_nhwc(op: OperatorIR) -> bool:
        if len(op.inputs) < 2:
            return False
        axes_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        axes_vals = _read_const_ints_from_tensor(axes_tensor)
        if axes_vals is None or len(axes_vals) == 0:
            return False
        mapped_axes: List[int] = []
        for axis in axes_vals:
            normalized = _normalize_axis(int(axis), 4)
            if normalized is None:
                return False
            mapped_axes.append(int(perm_nhwc_to_nchw[int(normalized)]))
        return bool(_write_const_ints_to_tensor(axes_tensor, [int(v) for v in mapped_axes]))

    def _rewrite_pad_pairs_nchw_to_nhwc(op: OperatorIR) -> bool:
        if len(op.inputs) < 2:
            return False
        pads_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if pads_tensor is None or pads_tensor.data is None:
            return False
        try:
            pads_array = np.asarray(pads_tensor.data)
            pads_pairs = np.asarray(pads_array).reshape(4, 2)
        except Exception:
            return False
        if int(pads_pairs.size) != 8:
            return False
        pads_nhwc = np.asarray(
            [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
            dtype=pads_array.dtype,
        )
        pads_tensor.data = np.asarray(pads_nhwc)
        pads_tensor.shape = [4, 2]
        pads_tensor.shape_signature = [4, 2]
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = {str(v) for v in list(model_ir.outputs)}

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if (
                str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.inputs) != 2
                or len(concat_op.outputs) != 1
            ):
                continue
            concat_axis = int(concat_op.options.get("axis", 0))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue

            concat_input_names = [str(v) for v in list(concat_op.inputs)]
            concat_output_name = str(concat_op.outputs[0])
            if concat_output_name in model_outputs:
                continue

            input_producers = [producers.get(name, None) for name in concat_input_names]
            if any(idx is None for idx in input_producers):
                continue
            lhs_op = model_ir.operators[int(input_producers[0])]
            rhs_op = model_ir.operators[int(input_producers[1])]
            producer_pair = {str(lhs_op.op_type), str(rhs_op.op_type)}
            if producer_pair != {"MEAN", "REDUCE_MAX"}:
                continue

            mean_op = lhs_op if str(lhs_op.op_type) == "MEAN" else rhs_op
            max_op = rhs_op if str(rhs_op.op_type) == "REDUCE_MAX" else lhs_op
            mean_out_name = str(mean_op.outputs[0])
            max_out_name = str(max_op.outputs[0])
            mean_tensor = model_ir.tensors.get(mean_out_name, None)
            max_tensor = model_ir.tensors.get(max_out_name, None)
            mean_input_tensor = model_ir.tensors.get(str(mean_op.inputs[0]), None)
            max_input_tensor = model_ir.tensors.get(str(max_op.inputs[0]), None)
            if (
                mean_tensor is None
                or max_tensor is None
                or mean_input_tensor is None
                or max_input_tensor is None
                or len(list(mean_tensor.shape)) != 4
                or len(list(max_tensor.shape)) != 4
            ):
                continue

            mean_input_name = str(mean_op.inputs[0])
            max_input_name = str(max_op.inputs[0])
            source_nhwc_name: Optional[str] = None
            pre_idx = producers.get(max_input_name, None)
            if pre_idx is not None:
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) == "TRANSPOSE"
                    and len(pre_op.inputs) >= 2
                    and len(pre_op.outputs) == 1
                    and str(pre_op.outputs[0]) == max_input_name
                    and str(pre_op.inputs[0]) == mean_input_name
                    and _read_transpose_perm(model_ir, pre_op) == perm_nhwc_to_nchw
                ):
                    source_nhwc_name = str(mean_input_name)
            if source_nhwc_name is None:
                pre_idx = producers.get(mean_input_name, None)
                if pre_idx is None:
                    continue
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != mean_input_name
                    or str(pre_op.inputs[0]) != max_input_name
                    or _read_transpose_perm(model_ir, pre_op) != perm_nchw_to_nhwc
                ):
                    continue
                source_nhwc_name = str(mean_input_name)
            if not bool(mean_op.options.get("keepDims", False)) or not bool(max_op.options.get("keepDims", False)):
                continue

            mirror_users = [int(v) for v in consumers.get(concat_output_name, [])]
            if len(mirror_users) != 1:
                continue
            mirror_idx = int(mirror_users[0])
            mirror_op = model_ir.operators[int(mirror_idx)]
            if (
                str(mirror_op.op_type) != "MIRROR_PAD"
                or len(mirror_op.inputs) < 2
                or len(mirror_op.outputs) != 1
                or str(mirror_op.inputs[0]) != concat_output_name
            ):
                continue

            mirror_output_name = str(mirror_op.outputs[0])
            if mirror_output_name in model_outputs:
                continue
            transpose_users = [int(v) for v in consumers.get(mirror_output_name, [])]
            if len(transpose_users) != 1:
                continue
            post_idx = int(transpose_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != mirror_output_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            conv_input_name = str(post_op.outputs[0])
            conv_users = [int(v) for v in consumers.get(conv_input_name, [])]
            if len(conv_users) != 1:
                continue
            conv_idx = int(conv_users[0])
            conv_op = model_ir.operators[int(conv_idx)]
            if (
                str(conv_op.op_type) != "CONV_2D"
                or len(conv_op.inputs) < 1
                or str(conv_op.inputs[0]) != conv_input_name
            ):
                continue

            if not _rewrite_reduce_axes_nchw_to_nhwc(max_op):
                continue
            if not _rewrite_pad_pairs_nchw_to_nhwc(mirror_op):
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=max_op,
                input_index=0,
                new_input_name=str(source_nhwc_name),
            )
            concat_op.options["axis"] = 3
            _replace_operator_input_at(
                model_ir=model_ir,
                op=conv_op,
                input_index=0,
                new_input_name=str(mirror_output_name),
            )

            for tensor_name in [str(max_out_name), str(concat_output_name), str(mirror_output_name)]:
                tensor = model_ir.tensors.get(str(tensor_name), None)
                _permute_tensor_metadata_if_rank_matches(tensor, perm_nchw_to_nhwc)
                if tensor is not None and len(list(tensor.shape)) == 4:
                    tensor.logical_layout = "NHWC"
            max_input_tensor.logical_layout = "NCHW"
            mean_input_tensor.logical_layout = "NHWC"

            del model_ir.operators[int(post_idx)]
            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains": int(optimized)}


def _optimize_attention_qkv_slice_replace_gather_reshape_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Replace attention QKV/KV branch `GATHER -> RESHAPE` with direct `SLICE`.

    Target (post hoist):
      qkv_htc[N,H,T,C] --GATHER(i, axis=0)--> g_i[H,T,C]
                       --RESHAPE([1,H,T,C])--> z_i      (N in {2,3})

    Rewrite:
      qkv_htc --SLICE(begin=[i,0,0,0], size=[1,H,T,C])--> z_i
    """
    rewritten = 0
    candidate_branch_counts = [3, 2]

    def _shape_list(name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.shape is None:
            return None
        return [int(v) for v in list(tensor.shape)]

    def _dims_compatible(a: int, b: int) -> bool:
        if int(a) < 0 or int(b) < 0:
            return True
        return int(a) == int(b)

    def _shape_compatible(a: List[int], b: List[int]) -> bool:
        if len(a) != len(b):
            return False
        return all(_dims_compatible(int(x), int(y)) for x, y in zip(a, b))

    def _gather_axis(op: OperatorIR) -> int:
        opts = dict(op.options) if isinstance(op.options, dict) else {}
        axis_value = opts.get("axis", 0)
        if isinstance(axis_value, list):
            if len(axis_value) != 1:
                return 0
            axis_value = axis_value[0]
        try:
            return int(axis_value)
        except Exception:
            return 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for source_name, user_indices in list(consumers.items()):
            source_shape = _shape_list(str(source_name))
            if source_shape is None or len(source_shape) != 4:
                continue
            n0, h, t, c = [int(v) for v in list(source_shape)]
            if not _is_fully_known_positive_shape([h, t, c]):
                continue

            gather_indices: set[int] = set()
            branch_by_index: Dict[int, Dict[str, Any]] = {}
            for candidate_branch_count in list(candidate_branch_counts):
                if not _dims_compatible(n0, int(candidate_branch_count)):
                    continue
                expected_indices = set(range(int(candidate_branch_count)))
                candidate_branch_by_index: Dict[int, Dict[str, Any]] = {}

                for user_idx in [int(v) for v in user_indices]:
                    gather_op = model_ir.operators[int(user_idx)]
                    if (
                        str(gather_op.op_type) != "GATHER"
                        or len(gather_op.inputs) < 2
                        or len(gather_op.outputs) != 1
                        or str(gather_op.inputs[0]) != str(source_name)
                        or _gather_axis(gather_op) != 0
                    ):
                        continue
                    index_vals = _read_const_ints_from_tensor(
                        model_ir.tensors.get(str(gather_op.inputs[1]), None)
                    )
                    if index_vals is None or len(index_vals) != 1:
                        continue
                    gather_index = int(index_vals[0])
                    if gather_index not in expected_indices or gather_index in candidate_branch_by_index:
                        continue

                    gather_out_name = str(gather_op.outputs[0])
                    if gather_out_name in model_outputs:
                        continue
                    gather_out_shape = _shape_list(gather_out_name)
                    if gather_out_shape is None or len(gather_out_shape) != 3:
                        continue
                    if not _shape_compatible(gather_out_shape, [h, t, c]):
                        continue

                    gather_out_users = [int(v) for v in consumers.get(gather_out_name, [])]
                    if len(gather_out_users) != 1:
                        continue
                    reshape_idx = int(gather_out_users[0])
                    reshape_op = model_ir.operators[int(reshape_idx)]
                    if (
                        str(reshape_op.op_type) != "RESHAPE"
                        or len(reshape_op.inputs) < 1
                        or len(reshape_op.outputs) != 1
                        or str(reshape_op.inputs[0]) != gather_out_name
                    ):
                        continue
                    reshape_out_name = str(reshape_op.outputs[0])
                    reshape_out_shape = _shape_list(reshape_out_name)
                    if reshape_out_shape is None or len(reshape_out_shape) != 4:
                        continue
                    if not _shape_compatible(reshape_out_shape, [1, h, t, c]):
                        continue

                    candidate_branch_by_index[gather_index] = {
                        "gather_idx": int(user_idx),
                        "gather_op": gather_op,
                        "reshape_idx": int(reshape_idx),
                        "reshape_op": reshape_op,
                        "reshape_out_name": str(reshape_out_name),
                    }

                if set(candidate_branch_by_index.keys()) == expected_indices:
                    gather_indices = set(expected_indices)
                    branch_by_index = candidate_branch_by_index
                    break

            if len(gather_indices) <= 0:
                continue

            remove_indices: set[int] = set()
            for gather_index in sorted(list(gather_indices)):
                branch = branch_by_index[gather_index]
                reshape_op = branch["reshape_op"]
                reshape_out_name = str(branch["reshape_out_name"])

                begin_name = _unique_tensor_name(f"{reshape_out_name}_slice_begin")
                size_name = _unique_tensor_name(f"{reshape_out_name}_slice_size")
                model_ir.tensors[begin_name] = TensorIR(
                    name=begin_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray([int(gather_index), 0, 0, 0], dtype=np.int32),
                    is_variable=False,
                )
                model_ir.tensors[size_name] = TensorIR(
                    name=size_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray([1, int(h), int(t), int(c)], dtype=np.int32),
                    is_variable=False,
                )

                reshape_op.op_type = "SLICE"
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=reshape_op,
                    new_inputs=[str(source_name), str(begin_name), str(size_name)],
                )
                reshape_op.options = {}
                out_tensor = model_ir.tensors.get(reshape_out_name, None)
                if out_tensor is not None:
                    out_tensor.shape = [1, int(h), int(t), int(c)]
                    out_tensor.shape_signature = [1, int(h), int(t), int(c)]

                gather_remove_idx = next(
                    (idx for idx, op in enumerate(model_ir.operators) if op is branch["gather_op"]),
                    None,
                )
                if gather_remove_idx is not None:
                    remove_indices.add(int(gather_remove_idx))

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_attention_qkv_slice_replace_gather_reshape_chains": int(rewritten)}


def _optimize_attention_qkv_slice_to_split_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Replace attention QKV/KV sibling `SLICE` groups with a single `SPLIT`.

    Target (generalized):
      src[...]
        --SLICE(begin,size)--> out_i  (i in 0..N-1, N in {2,3})
      where all branches slice along one axis with equal chunk size and
      non-overlapping contiguous starts.

    Rewrite:
      SPLIT(numSplits=N, axis=split_axis): src -> [out_0, ..., out_{N-1}]
    """
    rewritten = 0
    candidate_branch_counts = [3, 2]

    def _shape_list(name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.shape is None:
            return None
        return [int(v) for v in list(tensor.shape)]

    def _dims_compatible(a: int, b: int) -> bool:
        if int(a) < 0 or int(b) < 0:
            return True
        return int(a) == int(b)

    def _shape_compatible(a: List[int], b: List[int]) -> bool:
        if len(a) != len(b):
            return False
        return all(_dims_compatible(int(x), int(y)) for x, y in zip(a, b))

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for source_name, user_indices in list(consumers.items()):
            source_shape = _shape_list(str(source_name))
            if source_shape is None or len(source_shape) <= 0:
                continue
            if not _is_fully_known_positive_shape([int(v) for v in list(source_shape)]):
                continue
            rank = int(len(source_shape))

            branch_count: Optional[int] = None
            split_axis: Optional[int] = None
            branch_by_index: Dict[int, Dict[str, Any]] = {}
            for candidate_branch_count in list(candidate_branch_counts):
                expected_indices = set(range(int(candidate_branch_count)))
                for axis in range(rank):
                    axis_dim = int(source_shape[axis])
                    if int(axis_dim) <= 0 or int(axis_dim) % int(candidate_branch_count) != 0:
                        continue
                    chunk = int(axis_dim // int(candidate_branch_count))
                    if int(chunk) <= 0:
                        continue

                    candidate_branch_by_index: Dict[int, Dict[str, Any]] = {}
                    for user_idx in [int(v) for v in user_indices]:
                        slice_op = model_ir.operators[int(user_idx)]
                        if (
                            str(slice_op.op_type) != "SLICE"
                            or len(slice_op.inputs) < 3
                            or len(slice_op.outputs) != 1
                            or str(slice_op.inputs[0]) != str(source_name)
                        ):
                            continue

                        begin_vals = _read_const_ints_from_tensor(
                            model_ir.tensors.get(str(slice_op.inputs[1]), None)
                        )
                        size_vals = _read_const_ints_from_tensor(
                            model_ir.tensors.get(str(slice_op.inputs[2]), None)
                        )
                        if (
                            begin_vals is None
                            or size_vals is None
                            or len(begin_vals) != rank
                            or len(size_vals) != rank
                        ):
                            continue
                        begin = [int(v) for v in list(begin_vals)]
                        size = [int(v) for v in list(size_vals)]

                        valid = True
                        for dim_idx in range(rank):
                            dim = int(source_shape[dim_idx])
                            b = int(begin[dim_idx])
                            s = int(size[dim_idx])
                            if dim_idx == axis:
                                if int(b) < 0 or int(b) % int(chunk) != 0:
                                    valid = False
                                    break
                                if int(s) != int(chunk):
                                    valid = False
                                    break
                            else:
                                if int(b) != 0 or int(s) != int(dim):
                                    valid = False
                                    break
                        if not valid:
                            continue

                        slice_index = int(begin[axis] // chunk)
                        if slice_index not in expected_indices or slice_index in candidate_branch_by_index:
                            continue

                        out_name = str(slice_op.outputs[0])
                        out_shape = _shape_list(out_name)
                        expected_out_shape = [int(v) for v in list(source_shape)]
                        expected_out_shape[axis] = int(chunk)
                        if out_shape is None or len(out_shape) != rank:
                            continue
                        if not _shape_compatible(out_shape, expected_out_shape):
                            continue

                        candidate_branch_by_index[slice_index] = {
                            "slice_idx": int(user_idx),
                            "slice_op": slice_op,
                            "out_name": out_name,
                        }

                    if set(candidate_branch_by_index.keys()) == expected_indices:
                        branch_count = int(candidate_branch_count)
                        split_axis = int(axis)
                        branch_by_index = candidate_branch_by_index
                        break
                if branch_count is not None:
                    break

            if branch_count is None:
                continue
            if split_axis is None:
                continue

            axis_name = _unique_tensor_name(f"{source_name}_qkv_split_axis")
            model_ir.tensors[axis_name] = TensorIR(
                name=axis_name,
                dtype="INT32",
                shape=[1],
                shape_signature=[1],
                data=np.asarray([int(split_axis)], dtype=np.int32),
                is_variable=False,
            )

            split_outputs = [str(branch_by_index[idx]["out_name"]) for idx in sorted(list(branch_by_index.keys()))]
            split_op = OperatorIR(
                op_type="SPLIT",
                inputs=[str(axis_name), str(source_name)],
                outputs=split_outputs,
                options={"numSplits": int(branch_count)},
            )
            insert_base_idx = min(int(branch_by_index[idx]["slice_idx"]) for idx in sorted(list(branch_by_index.keys())))
            model_ir.operators.insert(int(insert_base_idx), split_op)

            remove_indices: set[int] = set()
            chunk = int(source_shape[int(split_axis)] // int(branch_count))
            for idx in sorted(list(branch_by_index.keys())):
                slice_op = branch_by_index[idx]["slice_op"]
                slice_remove_idx = next(
                    (op_idx for op_idx, op in enumerate(model_ir.operators) if op is slice_op),
                    None,
                )
                if slice_remove_idx is not None:
                    remove_indices.add(int(slice_remove_idx))

                out_name = str(branch_by_index[idx]["out_name"])
                out_tensor = model_ir.tensors.get(out_name, None)
                if out_tensor is not None:
                    out_shape = [int(v) for v in list(source_shape)]
                    out_shape[int(split_axis)] = int(chunk)
                    out_tensor.shape = [int(v) for v in list(out_shape)]
                    out_tensor.shape_signature = [int(v) for v in list(out_shape)]

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_attention_qkv_slice_to_split_chains": int(rewritten)}
