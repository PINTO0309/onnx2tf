from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _permute_shape,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
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


def _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Hoist sibling NHWC->NCHW transposes on Q/K/V branches to a single
    pre-slice transpose on the shared source tensor.

    Target (rank-4):
      src_nhwc --SLICE(c-range for q)--> q_nhwc --RESHAPE--> q_nchw_like --SOFTMAX
      src_nhwc --SLICE(c-range for k)--> k_nhwc --T(0,3,1,2)--> k_nchw
      src_nhwc --SLICE(c-range for v)--> v_nhwc --RELU--> v_relu_nhwc --T(0,3,1,2)--> v_nchw

    Rewrite:
      src_nhwc --T(0,3,1,2)--> src_nchw
      SLICE begin/size are remapped to NCHW indexing for q/k/v slices.
      Drop per-branch transposes on k and v-relu branches.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        serial = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{serial}"
            serial += 1
        return candidate

    def _permute_begin_or_size_nhwc_to_nchw(values: List[int]) -> List[int]:
        # [N,H,W,C] -> [N,C,H,W]
        return [int(values[0]), int(values[3]), int(values[1]), int(values[2])]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for k_transpose_idx, k_transpose_op in enumerate(model_ir.operators):
            if (
                str(k_transpose_op.op_type) != "TRANSPOSE"
                or len(k_transpose_op.inputs) < 2
                or len(k_transpose_op.outputs) != 1
                or _read_transpose_perm(model_ir, k_transpose_op) != perm_nhwc_to_nchw
            ):
                continue

            k_slice_out_name = str(k_transpose_op.inputs[0])
            k_transpose_out_name = str(k_transpose_op.outputs[0])
            shared_perm_name = str(k_transpose_op.inputs[1])
            if k_slice_out_name in model_outputs or k_transpose_out_name in model_outputs:
                continue

            k_slice_idx = producers.get(k_slice_out_name, None)
            if k_slice_idx is None:
                continue
            k_slice_op = model_ir.operators[int(k_slice_idx)]
            if (
                str(k_slice_op.op_type) != "SLICE"
                or len(k_slice_op.inputs) < 3
                or len(k_slice_op.outputs) != 1
                or str(k_slice_op.outputs[0]) != k_slice_out_name
            ):
                continue

            src_name = str(k_slice_op.inputs[0])
            if src_name in model_outputs:
                continue
            src_users = [int(v) for v in consumers.get(src_name, [])]
            if len(src_users) < 3:
                continue

            slice_user_indices = []
            for user_idx in src_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "SLICE"
                    and len(user_op.inputs) >= 3
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == src_name
                ):
                    slice_user_indices.append(int(user_idx))
            if len(slice_user_indices) != 3 or set(int(v) for v in src_users) != set(slice_user_indices):
                continue

            slice_by_out_name: Dict[str, Dict[str, Any]] = {}
            valid_slices = True
            for slice_idx in slice_user_indices:
                slice_op = model_ir.operators[int(slice_idx)]
                begin_name = str(slice_op.inputs[1])
                size_name = str(slice_op.inputs[2])
                begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(begin_name, None))
                size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(size_name, None))
                if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
                    valid_slices = False
                    break
                out_name = str(slice_op.outputs[0])
                slice_by_out_name[out_name] = {
                    "idx": int(slice_idx),
                    "begin_name": begin_name,
                    "size_name": size_name,
                    "begin_vals": [int(v) for v in list(begin_vals)],
                    "size_vals": [int(v) for v in list(size_vals)],
                }
            if not valid_slices:
                continue

            if k_slice_out_name not in slice_by_out_name:
                continue
            if set(int(v) for v in consumers.get(k_slice_out_name, [])) != {int(k_transpose_idx)}:
                continue

            v_relu_idx: Optional[int] = None
            v_transpose_idx: Optional[int] = None
            v_transpose_out_name = ""
            v_relu_out_name = ""
            v_slice_out_name = ""
            q_slice_out_name = ""
            for out_name, meta in list(slice_by_out_name.items()):
                if out_name == k_slice_out_name:
                    continue
                user_indices = [int(v) for v in consumers.get(str(out_name), [])]
                if len(user_indices) == 1:
                    only_user_idx = int(user_indices[0])
                    only_user_op = model_ir.operators[int(only_user_idx)]
                    if (
                        str(only_user_op.op_type) == "RESHAPE"
                        and len(only_user_op.outputs) == 1
                    ):
                        reshape_out_name = str(only_user_op.outputs[0])
                        softmax_users = [int(v) for v in consumers.get(reshape_out_name, [])]
                        if len(softmax_users) == 1:
                            softmax_op = model_ir.operators[int(softmax_users[0])]
                            if str(softmax_op.op_type) == "SOFTMAX":
                                q_slice_out_name = str(out_name)
                                continue
                if len(user_indices) == 1:
                    relu_candidate_idx = int(user_indices[0])
                    relu_candidate_op = model_ir.operators[int(relu_candidate_idx)]
                    if (
                        str(relu_candidate_op.op_type) == "RELU"
                        and len(relu_candidate_op.outputs) == 1
                    ):
                        relu_out_name = str(relu_candidate_op.outputs[0])
                        relu_users = [int(v) for v in consumers.get(relu_out_name, [])]
                        if len(relu_users) == 1:
                            t_idx = int(relu_users[0])
                            t_op = model_ir.operators[int(t_idx)]
                            if (
                                str(t_op.op_type) == "TRANSPOSE"
                                and len(t_op.inputs) >= 2
                                and len(t_op.outputs) == 1
                                and str(t_op.inputs[0]) == relu_out_name
                                and _read_transpose_perm(model_ir, t_op) == perm_nhwc_to_nchw
                                and str(t_op.outputs[0]) not in model_outputs
                            ):
                                v_slice_out_name = str(out_name)
                                v_relu_idx = int(relu_candidate_idx)
                                v_relu_out_name = relu_out_name
                                v_transpose_idx = int(t_idx)
                                v_transpose_out_name = str(t_op.outputs[0])
                                continue

            if (
                q_slice_out_name == ""
                or v_slice_out_name == ""
                or v_relu_idx is None
                or v_transpose_idx is None
                or v_relu_out_name == ""
                or v_transpose_out_name == ""
            ):
                continue

            src_tensor = model_ir.tensors.get(src_name, None)
            if src_tensor is None or src_tensor.shape is None or len(list(src_tensor.shape)) != 4:
                continue

            src_shape_nhwc = [int(v) for v in list(src_tensor.shape)]
            src_shape_nchw = _permute_shape(src_shape_nhwc, perm_nhwc_to_nchw)
            if src_shape_nchw is None:
                continue
            src_sig_nhwc = (
                [int(v) for v in list(src_tensor.shape_signature)]
                if src_tensor.shape_signature is not None
                else [int(v) for v in list(src_shape_nhwc)]
            )
            src_sig_nchw = _permute_shape(src_sig_nhwc, perm_nhwc_to_nchw)
            if src_sig_nchw is None:
                src_sig_nchw = [int(v) for v in list(src_shape_nchw)]

            shared_nchw_name = _unique_tensor_name(f"{src_name}_qkv_nchw")
            model_ir.tensors[shared_nchw_name] = TensorIR(
                name=shared_nchw_name,
                dtype=str(src_tensor.dtype),
                shape=[int(v) for v in list(src_shape_nchw)],
                shape_signature=[int(v) for v in list(src_sig_nchw)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(src_tensor.quantization),
            )

            shared_transpose_op = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(src_name), str(shared_perm_name)],
                outputs=[str(shared_nchw_name)],
            )

            for _, branch_meta in list(slice_by_out_name.items()):
                branch_slice_op = model_ir.operators[int(branch_meta["idx"])]
                begin_name = str(branch_meta["begin_name"])
                size_name = str(branch_meta["size_name"])
                begin_vals = [int(v) for v in list(branch_meta["begin_vals"])]
                size_vals = [int(v) for v in list(branch_meta["size_vals"])]
                mapped_begin = _permute_begin_or_size_nhwc_to_nchw(begin_vals)
                mapped_size = _permute_begin_or_size_nhwc_to_nchw(size_vals)
                if not (
                    _write_const_ints_to_tensor(model_ir.tensors.get(begin_name, None), mapped_begin)
                    or _read_const_ints_from_tensor(model_ir.tensors.get(begin_name, None)) == mapped_begin
                ):
                    valid_slices = False
                    break
                if not (
                    _write_const_ints_to_tensor(model_ir.tensors.get(size_name, None), mapped_size)
                    or _read_const_ints_from_tensor(model_ir.tensors.get(size_name, None)) == mapped_size
                ):
                    valid_slices = False
                    break
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=branch_slice_op,
                    new_inputs=[str(shared_nchw_name), begin_name, size_name],
                )
                out_name = str(branch_slice_op.outputs[0])
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_nhwc_to_nchw,
                )
            if not valid_slices:
                continue

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(v_relu_out_name, None),
                perm_nhwc_to_nchw,
            )

            _replace_tensor_inputs(model_ir, k_transpose_out_name, k_slice_out_name)
            _replace_tensor_inputs(model_ir, v_transpose_out_name, v_relu_out_name)

            remove_indices = sorted([int(k_transpose_idx), int(v_transpose_idx)], reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            insert_index = min(int(v) for v in list(slice_user_indices))
            removed_before_insert = sum(1 for v in remove_indices if int(v) < int(insert_index))
            adjusted_insert_index = int(insert_index) - int(removed_before_insert)
            model_ir.operators.insert(int(adjusted_insert_index), shared_transpose_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": int(rewritten)}


def _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Convert NCHW-side weighted-sum attention bridge to NHWC and remove
    redundant branch transposes.

    Target:
      k_nhwc --T(0,3,1,2)--> k_nchw
      soft_nchw --(from SOFTMAX)-->
      MUL(k_nchw, soft_nchw) -> m_nchw
      SUM(m_nchw, axis=[3], keepDims=True) -> s_nchw
      MUL(s_nchw, ones_nchw_const) -> e_nchw
      e_nchw --T(0,2,3,1)--> e_nhwc
      MUL(relu_nhwc, e_nhwc) -> out_nhwc

    Rewrite:
      soft_nchw --T(0,2,3,1)--> soft_nhwc
      MUL(k_nhwc, soft_nhwc) -> m_nhwc
      SUM(axis=[2], keepDims=True) -> s_nhwc
      MUL(s_nhwc, ones_nhwc_const) -> e_nhwc
      MUL(relu_nhwc, e_nhwc) -> out_nhwc
      Remove k/e bridge transposes.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        serial = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{serial}"
            serial += 1
        return candidate

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_t_idx, post_t_op in enumerate(model_ir.operators):
            if (
                str(post_t_op.op_type) != "TRANSPOSE"
                or len(post_t_op.inputs) < 2
                or len(post_t_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_t_op) != perm_nchw_to_nhwc
            ):
                continue

            expand_nchw_name = str(post_t_op.inputs[0])
            expand_nhwc_name = str(post_t_op.outputs[0])
            perm_post_name = str(post_t_op.inputs[1])
            if expand_nchw_name in model_outputs or expand_nhwc_name in model_outputs:
                continue

            relu_mul_users = [int(v) for v in consumers.get(expand_nhwc_name, [])]
            if len(relu_mul_users) != 1:
                continue
            relu_mul_idx = int(relu_mul_users[0])
            relu_mul_op = model_ir.operators[int(relu_mul_idx)]
            if str(relu_mul_op.op_type) != "MUL" or len(relu_mul_op.inputs) != 2 or len(relu_mul_op.outputs) != 1:
                continue
            if str(relu_mul_op.outputs[0]) in model_outputs:
                continue

            relu_mul_inputs = [str(v) for v in list(relu_mul_op.inputs)]
            if relu_mul_inputs[0] == expand_nhwc_name:
                relu_input_name = str(relu_mul_inputs[1])
                relu_mul_expand_input_index = 0
            elif relu_mul_inputs[1] == expand_nhwc_name:
                relu_input_name = str(relu_mul_inputs[0])
                relu_mul_expand_input_index = 1
            else:
                continue
            relu_prod_idx = producers.get(relu_input_name, None)
            if relu_prod_idx is None or str(model_ir.operators[int(relu_prod_idx)].op_type) != "RELU":
                continue

            expand_mul_idx = producers.get(expand_nchw_name, None)
            if expand_mul_idx is None:
                continue
            expand_mul_op = model_ir.operators[int(expand_mul_idx)]
            if (
                str(expand_mul_op.op_type) != "MUL"
                or len(expand_mul_op.inputs) != 2
                or len(expand_mul_op.outputs) != 1
                or str(expand_mul_op.outputs[0]) != expand_nchw_name
            ):
                continue

            expand_mul_inputs = [str(v) for v in list(expand_mul_op.inputs)]
            sum_out_name = ""
            expand_const_name = ""
            for input_name, other_name in [
                (str(expand_mul_inputs[0]), str(expand_mul_inputs[1])),
                (str(expand_mul_inputs[1]), str(expand_mul_inputs[0])),
            ]:
                other_tensor = model_ir.tensors.get(other_name, None)
                if other_tensor is not None and other_tensor.data is not None:
                    sum_out_name = str(input_name)
                    expand_const_name = str(other_name)
                    break
            if sum_out_name == "" or expand_const_name == "":
                continue
            if set(int(v) for v in consumers.get(sum_out_name, [])) != {int(expand_mul_idx)}:
                continue

            sum_idx = producers.get(sum_out_name, None)
            if sum_idx is None:
                continue
            sum_op = model_ir.operators[int(sum_idx)]
            if (
                str(sum_op.op_type) != "SUM"
                or len(sum_op.inputs) < 2
                or len(sum_op.outputs) != 1
                or str(sum_op.outputs[0]) != sum_out_name
                or not bool(sum_op.options.get("keepDims", False))
            ):
                continue
            sum_axes_name = str(sum_op.inputs[1])
            sum_axes_vals = _read_const_ints_from_tensor(model_ir.tensors.get(sum_axes_name, None))
            if sum_axes_vals is None or [int(v) for v in list(sum_axes_vals)] != [3]:
                continue

            kmul_out_name = str(sum_op.inputs[0])
            kmul_idx = producers.get(kmul_out_name, None)
            if kmul_idx is None:
                continue
            kmul_op = model_ir.operators[int(kmul_idx)]
            if (
                str(kmul_op.op_type) != "MUL"
                or len(kmul_op.inputs) != 2
                or len(kmul_op.outputs) != 1
                or str(kmul_op.outputs[0]) != kmul_out_name
            ):
                continue

            kmul_inputs = [str(v) for v in list(kmul_op.inputs)]
            k_nchw_name = ""
            soft_nchw_name = ""
            k_input_index = None
            soft_input_index = None
            for idx, input_name in enumerate(kmul_inputs):
                prod_idx = producers.get(str(input_name), None)
                if prod_idx is None:
                    continue
                prod_op = model_ir.operators[int(prod_idx)]
                if (
                    str(prod_op.op_type) == "TRANSPOSE"
                    and len(prod_op.inputs) >= 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(input_name)
                    and _read_transpose_perm(model_ir, prod_op) == perm_nhwc_to_nchw
                ):
                    k_nchw_name = str(input_name)
                    k_input_index = int(idx)
                    soft_nchw_name = str(kmul_inputs[1 - idx])
                    soft_input_index = int(1 - idx)
                    break
            if (
                k_nchw_name == ""
                or soft_nchw_name == ""
                or k_input_index is None
                or soft_input_index is None
            ):
                continue

            k_pre_t_idx = producers.get(k_nchw_name, None)
            if k_pre_t_idx is None:
                continue
            k_pre_t_op = model_ir.operators[int(k_pre_t_idx)]
            if (
                str(k_pre_t_op.op_type) != "TRANSPOSE"
                or len(k_pre_t_op.inputs) < 2
                or len(k_pre_t_op.outputs) != 1
                or str(k_pre_t_op.outputs[0]) != k_nchw_name
                or _read_transpose_perm(model_ir, k_pre_t_op) != perm_nhwc_to_nchw
                or str(k_pre_t_op.outputs[0]) in model_outputs
            ):
                continue
            k_nhwc_name = str(k_pre_t_op.inputs[0])
            if k_nhwc_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(k_nchw_name, [])) != {int(kmul_idx)}:
                continue

            soft_prod_idx = producers.get(soft_nchw_name, None)
            if soft_prod_idx is None:
                continue
            soft_prod_op = model_ir.operators[int(soft_prod_idx)]
            if str(soft_prod_op.op_type) != "SOFTMAX":
                continue
            if set(int(v) for v in consumers.get(soft_nchw_name, [])) != {int(kmul_idx)}:
                continue

            if set(int(v) for v in consumers.get(expand_nchw_name, [])) != {int(post_t_idx)}:
                continue

            expand_const_tensor = model_ir.tensors.get(expand_const_name, None)
            if expand_const_tensor is None or expand_const_tensor.data is None:
                continue
            if set(int(v) for v in consumers.get(expand_const_name, [])) != {int(expand_mul_idx)}:
                continue
            expand_const_data = np.asarray(expand_const_tensor.data)
            if int(expand_const_data.ndim) != 4:
                continue
            rotated_const = np.transpose(expand_const_data, axes=perm_nchw_to_nhwc).astype(
                expand_const_data.dtype,
                copy=False,
            )

            kmul_out_tensor = model_ir.tensors.get(kmul_out_name, None)
            sum_out_tensor = model_ir.tensors.get(sum_out_name, None)
            expand_out_tensor = model_ir.tensors.get(expand_nchw_name, None)
            if kmul_out_tensor is None or sum_out_tensor is None or expand_out_tensor is None:
                continue
            if (
                kmul_out_tensor.shape is None
                or sum_out_tensor.shape is None
                or expand_out_tensor.shape is None
                or len(list(kmul_out_tensor.shape)) != 4
                or len(list(sum_out_tensor.shape)) != 4
                or len(list(expand_out_tensor.shape)) != 4
            ):
                continue

            soft_nhwc_name = _unique_tensor_name(f"{soft_nchw_name}_to_nhwc")
            soft_nhwc_shape = _permute_shape(
                [int(v) for v in list(model_ir.tensors[soft_nchw_name].shape)],
                perm_nchw_to_nhwc,
            )
            if soft_nhwc_shape is None:
                continue
            soft_nhwc_sig = _permute_shape(
                (
                    [int(v) for v in list(model_ir.tensors[soft_nchw_name].shape_signature)]
                    if model_ir.tensors[soft_nchw_name].shape_signature is not None
                    else [int(v) for v in list(model_ir.tensors[soft_nchw_name].shape)]
                ),
                perm_nchw_to_nhwc,
            )
            if soft_nhwc_sig is None:
                soft_nhwc_sig = [int(v) for v in list(soft_nhwc_shape)]
            model_ir.tensors[soft_nhwc_name] = TensorIR(
                name=soft_nhwc_name,
                dtype=str(model_ir.tensors[soft_nchw_name].dtype),
                shape=[int(v) for v in list(soft_nhwc_shape)],
                shape_signature=[int(v) for v in list(soft_nhwc_sig)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(model_ir.tensors[soft_nchw_name].quantization),
            )
            soft_to_nhwc_op = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(soft_nchw_name), str(perm_post_name)],
                outputs=[str(soft_nhwc_name)],
            )

            kmul_new_inputs = [str(v) for v in list(kmul_inputs)]
            kmul_new_inputs[int(k_input_index)] = str(k_nhwc_name)
            kmul_new_inputs[int(soft_input_index)] = str(soft_nhwc_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=kmul_op,
                new_inputs=kmul_new_inputs,
            )

            if not (
                _write_const_ints_to_tensor(model_ir.tensors.get(sum_axes_name, None), [2])
                or _read_const_ints_from_tensor(model_ir.tensors.get(sum_axes_name, None)) == [2]
            ):
                continue

            expand_const_tensor.data = np.asarray(rotated_const)
            expand_const_tensor.shape = [int(v) for v in list(rotated_const.shape)]
            expand_const_tensor.shape_signature = [int(v) for v in list(rotated_const.shape)]

            _permute_tensor_metadata_if_rank_matches(kmul_out_tensor, perm_nchw_to_nhwc)
            _permute_tensor_metadata_if_rank_matches(sum_out_tensor, perm_nchw_to_nhwc)
            _permute_tensor_metadata_if_rank_matches(expand_out_tensor, perm_nchw_to_nhwc)

            _replace_operator_input_at(
                model_ir=model_ir,
                op=relu_mul_op,
                input_index=int(relu_mul_expand_input_index),
                new_input_name=str(expand_nchw_name),
            )

            remove_indices = sorted([int(k_pre_t_idx), int(post_t_idx)], reverse=True)
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            insert_index = int(kmul_idx)
            removed_before_insert = sum(1 for v in remove_indices if int(v) < int(insert_index))
            adjusted_insert_index = int(insert_index) - int(removed_before_insert)
            model_ir.operators.insert(int(adjusted_insert_index), soft_to_nhwc_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": int(rewritten)}
