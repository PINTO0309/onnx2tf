from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _is_singleton_constant_tensor,
    _permute_tensor_metadata_if_rank_matches,
    _permute_shape,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers
        producers = graph_index.producers
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
                graph_index=graph_index,
            )
            concat_op.options["axis"] = 3
            _replace_operator_input_at(
                model_ir=model_ir,
                op=conv_op,
                input_index=0,
                new_input_name=str(mirror_output_name),
                graph_index=graph_index,
            )

            for tensor_name in [str(max_out_name), str(concat_output_name), str(mirror_output_name)]:
                tensor = model_ir.tensors.get(str(tensor_name), None)
                _permute_tensor_metadata_if_rank_matches(tensor, perm_nchw_to_nhwc)
                if tensor is not None and len(list(tensor.shape)) == 4:
                    tensor.logical_layout = "NHWC"
                    if layout_state is not None:
                        layout_state.set(tensor_name, logical="NHWC")
            max_input_tensor.logical_layout = "NCHW"
            mean_input_tensor.logical_layout = "NHWC"
            if layout_state is not None:
                layout_state.set(max_input_name, logical="NCHW")
                layout_state.set(mean_input_name, logical="NHWC")

            graph_index.remove_operator(post_idx)
            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains": int(optimized)}


def run_mixed_attention_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run the mixed reduction/MirrorPad rewrite as an ordered layout pass."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        return preflight_required_op_types(
            candidate_model,
            {
                "MEAN",
                "REDUCE_MAX",
                "CONCATENATION",
                "MIRROR_PAD",
                "TRANSPOSE",
                "CONV_2D",
            },
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return _preflight(pass_state.model_ir)

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.mixed_attention_mirrorpad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


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
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers
        producers = graph_index.producers
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
                    graph_index=graph_index,
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

            _replace_tensor_inputs(
                model_ir,
                k_transpose_out_name,
                k_slice_out_name,
                graph_index=graph_index,
            )
            _replace_tensor_inputs(
                model_ir,
                v_transpose_out_name,
                v_relu_out_name,
                graph_index=graph_index,
            )

            remove_indices = sorted([int(k_transpose_idx), int(v_transpose_idx)], reverse=True)
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            insert_index = min(int(v) for v in list(slice_user_indices))
            removed_before_insert = sum(1 for v in remove_indices if int(v) < int(insert_index))
            adjusted_insert_index = int(insert_index) - int(removed_before_insert)
            graph_index.insert_operator(int(adjusted_insert_index), shared_transpose_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": int(rewritten)}


def _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers
        producers = graph_index.producers
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
                graph_index=graph_index,
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
                graph_index=graph_index,
            )

            remove_indices = sorted([int(k_pre_t_idx), int(post_t_idx)], reverse=True)
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            insert_index = int(kmul_idx)
            removed_before_insert = sum(1 for v in remove_indices if int(v) < int(insert_index))
            adjusted_insert_index = int(insert_index) - int(removed_before_insert)
            graph_index.insert_operator(int(adjusted_insert_index), soft_to_nhwc_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": int(rewritten)}


def run_qkv_attention_bridge_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run the contiguous QKV shared-layout and weighted-sum bridge pair."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        transpose_count = 0
        slice_count = 0
        mul_count = 0
        has_sum = False
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            if op_type == "TRANSPOSE":
                transpose_count += 1
            elif op_type == "SLICE":
                slice_count += 1
            elif op_type == "MUL":
                mul_count += 1
            elif op_type == "SUM":
                has_sum = True
            shared_possible = transpose_count >= 2 and slice_count >= 3
            weighted_possible = transpose_count >= 2 and mul_count >= 2 and has_sum
            if shared_possible or weighted_possible:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_shared_candidate(pass_state: ModelIRPassState) -> bool:
        for op in pass_state.model_ir.operators:
            if (
                str(op.op_type) != "TRANSPOSE"
                or len(op.inputs) < 2
                or _read_transpose_perm(pass_state.model_ir, op) != [0, 3, 1, 2]
            ):
                continue
            slice_idx = pass_state.graph_index.producers.get(str(op.inputs[0]))
            if slice_idx is None:
                continue
            slice_op = pass_state.model_ir.operators[int(slice_idx)]
            if str(slice_op.op_type) != "SLICE" or len(slice_op.inputs) < 3:
                continue
            source_users = pass_state.graph_index.consumer_indices(str(slice_op.inputs[0]))
            if len(source_users) != 3:
                continue
            if all(
                str(pass_state.model_ir.operators[int(index)].op_type) == "SLICE"
                for index in source_users
            ):
                return True
        return False

    def _has_weighted_candidate(pass_state: ModelIRPassState) -> bool:
        for op in pass_state.model_ir.operators:
            if (
                str(op.op_type) != "TRANSPOSE"
                or len(op.inputs) < 2
                or len(op.outputs) != 1
                or _read_transpose_perm(pass_state.model_ir, op) != [0, 2, 3, 1]
            ):
                continue
            producer_idx = pass_state.graph_index.producers.get(str(op.inputs[0]))
            if producer_idx is None:
                continue
            if str(pass_state.model_ir.operators[int(producer_idx)].op_type) != "MUL":
                continue
            output_users = pass_state.graph_index.consumer_indices(str(op.outputs[0]))
            if len(output_users) != 1:
                continue
            if str(pass_state.model_ir.operators[int(output_users[0])].op_type) == "MUL":
                return True
        return False

    def _run_shared(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains",
                    0,
                )
            ),
        }

    def _run_weighted(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.qkv_shared_pretranspose",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_shared,
                precondition=_has_shared_candidate,
                priority=10,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.qkv_weighted_sum_bridge",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_weighted,
                precondition=_has_weighted_candidate,
                priority=20,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": 0,
            "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_attention_qkv_gather_reshape_transpose_hoist_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Hoist shared QKV/KV layout transforms before per-branch GATHERs.

    Target branches (`N in {2,3}`, scalar gather indices `0..N-1`):
      src[N,T,1,D] --GATHER(i)--> g_i[T,1,D]
                    --RESHAPE([T,H,C])--> r_i[T,H,C]
                    --TRANSPOSE([1,0,2])--> t_i[H,T,C]
                    --RESHAPE([1,H,T,C])--> z_i

    Rewrite:
      src --RESHAPE([N,T,H,C])--> src_qkv
          --TRANSPOSE([0,2,1,3])--> src_qkv_ht
      src_qkv_ht --GATHER(i)--> g_i'[H,T,C]
      g_i' --RESHAPE([1,H,T,C])--> z_i
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
            branch_count: Optional[int] = None
            gather_branches: Dict[int, Dict[str, Any]] = {}
            gather_indices: set[int] = set()

            for candidate_branch_count in list(candidate_branch_counts):
                if not _dims_compatible(int(source_shape[0]), int(candidate_branch_count)):
                    continue
                expected_indices = set(range(int(candidate_branch_count)))
                candidate_branches: Dict[int, Dict[str, Any]] = {}

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
                    if gather_index not in expected_indices or gather_index in candidate_branches:
                        continue

                    gather_out_name = str(gather_op.outputs[0])
                    if gather_out_name in model_outputs:
                        continue
                    gather_out_users = [int(v) for v in consumers.get(gather_out_name, [])]
                    if len(gather_out_users) != 1:
                        continue
                    reshape1_idx = int(gather_out_users[0])
                    reshape1_op = model_ir.operators[int(reshape1_idx)]
                    if (
                        str(reshape1_op.op_type) != "RESHAPE"
                        or len(reshape1_op.inputs) < 1
                        or len(reshape1_op.outputs) != 1
                        or str(reshape1_op.inputs[0]) != gather_out_name
                    ):
                        continue
                    reshape1_out_name = str(reshape1_op.outputs[0])
                    if reshape1_out_name in model_outputs:
                        continue
                    reshape1_out_users = [int(v) for v in consumers.get(reshape1_out_name, [])]
                    if len(reshape1_out_users) != 1:
                        continue
                    transpose_idx = int(reshape1_out_users[0])
                    transpose_op = model_ir.operators[int(transpose_idx)]
                    if (
                        str(transpose_op.op_type) != "TRANSPOSE"
                        or len(transpose_op.inputs) < 2
                        or len(transpose_op.outputs) != 1
                        or str(transpose_op.inputs[0]) != reshape1_out_name
                        or _read_transpose_perm(model_ir, transpose_op) != [1, 0, 2]
                    ):
                        continue
                    transpose_out_name = str(transpose_op.outputs[0])
                    if transpose_out_name in model_outputs:
                        continue
                    transpose_out_users = [int(v) for v in consumers.get(transpose_out_name, [])]
                    if len(transpose_out_users) != 1:
                        continue
                    reshape2_idx = int(transpose_out_users[0])
                    reshape2_op = model_ir.operators[int(reshape2_idx)]
                    if (
                        str(reshape2_op.op_type) != "RESHAPE"
                        or len(reshape2_op.inputs) < 1
                        or len(reshape2_op.outputs) != 1
                        or str(reshape2_op.inputs[0]) != transpose_out_name
                    ):
                        continue

                    gather_shape = _shape_list(gather_out_name)
                    reshape1_shape = _shape_list(reshape1_out_name)
                    transpose_shape = _shape_list(transpose_out_name)
                    reshape2_shape = _shape_list(str(reshape2_op.outputs[0]))
                    if (
                        gather_shape is None
                        or reshape1_shape is None
                        or transpose_shape is None
                        or reshape2_shape is None
                        or len(gather_shape) != 3
                        or len(reshape1_shape) != 3
                        or len(transpose_shape) != 3
                        or len(reshape2_shape) != 4
                    ):
                        continue

                    candidate_branches[gather_index] = {
                        "gather_idx": int(user_idx),
                        "gather_op": gather_op,
                        "gather_out_name": gather_out_name,
                        "reshape1_idx": int(reshape1_idx),
                        "reshape1_op": reshape1_op,
                        "reshape1_out_name": reshape1_out_name,
                        "transpose_idx": int(transpose_idx),
                        "transpose_op": transpose_op,
                        "transpose_out_name": transpose_out_name,
                        "reshape2_idx": int(reshape2_idx),
                        "reshape2_op": reshape2_op,
                        "gather_shape": [int(v) for v in gather_shape],
                        "reshape1_shape": [int(v) for v in reshape1_shape],
                        "transpose_shape": [int(v) for v in transpose_shape],
                        "reshape2_shape": [int(v) for v in reshape2_shape],
                    }

                if set(candidate_branches.keys()) == expected_indices:
                    branch_count = int(candidate_branch_count)
                    gather_branches = candidate_branches
                    gather_indices = set(expected_indices)
                    break

            if branch_count is None:
                continue

            branch0 = gather_branches[0]
            t = int(branch0["reshape1_shape"][0])
            h = int(branch0["reshape1_shape"][1])
            c = int(branch0["reshape1_shape"][2])
            if not _shape_compatible(branch0["transpose_shape"], [h, t, c]):
                continue
            if not _shape_compatible(branch0["reshape2_shape"], [1, h, t, c]):
                continue

            is_consistent = True
            for gather_index in sorted(list(gather_indices)):
                branch = gather_branches[gather_index]
                if not _shape_compatible(branch["gather_shape"], [t, 1, h * c]):
                    is_consistent = False
                    break
                if not _shape_compatible(branch["reshape1_shape"], [t, h, c]):
                    is_consistent = False
                    break
                if not _shape_compatible(branch["transpose_shape"], [h, t, c]):
                    is_consistent = False
                    break
                if not _shape_compatible(branch["reshape2_shape"], [1, h, t, c]):
                    is_consistent = False
                    break
            if not is_consistent:
                continue

            n = int(source_shape[0])
            src_t = int(source_shape[1])
            src_one = int(source_shape[2])
            src_d = int(source_shape[3])
            if not _dims_compatible(n, int(branch_count)):
                continue
            if not _dims_compatible(src_t, t) or not _dims_compatible(src_one, 1):
                continue
            if int(src_d) > 0 and int(h) > 0 and int(c) > 0 and int(src_d) != int(h) * int(c):
                continue

            shared_shape_name = _unique_tensor_name(f"{source_name}_qkv_reshape_shape")
            shared_reshape_out_name = _unique_tensor_name(f"{source_name}_qkv_thc")
            shared_perm_name = _unique_tensor_name(f"{source_name}_qkv_perm")
            shared_transpose_out_name = _unique_tensor_name(f"{source_name}_qkv_htc")

            model_ir.tensors[shared_shape_name] = TensorIR(
                name=shared_shape_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray([int(branch_count), int(t), int(h), int(c)], dtype=np.int32),
                is_variable=False,
            )
            source_tensor = model_ir.tensors.get(str(source_name), None)
            model_ir.tensors[shared_reshape_out_name] = TensorIR(
                name=shared_reshape_out_name,
                dtype=str(source_tensor.dtype if source_tensor is not None else "FLOAT32"),
                shape=[int(branch_count), int(t), int(h), int(c)],
                shape_signature=[int(branch_count), int(t), int(h), int(c)],
            )
            model_ir.tensors[shared_perm_name] = TensorIR(
                name=shared_perm_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray([0, 2, 1, 3], dtype=np.int32),
                is_variable=False,
            )
            model_ir.tensors[shared_transpose_out_name] = TensorIR(
                name=shared_transpose_out_name,
                dtype=str(source_tensor.dtype if source_tensor is not None else "FLOAT32"),
                shape=[int(branch_count), int(h), int(t), int(c)],
                shape_signature=[int(branch_count), int(h), int(t), int(c)],
            )

            shared_reshape_op = OperatorIR(
                op_type="RESHAPE",
                inputs=[str(source_name), str(shared_shape_name)],
                outputs=[str(shared_reshape_out_name)],
                options={"newShape": [int(branch_count), int(t), int(h), int(c)]},
            )
            shared_transpose_op = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(shared_reshape_out_name), str(shared_perm_name)],
                outputs=[str(shared_transpose_out_name)],
                options={},
            )
            insert_base_idx = min(int(gather_branches[idx]["gather_idx"]) for idx in sorted(list(gather_indices)))
            model_ir.operators.insert(int(insert_base_idx), shared_reshape_op)
            model_ir.operators.insert(int(insert_base_idx) + 1, shared_transpose_op)

            remove_indices: set[int] = set()
            for gather_index in sorted(list(gather_indices)):
                branch = gather_branches[gather_index]
                gather_op = branch["gather_op"]
                gather_out_name = str(branch["gather_out_name"])
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=gather_op,
                    input_index=0,
                    new_input_name=str(shared_transpose_out_name),
                )
                gather_tensor = model_ir.tensors.get(gather_out_name, None)
                if gather_tensor is not None:
                    gather_tensor.shape = [int(h), int(t), int(c)]
                    gather_tensor.shape_signature = [int(h), int(t), int(c)]

                _replace_tensor_inputs(
                    model_ir,
                    str(branch["transpose_out_name"]),
                    gather_out_name,
                )
                reshape1_remove_idx = next(
                    (idx for idx, op in enumerate(model_ir.operators) if op is branch["reshape1_op"]),
                    None,
                )
                transpose_remove_idx = next(
                    (idx for idx, op in enumerate(model_ir.operators) if op is branch["transpose_op"]),
                    None,
                )
                if reshape1_remove_idx is not None:
                    remove_indices.add(int(reshape1_remove_idx))
                if transpose_remove_idx is not None:
                    remove_indices.add(int(transpose_remove_idx))

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_attention_qkv_gather_reshape_transpose_hoist_chains": int(rewritten)}


def _optimize_transpose_conv_attention_nhwc_propagation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Propagate NHWC layout through Conv-attention motifs and remove bridge transposes.

    Target motif (rank-4):
      x_nhwc --T(0,3,1,2)--> x_nchw --ACT--> a_nchw
      a_nchw --MEAN(keepDims=True, axes in NCHW)--> m_nchw --T(0,2,3,1)--> m_nhwc --CONV--CONV--> c_nhwc
      c_nhwc --T(0,3,1,2)--> c_nchw --GATE--> s_nchw
      MUL(a_nchw, s_nchw) --> y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - bypass both NCHW pre-transposes and all post-transposes in the motif
      - remap MEAN axes from NCHW to NHWC
      - keep ACT/MEAN/GATE/MUL in NHWC

    Supported GATE:
      - LOGISTIC
      - HardSigmoid expansion: MUL(const)->ADD(const)->RELU_0_TO_1
      - HardSigmoid expansion: MUL(const)->ADD(const)->MAXIMUM(const)->MINIMUM(const)
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw_const_name = "__nhwc_to_nchw_perm_rank4__"
    activation_ops = {"RELU", "LEAKY_RELU", "RELU6"}
    legacy_consumer_safe_ops = {
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "RELU",
        "RELU6",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "NEG",
        "ABS",
        "SQRT",
        "EXP",
    }

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    if perm_nhwc_to_nchw_const_name not in model_ir.tensors:
        model_ir.tensors[perm_nhwc_to_nchw_const_name] = TensorIR(
            name=perm_nhwc_to_nchw_const_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        def _match_hardsigmoid_gate_from_output(gate_output_name: str) -> Optional[Dict[str, Any]]:
            gate_idx = producers.get(str(gate_output_name), None)
            if gate_idx is None:
                return None
            gate_op = model_ir.operators[int(gate_idx)]
            gate_type = str(gate_op.op_type)
            add_output_name: Optional[str] = None
            chain_indices: List[int] = []
            metadata_names: List[str] = []

            if gate_type == "RELU_0_TO_1":
                if len(gate_op.inputs) != 1 or len(gate_op.outputs) != 1:
                    return None
                add_output_name = str(gate_op.inputs[0])
                if set(int(v) for v in consumers.get(add_output_name, [])) != {int(gate_idx)}:
                    return None
                chain_indices.append(int(gate_idx))
                metadata_names.append(str(gate_output_name))
            elif gate_type == "MINIMUM":
                if len(gate_op.inputs) != 2 or len(gate_op.outputs) != 1:
                    return None
                min_inputs = [str(v) for v in list(gate_op.inputs)]
                max_output_name: Optional[str] = None
                min_side_name: Optional[str] = None
                for candidate_name, other_name in [(min_inputs[0], min_inputs[1]), (min_inputs[1], min_inputs[0])]:
                    producer_idx = producers.get(candidate_name, None)
                    if producer_idx is None:
                        continue
                    producer_op = model_ir.operators[int(producer_idx)]
                    if str(producer_op.op_type) != "MAXIMUM":
                        continue
                    max_output_name = str(candidate_name)
                    min_side_name = str(other_name)
                    break
                if max_output_name is None or min_side_name is None:
                    return None
                if not _is_singleton_constant_tensor(model_ir, min_side_name):
                    return None
                if set(int(v) for v in consumers.get(max_output_name, [])) != {int(gate_idx)}:
                    return None

                max_idx = producers.get(max_output_name, None)
                if max_idx is None:
                    return None
                max_op = model_ir.operators[int(max_idx)]
                if str(max_op.op_type) != "MAXIMUM" or len(max_op.inputs) != 2 or len(max_op.outputs) != 1:
                    return None
                max_inputs = [str(v) for v in list(max_op.inputs)]
                for candidate_name, other_name in [(max_inputs[0], max_inputs[1]), (max_inputs[1], max_inputs[0])]:
                    producer_idx = producers.get(candidate_name, None)
                    if producer_idx is None:
                        continue
                    producer_op = model_ir.operators[int(producer_idx)]
                    if str(producer_op.op_type) != "ADD":
                        continue
                    add_output_name = str(candidate_name)
                    if not _is_singleton_constant_tensor(model_ir, str(other_name)):
                        return None
                    break
                if add_output_name is None:
                    return None
                if set(int(v) for v in consumers.get(add_output_name, [])) != {int(max_idx)}:
                    return None
                chain_indices.extend([int(max_idx), int(gate_idx)])
                metadata_names.extend([max_output_name, str(gate_output_name)])
            else:
                return None

            add_idx = producers.get(str(add_output_name), None)
            if add_idx is None:
                return None
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                return None
            add_inputs = [str(v) for v in list(add_op.inputs)]
            mul_output_name: Optional[str] = None
            for candidate_name, other_name in [(add_inputs[0], add_inputs[1]), (add_inputs[1], add_inputs[0])]:
                producer_idx = producers.get(candidate_name, None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if str(producer_op.op_type) != "MUL":
                    continue
                if not _is_singleton_constant_tensor(model_ir, str(other_name)):
                    return None
                mul_output_name = str(candidate_name)
                break
            if mul_output_name is None:
                return None
            if set(int(v) for v in consumers.get(mul_output_name, [])) != {int(add_idx)}:
                return None
            chain_indices.append(int(add_idx))
            metadata_names.insert(0, str(add_output_name))

            mul_idx = producers.get(mul_output_name, None)
            if mul_idx is None:
                return None
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                return None
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            main_input_name: Optional[str] = None
            main_input_index: Optional[int] = None
            for input_index, input_name in enumerate(mul_inputs):
                side_name = str(mul_inputs[1 - input_index])
                if _is_singleton_constant_tensor(model_ir, side_name):
                    main_input_name = str(input_name)
                    main_input_index = int(input_index)
                    break
            if main_input_name is None or main_input_index is None:
                return None

            chain_indices.append(int(mul_idx))
            metadata_names.insert(0, str(mul_output_name))

            return {
                "head_idx": int(mul_idx),
                "head_input_name": str(main_input_name),
                "head_input_index": int(main_input_index),
                "output_name": str(gate_output_name),
                "chain_indices": [int(v) for v in chain_indices],
                "metadata_names": [str(v) for v in metadata_names],
            }

        def _match_hardswish_activation_from_output(activation_output_name: str) -> Optional[Dict[str, Any]]:
            activation_idx = producers.get(str(activation_output_name), None)
            if activation_idx is None:
                return None
            activation_op = model_ir.operators[int(activation_idx)]
            if (
                str(activation_op.op_type) != "MUL"
                or len(activation_op.inputs) != 2
                or len(activation_op.outputs) != 1
                or str(activation_op.outputs[0]) != str(activation_output_name)
            ):
                return None

            activation_inputs = [str(v) for v in list(activation_op.inputs)]
            for data_input_index in [0, 1]:
                data_input_name = str(activation_inputs[int(data_input_index)])
                gate_output_name = str(activation_inputs[1 - int(data_input_index)])

                gate_match = _match_hardsigmoid_gate_from_output(gate_output_name)
                if gate_match is None:
                    continue
                if str(gate_match["head_input_name"]) != str(data_input_name):
                    continue

                gate_users = set(int(v) for v in consumers.get(str(gate_output_name), []))
                if gate_users != {int(activation_idx)}:
                    continue

                expected_data_users = {int(activation_idx), int(gate_match["head_idx"])}
                data_users = set(int(v) for v in consumers.get(str(data_input_name), []))
                if data_users != expected_data_users:
                    continue

                activation_metadata_names = [str(v) for v in list(gate_match.get("metadata_names", []))]
                activation_metadata_names.append(str(activation_output_name))
                rewire_specs = [
                    (int(gate_match["head_idx"]), int(gate_match["head_input_index"])),
                    (int(activation_idx), int(data_input_index)),
                ]

                return {
                    "activation_idx": int(activation_idx),
                    "activation_input_name": str(data_input_name),
                    "activation_input_expected_users": sorted(list(expected_data_users)),
                    "activation_output_name": str(activation_output_name),
                    "activation_metadata_names": activation_metadata_names,
                    "rewire_specs": rewire_specs,
                }
            return None

        def _match_self_hardswish_from_source(source_nchw_name: str) -> Optional[Dict[str, Any]]:
            source_users = [int(v) for v in consumers.get(str(source_nchw_name), [])]
            if len(source_users) == 0:
                return None

            add_idx: Optional[int] = None
            mul_idx: Optional[int] = None
            add_out_name: Optional[str] = None
            gate_name: Optional[str] = None

            for user_idx in source_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "ADD":
                    if len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                        return None
                    add_inputs = [str(v) for v in list(user_op.inputs)]
                    if source_nchw_name == add_inputs[0]:
                        add_side_name = str(add_inputs[1])
                    elif source_nchw_name == add_inputs[1]:
                        add_side_name = str(add_inputs[0])
                    else:
                        return None
                    if not _is_singleton_constant_tensor(model_ir, add_side_name):
                        return None
                    if add_idx is not None:
                        return None
                    add_idx = int(user_idx)
                    add_out_name = str(user_op.outputs[0])
                    continue
                if user_type == "MUL":
                    if len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                        return None
                    mul_inputs = [str(v) for v in list(user_op.inputs)]
                    if source_nchw_name == mul_inputs[0]:
                        other_name = str(mul_inputs[1])
                    elif source_nchw_name == mul_inputs[1]:
                        other_name = str(mul_inputs[0])
                    else:
                        return None
                    if mul_idx is not None:
                        return None
                    mul_idx = int(user_idx)
                    gate_name = str(other_name)
                    continue
                return None

            if add_idx is None or mul_idx is None or add_out_name is None or gate_name is None:
                return None
            if set(int(v) for v in source_users) != {int(add_idx), int(mul_idx)}:
                return None

            scale_idx = producers.get(str(gate_name), None)
            if scale_idx is None:
                return None
            scale_op = model_ir.operators[int(scale_idx)]
            if (
                str(scale_op.op_type) not in {"MUL", "DIV"}
                or len(scale_op.inputs) != 2
                or len(scale_op.outputs) != 1
                or str(scale_op.outputs[0]) != str(gate_name)
            ):
                return None
            scale_inputs = [str(v) for v in list(scale_op.inputs)]
            if str(add_out_name) == scale_inputs[0]:
                scale_side_name = str(scale_inputs[1])
            elif str(add_out_name) == scale_inputs[1]:
                scale_side_name = str(scale_inputs[0])
            else:
                return None
            if not _is_singleton_constant_tensor(model_ir, scale_side_name):
                return None
            if set(int(v) for v in consumers.get(str(add_out_name), [])) != {int(scale_idx)}:
                return None
            if set(int(v) for v in consumers.get(str(gate_name), [])) != {int(mul_idx)}:
                return None

            mul_op = model_ir.operators[int(mul_idx)]
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            if not (
                (str(source_nchw_name) == mul_inputs[0] and str(gate_name) == mul_inputs[1])
                or (str(source_nchw_name) == mul_inputs[1] and str(gate_name) == mul_inputs[0])
            ):
                return None

            return {
                "add_idx": int(add_idx),
                "scale_idx": int(scale_idx),
                "mul_idx": int(mul_idx),
                "add_out_name": str(add_out_name),
                "scale_out_name": str(gate_name),
                "mul_out_name": str(mul_op.outputs[0]),
            }

        for candidate_post_idx, candidate_post_op in enumerate(model_ir.operators):
            if (
                str(candidate_post_op.op_type) != "TRANSPOSE"
                or len(candidate_post_op.inputs) < 2
                or len(candidate_post_op.outputs) != 1
                or _read_transpose_perm(model_ir, candidate_post_op) != perm_nchw_to_nhwc
            ):
                continue

            mul_output_name = str(candidate_post_op.inputs[0])
            if mul_output_name in model_outputs:
                continue
            mul_idx = producers.get(mul_output_name, None)
            if mul_idx is None:
                continue
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            if str(mul_op.outputs[0]) != mul_output_name:
                continue

            mul_output_users = [int(v) for v in consumers.get(mul_output_name, [])]
            if len(mul_output_users) == 0:
                continue
            post_pairs: List[Tuple[int, OperatorIR]] = []
            legacy_slots: List[Tuple[int, int]] = []
            valid_posts = True
            for user_idx in mul_output_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != mul_output_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    if str(user_op.op_type) not in legacy_consumer_safe_ops:
                        valid_posts = False
                        break
                    input_indices = [
                        int(input_idx)
                        for input_idx, input_name in enumerate(list(user_op.inputs))
                        if str(input_name) == str(mul_output_name)
                    ]
                    if len(input_indices) == 0:
                        valid_posts = False
                        break
                    for input_idx in input_indices:
                        legacy_slots.append((int(id(user_op)), int(input_idx)))
                    continue
                post_pairs.append((int(user_idx), user_op))
            if not valid_posts or len(post_pairs) == 0:
                continue
            post_pairs = sorted(post_pairs, key=lambda v: int(v[0]))
            post_indices = [int(v[0]) for v in post_pairs]
            post_output_names = [str(v[1].outputs[0]) for v in post_pairs]

            input0_name = str(mul_op.inputs[0])
            input1_name = str(mul_op.inputs[1])
            activation_idx: Optional[int] = None
            activation_input_name: Optional[str] = None
            activation_input_expected_users: Optional[set[int]] = None
            activation_output_name: Optional[str] = None
            activation_rewire_specs: List[Tuple[int, int]] = []
            activation_metadata_names: List[str] = []
            gate_kind: Optional[str] = None
            gate_head_idx: Optional[int] = None
            gate_head_input_name: Optional[str] = None
            gate_head_input_index: Optional[int] = None
            gate_output_name: Optional[str] = None
            gate_metadata_names: List[str] = []
            for lhs_name, rhs_name in [(input0_name, input1_name), (input1_name, input0_name)]:
                lhs_prod_idx = producers.get(lhs_name, None)
                if lhs_prod_idx is None:
                    continue
                lhs_op = model_ir.operators[int(lhs_prod_idx)]
                lhs_activation_match: Optional[Dict[str, Any]] = None
                if str(lhs_op.op_type) in activation_ops:
                    if len(lhs_op.inputs) == 1 and len(lhs_op.outputs) == 1 and str(lhs_op.outputs[0]) == str(lhs_name):
                        lhs_activation_match = {
                            "activation_idx": int(lhs_prod_idx),
                            "activation_input_name": str(lhs_op.inputs[0]),
                            "activation_input_expected_users": [int(lhs_prod_idx)],
                            "activation_output_name": str(lhs_name),
                            "activation_metadata_names": [str(lhs_name)],
                            "rewire_specs": [(int(lhs_prod_idx), 0)],
                        }
                else:
                    lhs_activation_match = _match_hardswish_activation_from_output(str(lhs_name))
                if lhs_activation_match is None:
                    continue

                rhs_prod_idx = producers.get(rhs_name, None)
                if rhs_prod_idx is None:
                    continue
                rhs_op = model_ir.operators[int(rhs_prod_idx)]

                if str(rhs_op.op_type) == "LOGISTIC":
                    if len(rhs_op.inputs) != 1 or len(rhs_op.outputs) != 1:
                        continue
                    if str(rhs_op.outputs[0]) != rhs_name:
                        continue
                    activation_idx = int(lhs_activation_match["activation_idx"])
                    activation_input_name = str(lhs_activation_match["activation_input_name"])
                    activation_input_expected_users = set(
                        int(v) for v in list(lhs_activation_match["activation_input_expected_users"])
                    )
                    activation_output_name = str(lhs_activation_match["activation_output_name"])
                    activation_rewire_specs = [
                        (int(v[0]), int(v[1])) for v in list(lhs_activation_match["rewire_specs"])
                    ]
                    activation_metadata_names = [
                        str(v) for v in list(lhs_activation_match.get("activation_metadata_names", []))
                    ]
                    gate_kind = "LOGISTIC"
                    gate_head_idx = int(rhs_prod_idx)
                    gate_head_input_name = str(rhs_op.inputs[0])
                    gate_head_input_index = 0
                    gate_output_name = str(rhs_name)
                    gate_metadata_names = [str(rhs_name)]
                    break

                hsig_match = _match_hardsigmoid_gate_from_output(str(rhs_name))
                if hsig_match is None:
                    continue
                activation_idx = int(lhs_activation_match["activation_idx"])
                activation_input_name = str(lhs_activation_match["activation_input_name"])
                activation_input_expected_users = set(
                    int(v) for v in list(lhs_activation_match["activation_input_expected_users"])
                )
                activation_output_name = str(lhs_activation_match["activation_output_name"])
                activation_rewire_specs = [
                    (int(v[0]), int(v[1])) for v in list(lhs_activation_match["rewire_specs"])
                ]
                activation_metadata_names = [
                    str(v) for v in list(lhs_activation_match.get("activation_metadata_names", []))
                ]
                gate_kind = "HARDSIGMOID"
                gate_head_idx = int(hsig_match["head_idx"])
                gate_head_input_name = str(hsig_match["head_input_name"])
                gate_head_input_index = int(hsig_match["head_input_index"])
                gate_output_name = str(hsig_match["output_name"])
                gate_metadata_names = [str(v) for v in list(hsig_match.get("metadata_names", []))]
                break
            if (
                activation_idx is None
                or activation_input_name is None
                or activation_input_expected_users is None
                or activation_output_name is None
                or gate_kind is None
                or gate_head_idx is None
                or gate_head_input_name is None
                or gate_head_input_index is None
                or gate_output_name is None
            ):
                continue

            gate_head_op = model_ir.operators[int(gate_head_idx)]
            if (
                activation_input_name in model_outputs
                or activation_output_name in model_outputs
                or gate_head_input_name in model_outputs
                or gate_output_name in model_outputs
            ):
                continue

            pre_activation_idx = producers.get(activation_input_name, None)
            if pre_activation_idx is None:
                continue
            pre_activation_op = model_ir.operators[int(pre_activation_idx)]
            if (
                str(pre_activation_op.op_type) != "TRANSPOSE"
                or len(pre_activation_op.inputs) < 2
                or len(pre_activation_op.outputs) != 1
                or str(pre_activation_op.outputs[0]) != activation_input_name
                or _read_transpose_perm(model_ir, pre_activation_op) != perm_nhwc_to_nchw
            ):
                continue
            source_nhwc_name = str(pre_activation_op.inputs[0])
            if source_nhwc_name in model_outputs:
                continue

            mean_idx_candidates = [
                int(v)
                for v in consumers.get(activation_output_name, [])
                if int(v) != int(mul_idx)
            ]
            if len(mean_idx_candidates) != 1:
                continue
            mean_idx = int(mean_idx_candidates[0])
            mean_op = model_ir.operators[int(mean_idx)]
            if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                continue
            if str(mean_op.inputs[0]) != activation_output_name:
                continue
            if not bool(mean_op.options.get("keepDims", False)):
                continue
            mean_output_name = str(mean_op.outputs[0])
            if mean_output_name in model_outputs:
                continue

            mean_post_users = [int(v) for v in consumers.get(mean_output_name, [])]
            if len(mean_post_users) != 1:
                continue
            mean_post_idx = int(mean_post_users[0])
            mean_post_op = model_ir.operators[int(mean_post_idx)]
            if (
                str(mean_post_op.op_type) != "TRANSPOSE"
                or len(mean_post_op.inputs) < 2
                or len(mean_post_op.outputs) != 1
                or str(mean_post_op.inputs[0]) != mean_output_name
                or _read_transpose_perm(model_ir, mean_post_op) != perm_nchw_to_nhwc
            ):
                continue
            conv1_input_name = str(mean_post_op.outputs[0])
            if conv1_input_name in model_outputs:
                continue

            conv1_users = [int(v) for v in consumers.get(conv1_input_name, [])]
            if len(conv1_users) != 1:
                continue
            conv1_idx = int(conv1_users[0])
            conv1_op = model_ir.operators[int(conv1_idx)]
            if str(conv1_op.op_type) != "CONV_2D" or len(conv1_op.inputs) < 1 or len(conv1_op.outputs) != 1:
                continue
            if str(conv1_op.inputs[0]) != conv1_input_name:
                continue
            conv1_output_name = str(conv1_op.outputs[0])
            if conv1_output_name in model_outputs:
                continue
            conv1_output_users = [int(v) for v in consumers.get(conv1_output_name, [])]
            if len(conv1_output_users) != 1:
                continue
            conv2_idx = int(conv1_output_users[0])
            conv2_op = model_ir.operators[int(conv2_idx)]
            if str(conv2_op.op_type) != "CONV_2D" or len(conv2_op.inputs) < 1 or len(conv2_op.outputs) != 1:
                continue
            if str(conv2_op.inputs[0]) != conv1_output_name:
                continue
            conv2_output_name = str(conv2_op.outputs[0])
            if conv2_output_name in model_outputs:
                continue

            pre_gate_idx = producers.get(gate_head_input_name, None)
            if pre_gate_idx is None:
                continue
            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
            if (
                str(pre_gate_op.op_type) != "TRANSPOSE"
                or len(pre_gate_op.inputs) < 2
                or len(pre_gate_op.outputs) != 1
                or str(pre_gate_op.outputs[0]) != gate_head_input_name
                or _read_transpose_perm(model_ir, pre_gate_op) != perm_nhwc_to_nchw
                or str(pre_gate_op.inputs[0]) != conv2_output_name
            ):
                continue

            # Locality guards: this rewrite assumes a closed motif.
            if set(int(v) for v in consumers.get(activation_input_name, [])) != set(int(v) for v in activation_input_expected_users):
                continue
            if set(int(v) for v in consumers.get(activation_output_name, [])) != {int(mean_idx), int(mul_idx)}:
                continue
            if set(int(v) for v in consumers.get(mean_output_name, [])) != {int(mean_post_idx)}:
                continue
            if set(int(v) for v in consumers.get(conv1_input_name, [])) != {int(conv1_idx)}:
                continue
            if set(int(v) for v in consumers.get(conv1_output_name, [])) != {int(conv2_idx)}:
                continue
            if set(int(v) for v in consumers.get(conv2_output_name, [])) != {int(pre_gate_idx)}:
                continue
            if set(int(v) for v in consumers.get(gate_head_input_name, [])) != {int(gate_head_idx)}:
                continue
            if set(int(v) for v in consumers.get(gate_output_name, [])) != {int(mul_idx)}:
                continue

            # Remap MEAN axes from NCHW to NHWC.
            mean_axes_tensor = model_ir.tensors.get(str(mean_op.inputs[1]), None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None or len(mean_axes_vals) == 0:
                continue
            source_tensor = model_ir.tensors.get(source_nhwc_name, None)
            rank = 4
            if source_tensor is not None and source_tensor.shape is not None and len(list(source_tensor.shape)) > 0:
                rank = int(len(list(source_tensor.shape)))
            if rank != 4:
                continue
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes_vals:
                a = int(axis)
                if a < 0:
                    a += int(rank)
                if a < 0 or a >= int(rank):
                    valid_axes = False
                    break
                normalized_axes.append(int(a))
            if not valid_axes:
                continue
            mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]
            if not _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes]):
                continue

            # Rewire to NHWC.
            for rewire_op_idx, rewire_input_idx in activation_rewire_specs:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=model_ir.operators[int(rewire_op_idx)],
                    input_index=int(rewire_input_idx),
                    new_input_name=source_nhwc_name,
                )
            conv1_inputs = [str(v) for v in list(conv1_op.inputs)]
            conv1_inputs[0] = str(mean_output_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv1_op,
                new_inputs=conv1_inputs,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=gate_head_op,
                input_index=int(gate_head_input_index),
                new_input_name=conv2_output_name,
            )

            # Update metadata of tensors that switch from NCHW to NHWC.
            old_mul_tensor = model_ir.tensors.get(mul_output_name, None)
            legacy_mul_shape = (
                [int(v) for v in list(old_mul_tensor.shape)]
                if old_mul_tensor is not None and old_mul_tensor.shape is not None
                else None
            )
            legacy_mul_shape_signature = None
            if old_mul_tensor is not None:
                legacy_mul_shape_signature = (
                    [int(v) for v in list(old_mul_tensor.shape_signature)]
                    if old_mul_tensor.shape_signature is not None
                    else (
                        [int(v) for v in list(old_mul_tensor.shape)]
                        if old_mul_tensor.shape is not None
                        else None
                    )
                )
            metadata_targets = [str(v) for v in list(activation_metadata_names)]
            metadata_targets.append(str(mean_output_name))
            metadata_targets.extend([str(v) for v in list(gate_metadata_names)])
            metadata_targets.append(str(mul_output_name))
            for tensor_name in list(dict.fromkeys(metadata_targets)):
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[canonical_post_output_name],
            )
            _replace_tensor_inputs(model_ir, mul_output_name, canonical_post_output_name)
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name)

            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_mul_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_mul_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_mul_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_mul_tensor.shape_signature)]
                    if old_mul_tensor.shape_signature is not None
                    else [int(v) for v in list(old_mul_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_nchw_to_nhwc,
                )

            pending_legacy_slots = list(legacy_slots)

            remove_indices = {
                int(pre_activation_idx),
                int(mean_post_idx),
                int(pre_gate_idx),
                *[int(v) for v in post_indices],
            }
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            if len(pending_legacy_slots) > 0:
                op_index_by_id = {int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)}
                valid_legacy_slots: List[Tuple[int, int]] = []
                for op_id, input_idx in pending_legacy_slots:
                    op_idx = op_index_by_id.get(int(op_id), None)
                    if op_idx is None:
                        continue
                    if int(op_idx) < 0 or int(op_idx) >= len(model_ir.operators):
                        continue
                    legacy_op = model_ir.operators[int(op_idx)]
                    if int(input_idx) < 0 or int(input_idx) >= len(legacy_op.inputs):
                        continue
                    if str(legacy_op.inputs[int(input_idx)]) != str(canonical_post_output_name):
                        continue
                    valid_legacy_slots.append((int(op_idx), int(input_idx)))

                if len(valid_legacy_slots) > 0:
                    adapter_name = _unique_tensor_name(f"{mul_output_name}_nchw_adapter")
                    adapter_dtype = str(old_mul_tensor.dtype) if old_mul_tensor is not None else "FLOAT32"
                    adapter_quant = (
                        _clone_quantization(old_mul_tensor.quantization)
                        if old_mul_tensor is not None
                        else None
                    )
                    adapter_shape = (
                        [int(v) for v in list(legacy_mul_shape)]
                        if legacy_mul_shape is not None
                        else [1]
                    )
                    adapter_shape_signature = (
                        [int(v) for v in list(legacy_mul_shape_signature)]
                        if legacy_mul_shape_signature is not None
                        else [int(v) for v in list(adapter_shape)]
                    )
                    model_ir.tensors[adapter_name] = TensorIR(
                        name=adapter_name,
                        dtype=str(adapter_dtype),
                        shape=[int(v) for v in list(adapter_shape)],
                        shape_signature=[int(v) for v in list(adapter_shape_signature)],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(adapter_quant),
                    )
                    adapter_op = OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(canonical_post_output_name), perm_nhwc_to_nchw_const_name],
                        outputs=[str(adapter_name)],
                    )
                    for op_idx, input_idx in valid_legacy_slots:
                        legacy_op = model_ir.operators[int(op_idx)]
                        new_inputs = [str(v) for v in list(legacy_op.inputs)]
                        new_inputs[int(input_idx)] = str(adapter_name)
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=legacy_op,
                            new_inputs=new_inputs,
                        )
                    insert_index = int(min(v[0] for v in valid_legacy_slots))
                    model_ir.operators.insert(int(insert_index), adapter_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            for pre_idx, pre_op in enumerate(model_ir.operators):
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                ):
                    continue
                source_nhwc_name = str(pre_op.inputs[0])
                source_nchw_name = str(pre_op.outputs[0])
                if source_nhwc_name in model_outputs or source_nchw_name in model_outputs:
                    continue

                hs0_match = _match_self_hardswish_from_source(source_nchw_name)
                if hs0_match is None:
                    continue
                hsw0_out_name = str(hs0_match["mul_out_name"])
                if hsw0_out_name in model_outputs:
                    continue

                hsw0_users = [int(v) for v in consumers.get(hsw0_out_name, [])]
                if len(hsw0_users) != 1:
                    continue
                mean0_idx = int(hsw0_users[0])
                mean0_op = model_ir.operators[int(mean0_idx)]
                if (
                    str(mean0_op.op_type) != "MEAN"
                    or len(mean0_op.inputs) < 2
                    or len(mean0_op.outputs) != 1
                    or str(mean0_op.inputs[0]) != str(hsw0_out_name)
                    or not bool(mean0_op.options.get("keepDims", False))
                ):
                    continue
                mean0_out_name = str(mean0_op.outputs[0])
                if mean0_out_name in model_outputs:
                    continue
                mean0_axes_tensor = model_ir.tensors.get(str(mean0_op.inputs[1]), None)
                mean0_axes_vals = _read_const_ints_from_tensor(mean0_axes_tensor)
                if mean0_axes_vals is None or len(mean0_axes_vals) == 0:
                    continue

                mean0_users = [int(v) for v in consumers.get(mean0_out_name, [])]
                if len(mean0_users) != 1:
                    continue
                mean0_post_idx = int(mean0_users[0])
                mean0_post_op = model_ir.operators[int(mean0_post_idx)]
                if (
                    str(mean0_post_op.op_type) != "TRANSPOSE"
                    or len(mean0_post_op.inputs) < 2
                    or len(mean0_post_op.outputs) != 1
                    or str(mean0_post_op.inputs[0]) != str(mean0_out_name)
                    or _read_transpose_perm(model_ir, mean0_post_op) != perm_nchw_to_nhwc
                ):
                    continue
                conv_input_name = str(mean0_post_op.outputs[0])
                if conv_input_name in model_outputs:
                    continue

                conv_users = [int(v) for v in consumers.get(conv_input_name, [])]
                if len(conv_users) != 1:
                    continue
                conv_idx = int(conv_users[0])
                conv_op = model_ir.operators[int(conv_idx)]
                if (
                    str(conv_op.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}
                    or len(conv_op.inputs) < 1
                    or len(conv_op.outputs) != 1
                    or str(conv_op.inputs[0]) != str(conv_input_name)
                ):
                    continue
                conv_out_name = str(conv_op.outputs[0])
                if conv_out_name in model_outputs:
                    continue

                conv_out_users = [int(v) for v in consumers.get(conv_out_name, [])]
                if len(conv_out_users) != 1:
                    continue
                pre1_idx = int(conv_out_users[0])
                pre1_op = model_ir.operators[int(pre1_idx)]
                if (
                    str(pre1_op.op_type) != "TRANSPOSE"
                    or len(pre1_op.inputs) < 2
                    or len(pre1_op.outputs) != 1
                    or str(pre1_op.inputs[0]) != str(conv_out_name)
                    or _read_transpose_perm(model_ir, pre1_op) != perm_nhwc_to_nchw
                ):
                    continue
                gate_nchw_name = str(pre1_op.outputs[0])
                if gate_nchw_name in model_outputs:
                    continue

                hs1_match = _match_self_hardswish_from_source(gate_nchw_name)
                if hs1_match is None:
                    continue
                hsw1_out_name = str(hs1_match["mul_out_name"])
                if hsw1_out_name in model_outputs:
                    continue

                hsw1_users = [int(v) for v in consumers.get(hsw1_out_name, [])]
                if len(hsw1_users) != 1:
                    continue
                mean1_idx = int(hsw1_users[0])
                mean1_op = model_ir.operators[int(mean1_idx)]
                if (
                    str(mean1_op.op_type) != "MEAN"
                    or len(mean1_op.inputs) < 2
                    or len(mean1_op.outputs) != 1
                    or str(mean1_op.inputs[0]) != str(hsw1_out_name)
                    or not bool(mean1_op.options.get("keepDims", False))
                ):
                    continue
                mean1_out_name = str(mean1_op.outputs[0])
                if mean1_out_name in model_outputs:
                    continue
                mean1_axes_tensor = model_ir.tensors.get(str(mean1_op.inputs[1]), None)
                mean1_axes_vals = _read_const_ints_from_tensor(mean1_axes_tensor)
                if mean1_axes_vals is None or len(mean1_axes_vals) == 0:
                    continue
                mean1_users = [int(v) for v in consumers.get(mean1_out_name, [])]
                if len(mean1_users) == 0:
                    continue
                # Keep this rewrite conservative: terminal mean should not feed post-transpose bridges.
                if any(str(model_ir.operators[int(v)].op_type) == "TRANSPOSE" for v in mean1_users):
                    continue

                source_tensor = model_ir.tensors.get(source_nhwc_name, None)
                rank = 4
                if source_tensor is not None and source_tensor.shape is not None and len(list(source_tensor.shape)) > 0:
                    rank = int(len(list(source_tensor.shape)))
                if rank != 4:
                    continue

                def _map_axes_to_nhwc_side(axes_vals: List[int]) -> Optional[List[int]]:
                    normalized_axes: List[int] = []
                    for axis in axes_vals:
                        a = int(axis)
                        if a < 0:
                            a += int(rank)
                        if a < 0 or a >= int(rank):
                            return None
                        normalized_axes.append(int(a))
                    return [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]

                mapped_axes0 = _map_axes_to_nhwc_side([int(v) for v in list(mean0_axes_vals)])
                mapped_axes1 = _map_axes_to_nhwc_side([int(v) for v in list(mean1_axes_vals)])
                if mapped_axes0 is None or mapped_axes1 is None:
                    continue
                if not (
                    _write_const_ints_to_tensor(mean0_axes_tensor, [int(v) for v in mapped_axes0])
                    or _read_const_ints_from_tensor(mean0_axes_tensor) == [int(v) for v in mapped_axes0]
                ):
                    continue
                if not (
                    _write_const_ints_to_tensor(mean1_axes_tensor, [int(v) for v in mapped_axes1])
                    or _read_const_ints_from_tensor(mean1_axes_tensor) == [int(v) for v in mapped_axes1]
                ):
                    continue

                hs0_add_op = model_ir.operators[int(hs0_match["add_idx"])]
                hs0_add_inputs = [
                    str(source_nhwc_name) if str(v) == str(source_nchw_name) else str(v)
                    for v in list(hs0_add_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=hs0_add_op,
                    new_inputs=hs0_add_inputs,
                )
                hs0_mul_op = model_ir.operators[int(hs0_match["mul_idx"])]
                hs0_mul_inputs = [
                    str(source_nhwc_name) if str(v) == str(source_nchw_name) else str(v)
                    for v in list(hs0_mul_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=hs0_mul_op,
                    new_inputs=hs0_mul_inputs,
                )

                conv_inputs = [str(v) for v in list(conv_op.inputs)]
                conv_inputs[0] = str(mean0_out_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=conv_op,
                    new_inputs=conv_inputs,
                )

                hs1_add_op = model_ir.operators[int(hs1_match["add_idx"])]
                hs1_add_inputs = [
                    str(conv_out_name) if str(v) == str(gate_nchw_name) else str(v)
                    for v in list(hs1_add_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=hs1_add_op,
                    new_inputs=hs1_add_inputs,
                )
                hs1_mul_op = model_ir.operators[int(hs1_match["mul_idx"])]
                hs1_mul_inputs = [
                    str(conv_out_name) if str(v) == str(gate_nchw_name) else str(v)
                    for v in list(hs1_mul_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=hs1_mul_op,
                    new_inputs=hs1_mul_inputs,
                )

                for tensor_name in [
                    str(hs0_match["add_out_name"]),
                    str(hs0_match["scale_out_name"]),
                    str(hs0_match["mul_out_name"]),
                    str(mean0_out_name),
                    str(hs1_match["add_out_name"]),
                    str(hs1_match["scale_out_name"]),
                    str(hs1_match["mul_out_name"]),
                    str(mean1_out_name),
                ]:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(tensor_name), None),
                        perm_nchw_to_nhwc,
                    )

                remove_indices = {int(mean0_post_idx), int(pre1_idx)}
                pre_remaining_users = [
                    int(v)
                    for v in consumers.get(source_nchw_name, [])
                    if int(v) not in {int(hs0_match["add_idx"]), int(hs0_match["mul_idx"])}
                ]
                if len(pre_remaining_users) == 0:
                    remove_indices.add(int(pre_idx))
                for remove_idx in sorted(remove_indices, reverse=True):
                    del model_ir.operators[int(remove_idx)]

                rewritten += 1
                changed = True
                break

        if changed:
            # This legacy rewrite still performs several batched input/output
            # mutations and optional adapter insertion. Refresh once after the
            # completed transaction iteration instead of rebuilding producer
            # and consumer maps independently at the next scan.
            graph_index.refresh()
        else:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_conv_attention_nhwc_propagation_chains": int(rewritten)}


def run_conv_attention_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run generic Conv-attention NHWC propagation transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        has_transpose = False
        has_mean = False
        has_conv = False
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            has_transpose = has_transpose or op_type == "TRANSPOSE"
            has_mean = has_mean or op_type == "MEAN"
            has_conv = has_conv or op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
            if has_transpose and has_mean and has_conv:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return bool(_preflight(pass_state.model_ir).matched)

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_conv_attention_nhwc_propagation_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_conv_attention_nhwc_propagation_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.conv_attention_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_conv_attention_nhwc_propagation_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_csp_attention_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate RTMDet-like CSP attention NCHW/NHWC bridge chains.

    Strict target motif (rank-4):
      short_nhwc --T(0,3,1,2)--> short_nchw --LOGISTIC--> s_sig --MUL(short_nchw, s_sig)--> short_branch
      main_nhwc  --T(0,3,1,2)--> main_nchw
      point_nhwc --T(0,3,1,2)--> point_nchw --LOGISTIC--> p_sig --MUL(point_nchw, p_sig)--> point_branch
      ADD(main_nchw, point_branch) -> add_nchw
      CONCAT(axis=1, [add_nchw, short_branch]) -> feat_nchw
      MEAN(keepDims=True, axes=[2,3]) -> gap_nchw --T(0,2,3,1)--> gap_nhwc --CONV_2D--> gate_nhwc
      gate_nhwc --T(0,3,1,2)--> gate_in_nchw --(LOGISTIC|HardSigmoid expansion)--> gate_nchw
      MUL(feat_nchw, gate_nchw) -> out_nchw --T(0,2,3,1)--> out_nhwc

    Rewrite:
      - keep all branch tensors in NHWC
      - remap CONCAT axis to NHWC channel axis and MEAN axes to NHWC
      - remove bridge transposes in the motif
      - preserve post-transpose output tensor names
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _is_singleton_spatial_nchw_to_nhwc_reshape(
        *,
        input_name: str,
        output_name: str,
    ) -> bool:
        in_tensor = model_ir.tensors.get(str(input_name), None)
        out_tensor = model_ir.tensors.get(str(output_name), None)
        if in_tensor is None or out_tensor is None:
            return False
        in_shape = [int(v) for v in list(in_tensor.shape)]
        out_shape = [int(v) for v in list(out_tensor.shape)]
        if len(in_shape) != 4 or len(out_shape) != 4:
            return False
        if any(int(v) < 0 for v in list(in_shape) + list(out_shape)):
            return False
        return (
            int(in_shape[0]) == int(out_shape[0])
            and int(in_shape[1]) == int(out_shape[3])
            and int(in_shape[2]) == 1
            and int(in_shape[3]) == 1
            and int(out_shape[1]) == 1
            and int(out_shape[2]) == 1
        )

    def _is_singleton_spatial_nhwc_to_nchw_reshape(
        *,
        input_name: str,
        output_name: str,
    ) -> bool:
        in_tensor = model_ir.tensors.get(str(input_name), None)
        out_tensor = model_ir.tensors.get(str(output_name), None)
        if in_tensor is None or out_tensor is None:
            return False
        in_shape = [int(v) for v in list(in_tensor.shape)]
        out_shape = [int(v) for v in list(out_tensor.shape)]
        if len(in_shape) != 4 or len(out_shape) != 4:
            return False
        if any(int(v) < 0 for v in list(in_shape) + list(out_shape)):
            return False
        return (
            int(in_shape[0]) == int(out_shape[0])
            and int(in_shape[1]) == 1
            and int(in_shape[2]) == 1
            and int(in_shape[3]) == int(out_shape[1])
            and int(out_shape[2]) == 1
            and int(out_shape[3]) == 1
        )

    def _match_sigmoid_self_mul_from_transpose_output(
        branch_output_name: str,
        consumers: Dict[str, List[int]],
        producers: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        mul_idx = producers.get(str(branch_output_name), None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
            or str(mul_op.outputs[0]) != str(branch_output_name)
        ):
            return None

        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        for candidate_input_idx in [0, 1]:
            transpose_output_name = str(mul_inputs[int(candidate_input_idx)])
            sigmoid_output_name = str(mul_inputs[1 - int(candidate_input_idx)])

            pre_idx = producers.get(transpose_output_name, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != transpose_output_name
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue

            sig_idx = producers.get(sigmoid_output_name, None)
            if sig_idx is None:
                continue
            sig_op = model_ir.operators[int(sig_idx)]
            if (
                str(sig_op.op_type) != "LOGISTIC"
                or len(sig_op.inputs) != 1
                or len(sig_op.outputs) != 1
                or str(sig_op.inputs[0]) != transpose_output_name
                or str(sig_op.outputs[0]) != sigmoid_output_name
            ):
                continue

            if set(int(v) for v in consumers.get(sigmoid_output_name, [])) != {int(mul_idx)}:
                continue
            if set(int(v) for v in consumers.get(transpose_output_name, [])) != {int(sig_idx), int(mul_idx)}:
                continue

            return {
                "transpose_idx": int(pre_idx),
                "transpose_output_name": str(transpose_output_name),
                "source_nhwc_name": str(pre_op.inputs[0]),
                "sigmoid_idx": int(sig_idx),
                "sigmoid_output_name": str(sigmoid_output_name),
                "mul_idx": int(mul_idx),
                "branch_output_name": str(branch_output_name),
            }
        return None

    def _match_hardsigmoid_gate_from_output(
        gate_output_name: str,
        consumers: Dict[str, List[int]],
        producers: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        gate_idx = producers.get(str(gate_output_name), None)
        if gate_idx is None:
            return None
        gate_op = model_ir.operators[int(gate_idx)]
        gate_type = str(gate_op.op_type)
        add_output_name: Optional[str] = None
        chain_indices: List[int] = []
        metadata_names: List[str] = []

        if gate_type == "LOGISTIC":
            if len(gate_op.inputs) != 1 or len(gate_op.outputs) != 1:
                return None
            return {
                "head_idx": int(gate_idx),
                "head_input_name": str(gate_op.inputs[0]),
                "head_input_index": 0,
                "output_name": str(gate_output_name),
                "chain_indices": [int(gate_idx)],
                "metadata_names": [str(gate_output_name)],
            }

        if gate_type == "RELU_0_TO_1":
            if len(gate_op.inputs) != 1 or len(gate_op.outputs) != 1:
                return None
            add_output_name = str(gate_op.inputs[0])
            if set(int(v) for v in consumers.get(add_output_name, [])) != {int(gate_idx)}:
                return None
            chain_indices.append(int(gate_idx))
            metadata_names.append(str(gate_output_name))
        elif gate_type == "MINIMUM":
            if len(gate_op.inputs) != 2 or len(gate_op.outputs) != 1:
                return None
            min_inputs = [str(v) for v in list(gate_op.inputs)]
            max_output_name: Optional[str] = None
            min_side_name: Optional[str] = None
            for candidate_name, other_name in [(min_inputs[0], min_inputs[1]), (min_inputs[1], min_inputs[0])]:
                producer_idx = producers.get(candidate_name, None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if str(producer_op.op_type) != "MAXIMUM":
                    continue
                max_output_name = str(candidate_name)
                min_side_name = str(other_name)
                break
            if max_output_name is None or min_side_name is None:
                return None
            if not _is_singleton_constant_tensor(model_ir, min_side_name):
                return None
            if set(int(v) for v in consumers.get(max_output_name, [])) != {int(gate_idx)}:
                return None

            max_idx = producers.get(max_output_name, None)
            if max_idx is None:
                return None
            max_op = model_ir.operators[int(max_idx)]
            if str(max_op.op_type) != "MAXIMUM" or len(max_op.inputs) != 2 or len(max_op.outputs) != 1:
                return None
            max_inputs = [str(v) for v in list(max_op.inputs)]
            for candidate_name, other_name in [(max_inputs[0], max_inputs[1]), (max_inputs[1], max_inputs[0])]:
                producer_idx = producers.get(candidate_name, None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if str(producer_op.op_type) != "ADD":
                    continue
                add_output_name = str(candidate_name)
                if not _is_singleton_constant_tensor(model_ir, str(other_name)):
                    return None
                break
            if add_output_name is None:
                return None
            if set(int(v) for v in consumers.get(add_output_name, [])) != {int(max_idx)}:
                return None
            chain_indices.extend([int(max_idx), int(gate_idx)])
            metadata_names.extend([str(max_output_name), str(gate_output_name)])
        else:
            return None

        add_idx = producers.get(str(add_output_name), None)
        if add_idx is None:
            return None
        add_op = model_ir.operators[int(add_idx)]
        if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
            return None
        add_inputs = [str(v) for v in list(add_op.inputs)]
        mul_output_name: Optional[str] = None
        for candidate_name, other_name in [(add_inputs[0], add_inputs[1]), (add_inputs[1], add_inputs[0])]:
            producer_idx = producers.get(candidate_name, None)
            if producer_idx is None:
                continue
            producer_op = model_ir.operators[int(producer_idx)]
            if str(producer_op.op_type) != "MUL":
                continue
            if not _is_singleton_constant_tensor(model_ir, str(other_name)):
                return None
            mul_output_name = str(candidate_name)
            break
        if mul_output_name is None:
            return None
        if set(int(v) for v in consumers.get(mul_output_name, [])) != {int(add_idx)}:
            return None
        chain_indices.append(int(add_idx))
        metadata_names.insert(0, str(add_output_name))

        mul_idx = producers.get(mul_output_name, None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
            return None
        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        main_input_name: Optional[str] = None
        main_input_index: Optional[int] = None
        for input_index, input_name in enumerate(mul_inputs):
            side_name = str(mul_inputs[1 - input_index])
            if _is_singleton_constant_tensor(model_ir, side_name):
                main_input_name = str(input_name)
                main_input_index = int(input_index)
                break
        if main_input_name is None or main_input_index is None:
            return None

        chain_indices.append(int(mul_idx))
        metadata_names.insert(0, str(mul_output_name))
        return {
            "head_idx": int(mul_idx),
            "head_input_name": str(main_input_name),
            "head_input_index": int(main_input_index),
            "output_name": str(gate_output_name),
            "chain_indices": [int(v) for v in chain_indices],
            "metadata_names": [str(v) for v in metadata_names],
        }

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for candidate_post_idx, candidate_post_op in enumerate(model_ir.operators):
            if (
                str(candidate_post_op.op_type) != "TRANSPOSE"
                or len(candidate_post_op.inputs) < 2
                or len(candidate_post_op.outputs) != 1
                or _read_transpose_perm(model_ir, candidate_post_op) != perm_nchw_to_nhwc
            ):
                continue

            mul_output_name = str(candidate_post_op.inputs[0])
            if mul_output_name in model_outputs:
                continue
            mul_idx = producers.get(mul_output_name, None)
            if mul_idx is None:
                continue
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            if str(mul_op.outputs[0]) != mul_output_name:
                continue

            mul_output_users = [int(v) for v in consumers.get(mul_output_name, [])]
            post_pairs: List[Tuple[int, OperatorIR]] = []
            valid_posts = len(mul_output_users) > 0
            for user_idx in mul_output_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != mul_output_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_posts = False
                    break
                post_pairs.append((int(user_idx), user_op))
            if not valid_posts or len(post_pairs) == 0:
                continue
            post_pairs = sorted(post_pairs, key=lambda v: int(v[0]))
            post_indices = [int(v[0]) for v in post_pairs]
            post_output_names = [str(v[1].outputs[0]) for v in post_pairs]

            concat_name: Optional[str] = None
            gate_output_name: Optional[str] = None
            input0_name = str(mul_op.inputs[0])
            input1_name = str(mul_op.inputs[1])
            for lhs_name, rhs_name in [(input0_name, input1_name), (input1_name, input0_name)]:
                lhs_prod_idx = producers.get(lhs_name, None)
                if lhs_prod_idx is None:
                    continue
                lhs_prod_op = model_ir.operators[int(lhs_prod_idx)]
                if str(lhs_prod_op.op_type) != "CONCATENATION":
                    continue
                concat_name = str(lhs_name)
                gate_output_name = str(rhs_name)
                break
            if concat_name is None or gate_output_name is None:
                continue

            concat_idx = producers.get(concat_name, None)
            if concat_idx is None:
                continue
            concat_op = model_ir.operators[int(concat_idx)]
            if (
                str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.inputs) != 2
                or len(concat_op.outputs) != 1
                or str(concat_op.outputs[0]) != concat_name
                or int(concat_op.options.get("axis", 1)) != 1
                or concat_name in model_outputs
            ):
                continue

            gate_match = _match_hardsigmoid_gate_from_output(
                gate_output_name=gate_output_name,
                consumers=consumers,
                producers=producers,
            )
            if gate_match is None:
                continue
            gate_head_idx = int(gate_match["head_idx"])
            gate_head_input_name = str(gate_match["head_input_name"])
            gate_head_input_index = int(gate_match["head_input_index"])
            gate_metadata_names = [str(v) for v in list(gate_match.get("metadata_names", []))]
            gate_head_op = model_ir.operators[int(gate_head_idx)]
            if gate_head_input_name in model_outputs or gate_output_name in model_outputs:
                continue

            pre_gate_idx = producers.get(gate_head_input_name, None)
            if pre_gate_idx is None:
                continue
            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
            gate_conv_output_name: Optional[str] = None
            if (
                str(pre_gate_op.op_type) == "TRANSPOSE"
                and len(pre_gate_op.inputs) >= 2
                and len(pre_gate_op.outputs) == 1
                and str(pre_gate_op.outputs[0]) == gate_head_input_name
                and _read_transpose_perm(model_ir, pre_gate_op) == perm_nhwc_to_nchw
            ):
                gate_conv_output_name = str(pre_gate_op.inputs[0])
            elif (
                str(pre_gate_op.op_type) == "RESHAPE"
                and len(pre_gate_op.inputs) >= 2
                and len(pre_gate_op.outputs) == 1
                and str(pre_gate_op.outputs[0]) == gate_head_input_name
                and _is_singleton_spatial_nhwc_to_nchw_reshape(
                    input_name=str(pre_gate_op.inputs[0]),
                    output_name=str(pre_gate_op.outputs[0]),
                )
            ):
                gate_conv_output_name = str(pre_gate_op.inputs[0])
            if gate_conv_output_name is None:
                continue
            if gate_conv_output_name in model_outputs:
                continue

            concat_users = set(int(v) for v in consumers.get(concat_name, []))
            if concat_users != {int(mul_idx)} and len(concat_users) != 2:
                continue
            mean_idx_candidates = [int(v) for v in concat_users if int(v) != int(mul_idx)]
            if len(mean_idx_candidates) != 1:
                continue
            mean_idx = int(mean_idx_candidates[0])
            mean_op = model_ir.operators[int(mean_idx)]
            if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                continue
            if str(mean_op.inputs[0]) != concat_name or not bool(mean_op.options.get("keepDims", False)):
                continue
            mean_output_name = str(mean_op.outputs[0])
            if mean_output_name in model_outputs:
                continue

            mean_post_users = [int(v) for v in consumers.get(mean_output_name, [])]
            if len(mean_post_users) != 1:
                continue
            mean_post_idx = int(mean_post_users[0])
            mean_post_op = model_ir.operators[int(mean_post_idx)]
            conv_input_name: Optional[str] = None
            if (
                str(mean_post_op.op_type) == "TRANSPOSE"
                and len(mean_post_op.inputs) >= 2
                and len(mean_post_op.outputs) == 1
                and str(mean_post_op.inputs[0]) == mean_output_name
                and _read_transpose_perm(model_ir, mean_post_op) == perm_nchw_to_nhwc
            ):
                conv_input_name = str(mean_post_op.outputs[0])
            elif (
                str(mean_post_op.op_type) == "RESHAPE"
                and len(mean_post_op.inputs) >= 2
                and len(mean_post_op.outputs) == 1
                and str(mean_post_op.inputs[0]) == mean_output_name
                and _is_singleton_spatial_nchw_to_nhwc_reshape(
                    input_name=str(mean_post_op.inputs[0]),
                    output_name=str(mean_post_op.outputs[0]),
                )
            ):
                conv_input_name = str(mean_post_op.outputs[0])
            if conv_input_name is None:
                continue
            if conv_input_name in model_outputs:
                continue

            conv_idx_candidates = [int(v) for v in consumers.get(conv_input_name, [])]
            if len(conv_idx_candidates) != 1:
                continue
            conv_idx = int(conv_idx_candidates[0])
            conv_op = model_ir.operators[int(conv_idx)]
            if (
                str(conv_op.op_type) != "CONV_2D"
                or len(conv_op.inputs) < 1
                or len(conv_op.outputs) != 1
                or str(conv_op.inputs[0]) != conv_input_name
            ):
                continue
            if str(conv_op.outputs[0]) != gate_conv_output_name:
                continue

            concat_input_names = [str(v) for v in list(concat_op.inputs)]
            use_add_main_path = False
            add_idx: Optional[int] = None
            add_op: Optional[OperatorIR] = None
            add_output_name: Optional[str] = None
            main_pre_idx: Optional[int] = None
            main_transpose_output_name: Optional[str] = None
            main_source_nhwc_name: Optional[str] = None
            point_branch_output_name: Optional[str] = None
            short_branch_output_name: Optional[str] = None

            # Variant A: CONCAT(ADD(main_nchw, point_branch_nchw), short_branch_nchw)
            for a_name, b_name in [(concat_input_names[0], concat_input_names[1]), (concat_input_names[1], concat_input_names[0])]:
                add_candidate_idx = producers.get(a_name, None)
                if add_candidate_idx is None:
                    continue
                add_candidate_op = model_ir.operators[int(add_candidate_idx)]
                if str(add_candidate_op.op_type) != "ADD":
                    continue
                add_input_names = [str(v) for v in list(add_candidate_op.inputs)]
                for m_name, p_name in [(add_input_names[0], add_input_names[1]), (add_input_names[1], add_input_names[0])]:
                    pre_idx = producers.get(m_name, None)
                    if pre_idx is None:
                        continue
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != m_name
                        or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                    ):
                        continue
                    add_idx = int(add_candidate_idx)
                    add_op = add_candidate_op
                    add_output_name = str(a_name)
                    short_branch_output_name = str(b_name)
                    main_pre_idx = int(pre_idx)
                    main_transpose_output_name = str(m_name)
                    main_source_nhwc_name = str(pre_op.inputs[0])
                    point_branch_output_name = str(p_name)
                    use_add_main_path = True
                    break
                if use_add_main_path:
                    break

            # Variant B: CONCAT(point_branch_nchw, short_branch_nchw)
            if not use_add_main_path:
                point_branch_output_name = str(concat_input_names[0])
                short_branch_output_name = str(concat_input_names[1])

            if point_branch_output_name is None or short_branch_output_name is None:
                continue

            if use_add_main_path:
                if (
                    add_idx is None
                    or add_op is None
                    or add_output_name is None
                    or main_pre_idx is None
                    or main_transpose_output_name is None
                    or main_source_nhwc_name is None
                ):
                    continue
                if (
                    str(add_op.op_type) != "ADD"
                    or len(add_op.inputs) != 2
                    or len(add_op.outputs) != 1
                    or str(add_op.outputs[0]) != add_output_name
                    or add_output_name in model_outputs
                    or main_source_nhwc_name in model_outputs
                ):
                    continue

            point_match = _match_sigmoid_self_mul_from_transpose_output(
                branch_output_name=str(point_branch_output_name),
                consumers=consumers,
                producers=producers,
            )
            short_match = _match_sigmoid_self_mul_from_transpose_output(
                branch_output_name=str(short_branch_output_name),
                consumers=consumers,
                producers=producers,
            )
            if point_match is None or short_match is None:
                # Try swapped interpretation for variant-B.
                if not use_add_main_path:
                    point_match = _match_sigmoid_self_mul_from_transpose_output(
                        branch_output_name=str(short_branch_output_name),
                        consumers=consumers,
                        producers=producers,
                    )
                    short_match = _match_sigmoid_self_mul_from_transpose_output(
                        branch_output_name=str(point_branch_output_name),
                        consumers=consumers,
                        producers=producers,
                    )
                    if point_match is not None and short_match is not None:
                        point_branch_output_name, short_branch_output_name = (
                            str(short_branch_output_name),
                            str(point_branch_output_name),
                        )
                if point_match is None or short_match is None:
                    continue

            if int(point_match["transpose_idx"]) == int(short_match["transpose_idx"]):
                continue
            if use_add_main_path and (
                int(point_match["transpose_idx"]) == int(main_pre_idx)
                or int(short_match["transpose_idx"]) == int(main_pre_idx)
            ):
                continue

            if use_add_main_path:
                if set(int(v) for v in consumers.get(str(main_transpose_output_name), [])) != {int(add_idx)}:
                    continue
                if set(int(v) for v in consumers.get(str(point_branch_output_name), [])) != {int(add_idx)}:
                    continue
                if set(int(v) for v in consumers.get(str(short_branch_output_name), [])) != {int(concat_idx)}:
                    continue
            else:
                if set(int(v) for v in consumers.get(str(point_branch_output_name), [])) != {int(concat_idx)}:
                    continue
                if set(int(v) for v in consumers.get(str(short_branch_output_name), [])) != {int(concat_idx)}:
                    continue
            if set(int(v) for v in consumers.get(mean_output_name, [])) != {int(mean_post_idx)}:
                continue
            if set(int(v) for v in consumers.get(conv_input_name, [])) != {int(conv_idx)}:
                continue
            if set(int(v) for v in consumers.get(gate_conv_output_name, [])) != {int(pre_gate_idx)}:
                continue
            if set(int(v) for v in consumers.get(gate_head_input_name, [])) != {int(gate_head_idx)}:
                continue
            if set(int(v) for v in consumers.get(gate_output_name, [])) != {int(mul_idx)}:
                continue

            mean_axes_tensor = model_ir.tensors.get(str(mean_op.inputs[1]), None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None or len(mean_axes_vals) == 0:
                continue
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes_vals:
                a = int(axis)
                if a < 0:
                    a += 4
                if a < 0 or a >= 4:
                    valid_axes = False
                    break
                normalized_axes.append(int(a))
            if not valid_axes:
                continue
            mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]
            if not _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes]):
                continue

            short_sig_op = model_ir.operators[int(short_match["sigmoid_idx"])]
            short_mul_op = model_ir.operators[int(short_match["mul_idx"])]
            _set_operator_inputs(
                model_ir=model_ir,
                op=short_sig_op,
                new_inputs=[str(short_match["source_nhwc_name"])],
            )
            short_mul_inputs = [
                str(short_match["source_nhwc_name"])
                if str(v) == str(short_match["transpose_output_name"])
                else str(v)
                for v in list(short_mul_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=short_mul_op,
                new_inputs=short_mul_inputs,
            )

            point_sig_op = model_ir.operators[int(point_match["sigmoid_idx"])]
            point_mul_op = model_ir.operators[int(point_match["mul_idx"])]
            _set_operator_inputs(
                model_ir=model_ir,
                op=point_sig_op,
                new_inputs=[str(point_match["source_nhwc_name"])],
            )
            point_mul_inputs = [
                str(point_match["source_nhwc_name"])
                if str(v) == str(point_match["transpose_output_name"])
                else str(v)
                for v in list(point_mul_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=point_mul_op,
                new_inputs=point_mul_inputs,
            )

            if use_add_main_path and add_op is not None:
                add_inputs = [
                    str(main_source_nhwc_name)
                    if str(v) == str(main_transpose_output_name)
                    else str(v)
                    for v in list(add_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_inputs=add_inputs,
                )

            concat_op.options = dict(concat_op.options) if isinstance(concat_op.options, dict) else {}
            concat_op.options["axis"] = 3

            conv_inputs = [str(v) for v in list(conv_op.inputs)]
            conv_inputs[0] = str(mean_output_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv_op,
                new_inputs=conv_inputs,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=gate_head_op,
                input_index=int(gate_head_input_index),
                new_input_name=str(gate_conv_output_name),
            )

            old_mul_tensor = model_ir.tensors.get(mul_output_name, None)
            old_mul_shape = (
                [int(v) for v in list(old_mul_tensor.shape)]
                if old_mul_tensor is not None and old_mul_tensor.shape is not None
                else None
            )
            old_mul_signature = (
                [int(v) for v in list(old_mul_tensor.shape_signature)]
                if old_mul_tensor is not None and old_mul_tensor.shape_signature is not None
                else (
                    [int(v) for v in list(old_mul_tensor.shape)]
                    if old_mul_tensor is not None and old_mul_tensor.shape is not None
                    else None
                )
            )

            metadata_targets = [
                str(short_match["sigmoid_output_name"]),
                str(short_match["branch_output_name"]),
                str(point_match["sigmoid_output_name"]),
                str(point_match["branch_output_name"]),
                str(concat_name),
                str(mean_output_name),
                *[str(v) for v in gate_metadata_names],
                str(mul_output_name),
            ]
            if use_add_main_path and add_output_name is not None:
                metadata_targets.insert(4, str(add_output_name))
            for tensor_name in list(dict.fromkeys(metadata_targets)):
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[canonical_post_output_name],
            )
            _replace_tensor_inputs(model_ir, mul_output_name, canonical_post_output_name)
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name)

            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if canonical_tensor is not None and old_mul_tensor is not None:
                canonical_tensor.dtype = str(old_mul_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                if old_mul_shape is not None:
                    canonical_tensor.shape = [int(v) for v in list(old_mul_shape)]
                if old_mul_signature is not None:
                    canonical_tensor.shape_signature = [int(v) for v in list(old_mul_signature)]
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {
                int(short_match["transpose_idx"]),
                int(point_match["transpose_idx"]),
                int(mean_post_idx),
                int(pre_gate_idx),
                *[int(v) for v in post_indices],
            }
            if use_add_main_path and main_pre_idx is not None:
                remove_indices.add(int(main_pre_idx))
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_csp_attention_nhwc_chains": int(rewritten)}
