from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


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
