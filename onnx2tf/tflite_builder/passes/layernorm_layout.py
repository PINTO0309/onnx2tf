from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _invert_perm,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR

def _optimize_transpose_layernorm_stats_nhwc_propagation_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Re-propagate NHWC through LayerNorm-statistics branches after token-mixing residual paths.

    Target (rank-4):
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      m1 = MEAN(x_nchw, axes=[1], keepDims=True)
      d  = SUB(x_nchw, m1)
      sq = MUL(d, d)
      v  = MEAN(sq, axes=[1], keepDims=True)

    Rewrite:
      m1 = MEAN(x_nhwc, axes=[3], keepDims=True)
      d  = SUB(x_nhwc, m1)
      v  = MEAN(sq, axes=[3], keepDims=True)

    Notes:
    - Pre-transpose is removed only when it became dead.
    - This pass focuses on the statistics path and keeps other residual users intact.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _map_mean_axes_to_nhwc(
        *,
        axes_tensor: Optional[TensorIR],
        rank: int,
    ) -> Optional[List[int]]:
        if axes_tensor is None:
            return None
        axes_vals = _read_const_ints_from_tensor(axes_tensor)
        if axes_vals is None or len(axes_vals) == 0:
            return None
        normalized_axes: List[int] = []
        for axis in axes_vals:
            a = int(axis)
            if a < 0:
                a += int(rank)
            if a < 0 or a >= int(rank):
                return None
            normalized_axes.append(int(a))
        return [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
            rank = int(len(list(pre_input_tensor.shape))) if pre_input_tensor is not None and pre_input_tensor.shape is not None else 4
            if int(rank) != 4:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) == 0:
                continue

            mean1_idx: Optional[int] = None
            sub_idx: Optional[int] = None

            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    mean1_idx is None
                    and str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == pre_output_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean1_idx = int(user_idx)
                    continue
                if (
                    sub_idx is None
                    and str(user_op.op_type) == "SUB"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and pre_output_name in {str(user_op.inputs[0]), str(user_op.inputs[1])}
                ):
                    sub_idx = int(user_idx)

            if mean1_idx is None or sub_idx is None:
                continue

            mean1_op = model_ir.operators[int(mean1_idx)]
            sub_op = model_ir.operators[int(sub_idx)]
            mean1_out_name = str(mean1_op.outputs[0])
            if mean1_out_name in model_outputs:
                continue
            if mean1_out_name not in {str(sub_op.inputs[0]), str(sub_op.inputs[1])}:
                continue

            mean1_axes_name = str(mean1_op.inputs[1])
            mean1_axes_tensor = model_ir.tensors.get(mean1_axes_name, None)
            mapped_axes1 = _map_mean_axes_to_nhwc(
                axes_tensor=mean1_axes_tensor,
                rank=rank,
            )
            if mapped_axes1 is None:
                continue

            sub_out_name = str(sub_op.outputs[0])
            if sub_out_name in model_outputs:
                continue
            sub_users = [int(v) for v in consumers.get(sub_out_name, [])]
            if len(sub_users) == 0:
                continue

            sq_idx: Optional[int] = None
            for user_idx in sub_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == sub_out_name
                    and str(user_op.inputs[1]) == sub_out_name
                ):
                    sq_idx = int(user_idx)
                    break
            if sq_idx is None:
                continue
            # Only rewrite variance-statistics branches.
            # If SUB output feeds non-square branches (e.g. InstanceNorm affine path),
            # keep original layout to avoid breaking downstream broadcast semantics.
            sub_other_users = [
                int(v) for v in sub_users if int(v) != int(sq_idx)
            ]
            if len(sub_other_users) > 0:
                continue

            sq_op = model_ir.operators[int(sq_idx)]
            sq_out_name = str(sq_op.outputs[0])
            if sq_out_name in model_outputs:
                continue

            sq_users = [int(v) for v in consumers.get(sq_out_name, [])]
            mean2_idx: Optional[int] = None
            for user_idx in sq_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == sq_out_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean2_idx = int(user_idx)
                    break
            if mean2_idx is None:
                continue

            mean2_op = model_ir.operators[int(mean2_idx)]
            mean2_out_name = str(mean2_op.outputs[0])
            if mean2_out_name in model_outputs:
                continue

            mean2_axes_name = str(mean2_op.inputs[1])
            mean2_axes_tensor = model_ir.tensors.get(mean2_axes_name, None)
            mapped_axes2 = _map_mean_axes_to_nhwc(
                axes_tensor=mean2_axes_tensor,
                rank=rank,
            )
            if mapped_axes2 is None:
                continue

            _write_const_ints_to_tensor(mean1_axes_tensor, [int(v) for v in mapped_axes1])
            _set_operator_inputs(
                model_ir=model_ir,
                op=mean1_op,
                new_inputs=[pre_input_name, mean1_axes_name],
            )

            new_sub_inputs = [
                pre_input_name if str(v) == pre_output_name else str(v)
                for v in list(sub_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=sub_op,
                new_inputs=new_sub_inputs,
            )

            _write_const_ints_to_tensor(mean2_axes_tensor, [int(v) for v in mapped_axes2])
            _set_operator_inputs(
                model_ir=model_ir,
                op=mean2_op,
                new_inputs=[sq_out_name, mean2_axes_name],
            )

            for tensor_name in [
                mean1_out_name,
                sub_out_name,
                sq_out_name,
                mean2_out_name,
            ]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            pre_remaining_users = [
                int(v) for v in pre_users if int(v) not in {int(mean1_idx), int(sub_idx)}
            ]
            if len(pre_remaining_users) == 0:
                del model_ir.operators[int(pre_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_layernorm_stats_nhwc_propagation_chains": int(rewritten)}

def _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Re-propagate NHWC through LayerNorm statistics when an NCHW tensor already has
    an existing post-transpose (NCHW->NHWC) projection.

    Target:
      x_nchw --TRANSPOSE(0,2,3,1)--> x_nhwc
      m1 = MEAN(x_nchw, axes=[1], keepDims=True)
      d  = SUB(x_nchw, m1)
      sq = MUL(d, d)
      v  = MEAN(sq, axes=[1], keepDims=True)

    Rewrite:
      m1 = MEAN(x_nhwc, axes=[3], keepDims=True)
      d  = SUB(x_nhwc, m1)
      v  = MEAN(sq, axes=[3], keepDims=True)
    """
    rewritten = 0
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    axis_map_nchw_to_nhwc = _invert_perm(perm_nchw_to_nhwc)
    if axis_map_nchw_to_nhwc is None:
        return {"optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains": 0}

    def _map_axes(
        *,
        axes_tensor: Optional[TensorIR],
        rank: int,
    ) -> Optional[List[int]]:
        if axes_tensor is None:
            return None
        axes_vals = _read_const_ints_from_tensor(axes_tensor)
        if axes_vals is None or len(axes_vals) == 0:
            return None
        normalized_axes: List[int] = []
        for axis in axes_vals:
            a = int(axis)
            if a < 0:
                a += int(rank)
            if a < 0 or a >= int(rank):
                return None
            normalized_axes.append(int(a))
        return [int(axis_map_nchw_to_nhwc[int(axis)]) for axis in normalized_axes]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            x_nchw_name = str(post_op.inputs[0])
            x_nhwc_name = str(post_op.outputs[0])
            if x_nchw_name in model_outputs:
                continue

            x_tensor = model_ir.tensors.get(x_nchw_name, None)
            rank = int(len(list(x_tensor.shape))) if x_tensor is not None and x_tensor.shape is not None else 4
            if int(rank) != 4:
                continue

            x_users = [int(v) for v in consumers.get(x_nchw_name, [])]
            if len(x_users) == 0:
                continue

            mean1_idx: Optional[int] = None
            sub_idx: Optional[int] = None
            for user_idx in x_users:
                if int(user_idx) == int(post_idx):
                    continue
                user_op = model_ir.operators[int(user_idx)]
                if (
                    mean1_idx is None
                    and str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == x_nchw_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean1_idx = int(user_idx)
                    continue
                if (
                    sub_idx is None
                    and str(user_op.op_type) == "SUB"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and x_nchw_name in {str(user_op.inputs[0]), str(user_op.inputs[1])}
                ):
                    sub_idx = int(user_idx)

            if mean1_idx is None or sub_idx is None:
                continue

            mean1_op = model_ir.operators[int(mean1_idx)]
            sub_op = model_ir.operators[int(sub_idx)]
            mean1_out_name = str(mean1_op.outputs[0])
            if mean1_out_name in model_outputs:
                continue
            if mean1_out_name not in {str(sub_op.inputs[0]), str(sub_op.inputs[1])}:
                continue

            mean1_axes_name = str(mean1_op.inputs[1])
            mean1_axes_tensor = model_ir.tensors.get(mean1_axes_name, None)
            mapped_axes1 = _map_axes(
                axes_tensor=mean1_axes_tensor,
                rank=rank,
            )
            if mapped_axes1 is None:
                continue

            sub_out_name = str(sub_op.outputs[0])
            if sub_out_name in model_outputs:
                continue
            sq_users = [int(v) for v in consumers.get(sub_out_name, [])]
            sq_idx: Optional[int] = None
            for user_idx in sq_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == sub_out_name
                    and str(user_op.inputs[1]) == sub_out_name
                ):
                    sq_idx = int(user_idx)
                    break
            if sq_idx is None:
                continue
            # Same safety guard as the transpose-driven variant above.
            sub_other_users = [
                int(v) for v in sq_users if int(v) != int(sq_idx)
            ]
            if len(sub_other_users) > 0:
                continue

            sq_op = model_ir.operators[int(sq_idx)]
            sq_out_name = str(sq_op.outputs[0])
            if sq_out_name in model_outputs:
                continue

            mean2_idx: Optional[int] = None
            for user_idx in consumers.get(sq_out_name, []):
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == sq_out_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean2_idx = int(user_idx)
                    break
            if mean2_idx is None:
                continue

            mean2_op = model_ir.operators[int(mean2_idx)]
            mean2_out_name = str(mean2_op.outputs[0])
            if mean2_out_name in model_outputs:
                continue

            mean2_axes_name = str(mean2_op.inputs[1])
            mean2_axes_tensor = model_ir.tensors.get(mean2_axes_name, None)
            mapped_axes2 = _map_axes(
                axes_tensor=mean2_axes_tensor,
                rank=rank,
            )
            if mapped_axes2 is None:
                continue

            _write_const_ints_to_tensor(mean1_axes_tensor, [int(v) for v in mapped_axes1])
            _set_operator_inputs(
                model_ir=model_ir,
                op=mean1_op,
                new_inputs=[x_nhwc_name, mean1_axes_name],
            )
            _write_const_ints_to_tensor(mean2_axes_tensor, [int(v) for v in mapped_axes2])
            _set_operator_inputs(
                model_ir=model_ir,
                op=mean2_op,
                new_inputs=[sq_out_name, mean2_axes_name],
            )

            new_sub_inputs = [
                x_nhwc_name if str(v) == x_nchw_name else str(v)
                for v in list(sub_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=sub_op,
                new_inputs=new_sub_inputs,
            )

            for tensor_name in [
                mean1_out_name,
                sub_out_name,
                sq_out_name,
                mean2_out_name,
            ]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains": int(rewritten)}
