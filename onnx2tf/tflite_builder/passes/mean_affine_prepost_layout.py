from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around TRANSPOSE->MEAN->MUL(const)->ADD(const)->TRANSPOSE chains.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --MEAN(axes_nchw, keepDims=True)--> m_nchw
      m_nchw --MUL(const)--> u_nchw --ADD(const)--> y_nchw
      y_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc

    Rewrite:
      x_nhwc --MEAN(axes_nhwc, keepDims=True)--> m_nhwc
      m_nhwc --MUL(const_nhwc)--> u_nhwc --ADD(const_nhwc)--> y_nhwc
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _rewrite_binary_const_to_nhwc(
        *,
        op_idx: int,
        side_input_index: int,
        side_input_name: str,
        target_broadcast_shape: Optional[List[int]],
        chain_index_set: set[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        side_tensor = model_ir.tensors.get(str(side_input_name), None)
        if side_tensor is None or side_tensor.data is None:
            return False
        side_data = np.asarray(side_tensor.data)
        if int(side_data.size) == 1:
            return True
        if side_data.ndim != 4:
            return False

        nhwc_data: Optional[np.ndarray] = None
        target_shape = (
            [int(v) for v in list(target_broadcast_shape)]
            if _is_fully_known_positive_shape(target_broadcast_shape)
            else None
        )
        if target_shape is not None:
            side_shape = [int(v) for v in list(side_data.shape)]
            if _broadcast_static_shapes(target_shape, side_shape) is not None:
                # Already NHWC-compatible. Keep idempotent across repeated sweeps.
                return True
            rotated = np.asarray(side_data)
            for _ in range(3):
                rotated = np.transpose(rotated, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_shape, rotated_shape) is not None:
                    nhwc_data = np.asarray(rotated)
                    break
            if nhwc_data is None:
                return False
        else:
            side_shape = [int(v) for v in list(side_data.shape)]
            if (
                len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 1
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            ):
                nhwc_data = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
            else:
                return True

        side_users = [int(v) for v in consumers.get(str(side_input_name), [])]
        shared_outside_chain = any(int(u) not in chain_index_set for u in side_users)

        target_name = str(side_input_name)
        if shared_outside_chain:
            target_name = _unique_tensor_name(f"{side_input_name}_nhwc")
            model_ir.tensors[target_name] = TensorIR(
                name=target_name,
                dtype=str(side_tensor.dtype),
                shape=[int(v) for v in list(nhwc_data.shape)],
                shape_signature=[int(v) for v in list(nhwc_data.shape)],
                data=np.asarray(nhwc_data),
                is_variable=False,
                quantization=_clone_quantization(side_tensor.quantization),
            )
        else:
            side_tensor.data = np.asarray(nhwc_data)
            side_tensor.shape = [int(v) for v in list(nhwc_data.shape)]
            side_tensor.shape_signature = [int(v) for v in list(nhwc_data.shape)]

        _replace_operator_input_at(
            model_ir=model_ir,
            op=model_ir.operators[int(op_idx)],
            input_index=int(side_input_index),
            new_input_name=str(target_name),
        )
        return True

    def _can_rewrite_binary_const_to_nhwc(
        *,
        side_input_name: str,
        target_broadcast_shape: Optional[List[int]],
    ) -> bool:
        side_tensor = model_ir.tensors.get(str(side_input_name), None)
        if side_tensor is None or side_tensor.data is None:
            return False
        side_data = np.asarray(side_tensor.data)
        if int(side_data.size) == 1:
            return True
        if side_data.ndim != 4:
            return False
        target_shape = (
            [int(v) for v in list(target_broadcast_shape)]
            if _is_fully_known_positive_shape(target_broadcast_shape)
            else None
        )
        if target_shape is None:
            return True
        side_shape = [int(v) for v in list(side_data.shape)]
        if _broadcast_static_shapes(target_shape, side_shape) is not None:
            return True
        rotated = np.asarray(side_data)
        for _ in range(3):
            rotated = np.transpose(rotated, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
            rotated_shape = [int(v) for v in list(rotated.shape)]
            if _broadcast_static_shapes(target_shape, rotated_shape) is not None:
                return True
        return False

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_input_name in model_outputs or pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) == 0:
                continue

            for mean_idx in pre_users:
                mean_op = model_ir.operators[int(mean_idx)]
                if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                    continue
                if str(mean_op.inputs[0]) != pre_output_name:
                    continue
                if not bool(mean_op.options.get("keepDims", False)):
                    continue

                mean_axes_name = str(mean_op.inputs[1])
                mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                mean_axes = _read_const_ints_from_tensor(mean_axes_tensor)
                if mean_axes is None or len(mean_axes) == 0:
                    continue

                mean_output_name = str(mean_op.outputs[0])
                if mean_output_name in model_outputs:
                    continue
                mul_users = [int(v) for v in consumers.get(mean_output_name, [])]
                if len(mul_users) != 1:
                    continue
                mul_idx = int(mul_users[0])
                mul_op = model_ir.operators[mul_idx]
                if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                    continue

                mul_in0 = str(mul_op.inputs[0])
                mul_in1 = str(mul_op.inputs[1])
                if mul_in0 == mean_output_name:
                    mul_side_input_index = 1
                    mul_side_name = mul_in1
                elif mul_in1 == mean_output_name:
                    mul_side_input_index = 0
                    mul_side_name = mul_in0
                else:
                    continue

                add_input_name = str(mul_op.outputs[0])
                if add_input_name in model_outputs:
                    continue
                add_users = [int(v) for v in consumers.get(add_input_name, [])]
                if len(add_users) != 1:
                    continue
                add_idx = int(add_users[0])
                add_op = model_ir.operators[add_idx]
                if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                    continue

                add_in0 = str(add_op.inputs[0])
                add_in1 = str(add_op.inputs[1])
                if add_in0 == add_input_name:
                    add_side_input_index = 1
                    add_side_name = add_in1
                elif add_in1 == add_input_name:
                    add_side_input_index = 0
                    add_side_name = add_in0
                else:
                    continue

                add_output_name = str(add_op.outputs[0])
                if add_output_name in model_outputs:
                    continue
                add_output_users = [int(v) for v in consumers.get(add_output_name, [])]
                if len(add_output_users) == 0:
                    continue

                post_indices: List[int] = []
                post_output_names: List[str] = []
                valid_posts = True
                for user_idx in add_output_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) == "TRANSPOSE"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == add_output_name
                        and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        post_indices.append(int(user_idx))
                        post_output_names.append(str(user_op.outputs[0]))
                    else:
                        valid_posts = False
                        break
                if not valid_posts or len(post_indices) == 0:
                    continue

                # Normalize NCHW-side axes and map them to NHWC-side axes.
                mapped_axes: List[int] = []
                valid_axes = True
                for axis in mean_axes:
                    a = int(axis)
                    if a < 0:
                        a += 4
                    if a < 0 or a >= 4:
                        valid_axes = False
                        break
                    # `perm_nchw_to_nhwc` lists old axes in new-axis order.
                    # Reduction axes need the inverse mapping (old axis -> new
                    # axis), which is the NHWC->NCHW permutation here.
                    mapped_axes.append(int(perm_nhwc_to_nchw[int(a)]))
                if not valid_axes or len(mapped_axes) == 0:
                    continue

                chain_index_set = {int(mean_idx), int(mul_idx), int(add_idx)}
                target_nhwc_shape = (
                    list(model_ir.tensors[pre_input_name].shape)
                    if pre_input_name in model_ir.tensors
                    else None
                )
                if not _can_rewrite_binary_const_to_nhwc(
                    side_input_name=str(mul_side_name),
                    target_broadcast_shape=target_nhwc_shape,
                ):
                    continue
                if not _can_rewrite_binary_const_to_nhwc(
                    side_input_name=str(add_side_name),
                    target_broadcast_shape=target_nhwc_shape,
                ):
                    continue
                if not _rewrite_binary_const_to_nhwc(
                    op_idx=int(mul_idx),
                    side_input_index=int(mul_side_input_index),
                    side_input_name=str(mul_side_name),
                    target_broadcast_shape=target_nhwc_shape,
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                ):
                    continue
                if not _rewrite_binary_const_to_nhwc(
                    op_idx=int(add_idx),
                    side_input_index=int(add_side_input_index),
                    side_input_name=str(add_side_name),
                    target_broadcast_shape=target_nhwc_shape,
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                ):
                    continue

                if not _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes]):
                    # Keep going even when unchanged; rewrite still valid.
                    pass

                mean_inputs = [str(v) for v in list(mean_op.inputs)]
                mean_inputs[0] = pre_input_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mean_op,
                    new_inputs=mean_inputs,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mean_output_name, None),
                    perm_nchw_to_nhwc,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(add_input_name, None),
                    perm_nchw_to_nhwc,
                )

                canonical_post_output_name = str(post_output_names[0])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_outputs=[canonical_post_output_name],
                )
                for alias_name in post_output_names[1:]:
                    _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name)

                old_add_out_tensor = model_ir.tensors.get(add_output_name, None)
                canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
                if old_add_out_tensor is not None and canonical_tensor is not None:
                    canonical_tensor.dtype = str(old_add_out_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(old_add_out_tensor.quantization)
                    canonical_tensor.shape = [int(v) for v in list(old_add_out_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(old_add_out_tensor.shape_signature)]
                        if old_add_out_tensor.shape_signature is not None
                        else [int(v) for v in list(old_add_out_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        canonical_tensor,
                        perm_nchw_to_nhwc,
                    )

                remove_indices = set(int(v) for v in post_indices)
                # Drop pre transpose only when this rewrite consumed its last user.
                pre_remaining_users = [int(v) for v in pre_users if int(v) != int(mean_idx)]
                if len(pre_remaining_users) == 0:
                    remove_indices.add(int(pre_idx))
                for remove_idx in sorted(remove_indices, reverse=True):
                    del model_ir.operators[int(remove_idx)]

                rewritten += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_mean_mul_add_const_prepost_nhwc_chains": int(rewritten)}
