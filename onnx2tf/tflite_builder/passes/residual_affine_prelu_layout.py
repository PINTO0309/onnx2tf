from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR

def optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Lift NCHW residual blocks back to NHWC when they follow:
      ADD -> MUL(const) -> ADD(const) -> PRELU -> TRANSPOSE(0,2,3,1)

    Inputs to the first ADD must be wrapped by TRANSPOSE(0,3,1,2) adapters.
    If PRELU keeps legacy NCHW consumers, preserve one adapter transpose.
    """
    optimized = 0
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
                return True
            # NCHW per-channel const [1,C,1,1] is mapped once to NHWC [1,1,1,C].
            # Avoid multi-step rotations, which can accidentally produce [1,1,C,1]
            # when target metadata is stale/non-canonical.
            if (
                len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 0
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            ):
                rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_shape, rotated_shape) is not None:
                    nhwc_data = np.asarray(rotated)
            else:
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
                # Unknown target rank-4 shape: avoid speculative re-rotation to keep
                # repeated optimization sweeps idempotent.
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
        if (
            len(side_shape) == 4
            and int(side_shape[0]) == 1
            and int(side_shape[1]) > 0
            and int(side_shape[2]) == 1
            and int(side_shape[3]) == 1
        ):
            rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
            rotated_shape = [int(v) for v in list(rotated.shape)]
            return _broadcast_static_shapes(target_shape, rotated_shape) is not None
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
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for add_idx, add_op in enumerate(model_ir.operators):
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue
            add_out_users = [int(v) for v in consumers.get(add_out_name, []) if int(v) != int(add_idx)]
            if len(add_out_users) != 1:
                continue
            mul_idx = int(add_out_users[0])
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_in0 = str(mul_op.inputs[0])
            mul_in1 = str(mul_op.inputs[1])
            if mul_in0 == add_out_name:
                mul_side_input_index = 1
                mul_side_name = mul_in1
            elif mul_in1 == add_out_name:
                mul_side_input_index = 0
                mul_side_name = mul_in0
            else:
                continue

            mul_out_name = str(mul_op.outputs[0])
            if mul_out_name in model_outputs:
                continue
            mul_out_users = [int(v) for v in consumers.get(mul_out_name, []) if int(v) != int(mul_idx)]
            if len(mul_out_users) != 1:
                continue
            add2_idx = int(mul_out_users[0])
            add2_op = model_ir.operators[int(add2_idx)]
            if str(add2_op.op_type) != "ADD" or len(add2_op.inputs) != 2 or len(add2_op.outputs) != 1:
                continue
            add2_in0 = str(add2_op.inputs[0])
            add2_in1 = str(add2_op.inputs[1])
            if add2_in0 == mul_out_name:
                add2_side_input_index = 1
                add2_side_name = add2_in1
            elif add2_in1 == mul_out_name:
                add2_side_input_index = 0
                add2_side_name = add2_in0
            else:
                continue

            add2_out_name = str(add2_op.outputs[0])
            if add2_out_name in model_outputs:
                continue
            add2_out_users = [int(v) for v in consumers.get(add2_out_name, []) if int(v) != int(add2_idx)]
            if len(add2_out_users) != 1:
                continue
            prelu_idx = int(add2_out_users[0])
            prelu_op = model_ir.operators[int(prelu_idx)]
            if (
                str(prelu_op.op_type) != "PRELU"
                or len(prelu_op.inputs) != 2
                or len(prelu_op.outputs) != 1
                or str(prelu_op.inputs[0]) != add2_out_name
            ):
                continue
            prelu_side_input_index = 1
            prelu_side_name = str(prelu_op.inputs[1])
            prelu_out_name = str(prelu_op.outputs[0])
            if prelu_out_name in model_outputs:
                continue

            prelu_users = [int(v) for v in consumers.get(prelu_out_name, []) if int(v) != int(prelu_idx)]
            if len(prelu_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            for user_idx in prelu_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == prelu_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))
            if len(post_indices) == 0:
                continue

            input_plans: List[Dict[str, Any]] = []
            rewritable_inputs = True
            for add_input_name in [str(v) for v in list(add_op.inputs)]:
                pre_idx = producers.get(str(add_input_name), None)
                if pre_idx is None:
                    rewritable_inputs = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != str(add_input_name)
                    or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                    or str(add_input_name) in model_outputs
                ):
                    rewritable_inputs = False
                    break
                users = [int(v) for v in consumers.get(str(add_input_name), [])]
                if int(add_idx) not in users:
                    rewritable_inputs = False
                    break
                input_plans.append(
                    {
                        "nhwc_input_name": str(pre_op.inputs[0]),
                        "pre_remove_indices": [int(pre_idx)] if set(users) == {int(add_idx)} else [],
                    }
                )
            if not rewritable_inputs or len(input_plans) != 2:
                continue

            target_nhwc_shape = (
                list(model_ir.tensors[str(input_plans[0]["nhwc_input_name"])].shape)
                if str(input_plans[0]["nhwc_input_name"]) in model_ir.tensors
                else None
            )
            chain_index_set = {int(mul_idx), int(add2_idx), int(prelu_idx)}
            if not _can_rewrite_binary_const_to_nhwc(
                side_input_name=str(mul_side_name),
                target_broadcast_shape=target_nhwc_shape,
            ):
                continue
            if not _can_rewrite_binary_const_to_nhwc(
                side_input_name=str(add2_side_name),
                target_broadcast_shape=target_nhwc_shape,
            ):
                continue
            if not _can_rewrite_binary_const_to_nhwc(
                side_input_name=str(prelu_side_name),
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
                op_idx=int(add2_idx),
                side_input_index=int(add2_side_input_index),
                side_input_name=str(add2_side_name),
                target_broadcast_shape=target_nhwc_shape,
                chain_index_set=chain_index_set,
                consumers=consumers,
            ):
                continue
            if not _rewrite_binary_const_to_nhwc(
                op_idx=int(prelu_idx),
                side_input_index=int(prelu_side_input_index),
                side_input_name=str(prelu_side_name),
                target_broadcast_shape=target_nhwc_shape,
                chain_index_set=chain_index_set,
                consumers=consumers,
            ):
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=[str(plan["nhwc_input_name"]) for plan in input_plans],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mul_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add2_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(prelu_out_name, None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=prelu_op,
                new_outputs=[canonical_post_output_name],
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name)

            old_prelu_out_tensor = model_ir.tensors.get(prelu_out_name, None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_prelu_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_prelu_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_prelu_out_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_prelu_out_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_prelu_out_tensor.shape_signature)]
                    if old_prelu_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_prelu_out_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_nchw_to_nhwc,
                )

            if len(legacy_users) > 0:
                keep_post_idx = int(post_indices[0])
                keep_post_op = model_ir.operators[int(keep_post_idx)]
                keep_perm_name = str(keep_post_op.inputs[1])
                keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                if keep_perm_tensor is not None:
                    keep_perm_tensor.data = np.asarray(perm_nhwc_to_nchw, dtype=np.int32)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_inputs=[canonical_post_output_name, keep_perm_name],
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[prelu_out_name],
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = set(int(v) for v in post_remove_indices)
            for plan in input_plans:
                for pre_remove_idx in list(plan.get("pre_remove_indices", [])):
                    remove_indices.add(int(pre_remove_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pre_add_mul_add_prelu_nhwc_chains": int(optimized)}
