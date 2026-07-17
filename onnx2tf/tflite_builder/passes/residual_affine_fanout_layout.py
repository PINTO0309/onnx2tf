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
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Lift NCHW residual add fanouts back to NHWC when they follow:
      [x_nhwc --T(0,3,1,2)--> x_nchw] + [y_nhwc --T(0,3,1,2)--> y_nchw]
      ADD(x_nchw, y_nchw) -> a_nchw
      branch_i: a_nchw -> MUL(const) -> ADD(const) -> T(0,2,3,1) -> out_i_nhwc

    The ADD output may also keep legacy NCHW consumers. In that case, keep one
    adapter transpose: a_nhwc --T(0,3,1,2)--> a_nchw(legacy).
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
            if (
                len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 0
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            ):
                rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(
                    side_data.dtype, copy=False
                )
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_shape, rotated_shape) is not None:
                    nhwc_data = np.asarray(rotated)
            else:
                rotated = np.asarray(side_data)
                for _ in range(3):
                    rotated = np.transpose(rotated, perm_nchw_to_nhwc).astype(
                        side_data.dtype, copy=False
                    )
                    rotated_shape = [int(v) for v in list(rotated.shape)]
                    if (
                        _broadcast_static_shapes(target_shape, rotated_shape)
                        is not None
                    ):
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
                nhwc_data = np.transpose(side_data, perm_nchw_to_nhwc).astype(
                    side_data.dtype, copy=False
                )
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
        if (
            len(side_shape) == 4
            and int(side_shape[0]) == 1
            and int(side_shape[1]) > 0
            and int(side_shape[2]) == 1
            and int(side_shape[3]) == 1
        ):
            rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(
                side_data.dtype, copy=False
            )
            rotated_shape = [int(v) for v in list(rotated.shape)]
            return _broadcast_static_shapes(target_shape, rotated_shape) is not None
        rotated = np.asarray(side_data)
        for _ in range(3):
            rotated = np.transpose(rotated, perm_nchw_to_nhwc).astype(
                side_data.dtype, copy=False
            )
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
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
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
                input_users = [int(v) for v in consumers.get(str(add_input_name), [])]
                if int(add_idx) not in input_users:
                    rewritable_inputs = False
                    break
                input_plans.append(
                    {
                        "nhwc_input_name": str(pre_op.inputs[0]),
                        "pre_remove_indices": [int(pre_idx)]
                        if set(input_users) == {int(add_idx)}
                        else [],
                    }
                )
            if not rewritable_inputs or len(input_plans) != 2:
                continue

            add_out_users = [
                int(v)
                for v in consumers.get(add_out_name, [])
                if int(v) != int(add_idx)
            ]
            if len(add_out_users) == 0:
                continue

            branch_plans: List[Dict[str, Any]] = []
            legacy_users: List[int] = []
            for user_idx in add_out_users:
                mul_op = model_ir.operators[int(user_idx)]
                if (
                    str(mul_op.op_type) != "MUL"
                    or len(mul_op.inputs) != 2
                    or len(mul_op.outputs) != 1
                ):
                    legacy_users.append(int(user_idx))
                    continue
                mul_in0 = str(mul_op.inputs[0])
                mul_in1 = str(mul_op.inputs[1])
                if mul_in0 == add_out_name:
                    mul_data_input_index = 0
                    mul_side_input_index = 1
                    mul_side_name = mul_in1
                elif mul_in1 == add_out_name:
                    mul_data_input_index = 1
                    mul_side_input_index = 0
                    mul_side_name = mul_in0
                else:
                    legacy_users.append(int(user_idx))
                    continue

                mul_out_name = str(mul_op.outputs[0])
                if mul_out_name in model_outputs:
                    legacy_users.append(int(user_idx))
                    continue
                mul_out_users = [
                    int(v)
                    for v in consumers.get(mul_out_name, [])
                    if int(v) != int(user_idx)
                ]
                if len(mul_out_users) != 1:
                    legacy_users.append(int(user_idx))
                    continue
                add2_idx = int(mul_out_users[0])
                add2_op = model_ir.operators[int(add2_idx)]
                if (
                    str(add2_op.op_type) != "ADD"
                    or len(add2_op.inputs) != 2
                    or len(add2_op.outputs) != 1
                ):
                    legacy_users.append(int(user_idx))
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
                    legacy_users.append(int(user_idx))
                    continue

                add2_out_name = str(add2_op.outputs[0])
                if add2_out_name in model_outputs:
                    legacy_users.append(int(user_idx))
                    continue
                add2_out_users = [
                    int(v)
                    for v in consumers.get(add2_out_name, [])
                    if int(v) != int(add2_idx)
                ]
                if len(add2_out_users) != 1:
                    legacy_users.append(int(user_idx))
                    continue
                post_idx = int(add2_out_users[0])
                post_op = model_ir.operators[int(post_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != add2_out_name
                    or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                    or str(post_op.outputs[0]) in model_outputs
                ):
                    legacy_users.append(int(user_idx))
                    continue

                branch_plans.append(
                    {
                        "mul_idx": int(user_idx),
                        "mul_data_input_index": int(mul_data_input_index),
                        "mul_side_input_index": int(mul_side_input_index),
                        "mul_side_name": str(mul_side_name),
                        "mul_out_name": str(mul_out_name),
                        "add2_idx": int(add2_idx),
                        "add2_side_input_index": int(add2_side_input_index),
                        "add2_side_name": str(add2_side_name),
                        "add2_out_name": str(add2_out_name),
                        "post_idx": int(post_idx),
                        "post_output_name": str(post_op.outputs[0]),
                    }
                )

            if len(branch_plans) == 0:
                continue

            planned_pre_remove_count = int(
                sum(
                    len([int(v) for v in list(plan.get("pre_remove_indices", []))])
                    for plan in input_plans
                )
            )
            planned_post_remove_count = int(len(branch_plans))
            planned_adapter_count = int(1 if len(legacy_users) > 0 else 0)
            if int(planned_pre_remove_count + planned_post_remove_count) <= int(
                planned_adapter_count
            ):
                continue

            target_nhwc_shape = (
                list(model_ir.tensors[str(input_plans[0]["nhwc_input_name"])].shape)
                if str(input_plans[0]["nhwc_input_name"]) in model_ir.tensors
                else None
            )
            chain_index_set: set[int] = set()
            for branch in branch_plans:
                chain_index_set.add(int(branch["mul_idx"]))
                chain_index_set.add(int(branch["add2_idx"]))
            constants_rewritable = True
            for branch in branch_plans:
                if not _can_rewrite_binary_const_to_nhwc(
                    side_input_name=str(branch["mul_side_name"]),
                    target_broadcast_shape=target_nhwc_shape,
                ):
                    constants_rewritable = False
                    break
                if not _can_rewrite_binary_const_to_nhwc(
                    side_input_name=str(branch["add2_side_name"]),
                    target_broadcast_shape=target_nhwc_shape,
                ):
                    constants_rewritable = False
                    break
            if not constants_rewritable:
                continue
            constants_applied = True
            for branch in branch_plans:
                if not _rewrite_binary_const_to_nhwc(
                    op_idx=int(branch["mul_idx"]),
                    side_input_index=int(branch["mul_side_input_index"]),
                    side_input_name=str(branch["mul_side_name"]),
                    target_broadcast_shape=target_nhwc_shape,
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                ):
                    constants_applied = False
                    break
                if not _rewrite_binary_const_to_nhwc(
                    op_idx=int(branch["add2_idx"]),
                    side_input_index=int(branch["add2_side_input_index"]),
                    side_input_name=str(branch["add2_side_name"]),
                    target_broadcast_shape=target_nhwc_shape,
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                ):
                    constants_applied = False
                    break
            if not constants_applied:
                continue

            add_nhwc_name = (
                _unique_tensor_name(f"{add_out_name}_nhwc")
                if len(legacy_users) > 0
                else str(add_out_name)
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=[str(plan["nhwc_input_name"]) for plan in input_plans],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_op,
                new_outputs=[str(add_nhwc_name)],
            )

            old_add_tensor = model_ir.tensors.get(add_out_name, None)
            if (
                add_nhwc_name != add_out_name
                and old_add_tensor is not None
                and add_nhwc_name not in model_ir.tensors
            ):
                model_ir.tensors[add_nhwc_name] = TensorIR(
                    name=add_nhwc_name,
                    dtype=str(old_add_tensor.dtype),
                    shape=[int(v) for v in list(old_add_tensor.shape)],
                    shape_signature=(
                        [int(v) for v in list(old_add_tensor.shape_signature)]
                        if old_add_tensor.shape_signature is not None
                        else [int(v) for v in list(old_add_tensor.shape)]
                    ),
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(old_add_tensor.quantization),
                )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_nhwc_name, None),
                perm_nchw_to_nhwc,
            )

            for branch in branch_plans:
                mul_op = model_ir.operators[int(branch["mul_idx"])]
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                data_input_index = int(branch["mul_data_input_index"])
                if int(data_input_index) >= len(mul_inputs):
                    constants_applied = False
                    break
                if str(mul_inputs[int(data_input_index)]) != add_out_name:
                    constants_applied = False
                    break
                mul_inputs[int(data_input_index)] = str(add_nhwc_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                )

                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(branch["mul_out_name"]), None),
                    perm_nchw_to_nhwc,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(branch["add2_out_name"]), None),
                    perm_nchw_to_nhwc,
                )

                add2_op = model_ir.operators[int(branch["add2_idx"])]
                canonical_post_output_name = str(branch["post_output_name"])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=add2_op,
                    new_outputs=[canonical_post_output_name],
                )

                old_add2_out_tensor = model_ir.tensors.get(
                    str(branch["add2_out_name"]), None
                )
                canonical_tensor = model_ir.tensors.get(
                    canonical_post_output_name, None
                )
                if old_add2_out_tensor is not None and canonical_tensor is not None:
                    canonical_tensor.dtype = str(old_add2_out_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(
                        old_add2_out_tensor.quantization
                    )
                    canonical_tensor.shape = [
                        int(v) for v in list(old_add2_out_tensor.shape)
                    ]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(old_add2_out_tensor.shape_signature)]
                        if old_add2_out_tensor.shape_signature is not None
                        else [int(v) for v in list(old_add2_out_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        canonical_tensor,
                        perm_nchw_to_nhwc,
                    )
            if not constants_applied:
                continue

            remove_indices: set[int] = set()
            for plan in input_plans:
                for pre_remove_idx in list(plan.get("pre_remove_indices", [])):
                    remove_indices.add(int(pre_remove_idx))
            for branch in branch_plans:
                remove_indices.add(int(branch["post_idx"]))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                if int(remove_idx) == int(add_idx):
                    continue
                del model_ir.operators[int(remove_idx)]

            if len(legacy_users) > 0:
                adapter_perm_name = _unique_tensor_name(f"{add_out_name}_adapter_perm")
                model_ir.tensors[adapter_perm_name] = TensorIR(
                    name=adapter_perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
                try:
                    add_current_idx = int(model_ir.operators.index(add_op))
                except ValueError:
                    add_current_idx = int(max(0, min(add_idx, len(model_ir.operators))))
                model_ir.operators.insert(
                    int(add_current_idx + 1),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(add_nhwc_name), str(adapter_perm_name)],
                        outputs=[str(add_out_name)],
                    ),
                )

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": int(
            optimized
        )
    }
